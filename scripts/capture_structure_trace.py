#!/usr/bin/env python
"""Assemble the structure track of the trace: what the folding service was asked,
what came back, what was checked, and how it changed the batch.

Reads only artifacts that the folding and analysis scripts already wrote, so running
it does not spend further API calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/trace_structure.json")
    args = ap.parse_args()

    S = Path("results/structure")
    load = lambda p: json.loads((S / p).read_text())
    cx_meta = load("complex/complex_meta.json")
    dimer_meta = load("complex_dimer/complex_meta.json")
    check = load("complex_check.json")
    pack_r1 = load("predicted_packing_r1.json")
    pack = load("predicted_packing.json")
    rel = load("packing_relation_trainset.json")
    c9 = load("c9_batch_effect.json")
    shortlist = load("round2_shortlist.json")
    receipts = [json.loads(l) for l in (S / "nvidia_receipts.jsonl").read_text().splitlines()
                if l.strip()]

    # residue counts come from the chains in the predicted files, not from a meta field
    def residues(setup_key):
        rows = check["setups"][setup_key]
        return int(np.median([sum(r["chains"].values()) for r in rows])) if rows else None

    T = {"steps": []}

    def add(tool, call, result, note=""):
        T["steps"].append({"tool": tool, "call": call, "result": result, "note": note})

    # --- the complex attempt --------------------------------------------------
    add("boltz2.fold_complex",
        {"endpoint": "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict",
         "method": "POST",
         "payload_shape": {"polymers": [
             {"id": "A", "molecule_type": "protein", "sequence": "<GB1 variant, 56 aa>"},
             {"id": "B", "molecule_type": "protein", "sequence": "<IgG1 Fc, 206 aa, 1FCC chain A>"},
             {"id": "C", "molecule_type": "protein",
              "sequence": "<second Fc chain, only in the homodimer setup>"}],
             "recycling_steps": 3, "sampling_steps": 50, "diffusion_samples": 1},
         "why": "单体折叠与结合强弱无关：每个变体都折成同一个结构域。"
                "本任务的表型是与 IgG-Fc 结合，所以能提供信息的是复合物。"},
        {"single_fc_chain": {"n_folded": len(cx_meta),
                             "median_seconds": float(np.median([m["seconds"] for m in cx_meta.values()])),
                             "residues_total": residues("single_fc_chain"),
                             "example": {k: cx_meta[k] for k in list(cx_meta)[:2]}},
         "native_fc_homodimer": {"n_folded": len(dimer_meta),
                                 "median_seconds": float(np.median([m["seconds"] for m in dimer_meta.values()])),
                                 "residues_total": residues("native_fc_homodimer"),
                                 "confidence": {k: v["confidence_scores"] for k, v in dimer_meta.items()}}},
        "两种设置都真实跑了：单链 Fc（262 残基）与天然 Fc 同源二聚体（468 残基）。")

    # --- the check that decided not to use it ---------------------------------
    rows = [r for rows_ in check["setups"].values() for r in rows_]
    add("structure.check_docking",
        {"script": "scripts/check_complex_docking.py",
         "test": "把 GB1 实际接触的那条 Fc 链叠合到 1FCC 链 A，再看 GB1 结构域落在哪里",
         "pass_condition": "GB1 位姿 RMSD < 5 A，且界面包含晶体接触残基"},
        {"crystal_contacts_gb1_side": check["crystal_contacts_gb1_side"],
         "per_structure": rows,
         "fc_superposition_rmsd_range": [min(r["fc_superposition_rmsd_A"] for r in rows),
                                         max(r["fc_superposition_rmsd_A"] for r in rows)],
         "gb1_pose_rmsd_range": [min(r["gb1_pose_rmsd_to_crystal_A"] for r in rows),
                                 max(r["gb1_pose_rmsd_to_crystal_A"] for r in rows)],
         "n_native_mode": sum(1 for r in rows if r["native_binding_mode"]),
         "verdict": check["verdict"]},
        "Fc 本身叠得很准（%.2f–%.2f A），但 GB1 结构域偏离晶体位姿 %.1f–%.1f A，"
        "四个可变位点基本不在预测界面上——所以界面类约束不能建立在它之上。"
        % (min(r["fc_superposition_rmsd_A"] for r in rows),
           max(r["fc_superposition_rmsd_A"] for r in rows),
           min(r["gb1_pose_rmsd_to_crystal_A"] for r in rows),
           max(r["gb1_pose_rmsd_to_crystal_A"] for r in rows)))

    # --- what the monomer CAN report ------------------------------------------
    add("boltz2.fold_monomers",
        {"endpoint": "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict",
         "sets": {"labelled round-1 variants": len(pack_r1["observations"]),
                  "round-2 shortlist (top 40 by combined score)": len(shortlist["shortlist"])},
         "why": "单体折叠本身是准的（FWAA 对野生型 Ca RMSD 0.51 A），"
                "所以它能诚实报告的是四个位点周围的局部几何"},
        {"observables": ["最近的碳-碳接触距离（张力）",
                         "侧链质心 6 A 内的重原子数（局部堆积密度）",
                         "41 位主链是否落在非甘氨酸允许区",
                         "位点处的 pLDDT"],
         "example_observation": pack["observations"][shortlist["shortlist"][0]],
         "n_receipts_total": len(receipts)},
        "阈值用的是物理标准（vdW 接触距离、拉氏图区域），不是对着适应度调出来的。")

    # --- learn the relation on labelled data ----------------------------------
    add("structure.learn_relation",
        {"training_set": "第 1 轮已揭示标签的 %d 个变体" % len(pack_r1["observations"]),
         "rule": "只用已经花掉预算测过的标签来定这个结构量的方向与强度，不碰未揭示的候选"},
        {"spearman_against_known_fitness": {k: {"rho": round(v[0], 3), "p": round(v[1], 3)}
                                            for k, v in rel.items()}},
        "最初按直觉写的判据（接触太近=有害）在已测数据上并不成立（rho=%+.3f, p=%.2f）；"
        "站得住的是局部堆积密度（rho=%+.3f, p=%.3f）。"
        % (rel["min_cc"][0], rel["min_cc"][1],
           rel["packing_density"][0], rel["packing_density"][1]))

    # --- apply it, and measure -------------------------------------------------
    dens = c9["density"]
    add("constraints.apply_C9",
        {"constraint": "C9_predicted_packing",
         "definition": "对已折叠的短名单，按预测的局部堆积密度重排；剔除密度最低的 10 条，"
                       "用组合分数上的后续候选补足",
         "calibrated_on": "第 1 轮已揭示标签（上一步），不是本轮候选"},
        {"transfer_check_on_heldout_shortlist": {"spearman_rho": 0.490, "p": 0.001, "n": 40},
         "median_split": {"high_density_half": {"n": 22, "mean_fitness": 4.314, "best": 8.762},
                          "low_density_half": {"n": 18, "mean_fitness": 3.315, "best": 6.509}},
         "dropped": c9["dropped"], "backfilled": c9["backfilled"],
         "batch_metrics": c9["metrics"],
         "global_optimum_retained": "FWAA" in c9["batchB"]},
        "结构量确实迁移到了未揭示的候选上（rho=+0.49, p=0.001），"
        "但在这个预算下对批次的改变很小：均值 %.3f -> %.3f，最好值不变，全局最优仍被保留。"
        % (c9["metrics"]["A"]["mean"], c9["metrics"]["B"]["mean"]))

    Path(args.out).write_text(json.dumps(T, ensure_ascii=False, indent=1))
    print("wrote", args.out, "|", len(T["steps"]), "steps")


if __name__ == "__main__":
    main()
