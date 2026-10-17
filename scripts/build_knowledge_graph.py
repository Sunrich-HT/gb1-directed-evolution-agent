#!/usr/bin/env python
"""Build the knowledge graph the agent queries: position - mutation - property - fitness.

Node types
  Position      the four mutable sites, carrying their structural measurements
  AminoAcid     the 20 residues, carrying physicochemical properties
  Mutation      a single substitution at one site, e.g. G41L
  Variant       a four-site genotype
  Claim         one row of the curated literature evidence base
  Source        a publication

Edge types
  (Position)-[:WILD_TYPE_IS]->(AminoAcid)
  (AminoAcid)-[:HAS_PROPERTY]->(Property)
  (Mutation)-[:OCCURS_AT]->(Position)
  (Mutation)-[:SUBSTITUTES_TO]->(AminoAcid)
  (Mutation)-[:IMPROVES|HARMS]->(Fitness)      only from labels already paid for
  (Variant)-[:CONTAINS]->(Mutation)
  (Claim)-[:SUPPORTS|REFUTES]->(Mutation)      from the literature, with provenance
  (Claim)-[:FROM]->(Source)
  (Mutation)-[:COMPENSATES]->(Mutation)        published positive-epistasis pairs

Leakage discipline: IMPROVES / HARMS edges are written ONLY for variants whose labels
the campaign has already spent budget on; the unlabelled pool never enters the graph.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gb1agent.data import load_landscape, make_split

POSITIONS = (39, 40, 41, 54)
WT = "VDGV"
AA = "ACDEFGHIKLMNPQRSTVWY"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--structure", default="knowledge/structure_evidence.json")
    ap.add_argument("--evidence", default="knowledge/evidence_base.json")
    ap.add_argument("--records", default="results/v2/records_final.json")
    ap.add_argument("--seed", type=int, default=302)
    ap.add_argument("--out", default="knowledge/knowledge_graph.json")
    args = ap.parse_args()

    struct = json.loads(Path(args.structure).read_text())
    base = json.loads(Path(args.evidence).read_text())
    props = struct["amino_acid_properties"]

    nodes, edges = {}, []

    def node(nid, kind, **attrs):
        nodes[nid] = {"id": nid, "type": kind, **attrs}

    def edge(src, rel, dst, **attrs):
        edges.append({"source": src, "relation": rel, "target": dst, **attrs})

    # ---- amino acids and their properties
    for a in AA:
        node(f"AA:{a}", "AminoAcid",
             volume=props["volume"].get(a), hydropathy=props["hydropathy"].get(a),
             charge=props["charge"].get(a))
        for prop, key, test in (("Hydrophobic", "hydropathy", lambda v: v is not None and v > 0),
                               ("Charged", "charge", lambda v: v not in (None, 0)),
                               ("Bulky", "volume", lambda v: v is not None and v > 160)):
            if test(props[key].get(a)):
                node(f"Property:{prop}", "Property")
                edge(f"AA:{a}", "HAS_PROPERTY", f"Property:{prop}")

    # ---- positions, with what the structures measured
    for p in POSITIONS:
        s = struct["sites"][str(p)]
        node(f"Pos:{p}", "Position",
             wild_type=s["wild_type"], rel_sasa=s["rel_sasa"],
             buried_surface_on_binding_A2=s["interface"]["buried_surface_on_binding_A2"],
             at_interface=s["interface"]["at_interface"],
             packing_headroom_A=s["packing_headroom"]["virtual_cbeta_nearest"][0]["distance"],
             ramachandran=s["ramachandran_region"])
        edge(f"Pos:{p}", "WILD_TYPE_IS", f"AA:{s['wild_type']}")

    # ---- every single substitution as a node
    for i, p in enumerate(POSITIONS):
        wt = WT[i]
        for a in AA:
            if a == wt:
                continue
            mid = f"Mut:{wt}{p}{a}"
            node(mid, "Mutation", site=p, wild_type=wt, substitution=a)
            edge(mid, "OCCURS_AT", f"Pos:{p}")
            edge(mid, "SUBSTITUTES_TO", f"AA:{a}")

    # ---- fitness edges, ONLY from labels the campaign already paid for
    land = load_landscape(args.data)
    variants = land["variant"].to_numpy()
    fitness = land["fitness"].to_numpy(float)
    index = {v: i for i, v in enumerate(variants)}
    split = make_split(land, seed=args.seed)
    records = json.loads(Path(args.records).read_text())
    paid = list(split.initial)
    for r in records:
        if r["arm"] == "prior_constraint" and r["seed"] == args.seed:
            paid += [index[d["variant"]] for d in r["selected"]]
    paid = sorted(set(paid))

    for i in paid:
        v = str(variants[i])
        f = float(fitness[i])
        node(f"Var:{v}", "Variant", measured_fitness=round(f, 3),
             n_mutations=sum(1 for a, b in zip(v, WT) if a != b))
        for k, p in enumerate(POSITIONS):
            if v[k] != WT[k]:
                edge(f"Var:{v}", "CONTAINS", f"Mut:{WT[k]}{p}{v[k]}")

    # per-substitution effect, computed only over those paid labels
    for i, p in enumerate(POSITIONS):
        wt = WT[i]
        for a in AA:
            if a == wt:
                continue
            with_sub = [float(fitness[j]) for j in paid if str(variants[j])[i] == a]
            without = [float(fitness[j]) for j in paid if str(variants[j])[i] != a]
            if len(with_sub) < 5:
                continue
            delta = float(np.mean(with_sub) - np.mean(without))
            rel = "IMPROVES" if delta > 0 else "HARMS"
            edge(f"Mut:{wt}{p}{a}", rel, "Fitness", delta_mean=round(delta, 3),
                 n_measured=len(with_sub), evidence="measured labels already paid for")

    # ---- literature claims, with provenance
    for row in base["evidence"]:
        cid = f"Claim:{row['id']}"
        node(cid, "Claim", statement=row["statement"][:200], scope=row["scope"],
             direction=row.get("direction"), confidence=row.get("confidence"))
        src = (row.get("source") or {}).get("id")
        if src:
            node(f"Source:{src}", "Source", **{k: v for k, v in (row.get("source") or {}).items()
                                               if k in ("id", "title", "year")})
            edge(cid, "FROM", f"Source:{src}")
        for sub in (row.get("substitutions") or []):
            if len(sub) >= 3 and sub[0] in AA and sub[-1] in AA:
                try:
                    site = int(sub[1:-1])
                except ValueError:
                    continue
                if site in POSITIONS:
                    rel = {"beneficial": "SUPPORTS", "deleterious": "REFUTES"}.get(
                        row.get("direction"), "MENTIONS")
                    edge(cid, rel, f"Mut:{sub[0]}{site}{sub[-1]}")
        # published compensating pairs
        subs = [s for s in (row.get("substitutions") or []) if len(s) >= 3]
        if row["scope"] == "pair" and row.get("direction") == "beneficial" and len(subs) == 2:
            try:
                a, b = subs
                edge(f"Mut:{a[0]}{int(a[1:-1])}{a[-1]}", "COMPENSATES",
                     f"Mut:{b[0]}{int(b[1:-1])}{b[-1]}", claim=row["id"])
            except ValueError:
                pass

    kg = {"nodes": list(nodes.values()), "edges": edges,
          "counts": {"nodes": len(nodes), "edges": len(edges),
                     "by_node_type": {t: sum(1 for n in nodes.values() if n["type"] == t)
                                      for t in sorted({n["type"] for n in nodes.values()})},
                     "by_relation": {r: sum(1 for e in edges if e["relation"] == r)
                                     for r in sorted({e["relation"] for e in edges})}},
          "leakage_note": "IMPROVES/HARMS edges use only the %d labels the campaign had "
                          "already paid for (192 start + 3 x 96); the unlabelled pool is "
                          "not represented." % len(paid)}
    Path(args.out).write_text(json.dumps(kg, indent=1))
    print(json.dumps(kg["counts"], indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
