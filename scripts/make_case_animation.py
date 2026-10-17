#!/usr/bin/env python
"""Render one complete case as a step-by-step animation (GIF + MP4).

Each frame is one tool call from the captured traces: what went in, what came
back, which structure was looked at, and the judgement that followed. Text and
numbers are read from results/trace_seed302.json and results/trace_structure.json;
the structure panels are the figures those steps produced.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
INK = (27, 39, 51)
MUT = (100, 116, 139)
LINE = (226, 230, 234)
ACCENT = (196, 78, 82)
BLUE = (91, 127, 166)
BG = (255, 255, 255)
SOFT = (247, 248, 249)

CJK = "/System/Library/Fonts/Hiragino Sans GB.ttc"
CJK_FALLBACK = "/System/Library/Fonts/Songti.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"


def font(size: int, weight: int = 0, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        return ImageFont.truetype(MONO, size, index=1 if weight else 0)
    try:
        return ImageFont.truetype(CJK, size, index=1 if weight else 0)
    except OSError:
        return ImageFont.truetype(CJK_FALLBACK, size, index=1 if weight else 0)


def wrap(text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """Character-wise wrapping: CJK has no spaces to break on."""
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        if f.getlength(cur + ch) > width and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def compact(value, limit: int = 78) -> str:
    if isinstance(value, dict):
        inner = ", ".join(f"{k}: {compact(v, 24)}" for k, v in list(value.items())[:4])
        s = "{" + inner + ("…" if len(value) > 4 else "") + "}"
    elif isinstance(value, list):
        head = ", ".join(compact(v, 18) for v in value[:3])
        s = "[" + head + (f", …] ({len(value)})" if len(value) > 3 else "]")
    elif isinstance(value, float):
        s = f"{value:.4g}"
    elif isinstance(value, bool):
        s = "true" if value else "false"
    else:
        s = str(value)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[:limit - 1] + "…"


def kv_lines(obj: dict, n: int, val_limit: int = 78) -> list[tuple[str, str]]:
    out = []
    for k, v in obj.items():
        out.append((k, compact(v, val_limit)))
        if len(out) >= n:
            break
    return out


def box(d: ImageDraw.ImageDraw, x, y, w, h, title, colour=BLUE):
    d.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=SOFT, outline=LINE, width=1)
    d.rectangle([x, y + 1, x + 4, y + h - 1], fill=colour)
    d.text((x + 16, y + 11), title, font=font(17, 1), fill=colour)


def slide(step: dict, idx: int, total: int, image: Image.Image | None, section: str) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    # header
    d.text((48, 36), section, font=font(17), fill=MUT)
    d.text((48, 62), f"STEP {idx} / {total}", font=font(31, 1), fill=INK)
    tw = d.textlength(f"STEP {idx} / {total}", font=font(31, 1))
    d.text((48 + tw + 18, 72), step["tool"], font=font(23, 0, mono=True), fill=ACCENT)

    # progress
    seg = (W - 96) / total
    for i in range(total):
        x0 = 48 + i * seg
        d.rounded_rectangle([x0, 112, x0 + seg - 5, 118], radius=3,
                            fill=ACCENT if i < idx else LINE)

    top, bot = 142, 556
    if image is not None:
        tw_box = 420
        box(d, 48, top, tw_box, bot - top, "输入 / 输出")
        yy = top + 44
        f_k, f_v = font(16, 1, mono=True), font(16)
        for src, obj in (("in", step["call"]), ("out", step["result"])):
            for k, v in kv_lines(obj, 3, 44):
                d.text((64, yy), k, font=f_k, fill=BLUE if src == "in" else ACCENT)
                yy += 21
                for ln in wrap(v, f_v, tw_box - 40)[:3]:
                    d.text((64, yy), ln, font=f_v, fill=INK)
                    yy += 20
                yy += 6
                if yy > bot - 40:
                    break
        iw = W - 48 - (48 + tw_box + 26)
        ih = bot - top
        ratio = min(iw / image.width, ih / image.height)
        shown = image.resize((int(image.width * ratio), int(image.height * ratio)), Image.LANCZOS)
        im.paste(shown, (48 + tw_box + 26 + (iw - shown.width) // 2,
                         top + (ih - shown.height) // 2))
    else:
        half = (W - 96 - 26) // 2
        box(d, 48, top, half, bot - top, "输入 · 传给工具的参数")
        box(d, 48 + half + 26, top, half, bot - top, "输出 · 工具返回的内容", ACCENT)
        for x0, obj, nmax in ((48, step["call"], 6), (48 + half + 26, step["result"], 6)):
            yy = top + 46
            for k, v in kv_lines(obj, nmax, 96):
                d.text((x0 + 16, yy), k, font=font(16, 1, mono=True), fill=MUT)
                yy += 22
                for ln in wrap(v, font(17), half - 36)[:3]:
                    d.text((x0 + 16, yy), ln, font=font(17), fill=INK)
                    yy += 22
                yy += 8
                if yy > bot - 44:
                    break

    # analysis band
    note = step.get("note") or ""
    d.rounded_rectangle([48, 576, W - 48, 684], radius=10, fill=(252, 246, 246),
                        outline=(240, 220, 220), width=1)
    d.text((66, 590), "分析", font=font(17, 1), fill=ACCENT)
    yy = 614
    for ln in wrap(note, font(18), W - 150)[:3]:
        d.text((66, yy), ln, font=font(18), fill=INK)
        yy += 24
    return im


def title_slide(n_steps: int) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((64, 214), "先验与约束驱动的蛋白质定向进化智能体", font=font(44, 1), fill=INK)
    d.text((64, 286), "一个完整回合：每一步工具调用、它的输入与输出、以及随之做出的判断",
           font=font(24), fill=MUT)
    d.line([64, 352, 1216, 352], fill=LINE, width=1)
    rows = [
        ("GB1 四位点适应度景观", "149,361 条实测变体作为虚拟评估器"),
        (f"{n_steps} 次真实工具调用", "RCSB 下载、结构计算、反折叠、语言模型、约束引擎、折叠服务"),
        ("结构进入决策", "托管 Boltz-2 折叠单体与复合物；一条界面约束被否决，一条堆积约束被采纳"),
        ("结果", "第 2 轮命中全局最优 FWAA，实测 8.762"),
    ]
    yy = 386
    for k, v in rows:
        d.text((64, yy), k, font=font(22, 1), fill=ACCENT)
        d.text((430, yy), v, font=font(21), fill=INK)
        yy += 46
    d.text((64, 640), "github.com/Sunrich-HT/gb1-directed-evolution-agent",
           font=font(19, 0, mono=True), fill=MUT)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="results/trace_seed302.json")
    ap.add_argument("--structure-trace", default="results/trace_structure.json")
    ap.add_argument("--demo", default="demo/index.html",
                    help="the structure panels are read back out of the demo page")
    ap.add_argument("--out", default="figures/animation/case.gif")
    ap.add_argument("--mp4", default="figures/animation/case.mp4")
    args = ap.parse_args()

    T = json.loads(Path(args.trace).read_text())
    S = json.loads(Path(args.structure_trace).read_text())
    steps = [(s, f"种子 {T['seed']} 的第 {T['round']} 轮 · 决策回路") for s in T["steps"]]
    steps += [(s, "同一条回路里的结构环节 · 托管 Boltz-2") for s in S["steps"]]

    # the four structure panels are already embedded in the demo page; decode them
    # from there instead of keeping a second copy on disk
    html = Path(args.demo).read_text()
    img = {i: Image.open(io.BytesIO(base64.b64decode(m.group(1)))).convert("RGB")
           for i, m in enumerate(
               re.finditer(r'<img src="data:image/jpeg;base64,([^"]+)"', html), 1)}
    if len(img) < 4:
        raise SystemExit(f"expected 4 embedded panels in {args.demo}, found {len(img)}")

    def crop(i, box_frac):
        im = img[i]
        l, t, r, b = box_frac
        return im.crop((int(l * im.width), int(t * im.height),
                        int(r * im.width), int(b * im.height)))

    # which structure panel belongs to which step
    visuals = {
        4: img[1],                               # prior.score
        10: img[2],                              # boltz2.predict (monomers)
        11: crop(3, (0.02, 0.0, 0.50, 0.745)),     # fold_complex, single Fc chain
        12: img[3],                              # check_docking, verdict included
        13: crop(4, (0.0, 0.0, 0.60, 0.84)),     # fold_monomers, shortlist folds
        14: crop(4, (0.605, 0.0, 1.0, 0.80)),     # learn_relation, the correlation
        15: img[4],                              # apply_C9
    }

    frames: list[Image.Image] = []
    durations: list[float] = []
    frames.append(title_slide(len(steps)))
    durations.append(4.2)
    for i, (step, section) in enumerate(steps, 1):
        frames.append(slide(step, i, len(steps), visuals.get(i), section))
        durations.append(4.6 if i in visuals else 3.4)
    durations[-1] = 5.5

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    gif = [f.resize((1040, 585), Image.LANCZOS).convert(
        "P", palette=Image.ADAPTIVE, colors=96) for f in frames]
    gif[0].save(out, save_all=True, append_images=gif[1:],
                duration=[int(d * 1000) for d in durations], loop=0,
                optimize=True, disposal=2)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(frames)} slides)")

    # mp4: hold each slide for its duration at 10 fps
    tmp = out.parent / "_mp4frames"
    tmp.mkdir(exist_ok=True)
    k = 0
    for f, dur in zip(frames, durations):
        for _ in range(int(round(dur * 10))):
            f.save(tmp / f"{k:05d}.png")
            k += 1
    mp4 = Path(args.mp4)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "10",
           "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "23", "-movflags", "+faststart", str(mp4)]
    try:
        subprocess.run(cmd, check=True)
        print(f"wrote {mp4} ({mp4.stat().st_size / 1e6:.2f} MB, {k} frames)")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"mp4 skipped ({exc})")
    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()


if __name__ == "__main__":
    main()
