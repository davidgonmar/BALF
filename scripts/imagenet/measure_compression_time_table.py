import argparse
import json
from pathlib import Path

SUPPORTED = [
    "resnet18",
    "resnet50",
    "mobilenet_v2",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "vit_b_16",
    "deit_b_16",
]
PRETTY = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v2": "MobileNetV2",
    "resnext50_32x4d": "ResNeXt-50 (32×4d)",
    "resnext101_32x8d": "ResNeXt-101 (32×8d)",
    "vit_b_16": "ViT-B/16",
    "deit_b_16": "DeiT-B/16",
}

p = argparse.ArgumentParser()
p.add_argument("--in_json", required=True)
p.add_argument("--out_tex", required=True)
args = p.parse_args()

with open(args.in_json, "r") as f:
    data = json.load(f)

if isinstance(data, dict) and "models" in data:
    rows = []
    for rec in data["models"]:
        t = rec.get("timings", rec)
        t["model"] = rec.get("model", t.get("model"))
        rows.append(t)
else:
    rows = data

rows_by_name = {r["model"]: r for r in rows if "model" in r}
ordered = [rows_by_name[n] for n in SUPPORTED if n in rows_by_name]

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\begin{tabular}{lrrrrr}")
lines.append(r"\hline")
lines.append(
    r"Model & Cache (s) & Fact+Whit (s) & Solver (s) & Total (s) & PeakMem (GiB) \\"
)
lines.append(r"\hline")

for r in ordered:
    m = PRETTY.get(r["model"], r["model"])
    cache = float(r.get("time_activation_cache", float("nan")))
    fac = float(r.get("time_factorization_and_whitening", float("nan")))
    sol = float(r.get("time_solver", float("nan")))
    tot = float(r.get("time_total", float("nan")))
    mem = float(r.get("peak_cuda_memory_bytes", 0.0)) / (1024**3)
    lines.append(
        f"{m} & {cache:.3f} & {fac:.3f} & {sol:.3f} & {tot:.3f} & {mem:.2f} \\\\"
    )

lines.append(r"\hline")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

Path(args.out_tex).parent.mkdir(parents=True, exist_ok=True)
with open(args.out_tex, "w") as f:
    f.write("\n".join(lines))
