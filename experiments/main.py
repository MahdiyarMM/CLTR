"""python main.py {cifar,clip} --config CONFIG [--seed S]"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from cltr.clip import run_clip
from cltr.train import run_cifar

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("track", choices=["cifar", "clip"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    cfg["seed"] = a.seed
    device = torch.device("cuda")
    res = (run_cifar if a.track == "cifar" else run_clip)(cfg, device)
    print(f"[val]  raw={res['val']['raw']:.2f} la={res['val']['la']:.2f} (a={res['la_coef']})")
    print(f"[test] raw={res['test']['raw']:.2f} la={res['test']['la']:.2f}")
    out = Path("results") / f"{Path(a.config).stem}_s{a.seed}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"config": cfg, **res}, indent=1))
