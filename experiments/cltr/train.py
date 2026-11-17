"""Algorithm 1 (CLTR) on CIFAR-LT with ResNet-32 and SGP.

    theta_1 <- Train(theta_0, D)                 Stage 1
    S_1     <- InitState(theta_1, D^1)           SGP bases + replay memory of D^1
    theta_2 <- Update(theta_1, D^2, S_1)         Tail stage
"""

import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from .data import AutoAugBatches, GPUBatches, ImageStore, build_split, load_cifar
from .models import ResNet32
from .sgp import SGP

LA_GRID = [round(0.05 * i, 2) for i in range(-60, 61)]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def balanced_acc(logits, labels, num_classes):
    pred = logits.argmax(1)
    total = torch.zeros(num_classes, device=labels.device).index_add_(
        0, labels, torch.ones_like(labels, dtype=torch.float))
    correct = torch.zeros(num_classes, device=labels.device).index_add_(
        0, labels, (pred == labels).float())
    present = total > 0
    return 100.0 * float((correct[present] / total[present]).mean())


def fit_la(val_logits, val_labels, log_prior, num_classes, tie=lambda t: -abs(t)):
    """Logit adjustment logits - a*log(prior); a chosen on the validation split."""
    curve = [(t, balanced_acc(val_logits - t * log_prior, val_labels, num_classes)) for t in LA_GRID]
    return max(curve, key=lambda kv: (round(kv[1], 6), tie(kv[0])))[0]


def lr_at(step, total, warmup, base):
    if warmup and step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1.0 + math.cos(math.pi * min((step - warmup) / max(total - warmup, 1), 1.0)))


def train_stage(model, batches, cfg, sgp=None, replay=None, log=print, tag=""):
    """SGD (+SAM) with coupled weight decay; SGP projects the gradients in the Tail stage."""
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=0.0)
    total = cfg["epochs"] * len(batches)
    warm = cfg.get("warmup_epochs", 0) * len(batches)
    rho = cfg.get("sam_rho", 0.0)
    replay_iter = None
    if replay is not None:
        def cycle():
            while True:
                yield from replay
        replay_iter = cycle()
    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        loss_sum, nb = 0.0, 0
        for x, y in batches:
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warm, cfg["lr"])
            if replay_iter is not None:
                rx, ry = next(replay_iter)
                x, y = torch.cat([x, rx]), torch.cat([y, ry])
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if rho > 0:  # SAM: gradient at the perturbed point
                with torch.no_grad():
                    gn = torch.norm(torch.stack([p.grad.norm() for p in params if p.grad is not None]))
                    eps_w = []
                    for p in params:
                        e = rho * p.grad / (gn + 1e-12) if p.grad is not None else None
                        if e is not None:
                            p.add_(e)
                        eps_w.append(e)
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(x), y).backward()
                with torch.no_grad():
                    for p, e in zip(params, eps_w):
                        if e is not None:
                            p.sub_(e)
            with torch.no_grad():
                for p in params:
                    if p.grad is not None:
                        p.grad.add_(p, alpha=cfg["weight_decay"])
            if sgp is not None:
                sgp.project(model)
            opt.step()
            step += 1
            loss_sum += float(loss.detach())
            nb += 1
        if (ep + 1) % 50 == 0 or ep + 1 == cfg["epochs"]:
            log(f"[{tag}] epoch {ep + 1}/{cfg['epochs']} loss={loss_sum / max(nb, 1):.4f}")


@torch.no_grad()
def predict(model, batches):
    was = model.training
    model.eval()
    feats, labels = [], []
    for x, y in batches:
        feats.append(model.features(x))
        labels.append(y)
    model.train(was)
    return model.fc(torch.cat(feats)), torch.cat(labels)


def run_cifar(cfg, device, log=print):
    seed = cfg["seed"]
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    raw = load_cifar(cfg["dataset"])
    train_x, train_y = raw["train_x"], raw["train_y"]
    split = build_split(cfg["dataset"], cfg["imb_factor"], train_y, 2, cfg["val_per_class"])
    C = len(split["counts"])
    store = ImageStore(train_x, train_y, cfg["dataset"], device)
    test_store = ImageStore(raw["test_x"], raw["test_y"], cfg["dataset"], device)
    counts = torch.tensor(split["counts"], dtype=torch.float32, device=device)
    log_prior = torch.log(counts / counts.sum())
    model = ResNet32(C).to(device)

    # Stage 1 on the whole long-tailed training set D
    s1 = cfg["stage1"]
    set_seed(seed)
    b1 = AutoAugBatches(store, train_x, train_y, [i for p in split["part_idx"] for i in p],
                        s1["batch_size"], cfg["dataset"], s1["cutout"], seed=seed)
    train_stage(model, b1, s1, log=log, tag="stage1")
    stage2(model, split, store, cfg, train_y, log)

    val = predict(model, GPUBatches(store, split["val_idx"], 1000, train=False))
    test = predict(model, GPUBatches(test_store, list(range(len(raw["test_y"]))), 1000, train=False))
    a = fit_la(val[0], val[1], log_prior, C)
    return {"la_coef": a,
            "val": {"raw": balanced_acc(*val, C), "la": balanced_acc(val[0] - a * log_prior, val[1], C)},
            "test": {"raw": balanced_acc(*test, C), "la": balanced_acc(test[0] - a * log_prior, test[1], C)}}


def stage2(model, split, store, cfg, train_y, log=print):
    """S_1 from D^1 only (SGP bases + 200 exemplars per Head class), then SGP + replay on D^2."""
    seed, t, u = cfg["seed"], 2, cfg["update"]
    head = split["part_idx"][0]
    prev = np.random.RandomState(20_000 + seed + t).permutation(np.asarray(head))
    set_seed(seed * 100 + t)
    sgp = SGP(**cfg["sgp"])
    sgp.update_state(model, GPUBatches(store, prev, 250, train=False), split["partitions"][0])
    rng = np.random.RandomState(10_000 + seed)
    memory, idx = [], np.asarray(head)
    for c in split["partitions"][0]:
        pool = idx[train_y[idx] == c]
        memory.extend(sorted(rng.choice(pool, size=min(cfg["replay_per_class"], len(pool)),
                                        replace=False).tolist()))
    replay = GPUBatches(store, memory, u["batch_size"], train=True, seed=seed * 100 + t + 50)
    tail = GPUBatches(store, split["part_idx"][1], u["batch_size"], train=True, seed=seed * 100 + t)
    train_stage(model, tail, u, sgp, replay, log=log, tag="stage2")
