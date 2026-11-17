"""CLTR (DualPrompt) on CIFAR-100-LT with a CLIP ViT-B/32 (OpenAI weights).

Head task, then Tail task. DualPrompt-style prefix prompts (G-prompts on the first
blocks, a key-selected E-prompt pool on the next ones), LayerNorm adaptation, a cosine
head initialised from CLIP text embeddings, balanced-softmax loss; the CL state after
the Head task is a replay memory of Head exemplars and the frozen Head-task model (LwF).
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import DATA_ROOT, build_split
from .train import balanced_acc, fit_la, set_seed

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# EPrompt: adapted from https://github.com/JH-LEE-KR/dualprompt-pytorch (Apache-2.0)
class EPrompt(nn.Module):
    def __init__(self, length, embed_dim, pool_size, top_k, num_layers, num_heads):
        super().__init__()
        self.top_k, self.pool_size = top_k, pool_size
        self.prompt = nn.Parameter(torch.randn(num_layers, 2, pool_size, length, num_heads,
                                               embed_dim // num_heads))
        nn.init.uniform_(self.prompt, -1, 1)
        self.prompt_key = nn.Parameter(torch.randn(pool_size, embed_dim))
        nn.init.uniform_(self.prompt_key, -1, 1)

    @staticmethod
    def l2_normalize(x, dim):
        return x * torch.rsqrt(torch.maximum(torch.sum(x ** 2, dim=dim, keepdim=True),
                                             torch.tensor(1e-12, device=x.device)))

    def forward(self, x_embed, cls_features):
        key_norm = self.l2_normalize(self.prompt_key, dim=-1)
        x_norm = self.l2_normalize(cls_features, dim=-1)
        _, idx = torch.topk(torch.matmul(key_norm, x_norm.t()).t(), k=self.top_k, dim=1)
        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
        if prompt_id.shape[0] < self.pool_size:  # batch-wise prompt selection
            pad = self.pool_size - prompt_id.shape[0]
            prompt_id = torch.cat([prompt_id, torch.full((pad,), torch.min(idx.flatten()),
                                                         device=prompt_id.device)])
            id_counts = torch.cat([id_counts, torch.full((pad,), 0, device=id_counts.device)])
        _, major = torch.topk(id_counts, k=self.top_k)
        idx = prompt_id[major].expand(x_embed.shape[0], -1).contiguous()
        raw = self.prompt[:, :, idx]
        L, dual, B, k, length, H, D = raw.shape
        reduce_sim = torch.sum(key_norm[idx] * x_norm.unsqueeze(1)) / x_embed.shape[0]
        return raw.reshape(L, B, dual, k * length, H, D), reduce_sim


class PrefixBlock(nn.Module):
    """OpenCLIP residual block whose attention accepts prefix keys/values."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        self.num_heads = block.attn.num_heads
        self.head_dim = block.attn.embed_dim // self.num_heads

    def _attn(self, x, prompt):
        b, n, c = x.shape
        attn = self.block.attn
        w = attn.in_proj_weight.chunk(3, dim=0)
        bias = attn.in_proj_bias.chunk(3, dim=0)
        q, k, v = (F.linear(x, wi, bi).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
                   for wi, bi in zip(w, bias))
        if prompt is not None:
            p = prompt.permute(1, 0, 3, 2, 4).contiguous()
            k, v = torch.cat([p[0], k], dim=2), torch.cat([p[1], v], dim=2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, c)
        return attn.out_proj(out)

    def forward(self, x, prompt=None):
        blk = self.block
        x = x + blk.ls_1(self._attn(blk.ln_1(x), prompt))
        return x + blk.ls_2(blk.mlp(blk.ln_2(x)))


class CLTRCLIP(nn.Module):
    def __init__(self, clip_model, class_names, tokenizer, cfg, device):
        super().__init__()
        self.visual = clip_model.visual
        for p in self.visual.parameters():
            p.requires_grad_(False)
        for m in self.visual.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad_(True)
        width = int(self.visual.conv1.out_channels)
        heads = self.visual.transformer.resblocks[0].attn.num_heads
        self.g_layers, self.e_layers = [0, 1], [2, 3, 4]
        self.blocks = nn.ModuleList([PrefixBlock(b) for b in self.visual.transformer.resblocks])
        self.g_prompt = nn.Parameter(torch.randn(len(self.g_layers), 2, 5, heads, width // heads) * 0.02)
        self.e_prompt = EPrompt(5, width, 10, 1, len(self.e_layers), heads)
        from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
        with torch.no_grad():
            cols = []
            for c in class_names:
                tok = tokenizer([t(c.replace("_", " ")) for t in OPENAI_IMAGENET_TEMPLATES]).to(device)
                e = clip_model.encode_text(tok).float()
                e = (e / e.norm(dim=-1, keepdim=True)).mean(0)
                cols.append(e / e.norm().clamp(min=1e-8))
        self.head_weight = nn.Parameter(torch.stack(cols).clone())
        self.logit_scale = nn.Parameter(clip_model.logit_scale.detach().clone().float())

    def forward(self, x, return_aux=False):
        vis = self.visual
        h = vis._embeds(x)
        with torch.no_grad():
            query = vis._pool(vis.transformer(vis._embeds(x)))[0].to(h.dtype)
        e_prompt, reduce_sim = self.e_prompt(h, query)
        gi = ei = -1
        for i, blk in enumerate(self.blocks):
            if i in self.g_layers:
                gi += 1
                h = blk(h, self.g_prompt[gi].unsqueeze(0).expand(h.shape[0], -1, -1, -1, -1))
            elif i in self.e_layers:
                ei += 1
                h = blk(h, e_prompt[ei])
            else:
                h = blk(h)
        pooled = (vis._pool(h)[0] @ vis.proj).float()
        f = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        w = self.head_weight / self.head_weight.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        logits = self.logit_scale.exp().clamp(max=100.0) * (f @ w.t())
        return (logits, reduce_sim) if return_aux else logits


def transforms(train):
    from torchvision import transforms as T
    bicubic = T.InterpolationMode.BICUBIC
    norm = [T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)]
    if not train:
        return T.Compose([T.Resize(224, interpolation=bicubic)] + norm)
    return T.Compose([T.RandomCrop(32, padding=4, padding_mode="reflect"), T.RandomHorizontalFlip(),
                      T.RandAugment(num_ops=2, magnitude=9), T.Resize(224, interpolation=bicubic)]
                     + norm + [T.RandomErasing(p=0.25, scale=(0.02, 0.2))])


class Indexed(Dataset):
    def __init__(self, base, indices, tf):
        self.base, self.idx, self.tf = base, list(indices), tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        img, y = self.base[self.idx[i]]
        return self.tf(img), y


@torch.no_grad()
def predict(model, loader, device):
    """Logits averaged over the image and its horizontal flip."""
    model.eval()
    lg, lb = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        lg.append((0.5 * (model(x).float() + model(torch.flip(x, dims=[3])).float())).cpu())
        lb.append(y)
    return torch.cat(lg), torch.cat(lb)


def train_task(model, loader, device, cfg, task_classes, log_prior, log, teacher=None, replay=None):
    head, backbone = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (backbone if n.startswith("visual.") or n.startswith("blocks.") else head).append(p)
    opt = torch.optim.AdamW([{"params": head, "lr": cfg["lr"]}, {"params": backbone, "lr": cfg["lr"] / 10}],
                            lr=cfg["lr"], weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    total, warm = max(len(loader) * cfg["epochs"], 1), 200
    mask = torch.tensor(np.setdiff1d(np.arange(100), task_classes), dtype=torch.long, device=device)
    lp = (cfg["bs_tau"] * log_prior).to(device)
    step = 0
    model.train()
    for ep in range(cfg["epochs"]):
        tot, seen = 0.0, 0
        for x, y in loader:
            for g, b in zip(opt.param_groups, base):
                g["lr"] = (b * (step + 1) / warm if step < warm else
                           b * 0.5 * (1 + math.cos(math.pi * min((step - warm) / max(total - warm, 1), 1.0))))
            step += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, reduce_sim = model(x, return_aux=True)
            loss = F.cross_entropy(logits.index_fill(1, mask, float("-inf")) + lp, y)
            if replay is not None:
                rx, ry = next(replay)
                rx, ry = rx.to(device, non_blocking=True), ry.to(device, non_blocking=True)
                loss = loss + F.cross_entropy(model(rx) + lp, ry)
            if teacher is not None:
                with torch.no_grad():
                    tl = teacher(x)
                T = 2.0
                loss = loss + cfg["distill_weight"] * T * T * F.kl_div(
                    F.log_softmax(logits / T, 1), F.softmax(tl / T, 1), reduction="batchmean")
            loss = loss - 0.5 * reduce_sim
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            tot += float(loss) * y.size(0)
            seen += y.size(0)
        log(f"  epoch {ep + 1}/{cfg['epochs']} loss={tot / max(seen, 1):.4f}")


def run_clip(cfg, device, log=print):
    import open_clip
    from torchvision.datasets import CIFAR100

    set_seed(cfg["seed"])
    train_raw = CIFAR100(DATA_ROOT, train=True, download=True)
    test_raw = CIFAR100(DATA_ROOT, train=False, download=True)
    ty = np.asarray(train_raw.targets)
    split = build_split("cifar100", cfg["imb_factor"], ty, 2, cfg["val_per_class"])
    head_cls, tail_cls = split["partitions"]
    head_idx, tail_idx = split["part_idx"]
    tf_train, tf_eval = transforms(True), transforms(False)

    def loader(base, idx, bs=64, shuffle=True, tf=tf_train):
        return DataLoader(Indexed(base, idx, tf), batch_size=bs, shuffle=shuffle, num_workers=8,
                          pin_memory=True)

    val_loader = loader(train_raw, split["val_idx"], 256, False, tf_eval)
    counts = torch.tensor(split["counts"], dtype=torch.float32)
    log_prior = torch.log(counts / counts.sum())
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    clip_model = clip_model.to(device)
    model = CLTRCLIP(clip_model, train_raw.classes, tokenizer, cfg, device).to(device)

    log(f"=== task 1: Head ({len(head_cls)} classes) ===")
    train_task(model, loader(train_raw, head_idx), device, cfg, head_cls, log_prior, log)
    rng = np.random.RandomState(10_000 + cfg["seed"])
    hi, mem = np.asarray(head_idx), []
    for c in head_cls:
        pool = hi[ty[hi] == c]
        mem.extend(rng.choice(pool, size=min(cfg["replay_per_class"], len(pool)), replace=False).tolist())
    rl = loader(train_raw, mem, bs=32)

    def cycle():
        while True:
            yield from rl

    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    log(f"=== task 2: Tail ({len(tail_cls)} classes), replay={len(mem)} ===")
    train_task(model, loader(train_raw, tail_idx), device, cfg, tail_cls, log_prior, log,
               teacher=teacher, replay=cycle())

    vl, vb = predict(model, val_loader, device)
    tl, tb = predict(model, loader(test_raw, range(len(test_raw)), 256, False, tf_eval), device)
    a = fit_la(vl, vb, log_prior, 100, tie=lambda t: -t)
    return {"la_coef": a,
            "val": {"raw": balanced_acc(vl, vb, 100), "la": balanced_acc(vl - a * log_prior, vb, 100)},
            "test": {"raw": balanced_acc(tl, tb, 100), "la": balanced_acc(tl - a * log_prior, tb, 100)}}
