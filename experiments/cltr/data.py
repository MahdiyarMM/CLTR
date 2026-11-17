"""Long-tailed CIFAR, held-out validation split and the CLTR partition (Prop. 3.10)."""

import math
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = str(Path(__file__).resolve().parent.parent / "data")
STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def load_cifar(dataset):
    from torchvision.datasets import CIFAR10, CIFAR100
    cls = {"cifar10": CIFAR10, "cifar100": CIFAR100}[dataset]
    out = {}
    for split, train in (("train", True), ("test", False)):
        ds = cls(DATA_ROOT, train=train, download=True)
        out[f"{split}_x"] = np.asarray(ds.data, dtype=np.uint8)
        out[f"{split}_y"] = np.asarray(ds.targets, dtype=np.int64)
        if train:
            out["class_names"] = list(ds.classes)
    return out


def partition_classes(counts, n_parts):
    """Sort classes by cardinality; cut into N consecutive groups with equal local IF."""
    order = sorted(range(len(counts)), key=lambda c: (-counts[c], c))
    n = np.array([counts[c] for c in order], dtype=np.float64)
    C = len(order)
    target = math.log(n[0] / n[-1]) / n_parts
    dp = np.full((n_parts + 1, C + 1), float("inf"))
    arg = np.zeros((n_parts + 1, C + 1), dtype=np.int64)
    dp[0][0] = 0.0
    for k in range(1, n_parts + 1):
        for j in range(k, C + 1):
            for i in range(k - 1, j):
                v = dp[k - 1][i] + (math.log(n[i] / n[j - 1]) - target) ** 2
                if v < dp[k][j] - 1e-12:
                    dp[k][j], arg[k][j] = v, i
    cuts, j = [], C
    for k in range(n_parts, 0, -1):
        j = int(arg[k][j])
        if k > 1:
            cuts.append(j)
    edges = [0] + sorted(cuts) + [C]
    return [order[edges[t]:edges[t + 1]] for t in range(n_parts)]


def build_split(dataset, imb_factor, train_y, n_parts=2, val_per_class=20, val_seed=0):
    """LDAM-DRW LT profile (first n images per class) and a balanced validation set
    drawn from the training images left out of the LT subset."""
    C = NUM_CLASSES[dataset]
    n_max = int(np.sum(train_y == 0))
    counts = [int(n_max * (1.0 / imb_factor) ** (c / (C - 1.0))) for c in range(C)]
    train_idx = []
    for c, k in enumerate(counts):
        train_idx.extend(np.where(train_y == c)[0][:k].tolist())
    rng = np.random.RandomState(val_seed)
    used = np.zeros(len(train_y), dtype=bool)
    used[train_idx] = True
    val_idx = []
    for c in range(C):
        pool = np.where((train_y == c) & ~used)[0]
        take = min(val_per_class, len(pool))
        if take:
            val_idx.extend(sorted(rng.choice(pool, size=take, replace=False).tolist()))
    parts = partition_classes(counts, n_parts)
    tr = np.asarray(train_idx)
    part_idx = [tr[np.isin(train_y[tr], p)].tolist() for p in parts]
    return {"counts": counts, "partitions": parts, "train_idx": train_idx,
            "part_idx": part_idx, "val_idx": val_idx}


class ImageStore:
    """Normalised images of one CIFAR split, resident on the GPU."""

    def __init__(self, x_uint8, y, dataset, device):
        mean, std = STATS[dataset]
        x = torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        self.x = (x - torch.tensor(mean, device=device).view(1, 3, 1, 1)) / \
            torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.y = torch.from_numpy(y).to(device)
        self.device = device


def augment(x, gen):
    """RandomCrop(32, padding=4) + horizontal flip, on the GPU."""
    b = x.size(0)
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))
    ox = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    oy = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ar = torch.arange(32, device=x.device)
    rows = (oy[:, None] + ar)[:, :, None]
    cols = (ox[:, None] + ar)[:, None, :]
    bi = torch.arange(b, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[bi, rows, cols].permute(0, 3, 1, 2)
    flip = torch.rand(b, device=x.device, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out).contiguous()


class GPUBatches:
    def __init__(self, store, indices, batch_size, train, seed=0):
        self.store = store
        self.idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=store.device)
        self.batch_size = batch_size
        self.train = train
        self.gen = torch.Generator(device=store.device)
        self.gen.manual_seed(int(seed))

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __iter__(self):
        idx = self.idx
        if self.train:
            idx = idx[torch.randperm(len(idx), device=idx.device, generator=self.gen)]
        for k in range(len(self)):
            b = idx[k * self.batch_size:(k + 1) * self.batch_size]
            x = self.store.x[b]
            yield (augment(x, self.gen) if self.train else x), self.store.y[b]


class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[1:]
        cy, cx = np.random.randint(h), np.random.randint(w)
        img = img.clone()
        img[:, max(cy - self.size // 2, 0):min(cy + self.size // 2, h),
            max(cx - self.size // 2, 0):min(cx + self.size // 2, w)] = 0.0
        return img


class ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, idx, tf):
        self.x, self.y, self.idx, self.tf = x, y, idx, tf

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        from PIL import Image
        j = int(self.idx[i])
        return self.tf(Image.fromarray(self.x[j])), int(self.y[j])


def _worker_init(_):
    import random
    s = torch.initial_seed() % 2 ** 31
    np.random.seed(s)
    random.seed(s)


class AutoAugBatches:
    """Crop + flip + AutoAugment (CIFAR policy) + Cutout, on CPU workers."""

    def __init__(self, store, x_uint8, y, indices, batch_size, dataset, cutout, seed=0, workers=6):
        from torchvision import transforms as T
        mean, std = STATS[dataset]
        tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                        T.AutoAugment(T.AutoAugmentPolicy.CIFAR10), T.ToTensor(),
                        T.Normalize(mean, std), Cutout(cutout)])
        g = torch.Generator()
        g.manual_seed(int(seed))
        self.device = store.device
        self.loader = torch.utils.data.DataLoader(
            ArrayDataset(x_uint8, y, np.asarray(indices, dtype=np.int64), tf),
            batch_size=batch_size, shuffle=True, num_workers=workers, generator=g,
            worker_init_fn=_worker_init, persistent_workers=True, pin_memory=True)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for x, y in self.loader:
            yield x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
