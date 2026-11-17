"""Scaled Gradient Projection (Saha & Roy, 2023) for the shared-head CLTR setting.

Representation matrices of every conv/linear layer are collected on samples of the
partitions already learned; gradients are projected away from their principal
subspace, each direction scaled by its importance (a+1)s/(a s + s_max). For the
classifier only the rows of already-learned classes are projected.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SGP:
    def __init__(self, eps=0.9, eps_inc=0.004, scale_coff=1.0, max_samples=1000):
        self.eps, self.eps_inc, self.a, self.max_samples = eps, eps_inc, scale_coff, max_samples
        self.bases, self.importance, self.seen_classes, self.n_tasks = {}, {}, [], 0

    @staticmethod
    def modules(model):
        return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    @staticmethod
    def _rep(module, x):
        if isinstance(module, nn.Conv2d):
            cols = F.unfold(x, module.kernel_size, dilation=module.dilation,
                            padding=module.padding, stride=module.stride)
            return cols.transpose(1, 2).reshape(-1, cols.size(1))
        x = x.reshape(-1, x.size(-1))
        return torch.cat([x, x.new_ones(x.size(0), 1)], dim=1) if module.bias is not None else x

    def _imp(self, s, s_max):
        return ((self.a + 1) * s / (self.a * s + s_max)).clamp(0, 1)

    @torch.no_grad()
    def update_state(self, model, prev_batches, classes):
        """InitState / state update from batches of already-learned partitions only."""
        self.seen_classes = sorted(set(self.seen_classes) | set(classes))
        thr = min(self.eps + self.n_tasks * self.eps_inc, 0.9999)
        self.n_tasks += 1
        cov = {}

        def hook(name, module):
            def fn(_m, inp, _out):
                r = self._rep(module, inp[0].detach()).double()
                cov[name] = cov[name] + r.t() @ r if name in cov else r.t() @ r
            return fn

        handles = [m.register_forward_hook(hook(n, m)) for n, m in self.modules(model)]
        was = model.training
        model.eval()
        seen = 0
        for x, _ in prev_batches:
            x = x[: max(self.max_samples - seen, 0)]
            if x.size(0) == 0:
                break
            model(x)
            seen += x.size(0)
        for h in handles:
            h.remove()
        model.train(was)

        for name, C in cov.items():
            ev, U = torch.linalg.eigh(C)
            ev, U = ev.flip(0).clamp(min=0), U.flip(1)
            total = ev.sum()
            if total <= 0:
                continue
            s = ev.sqrt()
            if name not in self.bases:
                r = min(int((torch.cumsum(ev, 0) / total < thr).sum()) + 1, ev.numel())
                self.bases[name] = U[:, :r].float()
                self.importance[name] = self._imp(s[:r], s[0]).float()
                continue
            B = self.bases[name].double()
            proj = B.t() @ C @ B
            P = torch.eye(C.size(0), dtype=C.dtype, device=C.device) - B @ B.t()
            evr, Ur = torch.linalg.eigh(P @ C @ P)
            evr, Ur = evr.flip(0).clamp(min=0), Ur.flip(1)
            acc, r = float(torch.trace(proj) / total), 0
            for e in (evr / total).tolist():
                if acc >= thr:
                    break
                acc += e
                r += 1
            if r > 0:
                newB, _ = torch.linalg.qr(Ur[:, :r] - B @ (B.t() @ Ur[:, :r]))
                self.bases[name] = torch.cat([B, newB], 1).float()
                self.importance[name] = torch.cat(
                    [self.importance[name], self._imp(evr[:r].sqrt(), s[0]).float()])
            upd = self._imp(torch.sqrt(torch.clamp(torch.diagonal(proj), min=0)), s[0]).float()
            self.importance[name][: upd.numel()] = torch.maximum(
                self.importance[name][: upd.numel()], upd)

    @torch.no_grad()
    def project(self, model):
        for name, m in self.modules(model):
            B = self.bases.get(name)
            if B is None or m.weight.grad is None:
                continue
            lam = self.importance[name]
            g = m.weight.grad.reshape(m.weight.size(0), -1)
            if isinstance(m, nn.Linear) and m.bias is not None:
                gb = m.bias.grad if m.bias.grad is not None else torch.zeros_like(m.bias)
                g = torch.cat([g, gb[:, None]], 1)
                rows = torch.as_tensor(self.seen_classes, device=g.device)
                g[rows] = g[rows] - ((g[rows] @ B) * lam) @ B.t()
                m.weight.grad.copy_(g[:, :-1].reshape_as(m.weight))
                if m.bias.grad is not None:
                    m.bias.grad.copy_(g[:, -1])
            else:
                m.weight.grad.copy_((g - ((g @ B) * lam) @ B.t()).reshape_as(m.weight))
