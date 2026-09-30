"""SAMADHAN GPU LP engine: restarted primal-dual hybrid gradient (PDLP), written from scratch.

Only sparse matrix-vector products and element-wise vector operations are used, so the whole
iteration runs on the GPU (PyTorch CUDA tensors) with no factorisation and no external solver.

Algorithm (Applegate et al., "Practical Large-Scale LP using PDHG", NeurIPS 2021; Lu & Yang,
cuPDLP, 2023):
  * Ruiz + Pock-Chambolle diagonal preconditioning
  * PDHG with adaptive step size and primal weight
  * weighted iterate averaging with KKT-based adaptive restarts
  * relative KKT termination on the unscaled problem
"""
import math
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import warnings
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

from .lp import LP


@dataclass
class Result:
    status: str
    x: np.ndarray
    y: np.ndarray
    primal_obj: float
    dual_obj: float
    rel_gap: float
    rel_primal_res: float
    rel_dual_res: float
    iterations: int
    matvecs: int
    restarts: int
    solve_time: float
    setup_time: float
    device: str


# --------------------------------------------------------------------------- preconditioning
def _scale(K: sp.csr_matrix, ruiz_iters=10):
    m, n = K.shape
    dr = np.ones(m)
    dc = np.ones(n)
    Ks = K.copy().tocsr()
    Ks.data = Ks.data.astype(np.float64)
    for _ in range(ruiz_iters):
        absK = abs(Ks)
        rmax = np.asarray(absK.max(axis=1).todense()).ravel() if m else np.zeros(0)
        cmax = np.asarray(absK.max(axis=0).todense()).ravel()
        rs = 1.0 / np.sqrt(np.where(rmax > 0, rmax, 1.0))
        cs = 1.0 / np.sqrt(np.where(cmax > 0, cmax, 1.0))
        Ks = sp.diags(rs) @ Ks @ sp.diags(cs)
        dr *= rs
        dc *= cs
    absK = abs(Ks)
    rsum = np.asarray(absK.sum(axis=1)).ravel()
    csum = np.asarray(absK.sum(axis=0)).ravel()
    rs = 1.0 / np.sqrt(np.where(rsum > 0, rsum, 1.0))
    cs = 1.0 / np.sqrt(np.where(csum > 0, csum, 1.0))
    Ks = (sp.diags(rs) @ Ks @ sp.diags(cs)).tocsr()
    return Ks, dr * rs, dc * cs


def _to_torch_csr(A: sp.csr_matrix, device, dtype):
    A = A.tocsr()
    A.sort_indices()
    return torch.sparse_csr_tensor(  # noqa: sparse CSR is beta in torch but stable for mv

        torch.from_numpy(A.indptr.astype(np.int64)),
        torch.from_numpy(A.indices.astype(np.int64)),
        torch.from_numpy(A.data.astype(np.float64)),
        size=A.shape, dtype=dtype, device=device)


# --------------------------------------------------------------------------- solver
class PDLP:
    def __init__(self, lp: LP, device="cuda", dtype=torch.float64):
        t0 = time.perf_counter()
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device, self.dtype, self.lp = device, dtype, lp
        m, n = lp.K.shape
        self.m, self.n, self.n_eq = m, n, lp.n_eq

        Ks, dr, dc = _scale(lp.K)
        T = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)
        self.K = _to_torch_csr(Ks, device, dtype)
        self.KT = _to_torch_csr(Ks.T.tocsr(), device, dtype)
        self.dr, self.dc = T(dr), T(dc)
        self.c = T(lp.c * dc)
        self.q = T(lp.q * dr)
        self.l = T(lp.l / dc)
        self.u = T(lp.u / dc)
        self.l_fin = torch.isfinite(self.l)
        self.u_fin = torch.isfinite(self.u)
        self.l0 = torch.where(self.l_fin, self.l, torch.zeros_like(self.l))
        self.u0 = torch.where(self.u_fin, self.u, torch.zeros_like(self.u))
        self.ineq = torch.arange(m, device=device) >= lp.n_eq
        Q = getattr(lp, "Q", None)
        self.Q = None
        if Q is not None and Q.nnz:
            self.Q = _to_torch_csr((sp.diags(dc) @ sp.csr_matrix(Q) @ sp.diags(dc)).tocsr(), device, dtype)
        self.c_norm = float(np.linalg.norm(lp.c))
        self.q_norm = float(np.linalg.norm(lp.q))
        if device == "cuda":
            torch.cuda.synchronize()
        self.setup_time = time.perf_counter() - t0

    # ------------------------------------------------------------------ helpers
    def _mv(self, A, v):
        return torch.mv(A, v)

    def _proj_x(self, x):
        return torch.minimum(torch.maximum(x, self.l), self.u)

    def _proj_y(self, y):
        if self.n_eq < self.m:
            y = torch.where(self.ineq, torch.clamp_min(y, 0.0), y)
        return y

    def _norm_K(self, iters=40):
        """Largest singular value of the scaled K by power iteration on K'K."""
        v = torch.ones(self.n, dtype=self.dtype, device=self.device)
        v /= torch.linalg.vector_norm(v)
        s = torch.zeros((), dtype=self.dtype, device=self.device)
        for _ in range(iters):
            w = self._mv(self.KT, self._mv(self.K, v))
            s = torch.linalg.vector_norm(w)
            v = w / s
        return math.sqrt(float(s))

    def _kkt(self, x, y, Kx, KTy, Qx=None):
        """Residual pieces in scaled space plus relative (unscaled) errors. With Q: objective 0.5 x'Qx + c'x,
        reduced costs Qx + c - K'y, Wolfe dual objective q'y - 0.5 x'Qx + bound terms."""
        r = self.q - Kx
        rp = r.clone()
        if self.n_eq < self.m:
            rp[self.n_eq:] = torch.clamp_min(r[self.n_eq:], 0.0)
        lam = self.c - KTy if Qx is None else self.c + Qx - KTy
        lam_pos = torch.clamp_min(lam, 0.0)
        lam_neg = torch.clamp_min(-lam, 0.0)
        rd = torch.where(self.l_fin, torch.zeros_like(lam), lam_pos) + \
             torch.where(self.u_fin, torch.zeros_like(lam), lam_neg)
        quad = 0.5 * torch.dot(x, Qx) if Qx is not None else torch.zeros((), dtype=x.dtype, device=x.device)
        pobj = torch.dot(self.c, x) + quad
        dobj = torch.dot(self.q, y) + torch.dot(self.l0, torch.where(self.l_fin, lam_pos, 0.0)) \
            - torch.dot(self.u0, torch.where(self.u_fin, lam_neg, 0.0)) - quad
        vals = torch.stack([
            torch.linalg.vector_norm(rp), torch.linalg.vector_norm(rd),
            torch.linalg.vector_norm(rp / self.dr), torch.linalg.vector_norm(rd / self.dc),
            pobj, dobj]).tolist()
        rp_s, rd_s, rp_o, rd_o, p, d = vals
        p += self.lp.obj_const
        d += self.lp.obj_const
        gap = abs(p - d)
        rel = (rp_o / (1 + self.q_norm), rd_o / (1 + self.c_norm), gap / (1 + abs(p) + abs(d)))
        return dict(rp=rp_s, rd=rd_s, gap=gap, p=p, d=d, rel=rel)

    @staticmethod
    def _kkt_w(k, w):
        return math.sqrt((w * k["rp"]) ** 2 + (k["rd"] / w) ** 2 + k["gap"] ** 2)

    # ------------------------------------------------------------------ constant-step path (LP and QP)
    def _step_inplace(self, S):
        """One constant-step PDHG iteration, entirely in place on the static buffers in S.
        With a quadratic objective the primal step uses the gradient Qx + c - K'y (linearised PDHG)."""
        grad = self.c - S["KTy"]
        if self.Q is not None:
            grad = grad + S["Qx"]
        xn = torch.clamp(S["x"] - S["tau"] * grad, min=self.l, max=self.u)
        Kxn = torch.mv(self.K, xn)
        v = S["y"] + S["sigma"] * (self.q - 2 * Kxn + S["Kx"])
        yn = torch.where(self.ineq, torch.clamp_min(v, 0.0), v)
        KTyn = torch.mv(self.KT, yn)
        S["x"].copy_(xn); S["Kx"].copy_(Kxn); S["y"].copy_(yn); S["KTy"].copy_(KTyn)
        S["k"].add_(1.0)
        a = 1.0 / S["k"]
        S["xs"].lerp_(S["x"], a); S["ys"].lerp_(S["y"], a)
        S["Kxs"].lerp_(S["Kx"], a); S["KTys"].lerp_(S["KTy"], a)
        if self.Q is not None:
            S["Qx"].copy_(torch.mv(self.Q, xn))
            S["Qxs"].lerp_(S["Qx"], a)

    def _norm_Q(self, iters=40):
        v = torch.ones(self.n, dtype=self.dtype, device=self.device)
        v /= torch.linalg.vector_norm(v)
        s = torch.zeros((), dtype=self.dtype, device=self.device)
        for _ in range(iters):
            w = torch.mv(self.Q, v)
            s = torch.linalg.vector_norm(w)
            if float(s) == 0.0:
                return 0.0
            v = w / s
        return float(s)

    def _steps(self, S, w, eta, nK, LQ):
        """Primal/dual step sizes for primal weight w; with Q, tau satisfies 1/tau >= LQ/2 + sigma*||K||^2."""
        if self.Q is None:
            S["tau"].fill_(eta / w); S["sigma"].fill_(eta * w)
        else:
            sigma = eta * w
            S["sigma"].fill_(sigma); S["tau"].fill_(0.998 / (LQ / 2 + sigma * nK * nK))

    def _solve_graph(self, tol, max_iter, time_limit, eval_every, verbose):
        dev, dt = self.device, self.dtype
        cuda = dev == "cuda"
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        z = lambda k: torch.zeros(k, dtype=dt, device=dev)
        S = dict(x=self._proj_x(z(self.n)), y=z(self.m))
        S.update(Kx=torch.mv(self.K, S["x"]), KTy=torch.mv(self.KT, S["y"]),
                 xs=z(self.n), ys=z(self.m), Kxs=z(self.m), KTys=z(self.n),
                 tau=torch.zeros((), dtype=dt, device=dev), sigma=torch.zeros((), dtype=dt, device=dev),
                 k=torch.zeros((), dtype=dt, device=dev))
        if self.Q is not None:
            S.update(Qx=torch.mv(self.Q, S["x"]), Qxs=z(self.n))
        cn = float(torch.linalg.vector_norm(self.c)); qn = float(torch.linalg.vector_norm(self.q))
        w = cn / qn if cn > 1e-10 and qn > 1e-10 else 1.0
        nK = max(self._norm_K(), 1e-12)
        LQ = self._norm_Q() if self.Q is not None else 0.0
        eta = 0.998 / nK
        self._steps(S, w, eta, nK, LQ)
        cur = ("x", "y", "Kx", "KTy") + (("Qx",) if self.Q is not None else ())
        avg = ("xs", "ys", "Kxs", "KTys") + (("Qxs",) if self.Q is not None else ())
        kkt = lambda nm: self._kkt(S[nm[0]], S[nm[1]], S[nm[2]], S[nm[3]], S[nm[4]] if len(nm) > 4 else None)

        if cuda:   # warm up on a side stream, then capture eval_every iterations as one graph
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._step_inplace(S)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(eval_every):
                    self._step_inplace(S)
            k_total = k_since = 3
            run_block = g.replay
        else:
            k_total = k_since = 0
            def run_block():
                for _ in range(eval_every):
                    self._step_inplace(S)

        x_rs, y_rs = S["x"].clone(), S["y"].clone()
        k0 = kkt(cur)
        kkt_rs, kkt_prev_cand = self._kkt_w(k0, w), float("inf")
        restarts, status = 0, "iteration_limit"
        best = (k0, S["x"].clone(), S["y"].clone())
        while k_total < max_iter:
            run_block()
            k_total += eval_every; k_since += eval_every
            kc, ka = kkt(cur), kkt(avg)
            use_avg = self._kkt_w(ka, w) < self._kkt_w(kc, w)
            kcand = ka if use_avg else kc
            names = avg if use_avg else cur
            if max(kcand["rel"]) < max(best[0]["rel"]):
                best = (kcand, S[names[0]].clone(), S[names[1]].clone())
            elapsed = time.perf_counter() - t0
            if verbose:
                r = kcand["rel"]
                print(f"iter {k_total:7d}  obj {kcand['p']:+.8e}  rel_p {r[0]:.1e}  rel_d {r[1]:.1e}  "
                      f"gap {r[2]:.1e}  w {w:.2e}  {elapsed:6.1f}s")
            if max(kcand["rel"]) <= tol:
                status = "optimal"; break
            if not all(math.isfinite(v) for v in kcand["rel"]):
                status = "numerical_error"; break
            if elapsed > time_limit:
                status = "time_limit"; break
            kw = self._kkt_w(kcand, w)
            if (kw <= 0.2 * kkt_rs) or (kw <= 0.8 * kkt_rs and kw > kkt_prev_cand) or (k_since >= 0.36 * k_total):
                if use_avg:
                    for a_, b_ in zip(cur, names):
                        S[a_].copy_(S[b_])
                ddx = float(torch.linalg.vector_norm(S["x"] - x_rs))
                ddy = float(torch.linalg.vector_norm(S["y"] - y_rs))
                if ddx > 1e-10 and ddy > 1e-10:
                    w = math.exp(0.5 * math.log(ddy / ddx) + 0.5 * math.log(w))
                self._steps(S, w, eta, nK, LQ)
                x_rs, y_rs = S["x"].clone(), S["y"].clone()
                for nm in avg + ("k",):
                    S[nm].zero_()
                kkt_rs, kkt_prev_cand, k_since = self._kkt_w(kcand, w), float("inf"), 0
                restarts += 1
            else:
                kkt_prev_cand = kw
        if cuda:
            torch.cuda.synchronize()
        solve_time = time.perf_counter() - t0
        kb, xb, yb = best
        return Result(status, (xb * self.dc).cpu().numpy(), (yb * self.dr).cpu().numpy(), kb["p"], kb["d"],
                      kb["rel"][2], kb["rel"][0], kb["rel"][1], k_total, 2 * k_total + 80, restarts,
                      solve_time, self.setup_time, dev + ("+graph" if cuda else ""))

    # ------------------------------------------------------------------ main loop
    def solve(self, tol=1e-4, max_iter=200_000, time_limit=600.0, eval_every=64, verbose=False, adaptive=True):
        """adaptive=True: PDLP adaptive step (one host sync per iteration).
        adaptive=False: constant step 0.998/||K||, no host sync between KKT checks; on CUDA each block of
        eval_every iterations is captured once as a CUDA graph and replayed (one launch per block)."""
        if not adaptive or self.Q is not None:   # QP always uses the constant-step (linearised) method
            return self._solve_graph(tol, max_iter, time_limit, eval_every, verbose)
        dev, dt = self.device, self.dtype
        sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
        sync()
        t0 = time.perf_counter()

        x = self._proj_x(torch.zeros(self.n, dtype=dt, device=dev))
        y = torch.zeros(self.m, dtype=dt, device=dev)
        Kx, KTy = self._mv(self.K, x), self._mv(self.KT, y)
        matvecs = 2

        # initial primal weight and step
        cn = float(torch.linalg.vector_norm(self.c))
        qn = float(torch.linalg.vector_norm(self.q))
        w = cn / qn if cn > 1e-10 and qn > 1e-10 else 1.0
        eta = 1.0 / max(float(self.K.values().abs().max()) if self.K._nnz() else 1.0, 1e-12)
        if not adaptive:
            eta = 0.998 / max(self._norm_K(), 1e-12)
            matvecs += 80

        # averaging / restart state
        xs, ys, Kxs, KTys, wsum = [torch.zeros_like(v) for v in (x, y, Kx, KTy)] + [0.0]
        x_rs, y_rs = x.clone(), y.clone()
        k0 = self._kkt(x, y, Kx, KTy)
        kkt_rs = self._kkt_w(k0, w)
        kkt_prev_cand = float("inf")
        k_total, k_since, restarts = 0, 0, 0
        status, best = "iteration_limit", (k0, x, y, Kx, KTy)

        while k_total < max_iter:
            # ---- one PDHG iteration
            while True:
                if not adaptive:
                    tau, sigma = eta / w, eta * w
                    xn = self._proj_x(x - tau * (self.c - KTy))
                    Kxn = self._mv(self.K, xn)
                    yn = self._proj_y(y + sigma * (self.q - 2 * Kxn + Kx))
                    KTyn = self._mv(self.KT, yn)
                    matvecs += 2
                    eta_new = eta
                    break
                tau, sigma = eta / w, eta * w
                xn = self._proj_x(x - tau * (self.c - KTy))
                Kxn = self._mv(self.K, xn)
                yn = self._proj_y(y + sigma * (self.q - 2 * Kxn + Kx))
                KTyn = self._mv(self.KT, yn)
                matvecs += 2
                dx, dy = xn - x, yn - y
                ip, dxx, dyy = torch.stack([torch.dot(dy, Kxn - Kx), torch.dot(dx, dx), torch.dot(dy, dy)]).tolist()
                inter = abs(ip)
                move = 0.5 * w * dxx + 0.5 * dyy / w
                eta_lim = move / inter if inter > 1e-30 else float("inf")
                kk = k_total + 1
                grow = (1 + kk ** -0.6) * eta
                eta_new = grow if not math.isfinite(eta_lim) else min((1 - (kk + 1) ** -0.3) * eta_lim, grow)
                if eta <= eta_lim:
                    break
                eta = eta_new
            x, y, Kx, KTy = xn, yn, Kxn, KTyn
            wsum += eta
            a = eta / wsum
            xs.lerp_(x, a); ys.lerp_(y, a); Kxs.lerp_(Kx, a); KTys.lerp_(KTy, a)
            eta = eta_new
            k_total += 1
            k_since += 1

            if k_total % eval_every:
                continue

            # ---- evaluate current and average, check termination and restart
            kc = self._kkt(x, y, Kx, KTy)
            ka = self._kkt(xs, ys, Kxs, KTys)
            use_avg = self._kkt_w(ka, w) < self._kkt_w(kc, w)
            kcand = ka if use_avg else kc
            cand = (xs, ys, Kxs, KTys) if use_avg else (x, y, Kx, KTy)
            if max(kcand["rel"]) < max(best[0]["rel"]):
                best = (kcand,) + tuple(v.clone() for v in cand)
            elapsed = time.perf_counter() - t0
            if verbose:
                r = kcand["rel"]
                print(f"iter {k_total:7d}  obj {kcand['p']:+.8e}  rel_p {r[0]:.1e}  rel_d {r[1]:.1e}  "
                      f"gap {r[2]:.1e}  w {w:.2e}  eta {eta:.2e}  {elapsed:6.1f}s")
            if max(kcand["rel"]) <= tol:
                status = "optimal"
                break
            if not all(math.isfinite(v) for v in kcand["rel"]):
                status = "numerical_error"
                break
            if elapsed > time_limit:
                status = "time_limit"
                break

            kw = self._kkt_w(kcand, w)
            if (kw <= 0.2 * kkt_rs) or (kw <= 0.8 * kkt_rs and kw > kkt_prev_cand) or (k_since >= 0.36 * k_total):
                x, y, Kx, KTy = (v.clone() for v in cand)
                ddx = float(torch.linalg.vector_norm(x - x_rs))
                ddy = float(torch.linalg.vector_norm(y - y_rs))
                if ddx > 1e-10 and ddy > 1e-10:
                    w = math.exp(0.5 * math.log(ddy / ddx) + 0.5 * math.log(w))
                x_rs, y_rs = x.clone(), y.clone()
                xs, ys, Kxs, KTys = (torch.zeros_like(v) for v in (x, y, Kx, KTy))
                wsum = 0.0
                kkt_rs = self._kkt_w(kcand, w)
                kkt_prev_cand = float("inf")
                k_since = 0
                restarts += 1
            else:
                kkt_prev_cand = kw

        sync()
        solve_time = time.perf_counter() - t0
        kb, xb, yb, _, _ = best
        x_orig = (xb * self.dc).cpu().numpy()
        y_orig = (yb * self.dr).cpu().numpy()
        return Result(status, x_orig, y_orig, kb["p"], kb["d"], kb["rel"][2], kb["rel"][0], kb["rel"][1],
                      k_total, matvecs, restarts, solve_time, self.setup_time, dev)


def solve(lp: LP, device="cuda", **kw) -> Result:
    return PDLP(lp, device=device).solve(**kw)
