"""Native CUDA kernels for the GPU engine's constant-step PDHG iteration.

The kernels are CUDA C++ compiled at run time by NVRTC (the copy that ships with PyTorch's CUDA wheels) and launched
through the CUDA driver API on PyTorch's current stream, so they need no CUDA toolkit and are captured by CUDA graphs
like any other kernel. One iteration is three fused kernels instead of about twenty PyTorch kernels:

    primal:  x <- proj_[l,u](x - tau (c - K'y)),          xs <- xs + a (x - xs)
    dual:    Kx' = K x;  y <- proj(y + sigma (q - 2 Kx' + Kx)),  Kx <- Kx',  ys, Kxs averaged
    kt:      K'y = K' y,                                   K'ys averaged

with a = 1/k (k counts the iterations since the last restart, on the device). The sparse products use a group of V
threads per row (V a power of two chosen from the mean row length) and warp shuffles for the sum. Everything is in
double precision; any failure to load or compile leaves the PyTorch path in use.
"""
import ctypes
import hashlib
import math
import os
from pathlib import Path

import torch

SRC = r"""
#define NAN_SAFE_CLAMP(v, lo, hi) do { if (v < lo) v = lo; if (v > hi) v = hi; } while (0)

extern "C" __global__ void pdhg_tick(double* k) { *k += 1.0; }

extern "C" __global__ void pdhg_primal(int n, double* __restrict__ x, double* __restrict__ xs,
                                       const double* __restrict__ KTy, const double* __restrict__ c,
                                       const double* __restrict__ l, const double* __restrict__ u,
                                       const double* __restrict__ tau, const double* __restrict__ k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    double a = 1.0 / *k;
    double v = x[j] - *tau * (c[j] - KTy[j]);
    NAN_SAFE_CLAMP(v, l[j], u[j]);
    x[j] = v;
    xs[j] += a * (v - xs[j]);
}

template <int V>
__device__ __forceinline__ double row_dot(int row, int rows, const int* __restrict__ rp, const int* __restrict__ ci,
                                          const double* __restrict__ av, const double* __restrict__ v) {
    int lane = threadIdx.x & (V - 1);
    double s = 0.0;
    if (row < rows)
        for (int t = rp[row] + lane; t < rp[row + 1]; t += V) s += av[t] * v[ci[t]];
    for (int off = V / 2; off > 0; off >>= 1) s += __shfl_down_sync(0xffffffffu, s, off, V);
    return s;
}

template <int V>
__device__ void dual(int m, int n_eq, const int* rp, const int* ci, const double* av, const double* __restrict__ x,
                     double* __restrict__ y, double* __restrict__ ys, double* __restrict__ Kx,
                     double* __restrict__ Kxs, const double* __restrict__ q, const double* __restrict__ sigma,
                     const double* __restrict__ k) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / V;
    double s = row_dot<V>(row, m, rp, ci, av, x);
    if (row < m && (threadIdx.x & (V - 1)) == 0) {
        double a = 1.0 / *k;
        double v = y[row] + *sigma * (q[row] - 2.0 * s + Kx[row]);
        if (row >= n_eq && v < 0.0) v = 0.0;
        Kx[row] = s;
        Kxs[row] += a * (s - Kxs[row]);
        y[row] = v;
        ys[row] += a * (v - ys[row]);
    }
}

template <int V>
__device__ void kt(int n, const int* rp, const int* ci, const double* av, const double* __restrict__ y,
                   double* __restrict__ KTy, double* __restrict__ KTys, const double* __restrict__ k) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / V;
    double s = row_dot<V>(row, n, rp, ci, av, y);
    if (row < n && (threadIdx.x & (V - 1)) == 0) {
        double a = 1.0 / *k;
        KTy[row] = s;
        KTys[row] += a * (s - KTys[row]);
    }
}

// ---- adaptive-step iteration: a candidate (xn, yn) in separate buffers, the three inner products of the step-size
// rule as per-block partial sums (added up in a fixed order by ad_finish), and the averaging of the previous
// accepted iterate (weight a_prev, 0 when none is pending) folded into the first attempt's kernels
__device__ __forceinline__ void block_sum2(double& u, double& v) {
    __shared__ double sh[2][32];
    for (int off = 16; off > 0; off >>= 1) {
        u += __shfl_down_sync(0xffffffffu, u, off);
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) { sh[0][w] = u; sh[1][w] = v; }
    __syncthreads();
    bool live = threadIdx.x < (blockDim.x >> 5);
    u = live ? sh[0][lane] : 0.0;
    v = live ? sh[1][lane] : 0.0;
    if (w == 0)
        for (int off = 16; off > 0; off >>= 1) {
            u += __shfl_down_sync(0xffffffffu, u, off);
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
}

extern "C" __global__ void ad_primal(int n, const double* __restrict__ x, double* __restrict__ xn,
                                     const double* __restrict__ KTy, const double* __restrict__ c,
                                     const double* __restrict__ l, const double* __restrict__ u, double tau,
                                     double* __restrict__ xs, double* __restrict__ KTys, double a_prev,
                                     double* __restrict__ part) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    double d2 = 0.0, unused = 0.0;
    if (j < n) {
        double xj = x[j], g = KTy[j];
        if (a_prev != 0.0) { xs[j] += a_prev * (xj - xs[j]); KTys[j] += a_prev * (g - KTys[j]); }
        double v = xj - tau * (c[j] - g);
        NAN_SAFE_CLAMP(v, l[j], u[j]);
        xn[j] = v;
        d2 = (v - xj) * (v - xj);
    }
    block_sum2(d2, unused);
    if (threadIdx.x == 0) part[blockIdx.x] = d2;
}

template <int V>
__device__ void ad_dual(int m, int n_eq, const int* rp, const int* ci, const double* av, const double* __restrict__ xn,
                        const double* __restrict__ y, double* __restrict__ yn, const double* __restrict__ Kx,
                        double* __restrict__ Kxn, const double* __restrict__ q, double sigma, double* __restrict__ ys,
                        double* __restrict__ Kxs, double a_prev, double* __restrict__ part_yy,
                        double* __restrict__ part_ip) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / V;
    double s = row_dot<V>(row, m, rp, ci, av, xn);
    double yy = 0.0, ip = 0.0;
    if (row < m && (threadIdx.x & (V - 1)) == 0) {
        double yi = y[row], kx = Kx[row];
        if (a_prev != 0.0) { ys[row] += a_prev * (yi - ys[row]); Kxs[row] += a_prev * (kx - Kxs[row]); }
        double v = yi + sigma * (q[row] - 2.0 * s + kx);
        if (row >= n_eq && v < 0.0) v = 0.0;
        Kxn[row] = s;
        yn[row] = v;
        double dy = v - yi;
        yy = dy * dy;
        ip = dy * (s - kx);
    }
    block_sum2(yy, ip);
    if (threadIdx.x == 0) { part_yy[blockIdx.x] = yy; part_ip[blockIdx.x] = ip; }
}

template <int V>
__device__ void spmv(int rows, const int* rp, const int* ci, const double* av, const double* __restrict__ v,
                     double* __restrict__ out) {
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / V;
    double s = row_dot<V>(row, rows, rp, ci, av, v);
    if (row < rows && (threadIdx.x & (V - 1)) == 0) out[row] = s;
}

// one block: each thread adds a strided share of the partial sums, then a block reduction (fixed order: repeatable)
extern "C" __global__ void ad_finish(int nb_x, int nb_y, const double* part_xx, const double* part_yy,
                                     const double* part_ip, double* out) {
    double xx = 0.0, yy = 0.0, ip = 0.0, unused = 0.0;
    for (int b = threadIdx.x; b < nb_x; b += blockDim.x) xx += part_xx[b];
    for (int b = threadIdx.x; b < nb_y; b += blockDim.x) { yy += part_yy[b]; ip += part_ip[b]; }
    block_sum2(xx, unused);
    __syncthreads();
    block_sum2(yy, ip);
    if (threadIdx.x == 0) { out[0] = ip; out[1] = xx; out[2] = yy; }
}

#define WRAP(V)                                                                                                    \
extern "C" __global__ void pdhg_dual_##V(int m, int n_eq, const int* rp, const int* ci, const double* av,          \
        const double* x, double* y, double* ys, double* Kx, double* Kxs, const double* q, const double* sigma,       \
        const double* k) { dual<V>(m, n_eq, rp, ci, av, x, y, ys, Kx, Kxs, q, sigma, k); }                          \
extern "C" __global__ void pdhg_kt_##V(int n, const int* rp, const int* ci, const double* av, const double* y,     \
        double* KTy, double* KTys, const double* k) { kt<V>(n, rp, ci, av, y, KTy, KTys, k); }                      \
extern "C" __global__ void ad_dual_##V(int m, int n_eq, const int* rp, const int* ci, const double* av,            \
        const double* xn, const double* y, double* yn, const double* Kx, double* Kxn, const double* q,              \
        double sigma, double* ys, double* Kxs, double a_prev, double* part_yy, double* part_ip) {                   \
    ad_dual<V>(m, n_eq, rp, ci, av, xn, y, yn, Kx, Kxn, q, sigma, ys, Kxs, a_prev, part_yy, part_ip); }            \
extern "C" __global__ void spmv_##V(int rows, const int* rp, const int* ci, const double* av, const double* v,    \
        double* out) { spmv<V>(rows, rp, ci, av, v, out); }
WRAP(1) WRAP(2) WRAP(4) WRAP(8) WRAP(16) WRAP(32)
"""

WIDTHS = (1, 2, 4, 8, 16, 32)
BLOCK = 256
CACHE = Path(__file__).resolve().parent / "_lib"     # compiled kernels, next to the C++ core library
_state = {}


def _load_libs():
    lib = Path(torch.__file__).parent / "lib"
    if os.name == "nt":
        nvrtc = next((p for p in sorted(lib.glob("nvrtc64_*.dll")) if ".alt" not in p.name), None)
        driver = "nvcuda.dll"
    else:
        site = Path(torch.__file__).parent.parent
        cands = sorted(lib.glob("libnvrtc*.so*")) + sorted((site / "nvidia" / "cuda_nvrtc" / "lib").glob("libnvrtc.so*"))
        nvrtc = next((p for p in cands if "builtins" not in p.name), None) or "libnvrtc.so.12"
        driver = "libcuda.so.1"
    if nvrtc is None:
        raise OSError("NVRTC not found next to PyTorch")
    return ctypes.CDLL(str(nvrtc)), ctypes.CDLL(driver)


def _compile(nvrtc, src: str, arch: str) -> bytes:
    prog = ctypes.c_void_p()
    if nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"pdhg.cu", 0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    try:
        opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17"]
        rc = nvrtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
        if rc:
            size = ctypes.c_size_t()
            nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
            log = ctypes.create_string_buffer(size.value)
            nvrtc.nvrtcGetProgramLog(prog, log)
            raise RuntimeError(f"NVRTC compile failed ({rc}): {log.value.decode(errors='replace')[:2000]}")
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size))
        image = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetCUBIN(prog, image)
        return image.raw
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))


def _cubin(nvrtc, arch: str) -> bytes:
    """The kernels compiled for arch, from the on-disk cache when the source and NVRTC version match."""
    major, minor = ctypes.c_int(), ctypes.c_int()
    nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    key = hashlib.sha256(f"{SRC}|{arch}|{major.value}.{minor.value}".encode()).hexdigest()[:16]
    path = CACHE / f"pdhg_{arch}_{key}.cubin"
    if path.exists():
        return path.read_bytes()
    image = _compile(nvrtc, SRC, arch)
    try:
        CACHE.mkdir(exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(image)
        os.replace(tmp, path)
    except OSError:
        pass                                            # read-only install: compile again next time
    return image


def _module(device_index: int):
    """Compiled kernels for one device (cached): {name: CUfunction} and the driver library."""
    key = ("module", device_index)
    if key in _state:
        return _state[key]
    nvrtc, cu = _state.setdefault("libs", _load_libs())
    major, minor = torch.cuda.get_device_capability(device_index)
    image = _cubin(nvrtc, f"sm_{major}{minor}")
    with torch.cuda.device(device_index):
        torch.cuda.current_stream().synchronize()          # makes the primary context current on this thread
        mod = ctypes.c_void_p()
        if cu.cuModuleLoadData(ctypes.byref(mod), image):
            raise RuntimeError("cuModuleLoadData failed")
        fns = {}
        names = ["pdhg_tick", "pdhg_primal", "ad_primal", "ad_finish"] +             [f"{k}_{v}" for k in ("pdhg_dual", "pdhg_kt", "ad_dual", "spmv") for v in WIDTHS]
        for nm in names:
            f = ctypes.c_void_p()
            if cu.cuModuleGetFunction(ctypes.byref(f), mod, nm.encode()):
                raise RuntimeError(f"kernel {nm} not found")
            fns[nm] = f
    _state[key] = (fns, cu, mod, image)
    return _state[key]


def available(device="cuda") -> bool:
    """True when the native kernels can be compiled and loaded for the current CUDA device."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return False
    try:
        _module(torch.cuda.current_device())
        return True
    except Exception:  # noqa: BLE001 (any loader / compiler problem means: use the PyTorch path)
        return False


def _width(indptr: torch.Tensor) -> int:
    """Threads per row: the largest power of two up to a quarter of the mean row length (measured on the refinery
    LPs: K rows of 17-25 entries are fastest with 4 threads, K' rows of 2 entries with 1). A fixed rule rather than
    a timed choice, so that the summation order, and with it every run, is repeatable."""
    rows = indptr.numel() - 1
    mean = float(indptr[-1]) / max(rows, 1)
    return min(32, 2 ** int(math.floor(math.log2(max(mean / 4.0, 1.0)))))


class NativePDHG:
    """The PDHG iterations of samadhan.pdlp on fused native kernels (LP, float64, CUDA): the constant-step
    iteration (step) and one attempt of the adaptive-step iteration (attempt)."""

    def __init__(self, K: torch.Tensor, KT: torch.Tensor, n_eq: int):
        self.fns, self.cu, _, _ = _module(K.device.index if K.device.index is not None else torch.cuda.current_device())
        i32 = lambda t: t.to(torch.int32).contiguous()
        self.m, self.n, self.n_eq = K.shape[0], K.shape[1], n_eq
        self.K = (i32(K.crow_indices()), i32(K.col_indices()), K.values().contiguous())
        self.KT = (i32(KT.crow_indices()), i32(KT.col_indices()), KT.values().contiguous())
        self.vK, self.vKT = _width(self.K[0]), _width(self.KT[0])
        self.nb_x = (self.n + BLOCK - 1) // BLOCK
        self.nb_y = (self.m * self.vK + BLOCK - 1) // BLOCK
        f64 = dict(dtype=torch.float64, device=K.device)
        self.part = torch.zeros(self.nb_x + 2 * self.nb_y, **f64)
        self.out = torch.zeros(3, **f64)

    def _launch(self, name, threads, args):
        vals = []
        for a in args:
            if isinstance(a, torch.Tensor):
                vals.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, float):
                vals.append(ctypes.c_double(a))
            else:
                vals.append(ctypes.c_int(a))
        params = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        grid = max(1, (threads + BLOCK - 1) // BLOCK)
        block = BLOCK if threads > 1 else 1
        rc = self.cu.cuLaunchKernel(self.fns[name], grid, 1, 1, block, 1, 1, 0, stream, params, None)
        if rc:
            raise RuntimeError(f"cuLaunchKernel({name}) failed with {rc}")

    def step(self, S: dict, c, l, u, q):
        """One iteration, in place on the buffers of samadhan.pdlp.PDLP._solve_graph."""
        self._launch("pdhg_tick", 1, [S["k"]])
        self._launch("pdhg_primal", self.n, [self.n, S["x"], S["xs"], S["KTy"], c, l, u, S["tau"], S["k"]])
        rp, ci, av = self.K
        self._launch(f"pdhg_dual_{self.vK}", self.m * self.vK,
                     [self.m, self.n_eq, rp, ci, av, S["x"], S["y"], S["ys"], S["Kx"], S["Kxs"], q, S["sigma"], S["k"]])
        rp, ci, av = self.KT
        self._launch(f"pdhg_kt_{self.vKT}", self.n * self.vKT,
                     [self.n, rp, ci, av, S["y"], S["KTy"], S["KTys"], S["k"]])

    def attempt(self, x, y, Kx, KTy, xn, yn, Kxn, KTyn, xs, ys, Kxs, KTys, c, l, u, q, tau, sigma, a_prev):
        """One attempt of the adaptive-step iteration: the candidate xn, yn, Kxn = K xn, KTyn = K'yn from the current
        x, y, Kx, KTy, and the inner products (dy'(Kxn - Kx), dx'dx, dy'dy) of the step-size rule. The averages
        xs, ys, Kxs, KTys first take in the current point with weight a_prev (0: nothing pending)."""
        p_xx = self.part[:self.nb_x]
        p_yy = self.part[self.nb_x:self.nb_x + self.nb_y]
        p_ip = self.part[self.nb_x + self.nb_y:]
        self._launch("ad_primal", self.n, [self.n, x, xn, KTy, c, l, u, float(tau), xs, KTys, float(a_prev), p_xx])
        rp, ci, av = self.K
        self._launch(f"ad_dual_{self.vK}", self.m * self.vK,
                     [self.m, self.n_eq, rp, ci, av, xn, y, yn, Kx, Kxn, q, float(sigma), ys, Kxs, float(a_prev),
                      p_yy, p_ip])
        rp, ci, av = self.KT
        self._launch(f"spmv_{self.vKT}", self.n * self.vKT, [self.n, rp, ci, av, yn, KTyn])
        self._launch("ad_finish", BLOCK, [self.nb_x, self.nb_y, p_xx, p_yy, p_ip, self.out])
        return self.out.tolist()
