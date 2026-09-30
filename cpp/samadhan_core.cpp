// SAMADHAN C++ core: bounded dual simplex + branch-and-cut for mixed-integer linear programs.
//
// Written from scratch (no solver library). Model:   min c'x   s.t.  rlo <= A x <= rhi,  lb <= x <= ub,
// some x integer. Every row i gets a logical variable r_i = a_i x, so the simplex works on [A  -I] with all
// bounds on columns. The basis inverse is kept dense (models up to a few thousand rows), updated by a pivot
// per iteration and refactorised every `refactor` iterations.
//
//   LP   : bounded dual simplex, dual steepest-edge pricing (exact weights), Harris two-pass ratio test,
//          artificial boxes for dual feasibility on unbounded columns.
//   MILP : best-first branch-and-bound with plunging; children are warm-started from the parent basis
//          (a bound change on a basic variable keeps the basis dual feasible, so no refactorisation);
//          pseudocost branching; rounding heuristic; Gomory mixed-integer cuts at the root.
//
// C ABI (see samadhan/core.py):  sm_solve(...)  -> status, fills x and an info array.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define SM_API extern "C" __declspec(dllexport)
#else
#define SM_API extern "C" __attribute__((visibility("default")))
#endif

namespace sm {

constexpr double INF = std::numeric_limits<double>::infinity();
enum : int8_t { BASIC = 0, AT_LB = 1, AT_UB = 2, AT_ZERO = 3 };
enum Result { OPTIMAL = 0, INFEASIBLE = 1, UNBOUNDED = 2, TIME_LIMIT = 3, ITER_LIMIT = 4, NUMERIC = 5 };

using Clock = std::chrono::steady_clock;
static double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

struct Options {
    double tol_p = 1e-7, tol_d = 1e-7, tol_piv = 1e-7, int_tol = 1e-6, gap = 1e-6;
    double time_limit = 300.0;
    long node_limit = 50000000;
    int cut_rounds = 8, max_cuts_per_round = 60, refactor = 100, verbose = 0;
    long max_lp_iter = 50000000;
};

// ---------------------------------------------------------------------------------------------- model
struct Model {
    int n = 0, m = 0;
    std::vector<int> cp, ri;      // CSC of A (structural columns)
    std::vector<double> cv;
    std::vector<double> c, lb, ub, rlo, rhi;
    std::vector<char> isint;

    // append rows given as sparse (col, val) lists
    void add_rows(const std::vector<std::vector<std::pair<int, double>>>& rows, const std::vector<double>& lo,
                  const std::vector<double>& hi) {
        std::vector<int> extra(n, 0);
        for (auto& r : rows)
            for (auto& e : r) extra[e.first]++;
        std::vector<int> ncp(n + 1, 0);
        for (int j = 0; j < n; ++j) ncp[j + 1] = ncp[j] + (cp[j + 1] - cp[j]) + extra[j];
        std::vector<int> nri(ncp[n]);
        std::vector<double> ncv(ncp[n]);
        std::vector<int> pos(n);
        for (int j = 0; j < n; ++j) {
            int k = ncp[j];
            for (int t = cp[j]; t < cp[j + 1]; ++t) { nri[k] = ri[t]; ncv[k] = cv[t]; ++k; }
            pos[j] = k;
        }
        for (size_t r = 0; r < rows.size(); ++r)
            for (auto& e : rows[r]) { nri[pos[e.first]] = m + (int)r; ncv[pos[e.first]] = e.second; pos[e.first]++; }
        cp.swap(ncp); ri.swap(nri); cv.swap(ncv);
        for (size_t r = 0; r < rows.size(); ++r) { rlo.push_back(lo[r]); rhi.push_back(hi[r]); }
        m += (int)rows.size();
    }

    // row-wise copy of A (for expanding logical variables inside cuts and for feasibility checks)
    void rows_of(std::vector<std::vector<std::pair<int, double>>>& R) const {
        R.assign(m, {});
        for (int j = 0; j < n; ++j)
            for (int t = cp[j]; t < cp[j + 1]; ++t) R[ri[t]].push_back({j, cv[t]});
    }
};

// ---------------------------------------------------------------------------------------------- simplex
struct Simplex {
    const Model* M = nullptr;
    Options opt;
    int n = 0, m = 0, N = 0;
    std::vector<double> c, lb, ub, x, d, Binv, w;   // w: dual steepest-edge weights ||row_k(Binv)||^2
    std::vector<char> art;                           // column has an artificial (box) bound
    std::vector<int> head;
    std::vector<int8_t> st;
    std::vector<double> ar, acol;                    // work: pivot row / entering column
    long iters = 0;
    int since_refactor = 0;
    double big = 1e7;
    Clock::time_point t_end;

    template <class F> void col(int j, F f) const {
        if (j < n) {
            for (int t = M->cp[j]; t < M->cp[j + 1]; ++t) f(M->ri[t], M->cv[t]);
        } else {
            f(j - n, -1.0);
        }
    }

    // (re)size for the current model; keeps existing basis info for old columns, new rows get basic logicals
    void resize_to_model() {
        int old_m = m;
        n = M->n; m = M->m; N = n + m;
        c.resize(N, 0.0); lb.resize(N); ub.resize(N); x.resize(N, 0.0); d.resize(N, 0.0);
        art.resize(N, 0); st.resize(N, BASIC); ar.resize(N); acol.resize(m); w.resize(m, 1.0);
        for (int j = 0; j < n; ++j) c[j] = M->c[j];
        for (int i = old_m; i < m; ++i) {
            lb[n + i] = M->rlo[i]; ub[n + i] = M->rhi[i]; c[n + i] = 0.0;
            head.push_back(n + i); st[n + i] = BASIC;
        }
    }

    void init_slack_basis(const std::vector<double>& slb, const std::vector<double>& sub) {
        n = M->n; m = M->m; N = n + m;
        c.assign(N, 0.0); lb.assign(N, 0.0); ub.assign(N, 0.0); x.assign(N, 0.0); d.assign(N, 0.0);
        art.assign(N, 0); st.assign(N, AT_LB); ar.assign(N, 0.0); acol.assign(m, 0.0); w.assign(m, 1.0);
        head.resize(m);
        for (int j = 0; j < n; ++j) { c[j] = M->c[j]; lb[j] = slb[j]; ub[j] = sub[j]; }
        for (int i = 0; i < m; ++i) { lb[n + i] = M->rlo[i]; ub[n + i] = M->rhi[i]; head[i] = n + i; st[n + i] = BASIC; }
        for (int j = 0; j < n; ++j) st[j] = AT_LB;
        refactor_full();
    }

    // Gauss-Jordan inversion of the dense basis. Returns false if singular.
    bool reinvert() {
        std::vector<double> B((size_t)m * m, 0.0);
        for (int k = 0; k < m; ++k) col(head[k], [&](int i, double v) { B[(size_t)i * m + k] = v; });
        Binv.assign((size_t)m * m, 0.0);
        for (int i = 0; i < m; ++i) Binv[(size_t)i * m + i] = 1.0;
        for (int k = 0; k < m; ++k) {
            int p = -1; double best = 0.0;
            for (int i = k; i < m; ++i) {
                double v = std::fabs(B[(size_t)i * m + k]);
                if (v > best) { best = v; p = i; }
            }
            if (best < 1e-11) return false;
            if (p != k) {
                for (int t = 0; t < m; ++t) {
                    std::swap(B[(size_t)p * m + t], B[(size_t)k * m + t]);
                    std::swap(Binv[(size_t)p * m + t], Binv[(size_t)k * m + t]);
                }
            }
            double piv = B[(size_t)k * m + k];
            double* Bk = &B[(size_t)k * m];
            double* Ik = &Binv[(size_t)k * m];
            for (int t = 0; t < m; ++t) { Bk[t] /= piv; Ik[t] /= piv; }
            for (int i = 0; i < m; ++i) {
                if (i == k) continue;
                double f = B[(size_t)i * m + k];
                if (f == 0.0) continue;
                double* Bi = &B[(size_t)i * m];
                double* Ii = &Binv[(size_t)i * m];
                for (int t = 0; t < m; ++t) { Bi[t] -= f * Bk[t]; Ii[t] -= f * Ik[t]; }
            }
        }
        return true;
    }

    void place_nonbasic(int j) {
        if (st[j] == AT_LB && !std::isfinite(lb[j])) st[j] = std::isfinite(ub[j]) ? AT_UB : AT_ZERO;
        if (st[j] == AT_UB && !std::isfinite(ub[j])) st[j] = std::isfinite(lb[j]) ? AT_LB : AT_ZERO;
        if (st[j] == AT_ZERO && std::isfinite(lb[j])) st[j] = AT_LB;
        x[j] = st[j] == AT_LB ? lb[j] : st[j] == AT_UB ? ub[j] : 0.0;
    }

    void compute_xB() {
        std::vector<double> rhs(m, 0.0);
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC) continue;
            place_nonbasic(j);
            double xj = x[j];
            if (xj != 0.0) col(j, [&](int i, double v) { rhs[i] -= v * xj; });
        }
        for (int k = 0; k < m; ++k) {
            const double* Bk = &Binv[(size_t)k * m];
            double s = 0.0;
            for (int i = 0; i < m; ++i) s += Bk[i] * rhs[i];
            x[head[k]] = s;
        }
    }

    void compute_duals() {
        std::vector<double> y(m, 0.0);
        for (int k = 0; k < m; ++k) {
            double cb = c[head[k]];
            if (cb == 0.0) continue;
            const double* Bk = &Binv[(size_t)k * m];
            for (int i = 0; i < m; ++i) y[i] += cb * Bk[i];
        }
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC) { d[j] = 0.0; continue; }
            double s = c[j];
            col(j, [&](int i, double v) { s -= y[i] * v; });
            d[j] = s;
        }
    }

    void compute_weights() {
        for (int k = 0; k < m; ++k) {
            const double* Bk = &Binv[(size_t)k * m];
            double s = 0.0;
            for (int i = 0; i < m; ++i) s += Bk[i] * Bk[i];
            w[k] = std::max(s, 1e-12);
        }
    }

    // flip nonbasic columns so every reduced cost has the sign its bound needs; box unbounded ones
    void make_dual_feasible() {
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC) continue;
            if (lb[j] == ub[j]) { st[j] = AT_LB; continue; }
            if (d[j] > opt.tol_d) {
                if (!std::isfinite(lb[j])) { lb[j] = std::min(-big, ub[j] - big); art[j] = 1; }
                st[j] = AT_LB;
            } else if (d[j] < -opt.tol_d) {
                if (!std::isfinite(ub[j])) { ub[j] = std::max(big, lb[j] + big); art[j] = 1; }
                st[j] = AT_UB;
            }
        }
        compute_xB();
    }

    bool refactor_full() {
        if (!reinvert()) {
            // singular basis: fall back to the all-logical basis
            for (int j = 0; j < N; ++j) if (st[j] == BASIC) st[j] = AT_LB;
            for (int i = 0; i < m; ++i) { head[i] = n + i; st[n + i] = BASIC; }
            if (!reinvert()) return false;
        }
        compute_xB();
        compute_duals();
        make_dual_feasible();
        compute_weights();
        since_refactor = 0;
        return true;
    }

    double objective() const {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += c[j] * x[j];
        return s;
    }

    double infeas(int j) const {
        if (x[j] < lb[j] - opt.tol_p) return lb[j] - x[j];
        if (x[j] > ub[j] + opt.tol_p) return x[j] - ub[j];
        return 0.0;
    }

    // tableau row of basis position r:  ar[j] = (Binv row r) . a_j  for nonbasic j
    void row_alpha(int r) {
        const double* rho = &Binv[(size_t)r * m];
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC) { ar[j] = 0.0; continue; }
            double s = 0.0;
            col(j, [&](int i, double v) { s += rho[i] * v; });
            ar[j] = s;
        }
    }

    void col_alpha(int q) {
        std::fill(acol.begin(), acol.end(), 0.0);
        col(q, [&](int i, double v) {
            for (int k = 0; k < m; ++k) acol[k] += Binv[(size_t)k * m + i] * v;
        });
    }

    void pivot_update(int r) {
        double piv = acol[r];
        double* Br = &Binv[(size_t)r * m];
        for (int t = 0; t < m; ++t) Br[t] /= piv;
        for (int k = 0; k < m; ++k) {
            if (k == r || acol[k] == 0.0) continue;
            double f = acol[k];
            double* Bk = &Binv[(size_t)k * m];
            for (int t = 0; t < m; ++t) Bk[t] -= f * Br[t];
        }
    }

    // bounded dual simplex from a dual feasible basis
    Result dual() {
        int boxes_grown = 0;
        for (;;) {
            if (since_refactor >= opt.refactor && !refactor_full()) return NUMERIC;
            if ((iters & 63) == 0 && Clock::now() > t_end) return TIME_LIMIT;
            if (iters >= opt.max_lp_iter) return ITER_LIMIT;
            // pricing: dual steepest edge
            int r = -1; double best = 0.0;
            for (int k = 0; k < m; ++k) {
                double v = infeas(head[k]);
                if (v <= 0.0) continue;
                double score = v * v / w[k];
                if (score > best) { best = score; r = k; }
            }
            if (r < 0) {
                // optimal for the boxed problem: an active artificial bound means the box was too small
                bool hit = false;
                for (int j = 0; j < N; ++j)
                    if (art[j] && st[j] != BASIC && std::fabs(x[j]) >= big * 0.999) hit = true;
                if (!hit) return OPTIMAL;
                if (++boxes_grown > 3) return UNBOUNDED;
                big *= 1000.0;
                for (int j = 0; j < N; ++j) {
                    if (!art[j]) continue;
                    if (lb[j] <= -big / 1000.0 * 0.999) lb[j] = -big;
                    if (ub[j] >= big / 1000.0 * 0.999) ub[j] = big;
                }
                compute_xB();
                continue;
            }
            int jl = head[r];
            bool to_lb = x[jl] < lb[jl];
            row_alpha(r);
            // Harris two-pass ratio test
            double tmax = INF;
            auto num_of = [&](int j) {
                if (st[j] == AT_ZERO) return std::fabs(d[j]);
                return std::max(0.0, st[j] == AT_LB ? d[j] : -d[j]);
            };
            auto is_cand = [&](int j) {
                if (st[j] == BASIC || lb[j] == ub[j]) return false;
                double a = ar[j];
                if (std::fabs(a) < opt.tol_piv) return false;
                if (st[j] == AT_ZERO) return true;
                if (to_lb) return (st[j] == AT_LB && a < 0) || (st[j] == AT_UB && a > 0);
                return (st[j] == AT_LB && a > 0) || (st[j] == AT_UB && a < 0);
            };
            for (int j = 0; j < N; ++j)
                if (is_cand(j)) tmax = std::min(tmax, (num_of(j) + opt.tol_d) / std::fabs(ar[j]));
            if (tmax == INF) return INFEASIBLE;
            int q = -1; double amax = 0.0;
            for (int j = 0; j < N; ++j) {
                if (!is_cand(j)) continue;
                double a = std::fabs(ar[j]);
                if (num_of(j) / a <= tmax && a > amax) { amax = a; q = j; }
            }
            if (q < 0) return NUMERIC;
            col_alpha(q);
            if (std::fabs(acol[r] - ar[q]) > 1e-6 * (1.0 + std::fabs(ar[q]))) {
                if (!refactor_full()) return NUMERIC;   // stale inverse: refresh and retry
                continue;
            }
            // dual update (cost shifting when the entering reduced cost has the wrong sign)
            double dq = d[q];
            if (num_of(q) == 0.0 && st[q] != AT_ZERO) dq = 0.0;
            double theta = dq / ar[q];
            for (int j = 0; j < N; ++j)
                if (st[j] != BASIC) d[j] -= theta * ar[j];
            d[q] = 0.0;
            d[jl] = -theta;
            // primal update
            double bnd = to_lb ? lb[jl] : ub[jl];
            double dxq = (x[jl] - bnd) / acol[r];
            for (int k = 0; k < m; ++k) x[head[k]] -= acol[k] * dxq;
            x[q] += dxq;
            x[jl] = bnd;
            st[jl] = to_lb ? AT_LB : AT_UB;
            st[q] = BASIC;
            head[r] = q;
            // exact dual steepest-edge weights after the inverse update
            pivot_update(r);
            double wr = 0.0;
            {
                const double* Br = &Binv[(size_t)r * m];
                for (int t = 0; t < m; ++t) wr += Br[t] * Br[t];
            }
            w[r] = std::max(wr, 1e-12);
            for (int k = 0; k < m; ++k) {
                if (k == r || acol[k] == 0.0) continue;
                const double* Bk = &Binv[(size_t)k * m];
                double s = 0.0;
                for (int t = 0; t < m; ++t) s += Bk[t] * Bk[t];
                w[k] = std::max(s, 1e-12);
            }
            ++iters; ++since_refactor;
        }
    }

    Result solve() {
        Result r = dual();
        if (r == OPTIMAL) {
            // clean finish on a fresh factorisation
            if (!refactor_full()) return NUMERIC;
            r = dual();
        }
        return r;
    }
};

// ---------------------------------------------------------------------------------------------- MILP
struct Node {
    double bound;
    long id;
    int branch_var, dir;       // dir: -1 down, +1 up (for pseudocost update)
    double frac, parent_obj;
    std::vector<std::pair<int, std::pair<double, double>>> bnds;
    std::vector<int> head;
    std::vector<int8_t> st;
};
struct NodeCmp {
    bool operator()(const Node* a, const Node* b) const {
        return a->bound > b->bound || (a->bound == b->bound && a->id < b->id);
    }
};

struct Info {
    double obj = INF, bound = -INF, gap = INF, nodes = 0, lp_iters = 0, time = 0, cuts = 0, root_bound = -INF;
    double root_bound_cuts = -INF;
};

struct Solver {
    Model M;
    Options opt;
    Simplex S;
    Clock::time_point t0;
    std::vector<double> root_lb, root_ub, inc;
    double inc_obj = INF;
    std::vector<double> pc_sum[2];
    std::vector<int> pc_cnt[2];
    std::vector<std::vector<std::pair<int, double>>> rowsR;   // row copy of the ORIGINAL model
    int m_orig = 0;

    bool feasible_original(const std::vector<double>& xs) const {
        for (int j = 0; j < M.n; ++j) {
            if (xs[j] < root_lb[j] - 1e-6 || xs[j] > root_ub[j] + 1e-6) return false;
            if (M.isint[j] && std::fabs(xs[j] - std::round(xs[j])) > opt.int_tol) return false;
        }
        for (int i = 0; i < m_orig; ++i) {
            double s = 0.0;
            for (auto& e : rowsR[i]) s += e.second * xs[e.first];
            double sc = 1.0 + std::max(std::fabs(M.rlo[i] == -INF ? 0 : M.rlo[i]), std::fabs(M.rhi[i] == INF ? 0 : M.rhi[i]));
            if (s < M.rlo[i] - 1e-6 * sc || s > M.rhi[i] + 1e-6 * sc) return false;
        }
        return true;
    }

    // prune threshold; +inf while there is no incumbent (avoids inf - inf = NaN)
    double cutoff() const {
        return std::isfinite(inc_obj) ? inc_obj - opt.gap * std::max(1.0, std::fabs(inc_obj)) : INF;
    }

    void try_incumbent(const std::vector<double>& xs) {
        if (!feasible_original(xs)) return;
        double o = 0.0;
        for (int j = 0; j < M.n; ++j) o += M.c[j] * xs[j];
        if (o < inc_obj - 1e-9) {
            inc_obj = o; inc = xs;
            if (opt.verbose) std::printf("  incumbent %.10g  (%.1fs)\n", o, seconds_since(t0));
        }
    }

    void rounding_heuristic() {
        std::vector<double> xs(S.x.begin(), S.x.begin() + M.n);
        for (int j = 0; j < M.n; ++j) if (M.isint[j]) xs[j] = std::round(xs[j]);
        try_incumbent(xs);
    }

    int most_fractional_or_pc(double obj) {
        (void)obj;
        double avg[2] = {1.0, 1.0};
        for (int s = 0; s < 2; ++s) {
            double sum = 0; int cnt = 0;
            for (int j = 0; j < M.n; ++j) if (pc_cnt[s][j]) { sum += pc_sum[s][j] / pc_cnt[s][j]; ++cnt; }
            if (cnt) avg[s] = sum / cnt;
        }
        int best = -1; double bscore = -1.0;
        for (int j = 0; j < M.n; ++j) {
            if (!M.isint[j]) continue;
            double v = S.x[j], f = v - std::floor(v);
            if (f < opt.int_tol || f > 1 - opt.int_tol) continue;
            double pd = pc_cnt[0][j] ? pc_sum[0][j] / pc_cnt[0][j] : avg[0];
            double pu = pc_cnt[1][j] ? pc_sum[1][j] / pc_cnt[1][j] : avg[1];
            double score = std::max(pd * f, 1e-6) * std::max(pu * (1 - f), 1e-6);
            score *= 1.0 + 1e-3 * (0.5 - std::fabs(f - 0.5));       // tie-break: more fractional
            if (score > bscore) { bscore = score; best = j; }
        }
        return best;
    }

    // ---- Gomory mixed-integer cuts from the optimal root tableau
    int add_gomory_round() {
        std::vector<std::vector<std::pair<int, double>>> R;
        M.rows_of(R);
        std::vector<std::pair<double, int>> cand;
        for (int k = 0; k < S.m; ++k) {
            int j = S.head[k];
            if (j >= M.n || !M.isint[j]) continue;
            double f = S.x[j] - std::floor(S.x[j]);
            if (f < 0.01 || f > 0.99) continue;
            cand.push_back({std::fabs(f - 0.5), k});
        }
        std::sort(cand.begin(), cand.end());
        std::vector<std::vector<std::pair<int, double>>> cuts;
        std::vector<double> lo, hi;
        std::vector<double> pi(M.n);
        for (auto& ck : cand) {
            if ((int)cuts.size() >= opt.max_cuts_per_round) break;
            int k = ck.second;
            int ib = S.head[k];
            double f0 = S.x[ib] - std::floor(S.x[ib]);
            S.row_alpha(k);
            std::fill(pi.begin(), pi.end(), 0.0);
            double pi0 = 1.0;
            bool ok = true;
            for (int j = 0; j < S.N && ok; ++j) {
                if (S.st[j] == BASIC) continue;
                double a = S.ar[j];
                if (std::fabs(a) < 1e-12) continue;
                if (S.lb[j] == S.ub[j]) continue;                     // fixed: s_j == 0
                if (S.st[j] == AT_ZERO || S.art[j]) { ok = false; break; }
                double sigma = S.st[j] == AT_LB ? 1.0 : -1.0;
                double bnd = S.st[j] == AT_LB ? S.lb[j] : S.ub[j];
                double abar = sigma * a;
                double g;
                bool integral = j < M.n && M.isint[j] && std::fabs(bnd - std::round(bnd)) < 1e-9;
                if (integral) {
                    double fj = abar - std::floor(abar);
                    g = fj <= f0 ? fj / f0 : (1 - fj) / (1 - f0);
                } else {
                    g = abar >= 0 ? abar / f0 : -abar / (1 - f0);
                }
                if (g == 0.0) continue;
                // g * s_j with s_j = sigma * (x_j - bnd)
                double coef = g * sigma;
                pi0 += coef * bnd;
                if (j < M.n) pi[j] += coef;
                else for (auto& e : R[j - M.n]) pi[e.first] += coef * e.second;   // logical = row activity
            }
            if (!ok) continue;
            // clean tiny coefficients by relaxing with bounds; reject badly scaled cuts
            double amax = 0.0;
            for (int j = 0; j < M.n; ++j) amax = std::max(amax, std::fabs(pi[j]));
            if (amax < 1e-9) continue;
            std::vector<std::pair<int, double>> row;
            double amin = INF;
            for (int j = 0; j < M.n && ok; ++j) {
                double v = pi[j];
                if (v == 0.0) continue;
                if (std::fabs(v) < 1e-9 * amax) {
                    double b = v > 0 ? root_ub[j] : root_lb[j];
                    if (!std::isfinite(b)) { ok = false; break; }
                    pi0 -= v * b;
                    continue;
                }
                row.push_back({j, v});
                amin = std::min(amin, std::fabs(v));
            }
            if (!ok || row.empty() || amax / amin > 1e6) continue;
            double act = 0.0, nrm = 0.0;
            for (auto& e : row) { act += e.second * S.x[e.first]; nrm += e.second * e.second; }
            double viol = (pi0 - act) / std::sqrt(nrm);
            if (viol < 1e-5) continue;
            for (auto& e : row) e.second /= amax;
            cuts.push_back(row); lo.push_back(pi0 / amax); hi.push_back(INF);
        }
        if (cuts.empty()) return 0;
        M.add_rows(cuts, lo, hi);
        S.resize_to_model();
        S.refactor_full();
        return (int)cuts.size();
    }

    int solve(Info& info, std::vector<double>& xout) {
        t0 = Clock::now();
        S.M = &M; S.opt = opt;
        S.t_end = t0 + std::chrono::milliseconds((long long)(opt.time_limit * 1000));
        root_lb = M.lb; root_ub = M.ub;
        m_orig = M.m;
        M.rows_of(rowsR);
        for (int s = 0; s < 2; ++s) { pc_sum[s].assign(M.n, 0.0); pc_cnt[s].assign(M.n, 0); }
        bool has_int = false;
        for (int j = 0; j < M.n; ++j) if (M.isint[j]) has_int = true;

        S.init_slack_basis(root_lb, root_ub);
        Result r = S.solve();
        info.lp_iters = (double)S.iters;
        if (r != OPTIMAL) {
            info.time = seconds_since(t0);
            return r == TIME_LIMIT ? 3 : r == INFEASIBLE ? 1 : r == UNBOUNDED ? 2 : 5;
        }
        info.root_bound = S.objective();
        if (!has_int) {
            xout.assign(S.x.begin(), S.x.begin() + M.n);
            info.obj = info.bound = S.objective(); info.gap = 0; info.time = seconds_since(t0);
            return 0;
        }
        // root cutting-plane loop
        double prev = S.objective();
        for (int round = 0; round < opt.cut_rounds; ++round) {
            rounding_heuristic();
            int added = add_gomory_round();
            if (!added) break;
            info.cuts += added;
            r = S.solve();
            if (r != OPTIMAL) break;
            double now = S.objective();
            if (opt.verbose) std::printf("  cut round %d: +%d cuts, bound %.10g\n", round + 1, added, now);
            if (now - prev < 1e-4 * (1.0 + std::fabs(now))) { prev = now; break; }
            prev = now;
        }
        if (r != OPTIMAL) {   // numerical trouble after cuts: restart without them
            info.time = seconds_since(t0);
            return 5;
        }
        info.root_bound_cuts = S.objective();

        // branch-and-bound with plunging
        std::priority_queue<Node*, std::vector<Node*>, NodeCmp> heap;
        long next_id = 0, nodes = 0;
        std::vector<double> cur_lb = root_lb, cur_ub = root_ub;
        auto apply_bounds = [&](const Node& nd) {
            cur_lb = root_lb; cur_ub = root_ub;
            for (auto& b : nd.bnds) { cur_lb[b.first] = b.second.first; cur_ub[b.first] = b.second.second; }
            for (int j = 0; j < M.n; ++j) {
                if (!S.art[j] || std::isfinite(root_lb[j])) S.lb[j] = cur_lb[j];
                if (!S.art[j] || std::isfinite(root_ub[j])) S.ub[j] = cur_ub[j];
            }
        };
        std::vector<std::pair<int, std::pair<double, double>>> path;   // bound changes of the current node
        double cur_bound = S.objective();
        bool have_node = true;       // the LP in S is the current node (root)
        int pc_var = -1, pc_dir = 0; double pc_frac = 0, pc_parent = 0;
        int status = 0;

        auto global_bound = [&]() {
            double gb = have_node ? cur_bound : INF;
            if (!heap.empty()) gb = std::min(gb, heap.top()->bound);
            return std::min(gb, inc_obj);
        };

        for (;;) {
            if (!have_node) {
                // pop best-bound node, discard pruned ones
                Node* nd = nullptr;
                while (!heap.empty()) {
                    nd = heap.top(); heap.pop();
                    if (nd->bound < cutoff()) break;
                    delete nd; nd = nullptr;
                }
                if (!nd) break;
                double popped_bound = nd->bound;
                apply_bounds(*nd);
                S.head = nd->head;
                for (int j = 0; j < S.N; ++j) S.st[j] = nd->st[j];
                path = nd->bnds;
                pc_var = nd->branch_var; pc_dir = nd->dir; pc_frac = nd->frac; pc_parent = nd->parent_obj;
                delete nd;
                if (!S.refactor_full()) { status = 5; break; }
                Result rr = S.dual();
                ++nodes;
                if (rr == TIME_LIMIT) { status = 3; cur_bound = popped_bound; have_node = true; break; }
                if (rr == INFEASIBLE) { have_node = false; continue; }
                if (rr != OPTIMAL) { have_node = false; continue; }
                cur_bound = S.objective();
                have_node = true;
                if (pc_var >= 0) {
                    int s = pc_dir < 0 ? 0 : 1;
                    double dist = pc_dir < 0 ? pc_frac : 1 - pc_frac;
                    pc_sum[s][pc_var] += std::max(0.0, cur_bound - pc_parent) / std::max(dist, 1e-6);
                    pc_cnt[s][pc_var]++;
                }
            }
            // current node is solved and stored in S
            if (Clock::now() > S.t_end) { status = 3; break; }
            if (nodes >= opt.node_limit) { status = 4; break; }
            if (cur_bound >= cutoff()) { have_node = false; continue; }
            int bv = most_fractional_or_pc(cur_bound);
            if (bv < 0) {
                std::vector<double> xs(S.x.begin(), S.x.begin() + M.n);
                try_incumbent(xs);
                have_node = false;
                continue;
            }
            if ((nodes & 15) == 0) rounding_heuristic();
            if (opt.verbose && (nodes % 2000) == 0) {
                double gb = global_bound();
                std::printf("  nodes %8ld  open %7zu  incumbent %.10g  bound %.10g  %.1fs\n", nodes, heap.size(),
                            inc_obj, gb, seconds_since(t0));
            }
            double v = S.x[bv], f = v - std::floor(v);
            double dn = std::floor(v), up = std::ceil(v);
            // plunge into the child with the smaller estimated degradation, push the other one
            double pd = pc_cnt[0][bv] ? pc_sum[0][bv] / pc_cnt[0][bv] : 1.0;
            double pu = pc_cnt[1][bv] ? pc_sum[1][bv] / pc_cnt[1][bv] : 1.0;
            int first = (pd * f < pu * (1 - f)) ? -1 : +1;
            Node* other = new Node();
            other->bound = cur_bound; other->id = next_id++;
            other->branch_var = bv; other->dir = -first; other->frac = f; other->parent_obj = cur_bound;
            other->bnds = path;
            double olb = cur_lb[bv], oub = cur_ub[bv];
            if (-first < 0) other->bnds.push_back({bv, {olb, dn}});
            else other->bnds.push_back({bv, {up, oub}});
            other->head = S.head;
            other->st.assign(S.st.begin(), S.st.end());
            heap.push(other);
            // plunge: change the bound of the basic branching variable, keep the factorisation
            if (first < 0) { cur_ub[bv] = dn; S.ub[bv] = dn; path.push_back({bv, {olb, dn}}); }
            else { cur_lb[bv] = up; S.lb[bv] = up; path.push_back({bv, {up, oub}}); }
            double parent = cur_bound;
            Result rr = S.dual();
            ++nodes;
            if (rr == TIME_LIMIT) { status = 3; cur_bound = parent; have_node = true; break; }
            if (rr != OPTIMAL) { have_node = false; continue; }
            cur_bound = S.objective();
            int s = first < 0 ? 0 : 1;
            pc_sum[s][bv] += std::max(0.0, cur_bound - parent) / std::max(first < 0 ? f : 1 - f, 1e-6);
            pc_cnt[s][bv]++;
        }
        double gb = inc_obj;
        if (!heap.empty()) gb = std::min(gb, heap.top()->bound);
        if (status == 0 || status == 4 || status == 3) {
            if (have_node) gb = std::min(gb, cur_bound);
        }
        while (!heap.empty()) { delete heap.top(); heap.pop(); }
        info.nodes = (double)nodes;
        info.lp_iters = (double)S.iters;
        info.time = seconds_since(t0);
        info.obj = inc_obj;
        info.bound = std::min(gb, inc_obj);
        info.gap = std::isfinite(inc_obj) ? (inc_obj - info.bound) / std::max(1.0, std::fabs(inc_obj)) : INF;
        if (std::isfinite(inc_obj)) xout = inc;
        if (status == 5) return 5;
        if (status == 3) return std::isfinite(inc_obj) ? 3 : 6;
        if (status == 4) return std::isfinite(inc_obj) ? 4 : 6;
        return std::isfinite(inc_obj) ? 0 : 1;
    }
};

}  // namespace sm

// ---------------------------------------------------------------------------------------------- C ABI
// opts: [time_limit, node_limit, cut_rounds, verbose, gap]
// info: [obj, bound, gap, nodes, lp_iters, time, cuts, root_bound, root_bound_after_cuts]
// return: 0 optimal, 1 infeasible, 2 unbounded, 3 time limit (with solution), 4 node limit (with solution),
//         5 numerical failure, 6 limit reached without a feasible solution
SM_API int sm_solve(int n, int m, const int* colptr, const int* rowidx, const double* vals, const double* c,
                    const double* lb, const double* ub, const double* rlo, const double* rhi, const char* isint,
                    const double* opts, double* x_out, double* info_out) {
    sm::Solver s;
    s.M.n = n; s.M.m = m;
    s.M.cp.assign(colptr, colptr + n + 1);
    s.M.ri.assign(rowidx, rowidx + colptr[n]);
    s.M.cv.assign(vals, vals + colptr[n]);
    s.M.c.assign(c, c + n); s.M.lb.assign(lb, lb + n); s.M.ub.assign(ub, ub + n);
    s.M.rlo.assign(rlo, rlo + m); s.M.rhi.assign(rhi, rhi + m);
    s.M.isint.assign(isint, isint + n);
    s.opt.time_limit = opts[0];
    s.opt.node_limit = (long)opts[1];
    s.opt.cut_rounds = (int)opts[2];
    s.opt.verbose = (int)opts[3];
    s.opt.gap = opts[4];
    sm::Info info;
    std::vector<double> x;
    int status = s.solve(info, x);
    for (int j = 0; j < n; ++j) x_out[j] = j < (int)x.size() ? x[j] : 0.0;
    double vals_out[9] = {info.obj, info.bound, info.gap, info.nodes, info.lp_iters, info.time, info.cuts,
                          info.root_bound, info.root_bound_cuts};
    for (int k = 0; k < 9; ++k) info_out[k] = vals_out[k];
    return status;
}
