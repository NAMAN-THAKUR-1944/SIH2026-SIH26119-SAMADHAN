// SAMADHAN C++ core: bounded dual simplex + branch-and-cut for mixed-integer linear programs.
//
// Written from scratch (no solver library). Model:   min c'x   s.t.  rlo <= A x <= rhi,  lb <= x <= ub,
// some x integer. Every row i gets a logical variable r_i = a_i x, so the simplex works on [A  -I] with all
// bounds on columns.
//
//   LU   : sparse LU factorisation of the basis (Markowitz pivot order, threshold partial pivoting),
//          product-form eta updates between refactorisations, repair of singular bases with logicals.
//   LP   : bounded dual simplex, dual steepest-edge pricing (Forrest-Goldfarb weight updates), Harris
//          two-pass ratio test, row-wise pricing for sparse pivot rows, artificial boxes for dual
//          feasibility on unbounded columns.
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
#include <cstring>
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
    double crash_tol = 1e-5;     // crossover: a value this close (relative) to a bound counts as at the bound
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

// ---------------------------------------------------------------------------------------------- sparse LU
// Factorisation of the basis matrix B (m x m, columns = basis positions) by right-looking Gaussian
// elimination. Step t pivots on row prow[t] and basis column pcol[t], chosen by Markowitz cost
// (r_i - 1)(c_j - 1) among entries with |a_ij| >= u * max_k |a_ik| (threshold pivoting relative to the row,
// which bounds the entries of U by 1/u times their pivot; the row maxima are cached, so the search stays
// cheap on dense kernels). The result is  E B = U  with E the product of the elimination steps (L) and U
// upper triangular in pivot order. Each later basis change is appended as a product-form eta vector.
struct Factor {
    int m = 0, rank = 0;
    double u = 0.1, abs_tol = 1e-9;     // the model is scaled: entries are O(1)
    bool symmetric = false;             // SPD matrix: diagonal pivots only (minimum-degree order, LDL')
    std::vector<int> prow, pcol;
    std::vector<int> lstart, lidx;           // L step t: x[lidx] -= lval * x[prow[t]]
    std::vector<double> lval;
    std::vector<int> ustart, uidx;           // U row t: diagonal udiag[t] at column pcol[t], others (column, value)
    std::vector<double> uval, udiag;
    std::vector<int> ucstart, ucrow;         // U by columns (for FTRAN): column c holds (pivot row, value)
    std::vector<double> ucval;
    std::vector<int> epos, estart, eidx;     // eta file: pivot position, pivot value, other (position, value)
    std::vector<double> epiv, eval;
    std::vector<int> bad_rows, bad_cols;     // unpivoted rows / columns when B is singular
    std::vector<double> work;

    struct Buckets {                         // objects linked into doubly linked lists by their count
        std::vector<int> head, next, prev;
        void init(int n) { head.assign(n + 2, -1); next.assign(n, -1); prev.assign(n, -1); }
        void add(int o, int k) {
            prev[o] = -1; next[o] = head[k];
            if (head[k] >= 0) prev[head[k]] = o;
            head[k] = o;
        }
        void remove(int o, int k) {
            if (prev[o] >= 0) next[prev[o]] = next[o]; else head[k] = next[o];
            if (next[o] >= 0) prev[next[o]] = prev[o];
        }
    };

    // B given column-wise (bs, bi, bv). Returns false if B is singular; bad_cols / bad_rows then pair up the
    // basis columns and rows that could not be pivoted.
    bool factor(int m_, const std::vector<int>& bs, const std::vector<int>& bi, const std::vector<double>& bv) {
        m = m_; rank = 0;
        prow.clear(); pcol.clear(); lstart.assign(1, 0); lidx.clear(); lval.clear();
        ustart.assign(1, 0); uidx.clear(); uval.clear(); udiag.clear();
        epos.clear(); estart.assign(1, 0); eidx.clear(); epiv.clear(); eval.clear();
        bad_rows.clear(); bad_cols.clear();
        // active submatrix: values stored by rows, patterns by columns (rows already pivoted are skipped)
        std::vector<std::vector<std::pair<int, double>>> R(m);
        std::vector<std::vector<int>> C(m);
        for (int c = 0; c < m; ++c)
            for (int t = bs[c]; t < bs[c + 1]; ++t)
                if (bv[t] != 0.0) { R[bi[t]].push_back({c, bv[t]}); C[c].push_back(bi[t]); }
        std::vector<int> rcnt(m), ccnt(m), rstep(m, -1), cstep(m, -1), mark(m, -1);
        Buckets RB, CB;
        RB.init(m); CB.init(m);
        for (int i = 0; i < m; ++i) { rcnt[i] = (int)R[i].size(); RB.add(i, rcnt[i]); }
        for (int c = 0; c < m; ++c) { ccnt[c] = (int)C[c].size(); CB.add(c, ccnt[c]); }
        auto value = [&](int i, int c) {
            for (auto& e : R[i]) if (e.first == c) return e.second;
            return 0.0;
        };
        std::vector<double> rmax(m, -1.0);      // cached largest |entry| of each active row (-1: recompute)
        auto rowmax = [&](int i) {
            if (rmax[i] < 0.0) {
                double mx = 0.0;
                for (auto& e : R[i]) mx = std::max(mx, std::fabs(e.second));
                rmax[i] = mx;
            }
            return rmax[i];
        };
        for (int t = 0; t < m; ++t) {
            // Markowitz search over columns and rows of increasing count (singletons first)
            int P = -1, Q = -1, seen = 0;
            long best = std::numeric_limits<long>::max();
            double bestv = 0.0;
            auto consider = [&](int i, int c, double v) {
                double a = std::fabs(v);
                if (symmetric ? (i != c || a < abs_tol) : (a < abs_tol || a < u * rowmax(i))) return;
                long cost = (long)(rcnt[i] - 1) * (long)(ccnt[c] - 1);
                if (cost < best || (cost == best && a > bestv)) { best = cost; P = i; Q = c; bestv = a; }
            };
            auto done = [&]() { return P >= 0 && (best == 0 || seen >= 4); };
            for (int k = 1; k <= m && !done(); ++k) {
                for (int c = CB.head[k]; c >= 0 && !done(); c = CB.next[c]) {
                    for (int i : C[c]) if (rstep[i] < 0) consider(i, c, value(i, c));
                    if (P >= 0) ++seen;
                }
                for (int i = RB.head[k]; i >= 0 && !done(); i = RB.next[i]) {
                    for (auto& e : R[i]) consider(i, e.first, e.second);
                    if (P >= 0) ++seen;
                }
                if (P >= 0 && best <= (long)k * k) break;     // any later candidate costs at least k*k
            }
            if (P < 0) break;                                   // what is left is (numerically) singular
            double piv = value(P, Q);
            prow.push_back(P); pcol.push_back(Q); rstep[P] = t; cstep[Q] = t;
            RB.remove(P, rcnt[P]); CB.remove(Q, ccnt[Q]);
            std::vector<std::pair<int, double>>& Rp = R[P];
            for (auto& e : Rp) if (e.first != Q) { CB.remove(e.first, ccnt[e.first]); ccnt[e.first]--; }
            // eliminate column Q from the other active rows:  row_i -= (a_iQ / piv) * row_P
            for (int i : C[Q]) {
                if (rstep[i] >= 0) continue;
                RB.remove(i, rcnt[i]);
                std::vector<std::pair<int, double>>& Ri = R[i];
                double a = 0.0;
                for (size_t s = 0; s < Ri.size(); ++s)
                    if (Ri[s].first == Q) { a = Ri[s].second; Ri[s] = Ri.back(); Ri.pop_back(); break; }
                if (a != 0.0) {
                    double l = a / piv;
                    lidx.push_back(i); lval.push_back(l);
                    for (size_t s = 0; s < Ri.size(); ++s) mark[Ri[s].first] = (int)s;
                    for (auto& e : Rp) {
                        int c = e.first;
                        if (c == Q) continue;
                        if (mark[c] >= 0) {
                            Ri[mark[c]].second -= l * e.second;
                        } else {
                            Ri.push_back({c, -l * e.second}); C[c].push_back(i); ccnt[c]++;
                        }
                    }
                    for (auto& e : Ri) mark[e.first] = -1;
                }
                rmax[i] = -1.0;
                rcnt[i] = (int)Ri.size();
                RB.add(i, rcnt[i]);
            }
            lstart.push_back((int)lidx.size());
            for (auto& e : Rp) if (e.first != Q) CB.add(e.first, ccnt[e.first]);
            udiag.push_back(piv);
            for (auto& e : Rp) if (e.first != Q) { uidx.push_back(e.first); uval.push_back(e.second); }
            ustart.push_back((int)uidx.size());
            std::vector<std::pair<int, double>>().swap(Rp);
            std::vector<int>().swap(C[Q]);
            ++rank;
        }
        if (rank < m) {
            for (int c = 0; c < m; ++c) if (cstep[c] < 0) bad_cols.push_back(c);
            for (int i = 0; i < m; ++i) if (rstep[i] < 0) bad_rows.push_back(i);
            return false;
        }
        // column-wise copy of U for the scatter form of the back substitution
        ucstart.assign(m + 1, 0);
        for (int c : uidx) ucstart[c + 1]++;
        for (int c = 0; c < m; ++c) ucstart[c + 1] += ucstart[c];
        ucrow.resize(uidx.size()); ucval.resize(uidx.size());
        std::vector<int> pos(ucstart.begin(), ucstart.end() - 1);
        for (int s = 0; s < m; ++s)
            for (int k = ustart[s]; k < ustart[s + 1]; ++k) {
                int c = uidx[k];
                ucrow[pos[c]] = prow[s]; ucval[pos[c]++] = uval[k];
            }
        return true;
    }

    // B x = b: on entry b is indexed by row, on exit it holds x indexed by basis position
    void ftran(std::vector<double>& b) {
        for (int t = 0; t < m; ++t) {
            double v = b[prow[t]];
            if (v == 0.0) continue;
            for (int k = lstart[t]; k < lstart[t + 1]; ++k) b[lidx[k]] -= lval[k] * v;
        }
        work.assign(m, 0.0);
        for (int t = m - 1; t >= 0; --t) {
            double v = b[prow[t]];
            if (v == 0.0) continue;
            int c = pcol[t];
            double xc = v / udiag[t];
            work[c] = xc;
            for (int k = ucstart[c]; k < ucstart[c + 1]; ++k) b[ucrow[k]] -= ucval[k] * xc;
        }
        for (size_t e = 0; e < epos.size(); ++e) {
            int r = epos[e];
            double xr = work[r] / epiv[e];
            work[r] = xr;
            if (xr != 0.0)
                for (int k = estart[e]; k < estart[e + 1]; ++k) work[eidx[k]] -= eval[k] * xr;
        }
        b.swap(work);
    }

    // B' y = d: on entry d is indexed by basis position, on exit it holds y indexed by row
    void btran(std::vector<double>& d) {
        for (int e = (int)epos.size() - 1; e >= 0; --e) {
            int r = epos[e];
            double s = d[r];
            for (int k = estart[e]; k < estart[e + 1]; ++k) s -= eval[k] * d[eidx[k]];
            d[r] = s / epiv[e];
        }
        work.assign(m, 0.0);
        for (int t = 0; t < m; ++t) {
            double v = d[pcol[t]];
            if (v == 0.0) continue;
            double wt = v / udiag[t];
            work[prow[t]] = wt;
            for (int k = ustart[t]; k < ustart[t + 1]; ++k) d[uidx[k]] -= uval[k] * wt;
        }
        for (int t = m - 1; t >= 0; --t) {
            double s = 0.0;
            for (int k = lstart[t]; k < lstart[t + 1]; ++k) s += lval[k] * work[lidx[k]];
            work[prow[t]] -= s;
        }
        d.swap(work);
    }

    // basis position r now holds a column with B^-1 a = alpha (alpha indexed by basis position)
    void add_eta(int r, const std::vector<double>& alpha) {
        epos.push_back(r); epiv.push_back(alpha[r]);
        for (int k = 0; k < m; ++k)
            if (k != r && std::fabs(alpha[k]) > 1e-14) { eidx.push_back(k); eval.push_back(alpha[k]); }
        estart.push_back((int)eidx.size());
    }

    long nnz() const { return (long)lidx.size() + (long)uidx.size() + m; }
};

// ---------------------------------------------------------------------------------------------- simplex
struct Simplex {
    const Model* M = nullptr;
    Options opt;
    int n = 0, m = 0, N = 0;
    std::vector<double> c, lb, ub, x, d, w;          // w: dual steepest-edge weights ||row_k(B^-1)||^2
    std::vector<char> art;                           // column has an artificial (box) bound
    std::vector<int> head;
    std::vector<int8_t> st;
    std::vector<double> ar, acol, rho, tau;          // work: pivot row, entering column, row of B^-1, B^-1 rho
    std::vector<int> rp, rc;                         // row-wise copy of A (pricing with sparse pivot rows)
    std::vector<double> rv;
    std::vector<int> bstart, bidx;                   // basis columns handed to the factorisation
    std::vector<double> bval;
    Factor F;
    long iters = 0;
    int since_refactor = 0;
    double big = 1e7;
    Clock::time_point t_end;

    template <class Fn> void col(int j, Fn f) const {
        if (j < n) {
            for (int t = M->cp[j]; t < M->cp[j + 1]; ++t) f(M->ri[t], M->cv[t]);
        } else {
            f(j - n, -1.0);
        }
    }

    void build_rows() {
        rp.assign(m + 1, 0);
        for (int t = 0; t < M->cp[n]; ++t) rp[M->ri[t] + 1]++;
        for (int i = 0; i < m; ++i) rp[i + 1] += rp[i];
        rc.resize(M->cp[n]); rv.resize(M->cp[n]);
        std::vector<int> pos(rp.begin(), rp.end() - 1);
        for (int j = 0; j < n; ++j)
            for (int t = M->cp[j]; t < M->cp[j + 1]; ++t) { rc[pos[M->ri[t]]] = j; rv[pos[M->ri[t]]++] = M->cv[t]; }
    }

    // (re)size for the current model; keeps existing basis info for old columns, new rows get basic logicals
    // (their dual steepest-edge weights are marked unknown and computed at the next refactorisation)
    void resize_to_model() {
        int old_m = m;
        n = M->n; m = M->m; N = n + m;
        c.resize(N, 0.0); lb.resize(N); ub.resize(N); x.resize(N, 0.0); d.resize(N, 0.0);
        art.resize(N, 0); st.resize(N, BASIC); ar.resize(N); acol.resize(m); w.resize(m, -1.0);
        for (int j = 0; j < n; ++j) c[j] = M->c[j];
        for (int i = old_m; i < m; ++i) {
            lb[n + i] = M->rlo[i]; ub[n + i] = M->rhi[i]; c[n + i] = 0.0;
            head.push_back(n + i); st[n + i] = BASIC;
        }
        build_rows();
    }

    void init_slack_basis(const std::vector<double>& slb, const std::vector<double>& sub) {
        n = M->n; m = M->m; N = n + m;
        c.assign(N, 0.0); lb.assign(N, 0.0); ub.assign(N, 0.0); x.assign(N, 0.0); d.assign(N, 0.0);
        art.assign(N, 0); st.assign(N, AT_LB); ar.assign(N, 0.0); acol.assign(m, 0.0); w.assign(m, 1.0);
        head.resize(m);
        for (int j = 0; j < n; ++j) { c[j] = M->c[j]; lb[j] = slb[j]; ub[j] = sub[j]; }
        for (int i = 0; i < m; ++i) { lb[n + i] = M->rlo[i]; ub[n + i] = M->rhi[i]; head[i] = n + i; st[n + i] = BASIC; }
        for (int j = 0; j < n; ++j) st[j] = AT_LB;
        build_rows();
        refactor_full();
    }

    // factorise the basis; a singular basis is repaired by swapping in the logicals of the unpivoted rows
    bool reinvert(bool& repaired) {
        for (int attempt = 0; attempt < 4; ++attempt) {
            bstart.assign(1, 0); bidx.clear(); bval.clear();
            for (int k = 0; k < m; ++k) {
                col(head[k], [&](int i, double v) { bidx.push_back(i); bval.push_back(v); });
                bstart.push_back((int)bidx.size());
            }
            if (F.factor(m, bstart, bidx, bval)) return true;
            repaired = true;
            for (size_t t = 0; t < F.bad_cols.size(); ++t) {
                int k = F.bad_cols[t], i = F.bad_rows[t];
                st[head[k]] = AT_LB;
                head[k] = n + i; st[n + i] = BASIC;
            }
        }
        return false;
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
        F.ftran(rhs);
        for (int k = 0; k < m; ++k) x[head[k]] = rhs[k];
    }

    void compute_duals() {
        std::vector<double> y(m);
        for (int k = 0; k < m; ++k) y[k] = c[head[k]];
        F.btran(y);
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC) { d[j] = 0.0; continue; }
            double s = c[j];
            col(j, [&](int i, double v) { s -= y[i] * v; });
            d[j] = s;
        }
    }

    // exact dual steepest-edge weights for the positions marked unknown (w < 0), or for all positions
    void exact_weights(bool all) {
        for (int k = 0; k < m; ++k) {
            if (!all && w[k] >= 0.0) continue;
            rho.assign(m, 0.0); rho[k] = 1.0;
            F.btran(rho);
            double s = 0.0;
            for (double v : rho) s += v * v;
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

    // refactorise and recompute primal and dual values; the weights are kept (the basis is unchanged)
    // unless the basis had to be repaired
    bool refactor_full() {
        bool repaired = false;
        if (!reinvert(repaired)) {
            for (int j = 0; j < N; ++j) if (st[j] == BASIC) st[j] = AT_LB;      // last resort: slack basis
            for (int i = 0; i < m; ++i) { head[i] = n + i; st[n + i] = BASIC; }
            if (!reinvert(repaired)) return false;
            w.assign(m, 1.0);
        } else if (repaired) {
            if (m <= 5000) exact_weights(true); else w.assign(m, 1.0);
        }
        compute_xB();
        compute_duals();
        make_dual_feasible();
        exact_weights(false);
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

    // tableau row of basis position r:  rho = (row r of B^-1),  ar[j] = rho . a_j  for nonbasic j
    void row_alpha(int r) {
        rho.assign(m, 0.0); rho[r] = 1.0;
        F.btran(rho);
        int nz = 0;
        for (int i = 0; i < m; ++i) if (rho[i] != 0.0) ++nz;
        if (nz * 10 < m) {                       // sparse rho: accumulate row by row
            std::fill(ar.begin(), ar.begin() + n, 0.0);
            for (int i = 0; i < m; ++i) {
                double v = rho[i];
                if (v == 0.0) continue;
                for (int t = rp[i]; t < rp[i + 1]; ++t) ar[rc[t]] += v * rv[t];
            }
        } else {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int t = M->cp[j]; t < M->cp[j + 1]; ++t) s += rho[M->ri[t]] * M->cv[t];
                ar[j] = s;
            }
        }
        for (int i = 0; i < m; ++i) ar[n + i] = -rho[i];
        for (int k = 0; k < m; ++k) ar[head[k]] = 0.0;
    }

    void col_alpha(int q) {
        acol.assign(m, 0.0);
        col(q, [&](int i, double v) { acol[i] += v; });
        F.ftran(acol);
    }

    // bounded dual simplex from a dual feasible basis
    Result dual() {
        int boxes_grown = 0;
        for (;;) {
            if (since_refactor >= opt.refactor && !refactor_full()) return NUMERIC;
            if (Clock::now() > t_end) return TIME_LIMIT;
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
                // optimal for the boxed problem. An artificial bound that is active with a reduced cost of the
                // wrong sign for the real (infinite) bound means the box was too small; with a zero reduced
                // cost the point is also optimal for the original problem.
                bool hit = false;
                for (int j = 0; j < N; ++j) {
                    if (!art[j] || st[j] == BASIC) continue;
                    if (st[j] == AT_UB && ub[j] >= big * 0.999 && d[j] < -opt.tol_d) hit = true;
                    if (st[j] == AT_LB && lb[j] <= -big * 0.999 && d[j] > opt.tol_d) hit = true;
                }
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
                if (since_refactor == 0) return NUMERIC;   // inconsistent even on a fresh factorisation
                if (!refactor_full()) return NUMERIC;   // stale factorisation: refresh and retry
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
            // dual steepest-edge weights (Forrest-Goldfarb):  w_k += -2 (a_k/a_r) tau_k + (a_k/a_r)^2 w_r
            tau = rho;
            F.ftran(tau);
            double wr = 0.0;
            for (double v : rho) wr += v * v;
            double alr = acol[r];
            for (int k = 0; k < m; ++k) {
                if (k == r || acol[k] == 0.0) continue;
                double g = acol[k] / alr;
                w[k] = std::max(w[k] - 2.0 * g * tau[k] + g * g * wr, 1e-12);
            }
            w[r] = std::max(wr / (alr * alr), 1e-12);
            F.add_eta(r, acol);
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

    // ------------------------------------------------------------------ primal simplex and crossover
    std::vector<double> cshift, wp;      // original costs while shifted (crossover); primal Devex weights

    // refactorise and recompute primal and dual values without touching the nonbasic bound statuses
    bool refactor_basis() {
        bool repaired = false;
        if (!reinvert(repaired)) return false;
        compute_xB();
        compute_duals();
        since_refactor = 0;
        return true;
    }

    double dual_infeas(int j) const {
        if (st[j] == BASIC || lb[j] == ub[j]) return 0.0;
        if (st[j] == AT_LB) return std::max(0.0, -d[j]);
        if (st[j] == AT_UB) return std::max(0.0, d[j]);
        return std::fabs(d[j]);                          // free nonbasic
    }

    // bounded primal simplex (phase 2) from a primal feasible basis: Devex pricing, Harris ratio test,
    // bound flips of boxed entering columns
    Result primal() {
        wp.assign(N, 1.0);
        for (;;) {
            if (since_refactor >= opt.refactor && !refactor_basis()) return NUMERIC;
            if (Clock::now() > t_end) return TIME_LIMIT;
            if (iters >= opt.max_lp_iter) return ITER_LIMIT;
            int q = -1; double best = 0.0;
            for (int j = 0; j < N; ++j) {
                double v = dual_infeas(j);
                if (v <= opt.tol_d) continue;
                double score = v * v / wp[j];
                if (score > best) { best = score; q = j; }
            }
            if (q < 0) return OPTIMAL;
            double dir = st[q] == AT_UB || (st[q] == AT_ZERO && d[q] > 0) ? -1.0 : 1.0;
            col_alpha(q);
            // Harris two-pass ratio test over the basic variables
            double tmax = INF;
            for (int k = 0; k < m; ++k) {
                double a = dir * acol[k];
                if (std::fabs(a) < opt.tol_piv) continue;
                int j = head[k];
                double room = a > 0 ? x[j] - lb[j] : ub[j] - x[j];      // x_j moves by -t * a
                if (std::isfinite(room)) tmax = std::min(tmax, (std::max(room, 0.0) + opt.tol_p) / std::fabs(a));
            }
            double flip = ub[q] - lb[q];                                  // entering column hits its other bound
            int r = -1; double amax = 0.0, t = 0.0;
            if (tmax < INF) {
                for (int k = 0; k < m; ++k) {
                    double a = dir * acol[k];
                    if (std::fabs(a) < opt.tol_piv) continue;
                    int j = head[k];
                    double room = a > 0 ? x[j] - lb[j] : ub[j] - x[j];
                    if (!std::isfinite(room)) continue;
                    double ratio = std::max(room, 0.0) / std::fabs(a);
                    if (ratio <= tmax && std::fabs(a) > amax) { amax = std::fabs(a); r = k; t = ratio; }
                }
            }
            if (std::isfinite(flip) && (r < 0 || flip <= t)) {
                for (int k = 0; k < m; ++k) x[head[k]] -= flip * dir * acol[k];
                st[q] = st[q] == AT_LB ? AT_UB : AT_LB;
                x[q] = st[q] == AT_LB ? lb[q] : ub[q];
                ++iters;
                continue;
            }
            if (r < 0) return UNBOUNDED;
            row_alpha(r);
            if (std::fabs(acol[r] - ar[q]) > 1e-6 * (1.0 + std::fabs(ar[q]))) {
                if (since_refactor == 0) return NUMERIC;
                if (!refactor_basis()) return NUMERIC;
                continue;
            }
            int jl = head[r];
            bool leave_lb = dir * acol[r] > 0;
            for (int k = 0; k < m; ++k) x[head[k]] -= t * dir * acol[k];
            x[q] += t * dir;
            x[jl] = leave_lb ? lb[jl] : ub[jl];
            // reduced costs and Devex weights
            double theta = d[q] / ar[q];
            double wq = std::max(wp[q], 1.0);
            for (int j = 0; j < N; ++j) {
                if (st[j] == BASIC || j == q) continue;
                d[j] -= theta * ar[j];
                double g = ar[j] / ar[q];
                wp[j] = std::max(wp[j], g * g * wq);
            }
            d[q] = 0.0;
            d[jl] = -theta;
            wp[jl] = std::max(wq / (ar[q] * ar[q]), 1.0);
            st[jl] = leave_lb ? AT_LB : AT_UB;
            st[q] = BASIC;
            head[r] = q;
            F.add_eta(r, acol);
            ++iters; ++since_refactor;
        }
    }

    // Shift the costs of nonbasic columns so that the basis becomes dual feasible, with every reduced cost at
    // least a small pseudo-random margin on the right side (cost perturbation against dual degeneracy and
    // cycling). The original costs are saved and restored exactly afterwards.
    void shift_costs() {
        cshift = c;
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC || lb[j] == ub[j]) continue;
            double eps = 1e-6 * (1.0 + (double)((uint32_t)j * 2654435761u % 1000u) / 1000.0);
            double target = st[j] == AT_LB ? std::max(d[j], eps) : st[j] == AT_UB ? std::min(d[j], -eps) : 0.0;
            c[j] += target - d[j];
            d[j] = target;
        }
    }

    void unshift_costs() {
        c = cshift;
        cshift.clear();
    }

    // crash basis from an approximate solution x0 (structural columns): columns and row activities strictly
    // inside their bounds become basic (the furthest from a bound first), the rest are placed at the nearer bound
    void init_from_point(const std::vector<double>& slb, const std::vector<double>& sub,
                         const std::vector<double>& x0) {
        n = M->n; m = M->m; N = n + m;
        c.assign(N, 0.0); lb.assign(N, 0.0); ub.assign(N, 0.0); x.assign(N, 0.0); d.assign(N, 0.0);
        art.assign(N, 0); st.assign(N, AT_LB); ar.assign(N, 0.0); acol.assign(m, 0.0);
        for (int j = 0; j < n; ++j) { c[j] = M->c[j]; lb[j] = slb[j]; ub[j] = sub[j]; }
        for (int i = 0; i < m; ++i) { lb[n + i] = M->rlo[i]; ub[n + i] = M->rhi[i]; }
        build_rows();
        std::vector<double> v(N, 0.0);
        for (int j = 0; j < n; ++j) {
            v[j] = x0[j];
            for (int t = M->cp[j]; t < M->cp[j + 1]; ++t) v[n + M->ri[t]] += M->cv[t] * x0[j];
        }
        std::vector<std::pair<double, int>> cand;
        for (int j = 0; j < N; ++j) {
            double tl = opt.crash_tol * (1.0 + std::fabs(lb[j])), tu = opt.crash_tol * (1.0 + std::fabs(ub[j]));
            bool at_l = std::isfinite(lb[j]) && v[j] <= lb[j] + tl, at_u = std::isfinite(ub[j]) && v[j] >= ub[j] - tu;
            if (at_l || at_u) {
                st[j] = at_l && (!at_u || v[j] - lb[j] <= ub[j] - v[j]) ? AT_LB : AT_UB;
                continue;
            }
            st[j] = std::isfinite(lb[j]) ? AT_LB : std::isfinite(ub[j]) ? AT_UB : AT_ZERO;
            cand.push_back({std::min(v[j] - lb[j], ub[j] - v[j]), j});
        }
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        head.clear();
        for (auto& e : cand) {
            if ((int)head.size() == m) break;
            head.push_back(e.second); st[e.second] = BASIC;
        }
        for (int i = 0; i < m && (int)head.size() < m; ++i)
            if (st[n + i] != BASIC) { head.push_back(n + i); st[n + i] = BASIC; }
        w.assign(m, m <= 5000 ? -1.0 : 1.0);
        // factorise with a strict pivot tolerance: near-dependent crash columns are swapped for logicals
        double tol = F.abs_tol;
        F.abs_tol = 1e-6;
        refactor_basis();
        F.abs_tol = tol;
        exact_weights(false);
    }
};

// ---------------------------------------------------------------------------------------------- MILP
struct Node {
    double bound;
    long id;
    int branch_var, dir;       // dir: -1 down, +1 up (for pseudocost update)
    double frac, parent_obj;
    int leaf = -1;                // last branching record on the path from the root (see Branch)
    std::vector<uint8_t> basis;   // warm start: 2-bit status of every column, packed 4 per byte

    void save(const std::vector<int8_t>& st) {
        basis.assign((st.size() + 3) / 4, 0);
        for (size_t j = 0; j < st.size(); ++j) basis[j >> 2] |= (uint8_t)(st[j] << ((j & 3) * 2));
    }
    // restore the column statuses; the basis order is rebuilt from the basic columns
    void load(std::vector<int8_t>& st, std::vector<int>& head) const {
        head.clear();
        for (size_t j = 0; j < st.size(); ++j) {
            st[j] = (int8_t)((basis[j >> 2] >> ((j & 3) * 2)) & 3);
            if (st[j] == BASIC) head.push_back((int)j);
        }
    }
};
// Branching decisions of all nodes, stored once: one bound change per record plus the index of its parent record
// (-1 = root). A node is identified by its last record, so memory grows with the number of nodes, not with
// nodes x depth.
struct Branch {
    int parent, var;
    double lo, hi;
};
struct NodeCmp {
    bool operator()(const Node* a, const Node* b) const {
        return a->bound > b->bound || (a->bound == b->bound && a->id < b->id);
    }
};
// best-bound queue of open nodes; clear() frees them in O(n) instead of popping one by one
struct NodeHeap : std::priority_queue<Node*, std::vector<Node*>, NodeCmp> {
    void clear() {
        for (Node* p : c) delete p;
        c.clear();
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
    std::vector<double> colscale;   // x_j (original) = colscale[j] * x_j (scaled)

    // Geometric-mean scaling (four passes over rows and columns), then equilibration so that the largest |a_ij|
    // of every row and column is 1. Factors are rounded to powers of two (no rounding error is introduced);
    // integer columns keep the factor 1 so that integrality is unchanged.
    void scale_model() {
        int n = M.n, m = M.m;
        std::vector<double> rs(m, 1.0), cs(n, 1.0);
        auto pass = [&](bool geometric) {
            std::vector<double> lo(m, INF), hi(m, 0.0);
            for (int j = 0; j < n; ++j)
                for (int t = M.cp[j]; t < M.cp[j + 1]; ++t) {
                    double v = std::fabs(M.cv[t]) * rs[M.ri[t]] * cs[j];
                    if (v == 0.0) continue;
                    lo[M.ri[t]] = std::min(lo[M.ri[t]], v); hi[M.ri[t]] = std::max(hi[M.ri[t]], v);
                }
            for (int i = 0; i < m; ++i) if (hi[i] > 0.0) rs[i] /= geometric ? std::sqrt(lo[i] * hi[i]) : hi[i];
            for (int j = 0; j < n; ++j) {
                if (M.isint[j]) continue;
                double l = INF, h = 0.0;
                for (int t = M.cp[j]; t < M.cp[j + 1]; ++t) {
                    double v = std::fabs(M.cv[t]) * rs[M.ri[t]] * cs[j];
                    if (v == 0.0) continue;
                    l = std::min(l, v); h = std::max(h, v);
                }
                if (h > 0.0) cs[j] /= geometric ? std::sqrt(l * h) : h;
            }
        };
        for (int k = 0; k < 4; ++k) pass(true);
        pass(false);
        for (auto& v : rs) v = std::exp2(std::round(std::log2(v)));
        for (auto& v : cs) v = std::exp2(std::round(std::log2(v)));
        for (int j = 0; j < n; ++j) {
            for (int t = M.cp[j]; t < M.cp[j + 1]; ++t) M.cv[t] *= rs[M.ri[t]] * cs[j];
            M.c[j] *= cs[j]; M.lb[j] /= cs[j]; M.ub[j] /= cs[j];
        }
        for (int i = 0; i < m; ++i) { M.rlo[i] *= rs[i]; M.rhi[i] *= rs[i]; }
        colscale.swap(cs);
    }

    int solve(Info& info, std::vector<double>& xout) {
        scale_model();
        int status = solve_scaled(info, xout);
        for (size_t j = 0; j < xout.size(); ++j) xout[j] *= colscale[j];
        return status;
    }

    // Crossover: approximate LP solution x0 (e.g. from the GPU engine) -> optimal vertex. Crash basis from x0,
    // dual simplex on shifted costs until primal feasible, then primal simplex without the shifts, then a final
    // dual simplex check on a fresh factorisation.
    int crossover(const std::vector<double>& x0, Info& info, std::vector<double>& xout) {
        scale_model();
        std::vector<double> xs(M.n);
        for (int j = 0; j < M.n; ++j) xs[j] = x0[j] / colscale[j];
        t0 = Clock::now();
        S.M = &M; S.opt = opt;
        S.t_end = t0 + std::chrono::milliseconds((long long)(opt.time_limit * 1000));
        S.init_from_point(M.lb, M.ub, xs);
        // iteration budget: a crossover that needs more than this is slower than a cold start (the caller then
        // falls back to one)
        S.opt.max_lp_iter = std::max(5000L, (long)S.m + S.n);
        long crash_basic = 0;
        for (int k = 0; k < S.m; ++k) crash_basic += S.head[k] < S.n;
        S.shift_costs();
        Result r = S.dual();
        long it_dual = S.iters;
        if (r == OPTIMAL) {
            S.unshift_costs();
            r = S.refactor_basis() ? S.primal() : NUMERIC;
        }
        long it_primal = S.iters - it_dual;
        if (r == OPTIMAL) {
            S.w.assign(S.m, 1.0);
            r = S.solve();
        }
        std::vector<double> xkeep;
        if (r == OPTIMAL) {                  // polish with 100x tighter tolerances, as for a cold LP solve
            xkeep = S.x;
            S.opt.tol_p = S.opt.tol_d = 1e-9;
            S.opt.max_lp_iter = S.iters + std::max(5000L, (long)S.m + S.n);
            if (!(S.refactor_full() && S.dual() == OPTIMAL)) S.x = xkeep;
            S.opt = opt;
        }
        if (opt.verbose)
            std::printf("crossover: %ld structural columns basic in the crash basis, %ld dual + %ld primal + %ld "
                        "cleanup iterations, %.2f s\n", crash_basic, it_dual, it_primal,
                        S.iters - it_dual - it_primal, seconds_since(t0));
        info.lp_iters = (double)S.iters;
        info.time = seconds_since(t0);
        if (r != OPTIMAL) return r == TIME_LIMIT ? 3 : r == INFEASIBLE ? 1 : r == UNBOUNDED ? 2 : 5;
        xout.assign(S.x.begin(), S.x.begin() + M.n);
        for (int j = 0; j < M.n; ++j) xout[j] *= colscale[j];
        info.obj = info.bound = S.objective();
        info.gap = 0.0;
        return 0;
    }

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

    int solve_scaled(Info& info, std::vector<double>& xout) {
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
            // polish: re-solve from the optimal basis with tolerances 100x tighter; keep the first answer if the
            // tight pass does not finish cleanly (ill-conditioned models)
            xout.assign(S.x.begin(), S.x.begin() + M.n);
            double obj = S.objective();
            S.opt.tol_p = S.opt.tol_d = 1e-9;
            if (S.refactor_full() && S.dual() == OPTIMAL) {
                xout.assign(S.x.begin(), S.x.begin() + M.n);
                obj = S.objective();
            }
            S.opt = opt;
            info.lp_iters = (double)S.iters;
            info.obj = info.bound = obj; info.gap = 0; info.time = seconds_since(t0);
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
        NodeHeap heap;
        long next_id = 0, nodes = 0;
        std::vector<double> cur_lb = root_lb, cur_ub = root_ub;
        std::vector<Branch> tree;
        std::vector<char> seen(M.n, 0);
        std::vector<int> touched;
        auto apply_bounds = [&](int leaf) {
            cur_lb = root_lb; cur_ub = root_ub;
            for (int b = leaf; b >= 0; b = tree[b].parent) {
                const Branch& br = tree[b];
                if (seen[br.var]) continue;            // a deeper record of the same column is tighter
                seen[br.var] = 1; touched.push_back(br.var);
                cur_lb[br.var] = br.lo; cur_ub[br.var] = br.hi;
            }
            for (int j : touched) seen[j] = 0;
            touched.clear();
            for (int j = 0; j < M.n; ++j) {
                if (!S.art[j] || std::isfinite(root_lb[j])) S.lb[j] = cur_lb[j];
                if (!S.art[j] || std::isfinite(root_ub[j])) S.ub[j] = cur_ub[j];
            }
        };
        int path = -1;               // last branching record of the current node
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
                apply_bounds(nd->leaf);
                nd->load(S.st, S.head);
                // steepest-edge weights of the restored basis: exact for small models, else restart from 1
                S.w.assign(S.m, S.m <= 1000 ? -1.0 : 1.0);
                path = nd->leaf;
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
            double olb = cur_lb[bv], oub = cur_ub[bv];
            if (-first < 0) tree.push_back({path, bv, olb, dn});
            else tree.push_back({path, bv, up, oub});
            other->leaf = (int)tree.size() - 1;
            other->save(S.st);
            heap.push(other);
            // plunge: change the bound of the basic branching variable, keep the factorisation
            if (first < 0) { cur_ub[bv] = dn; S.ub[bv] = dn; tree.push_back({path, bv, olb, dn}); }
            else { cur_lb[bv] = up; S.lb[bv] = up; tree.push_back({path, bv, up, oub}); }
            path = (int)tree.size() - 1;
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
        heap.clear();
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
static void load_model(sm::Solver& s, int n, int m, const int* colptr, const int* rowidx, const double* vals,
                       const double* c, const double* lb, const double* ub, const double* rlo, const double* rhi) {
    s.M.n = n; s.M.m = m;
    s.M.cp.assign(colptr, colptr + n + 1);
    s.M.ri.assign(rowidx, rowidx + colptr[n]);
    s.M.cv.assign(vals, vals + colptr[n]);
    s.M.c.assign(c, c + n); s.M.lb.assign(lb, lb + n); s.M.ub.assign(ub, ub + n);
    s.M.rlo.assign(rlo, rlo + m); s.M.rhi.assign(rhi, rhi + m);
    s.M.isint.assign(n, 0);
}

static void store_info(const sm::Info& info, double* info_out) {
    double vals_out[9] = {info.obj, info.bound, info.gap, info.nodes, info.lp_iters, info.time, info.cuts,
                          info.root_bound, info.root_bound_cuts};
    for (int k = 0; k < 9; ++k) info_out[k] = vals_out[k];
}

SM_API int sm_solve(int n, int m, const int* colptr, const int* rowidx, const double* vals, const double* c,
                    const double* lb, const double* ub, const double* rlo, const double* rhi, const char* isint,
                    const double* opts, double* x_out, double* info_out) {
    sm::Solver s;
    load_model(s, n, m, colptr, rowidx, vals, c, lb, ub, rlo, rhi);
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
    store_info(info, info_out);
    return status;
}

// Crossover from an approximate LP solution x0 to an optimal vertex. opts: [time_limit, verbose, crash_tol (0 =
// default)].
// Returns 0 optimal, 1 infeasible, 2 unbounded, 3 time limit, 5 numerical failure; info as for sm_solve.
SM_API int sm_crossover(int n, int m, const int* colptr, const int* rowidx, const double* vals, const double* c,
                        const double* lb, const double* ub, const double* rlo, const double* rhi, const double* x0,
                        const double* opts, double* x_out, double* info_out) {
    sm::Solver s;
    load_model(s, n, m, colptr, rowidx, vals, c, lb, ub, rlo, rhi);
    s.opt.time_limit = opts[0];
    s.opt.verbose = (int)opts[1];
    if (opts[2] > 0) s.opt.crash_tol = opts[2];
    sm::Info info;
    std::vector<double> x, xs(x0, x0 + n);
    int status = s.crossover(xs, info, x);
    for (int j = 0; j < n; ++j) x_out[j] = j < (int)x.size() ? x[j] : 0.0;
    store_info(info, info_out);
    return status;
}

// Sparse LU of a square CSC matrix (symmetric != 0: SPD, diagonal pivots), kept between calls (the interior-point method factorises its normal
// equations once per iteration and solves twice). Returns a handle, or null if the matrix is singular to the
// given absolute pivot tolerance; the rank is stored in rank_out either way.
SM_API void* sm_lu_create(int m, const int* colptr, const int* rowidx, const double* vals, double abs_tol,
                          int symmetric, int* rank_out) {
    auto* F = new sm::Factor();
    F->abs_tol = abs_tol;
    F->symmetric = symmetric != 0;
    std::vector<int> bs(colptr, colptr + m + 1), bi(rowidx, rowidx + colptr[m]);
    std::vector<double> bv(vals, vals + colptr[m]);
    bool ok = F->factor(m, bs, bi, bv);
    *rank_out = F->rank;
    if (!ok) { delete F; return nullptr; }
    return F;
}

// solve B x = b in place
SM_API void sm_lu_solve(void* handle, double* b) {
    auto* F = static_cast<sm::Factor*>(handle);
    std::vector<double> x(b, b + F->m);
    F->ftran(x);
    std::memcpy(b, x.data(), sizeof(double) * F->m);
}

SM_API long sm_lu_nnz(void* handle) { return static_cast<sm::Factor*>(handle)->nnz(); }

SM_API void sm_lu_free(void* handle) { delete static_cast<sm::Factor*>(handle); }

// Test hook for the sparse LU (tests/test_samadhan.py). Factorises the m x m CSC matrix B; if r >= 0 the basis
// column r is then replaced by the dense column `newcol` through an eta update. Solves B x = b and B' y = d in
// place. Returns the rank of the original B (the solves are skipped when it is singular).
SM_API int sm_lu_check(int m, const int* colptr, const int* rowidx, const double* vals, int r, const double* newcol,
                       double* b, double* d) {
    sm::Factor F;
    std::vector<int> bs(colptr, colptr + m + 1), bi(rowidx, rowidx + colptr[m]);
    std::vector<double> bv(vals, vals + colptr[m]);
    if (!F.factor(m, bs, bi, bv)) return F.rank;
    if (r >= 0) {
        std::vector<double> alpha(newcol, newcol + m);
        F.ftran(alpha);
        F.add_eta(r, alpha);
    }
    std::vector<double> x(b, b + m), y(d, d + m);
    F.ftran(x);
    F.btran(y);
    std::memcpy(b, x.data(), sizeof(double) * m);
    std::memcpy(d, y.data(), sizeof(double) * m);
    return F.rank;
}
