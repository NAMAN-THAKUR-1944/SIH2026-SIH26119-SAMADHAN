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
//          pseudocost branching; rounding heuristic; Gomory mixed-integer cuts at the root; with several
//          threads, a deterministic parallel tree search (rounds of fixed LP work, merged in thread order).
//
// C ABI (see samadhan/core.py):  sm_solve(...)  -> status, fills x and an info array.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <set>
#include <thread>
#include <tuple>
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
    int features = 2043;         // branch-and-cut: 1 feasibility pump, 2 diving, 4 cover cuts, 8 reliability branching,
                                 // 16 node domain propagation, 32 c-MIR cuts, 64 fix-and-propagate, 128 RINS,
                                 // 256 flow cover cuts, 512 local branching, 1024 network cut-set cuts,
                                 // 2048 root heuristics side by side when threads > 1
    double time_limit = 300.0;
    long node_limit = 50000000;
    int cut_rounds = 8, max_cuts_per_round = 60, refactor = 100, verbose = 0;
    int threads = 1;             // branch-and-bound threads (the root and everything before it run on one)
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

    // drop the rows from `keep` on (cuts appended after the original rows)
    void truncate_rows(int keep) {
        if (keep >= m) return;
        std::vector<int> ncp(n + 1, 0), nri;
        std::vector<double> ncv;
        for (int j = 0; j < n; ++j) {
            for (int t = cp[j]; t < cp[j + 1]; ++t)
                if (ri[t] < keep) { nri.push_back(ri[t]); ncv.push_back(cv[t]); }
            ncp[j + 1] = (int)nri.size();
        }
        cp.swap(ncp); ri.swap(nri); cv.swap(ncv);
        rlo.resize(keep); rhi.resize(keep);
        m = keep;
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
// upper triangular in the step order `order`.
// A basis change is a Forrest-Tomlin update: the entering column transformed by E and the row etas (the spike)
// replaces its column of U, the row of that step moves to the end of the order, and the entries of that row
// which now lie left of the diagonal are eliminated by one row eta. U stays sparse; only short row etas pile up.
struct Factor {
    int m = 0, rank = 0;
    double u = 0.1, abs_tol = 1e-9;     // the model is scaled: entries are O(1)
    bool symmetric = false;             // SPD matrix: diagonal pivots only (minimum-degree order, LDL')
    std::vector<int> prow, pcol;        // step t: pivot row and basis column
    std::vector<int> rstep, cstep;      // step of each row / basis column
    std::vector<int> order, pos;        // triangular order of the steps (changed by updates) and its inverse
    std::vector<int> lstart, lidx;      // L step t: x[lidx] -= lval * x[prow[t]]
    std::vector<double> lval;
    std::vector<double> udiag;          // U: diagonal of step t (row prow[t], column pcol[t])
    std::vector<std::vector<std::pair<int, double>>> urow;   // step t: off-diagonal entries (column, value)
    std::vector<std::vector<std::pair<int, double>>> ucol;   // column c: off-diagonal entries (step, value)
    std::vector<int> epos, estart, eidx;     // row etas: x[epos] -= sum eval * x[eidx]  (row indices)
    std::vector<double> eval;
    std::vector<int> bad_rows, bad_cols;     // unpivoted rows / columns when B is singular
    std::vector<double> work, spike, fwork;
    long u_nnz = 0, u_nnz0 = 0;              // off-diagonal entries of U: now / right after the factorisation
    double last_cost = 0.0;                  // work of the last factorisation, in entry operations (about 0.55 ns
                                             // each) plus 700 per row for its list set-up; deterministic, unlike
                                             // a timer, so runs repeat exactly
    bool stale = false;                      // an update failed: refactorise before the next solve
    bool refresh = false;                    // an update lost accuracy: refactorise soon
    double grow_tol = 1e-9, mult_tol = 1e4;  // ... when the diagonal check or a row-eta multiplier exceeds these

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

    static void remove_entry(std::vector<std::pair<int, double>>& v, int key) {
        for (size_t s = 0; s < v.size(); ++s)
            if (v[s].first == key) { v[s] = v.back(); v.pop_back(); return; }
    }

    // B given column-wise (bs, bi, bv). Returns false if B is singular; bad_cols / bad_rows then pair up the
    // basis columns and rows that could not be pivoted.
    bool factor(int m_, const std::vector<int>& bs, const std::vector<int>& bi, const std::vector<double>& bv) {
        m = m_; rank = 0; stale = refresh = false;
        double ops = (double)bs[m_] + 4.0 * m_;
        prow.clear(); pcol.clear(); lstart.assign(1, 0); lidx.clear(); lval.clear();
        udiag.clear(); urow.clear();
        epos.clear(); estart.assign(1, 0); eidx.clear(); eval.clear();
        bad_rows.clear(); bad_cols.clear();
        rstep.assign(m, -1); cstep.assign(m, -1);
        // active submatrix: values stored by rows, patterns by columns (rows already pivoted are skipped)
        std::vector<std::vector<std::pair<int, double>>> R(m);
        std::vector<std::vector<int>> C(m);
        for (int c = 0; c < m; ++c)
            for (int t = bs[c]; t < bs[c + 1]; ++t)
                if (bv[t] != 0.0) { R[bi[t]].push_back({c, bv[t]}); C[c].push_back(bi[t]); }
        std::vector<int> rcnt(m), ccnt(m), mark(m, -1);
        Buckets RB, CB;
        RB.init(m); CB.init(m);
        for (int i = 0; i < m; ++i) { rcnt[i] = (int)R[i].size(); RB.add(i, rcnt[i]); }
        for (int c = 0; c < m; ++c) { ccnt[c] = (int)C[c].size(); CB.add(c, ccnt[c]); }
        auto value = [&](int i, int c) {
            for (auto& e : R[i]) {
                ops += 1.0;
                if (e.first == c) return e.second;
            }
            return 0.0;
        };
        std::vector<double> rmax(m, -1.0);      // cached largest |entry| of each active row (-1: recompute)
        auto rowmax = [&](int i) {
            if (rmax[i] < 0.0) {
                ops += (double)R[i].size();
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
                ops += 1.0;
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
                ops += (double)Ri.size();
                double a = 0.0;
                for (size_t s = 0; s < Ri.size(); ++s)
                    if (Ri[s].first == Q) { a = Ri[s].second; Ri[s] = Ri.back(); Ri.pop_back(); break; }
                if (a != 0.0) {
                    double l = a / piv;
                    ops += 2.0 * (double)Ri.size() + (double)Rp.size();
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
            remove_entry(Rp, Q);                                // the pivot row, without its pivot, is row t of U
            urow.push_back(std::move(Rp));
            std::vector<std::pair<int, double>>().swap(Rp);
            std::vector<int>().swap(C[Q]);
            ++rank;
        }
        last_cost = ops + 700.0 * m;
        if (rank < m) {
            for (int c = 0; c < m; ++c) if (cstep[c] < 0) bad_cols.push_back(c);
            for (int i = 0; i < m; ++i) if (rstep[i] < 0) bad_rows.push_back(i);
            return false;
        }
        ucol.assign(m, {});
        u_nnz = 0;
        for (int t = 0; t < m; ++t) {
            for (auto& e : urow[t]) ucol[e.first].push_back({t, e.second});
            u_nnz += (long)urow[t].size();
        }
        u_nnz0 = u_nnz;
        order.resize(m); pos.resize(m);
        for (int t = 0; t < m; ++t) { order[t] = t; pos[t] = t; }
        return true;
    }

    // B x = b: on entry b is indexed by row, on exit it holds x indexed by basis position. With save_spike the
    // vector after L and the row etas is kept for a following replace_column.
    void ftran(std::vector<double>& b, bool save_spike = false) {
        for (int t = 0; t < m; ++t) {
            double v = b[prow[t]];
            if (v == 0.0) continue;
            for (int k = lstart[t]; k < lstart[t + 1]; ++k) b[lidx[k]] -= lval[k] * v;
        }
        for (size_t e = 0; e < epos.size(); ++e) {
            double s = 0.0;
            for (int k = estart[e]; k < estart[e + 1]; ++k) s += eval[k] * b[eidx[k]];
            b[epos[e]] -= s;
        }
        if (save_spike) spike = b;
        work.assign(m, 0.0);
        for (int p = m - 1; p >= 0; --p) {
            int t = order[p];
            double v = b[prow[t]];
            if (v == 0.0) continue;
            int c = pcol[t];
            double xc = v / udiag[t];
            work[c] = xc;
            for (auto& e : ucol[c]) b[prow[e.first]] -= e.second * xc;
        }
        b.swap(work);
    }

    // B' y = d: on entry d is indexed by basis position, on exit it holds y indexed by row
    void btran(std::vector<double>& d) {
        work.assign(m, 0.0);
        for (int p = 0; p < m; ++p) {
            int t = order[p];
            double v = d[pcol[t]];
            if (v == 0.0) continue;
            double wt = v / udiag[t];
            work[prow[t]] = wt;
            for (auto& e : urow[t]) d[e.first] -= e.second * wt;
        }
        for (int e = (int)epos.size() - 1; e >= 0; --e) {
            double v = work[epos[e]];
            if (v == 0.0) continue;
            for (int k = estart[e]; k < estart[e + 1]; ++k) work[eidx[k]] -= eval[k] * v;
        }
        for (int t = m - 1; t >= 0; --t) {
            double s = 0.0;
            for (int k = lstart[t]; k < lstart[t + 1]; ++k) s += lval[k] * work[lidx[k]];
            work[prow[t]] -= s;
        }
        d.swap(work);
    }

    // Forrest-Tomlin update: basis position r takes the column of the last ftran(..., save_spike = true), whose
    // entry r of B^-1 a is alpha_r. Returns false if the update is not numerically safe; the factorisation is
    // then unusable (stale) until the next factor().
    bool replace_column(int r, double alpha_r) {
        stale = true;
        int tr = cstep[r];
        double old_diag = udiag[tr];
        for (auto& e : ucol[r]) remove_entry(urow[e.first], r);      // the old column r leaves U
        u_nnz -= (long)ucol[r].size();
        ucol[r].clear();
        double d0 = 0.0;                                               // the spike becomes column r
        for (int i = 0; i < m; ++i) {
            double v = spike[i];
            if (std::fabs(v) < 1e-14) continue;
            int k = rstep[i];
            if (k == tr) { d0 = v; continue; }
            urow[k].push_back({r, v});
            ucol[r].push_back({k, v});
            ++u_nnz;
        }
        // row tr moves to the end of the order: eliminate its entries in the columns of the later steps
        fwork.assign(m, 0.0);
        for (auto& e : urow[tr]) { fwork[e.first] = e.second; remove_entry(ucol[e.first], tr); }
        u_nnz -= (long)urow[tr].size();
        urow[tr].clear();
        fwork[r] = d0;
        size_t first = eidx.size();
        double mmax = 0.0;
        for (int p = pos[tr] + 1; p < m; ++p) {
            int k = order[p], ck = pcol[k];
            double v = fwork[ck];
            if (v == 0.0) continue;
            fwork[ck] = 0.0;
            if (std::fabs(v) < 1e-14) continue;
            double mk = v / udiag[k];
            mmax = std::max(mmax, std::fabs(mk));
            eidx.push_back(prow[k]); eval.push_back(mk);
            for (auto& e : urow[k]) fwork[e.first] -= mk * e.second;
        }
        if (eidx.size() > first) { epos.push_back(prow[tr]); estart.push_back((int)eidx.size()); }
        double dnew = fwork[r], expect = old_diag * alpha_r;
        // the determinant says dnew = old_diag * alpha_r; a large difference means cancellation in the update
        double check = std::fabs(dnew - expect) / std::max(std::fabs(dnew), std::fabs(expect));
        if (!(std::fabs(dnew) > 1e-11) || !(check <= 1e-6)) return false;
        if (check > grow_tol || mmax > mult_tol) refresh = true;
        udiag[tr] = dnew;
        int p0 = pos[tr];
        order.erase(order.begin() + p0);
        order.push_back(tr);
        for (int p = p0; p < m; ++p) pos[order[p]] = p;
        stale = false;
        return true;
    }

    long nnz() const { return (long)lidx.size() + u_nnz + m; }
    // work added by the updates since the factorisation: row-eta entries and growth of U
    long eta_nnz() const { return (long)eidx.size() + (long)epos.size() + std::max(0L, u_nnz - u_nnz0); }
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
    long work_extra = 0;                             // other work in iteration equivalents (factorisations, weights)
    int since_refactor = 0;
    double eta_work = 0.0;                           // update entries processed by the solves since the refactor

    // Refactorise after opt.refactor updates (more on large bases, where a factorisation costs more than many
    // Forrest-Tomlin updates), earlier once the solves have spent more work on the updates than one
    // factorisation costs (an update entry costs about 2.7 factorisation operations), and at once when an
    // update failed or lost accuracy
    bool want_refactor() const {
        return F.stale || F.refresh || since_refactor >= std::max(opt.refactor, std::min(1000, m / 50)) ||
               (since_refactor >= 8 && 2.7 * eta_work > F.last_cost);
    }
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
    // after the model lost its last rows (cuts rolled back): the sizes follow the model again; bounds, costs and
    // the basis are then restored from a saved state
    void shrink_to_model() {
        n = M->n; m = M->m; N = n + m;
        x.resize(N); d.resize(N); ar.resize(N); acol.resize(m);
        build_rows();
    }

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
            if (k % 3 == 0) ++work_extra;
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
        work_extra += 3;
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
        eta_work = 0.0;
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
    // pivot row r of B^-1 [A -I] into ar. A sparse row (from a sparse row of B^-1) also lists its possibly nonzero
    // positions in ar_nz (ar_full false); every other entry of ar is then zero.
    std::vector<int> ar_nz;
    std::vector<char> ar_mark;
    bool ar_full = true;
    void row_alpha(int r) {
        rho.assign(m, 0.0); rho[r] = 1.0;
        F.btran(rho);
        int nz = 0;
        for (int i = 0; i < m; ++i) if (rho[i] != 0.0) ++nz;
        if (nz * 10 < m) {                       // sparse rho: accumulate row by row (the nonzero positions known)
            if (ar_full) std::fill(ar.begin(), ar.begin() + n, 0.0);
            else for (int j : ar_nz) if (j < n) ar[j] = 0.0;
            ar_mark.resize(n, 0);
            ar_nz.clear();
            for (int i = 0; i < m; ++i) {
                double v = rho[i];
                if (v == 0.0) continue;
                for (int t = rp[i]; t < rp[i + 1]; ++t) {
                    ar[rc[t]] += v * rv[t];
                    if (!ar_mark[rc[t]]) { ar_mark[rc[t]] = 1; ar_nz.push_back(rc[t]); }
                }
            }
            for (int j : ar_nz) ar_mark[j] = 0;
            for (int i = 0; i < m; ++i) if (rho[i] != 0.0) ar_nz.push_back(n + i);
            ar_full = false;
        } else {
            for (int j = 0; j < n; ++j) {
                double s = 0.0;
                for (int t = M->cp[j]; t < M->cp[j + 1]; ++t) s += rho[M->ri[t]] * M->cv[t];
                ar[j] = s;
            }
            ar_full = true;
        }
        for (int i = 0; i < m; ++i) ar[n + i] = -rho[i];
        for (int k = 0; k < m; ++k) ar[head[k]] = 0.0;
    }

    void col_alpha(int q) {
        acol.assign(m, 0.0);
        col(q, [&](int i, double v) { acol[i] += v; });
        F.ftran(acol, true);
    }

    // bounded dual simplex from a dual feasible basis
    Result dual() {
        int boxes_grown = 0;
        for (;;) {
            if (want_refactor() && !refactor_full()) return NUMERIC;
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
            F.replace_column(r, acol[r]);         // a failed update leaves F stale: refactorised next
            eta_work += 3.0 * F.eta_nnz();
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
        eta_work = 0.0;
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
        auto t_start = Clock::now();
        wp.assign(N, 1.0);
        // pricing candidates: the columns that may be dual infeasible, rebuilt after each refactorisation (which
        // recomputes the reduced costs); a column joins when an update makes it infeasible and leaves when pricing
        // finds it feasible. The entering column is the one a scan of all columns would choose.
        std::vector<int> cand;
        std::vector<char> incand(N, 0);
        auto consider = [&](int j) {
            if (!incand[j] && dual_infeas(j) > opt.tol_d) { incand[j] = 1; cand.push_back(j); }
        };
        auto rebuild = [&]() {
            for (int j : cand) incand[j] = 0;
            cand.clear();
            for (int j = 0; j < N; ++j) consider(j);
        };
        rebuild();
        for (;;) {
            if (want_refactor()) {
                if (!refactor_basis()) return NUMERIC;
                rebuild();
            }
            if (Clock::now() > t_end) return TIME_LIMIT;
            if (iters >= opt.max_lp_iter) return ITER_LIMIT;
            if (progress && iters % 5000 == 0) {
                double obj = 0.0;
                for (int j = 0; j < n; ++j) obj += c[j] * x[j];
                std::printf("    primal simplex: iteration %ld, objective %.10g, %zu candidates, %.1fs\n", iters, obj,
                            cand.size(), seconds_since(t_start));
            }
            int q = -1; double best = 0.0;
            for (size_t p = 0; p < cand.size();) {
                int j = cand[p];
                double v = dual_infeas(j);
                if (v <= opt.tol_d) { incand[j] = 0; cand[p] = cand.back(); cand.pop_back(); continue; }
                double score = v * v / wp[j];
                if (score > best || (score == best && j < q)) { best = score; q = j; }
                ++p;
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
                rebuild();
                continue;
            }
            int jl = head[r];
            bool leave_lb = dir * acol[r] > 0;
            for (int k = 0; k < m; ++k) x[head[k]] -= t * dir * acol[k];
            x[q] += t * dir;
            x[jl] = leave_lb ? lb[jl] : ub[jl];
            // reduced costs and Devex weights (only where the pivot row is nonzero, when that is known)
            double theta = d[q] / ar[q];
            double wq = std::max(wp[q], 1.0);
            auto update = [&](int j) {
                if (st[j] == BASIC || j == q) return;
                d[j] -= theta * ar[j];
                double g = ar[j] / ar[q];
                wp[j] = std::max(wp[j], g * g * wq);
                consider(j);
            };
            if (ar_full) for (int j = 0; j < N; ++j) update(j);
            else for (int j : ar_nz) update(j);
            d[q] = 0.0;
            d[jl] = -theta;
            wp[jl] = std::max(wq / (ar[q] * ar[q]), 1.0);
            st[jl] = leave_lb ? AT_LB : AT_UB;
            st[q] = BASIC;
            consider(jl);
            head[r] = q;
            F.replace_column(r, acol[r]);         // a failed update leaves F stale: refactorised next
            eta_work += 3.0 * F.eta_nnz();
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

    // Superbasic columns: nonbasic, but strictly inside their bounds. While they wait for push_superbasics their
    // bounds are fixed at their values (the real bounds are kept in sb_lo / sb_hi), so every solve sees them there.
    std::vector<int> superbasic;
    bool progress = false;                           // crossover with verbose output: report the long phases
    std::vector<int> nzk;                            // nonzero positions of the transformed column (push)
    std::vector<double> sb_lo, sb_hi;

    // crash basis from an approximate solution x0 (structural columns): columns and row activities strictly
    // inside their bounds become basic. With approximate reduced costs dj (all columns, logicals last) they are
    // ranked by the indicator p / (p + |d|), p the distance to the nearer bound, so the basis is made of columns
    // whose reduced cost is near zero and its duals stay close to the given ones; without, by p alone. On a
    // degenerate optimal face a first-order point has more such columns than there are rows; the ones that do
    // not fit stay at their values as superbasics (placing them at a bound would make the basis badly
    // infeasible). The rest go to the nearer bound.
    void init_from_point(const std::vector<double>& slb, const std::vector<double>& sub,
                         const std::vector<double>& x0, const std::vector<double>& dj = {}) {
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
            double p = std::min(v[j] - lb[j], ub[j] - v[j]);
            double key = p;
            if (!dj.empty()) key = std::isfinite(p) ? p / (p + std::fabs(dj[j])) : 1.0;
            cand.push_back({key, j});
        }
        std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
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
        // nonbasic columns still strictly inside their bounds (no room in the basis, or swapped out by the
        // repair) become superbasic at their values
        superbasic.clear();
        sb_lo.assign(N, 0.0); sb_hi.assign(N, 0.0);
        for (int j = 0; j < N; ++j) {
            if (st[j] == BASIC || lb[j] == ub[j]) continue;
            double tl = opt.crash_tol * (1.0 + std::fabs(lb[j])), tu = opt.crash_tol * (1.0 + std::fabs(ub[j]));
            if ((std::isfinite(lb[j]) && v[j] <= lb[j] + tl) || (std::isfinite(ub[j]) && v[j] >= ub[j] - tu)) continue;
            if (!std::isfinite(lb[j]) && !std::isfinite(ub[j]) && v[j] == 0.0) continue;    // free, at zero
            superbasic.push_back(j);
            sb_lo[j] = lb[j]; sb_hi[j] = ub[j];
            lb[j] = ub[j] = v[j]; st[j] = AT_LB;
        }
        if (!superbasic.empty()) compute_xB();
        exact_weights(false);
    }

    // Bound relaxation (crossover): a basic variable outside its bounds gets the violated bound moved to its
    // value, so the basis is exactly primal feasible for a slightly wider LP and ratio tests stay consistent.
    // restore_bounds puts the real bounds back (the dual simplex then repairs the violations).
    std::vector<std::pair<int, double>> relaxed_lo, relaxed_hi;
    long relax_infeasible_basics() {
        relaxed_lo.clear(); relaxed_hi.clear();
        for (int k = 0; k < m; ++k) {
            int j = head[k];
            if (x[j] < lb[j]) { relaxed_lo.push_back({j, lb[j]}); lb[j] = x[j]; }
            else if (x[j] > ub[j]) { relaxed_hi.push_back({j, ub[j]}); ub[j] = x[j]; }
        }
        return (long)(relaxed_lo.size() + relaxed_hi.size());
    }
    void restore_bounds() {
        for (auto& e : relaxed_lo) lb[e.first] = e.second;
        for (auto& e : relaxed_hi) ub[e.first] = e.second;
        relaxed_lo.clear(); relaxed_hi.clear();
    }

    // Primal push: every superbasic column moves to one of its bounds, the direction that does not raise the
    // objective first. A ratio test keeps the basic variables within their bounds; the basic variable that
    // blocks the move leaves the basis and the column enters in its place. Returns the number of basis changes,
    // or -1 if a refactorisation failed.
    long push_superbasics() {
        if (superbasic.empty()) return 0;
        compute_duals();
        long pivots = 0;
        auto t_push = Clock::now();
        for (size_t q = 0; q < superbasic.size(); ++q) {
            int j = superbasic[q];
            if (progress && q && q % 20000 == 0)
                std::printf("    pushed %zu of %zu superbasic columns (%ld basis changes), %.1fs\n", q,
                            superbasic.size(), pivots, seconds_since(t_push));
            // (refactorise while column j is still held at its value by its fixed bounds)
            if (want_refactor() && !refactor_basis()) return -1;
            double v = x[j];
            lb[j] = sb_lo[j]; ub[j] = sb_hi[j];
            if (Clock::now() > t_end) {                     // out of time: the rest goes to a bound
                place_nonbasic(j);
                continue;
            }
            col_alpha(j);
            // the ratio tests and the update only touch the nonzeros of the transformed column
            nzk.clear();
            for (int k = 0; k < m; ++k) if (acol[k] != 0.0) nzk.push_back(k);
            double s0 = d[j] < -opt.tol_d ? 1.0 : d[j] > opt.tol_d ? -1.0 : (ub[j] - v <= v - lb[j] ? 1.0 : -1.0);
            bool done = false;
            for (int pass = 0; pass < 2 && !done; ++pass) {
                double s = pass == 0 ? s0 : -s0;
                double dist = s > 0 ? ub[j] - v : v - lb[j];
                // Harris two-pass ratio test over the basic variables (x_B moves by -t s alpha)
                double tmax = INF;
                for (int k : nzk) {
                    double a = s * acol[k];
                    if (std::fabs(a) < opt.tol_piv) continue;
                    int jb = head[k];
                    double room = a > 0 ? x[jb] - lb[jb] : ub[jb] - x[jb];
                    if (std::isfinite(room)) tmax = std::min(tmax, (std::max(room, 0.0) + opt.tol_p) / std::fabs(a));
                }
                int r = -1; double t = 0.0, amax = 0.0;
                if (tmax < INF) {
                    for (int k : nzk) {
                        double a = s * acol[k];
                        if (std::fabs(a) < opt.tol_piv) continue;
                        int jb = head[k];
                        double room = a > 0 ? x[jb] - lb[jb] : ub[jb] - x[jb];
                        if (!std::isfinite(room)) continue;
                        double ratio = std::max(room, 0.0) / std::fabs(a);
                        if (ratio <= tmax && std::fabs(a) > amax) { amax = std::fabs(a); r = k; t = ratio; }
                    }
                }
                if (std::isfinite(dist) && (r < 0 || dist <= t)) {         // the column reaches its bound
                    for (int k : nzk) x[head[k]] -= dist * s * acol[k];
                    st[j] = s > 0 ? AT_UB : AT_LB;
                    x[j] = s > 0 ? ub[j] : lb[j];
                    done = true;
                } else if (r >= 0) {                                         // a basic variable blocks: swap
                    int jl = head[r];
                    bool leave_lb = s * acol[r] > 0;
                    for (int k : nzk) x[head[k]] -= t * s * acol[k];
                    x[j] = v + s * t;
                    x[jl] = leave_lb ? lb[jl] : ub[jl];
                    st[jl] = leave_lb ? AT_LB : AT_UB;
                    st[j] = BASIC;
                    head[r] = j;
                    F.replace_column(r, acol[r]);
                    eta_work += 3.0 * F.eta_nnz();
                    ++iters; ++since_refactor; ++pivots;
                    done = true;
                }
            }
            if (!done) {               // free column that nothing blocks in either direction: it goes to zero
                place_nonbasic(j);
                double dx = x[j] - v;
                for (int k : nzk) x[head[k]] -= dx * acol[k];
            }
        }
        superbasic.clear();
        if (!refactor_basis()) return -1;
        w.assign(m, m <= 5000 ? -1.0 : 1.0);
        exact_weights(false);
        return pivots;
    }
};

// ---------------------------------------------------------------------------------------------- MILP
struct BoundChange {
    int var;
    double lo, hi;
};
// Bound changes of the nodes in the parallel search: a chain of immutable blocks (one per branching or propagation
// step) that all nodes below share; a node's bounds are the root bounds with the changes of its chain applied,
// deeper ones winning. The reference counts are thread-safe, so a node can move between threads.
struct BoundBlock {
    mutable std::shared_ptr<const BoundBlock> parent;
    std::vector<BoundChange> ch;
    ~BoundBlock() {             // release a long chain in a loop (one destructor calling the next could overflow)
        std::shared_ptr<const BoundBlock> p = std::move(parent);
        while (p && p.use_count() == 1) p = std::move(p->parent);
    }
};
struct Node {
    double bound;
    long id;
    int branch_var, dir;       // dir: -1 down, +1 up (for pseudocost update)
    double frac, parent_obj;
    int leaf = -1;                // last branching record on the path from the root (see Branch)
    std::shared_ptr<const BoundBlock> chain;   // instead of leaf, in the parallel search
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
    std::vector<Node*> take() {         // empty the queue, handing over its nodes
        std::vector<Node*> v;
        v.swap(c);
        return v;
    }
};

// k threads that run job(w), w = 0..k-1, once per call of run(); w = 0 is the calling thread
struct Crew {
    std::vector<std::thread> th;
    std::mutex mu;
    std::condition_variable go, done;
    std::function<void(int)> job;
    long gen = 0;
    int pending = 0;
    bool quit = false;
    explicit Crew(int k) {
        for (int w = 1; w < k; ++w) th.emplace_back([this, w]() { loop(w); });
    }
    void loop(int w) {
        long seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> l(mu);
                go.wait(l, [&]() { return quit || gen != seen; });
                if (quit) return;
                seen = gen;
            }
            job(w);
            std::lock_guard<std::mutex> l(mu);
            if (--pending == 0) done.notify_one();
        }
    }
    void run(std::function<void(int)> f) {
        {
            std::lock_guard<std::mutex> l(mu);
            job = std::move(f);
            pending = (int)th.size();
            ++gen;
        }
        go.notify_all();
        job(0);
        std::unique_lock<std::mutex> l(mu);
        done.wait(l, [&]() { return pending == 0; });
    }
    ~Crew() {
        {
            std::lock_guard<std::mutex> l(mu);
            quit = true;
        }
        go.notify_all();
        for (auto& t : th) t.join();
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
    std::vector<double> orig_lb, orig_ub;     // bounds of the model itself (root bounds may be tightened)
    double inc_obj = INF;
    std::vector<double> pc_sum[2];
    std::vector<int> pc_cnt[2];
    std::vector<std::vector<std::pair<int, double>>> rowsR;   // row copy of the ORIGINAL model
    int m_orig = 0;
    std::vector<double> colscale;   // x_j (original) = colscale[j] * x_j (scaled)
    std::vector<double> rowscale;   // row i (scaled) = rowscale[i] * row i (original); y_i (scaled) = y_i / rowscale[i]

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
        rowscale.swap(rs);
    }

    int solve(Info& info, std::vector<double>& xout) {
        obj_step = objective_step();         // integer columns keep their scale, so it holds for the scaled model
        scale_model();
        int status = solve_scaled(info, xout);
        for (size_t j = 0; j < xout.size(); ++j) xout[j] *= colscale[j];
        return status;
    }

    // Crossover: approximate LP solution x0 (and duals y0, if given; e.g. from the GPU engine) -> optimal vertex.
    // Crash basis from x0, bounds of infeasible basic variables relaxed to their values, primal push of the
    // superbasic columns to their bounds, primal simplex to the optimum of the relaxed LP, real bounds back and
    // dual simplex, then a final dual simplex check on a fresh factorisation.
    int crossover(const std::vector<double>& x0, const std::vector<double>& y0, Info& info, std::vector<double>& xout) {
        scale_model();
        std::vector<double> xs(M.n);
        for (int j = 0; j < M.n; ++j) xs[j] = x0[j] / colscale[j];
        // reduced costs of the approximate duals in the scaled model (structural columns, then logicals: the
        // logical of row i is -e_i, so its reduced cost is y_i)
        std::vector<double> dj;
        if (!y0.empty()) {
            dj.assign(M.n + M.m, 0.0);
            for (int i = 0; i < M.m; ++i) dj[M.n + i] = y0[i] / rowscale[i];
            for (int j = 0; j < M.n; ++j) {
                double s = M.c[j];
                for (int t = M.cp[j]; t < M.cp[j + 1]; ++t) s -= M.cv[t] * dj[M.n + M.ri[t]];
                dj[j] = s;
            }
        }
        t0 = Clock::now();
        S.M = &M; S.opt = opt;
        S.t_end = t0 + std::chrono::milliseconds((long long)(opt.time_limit * 1000));
        S.init_from_point(M.lb, M.ub, xs, dj);
        long crash_basic = 0;
        for (int k = 0; k < S.m; ++k) crash_basic += S.head[k] < S.n;
        long n_relaxed = S.relax_infeasible_basics();
        long n_super = (long)S.superbasic.size();
        S.progress = opt.verbose != 0;
        if (opt.verbose)
            std::printf("crossover: %d rows, %d columns; crash basis %.1fs, %ld bounds relaxed, %ld superbasic to push\n",
                        S.m, S.n, seconds_since(t0), n_relaxed, n_super);
        long it_push = S.push_superbasics();
        if (opt.verbose) std::printf("crossover: push done %.1fs\n", seconds_since(t0));
        if (it_push < 0) {
            info.time = seconds_since(t0);
            return 5;
        }
        // iteration budget: a crossover that needs more than this is slower than a cold start (the caller then
        // falls back to one)
        S.opt.max_lp_iter = std::max(5000L, (long)S.m + S.n);
        // The pushed basis is primal feasible for the relaxed bounds: primal simplex on the true costs to the
        // optimum of the relaxed LP, then the real bounds back and the dual simplex from that dual feasible basis.
        Result r = S.primal();
        double t_primal = seconds_since(t0);
        long it_primal = S.iters - it_push, it_dual = 0;
        S.restore_bounds();
        bool shifted = false;
        if (r == OPTIMAL) {
            S.w.assign(S.m, S.m <= 5000 ? -1.0 : 1.0);
            r = S.refactor_full() ? S.dual() : NUMERIC;
            it_dual = S.iters - it_push - it_primal;
        } else if (r != TIME_LIMIT && r != ITER_LIMIT) {
            // primal simplex in trouble: dual simplex on shifted costs from the current basis, then the primal
            // simplex without the shifts
            shifted = true;
            r = S.refactor_basis() ? OPTIMAL : NUMERIC;
            if (r == OPTIMAL) {
                S.shift_costs();
                r = S.dual();
                it_dual = S.iters - it_push - it_primal;
            }
            if (r == OPTIMAL) {
                S.unshift_costs();
                r = S.refactor_basis() ? S.primal() : NUMERIC;
            }
        }
        long it_main = S.iters;
        double t_main = seconds_since(t0);
        if (r == OPTIMAL) {
            S.w.assign(S.m, 1.0);
            r = S.solve();
        }
        std::vector<double> xkeep;
        if (r == OPTIMAL) {                  // polish with 100x tighter tolerances, as for a cold LP solve
            xkeep = S.x;
            S.opt.tol_p = S.opt.tol_d = 1e-9;
            S.opt.max_lp_iter = S.iters + std::max(5000L, (long)S.m + S.n);
            if (!(S.refactor_full() && S.dual() == OPTIMAL && (S.since_refactor == 0 || S.solve() == OPTIMAL)))
                S.x = xkeep;
            S.opt = opt;
        }
        if (opt.verbose)
            std::printf("crossover: %ld structural columns basic in the crash basis, %ld bounds relaxed, %ld superbasic "
                        "pushed (%ld basis changes), %ld primal + %ld dual%s + %ld cleanup iterations, %.2f s\n",
                        crash_basic, n_relaxed, n_super, it_push, it_primal, it_dual,
                        shifted ? " (shifted costs)" : "", S.iters - it_main, seconds_since(t0));
        if (opt.verbose)
            std::printf("crossover: primal done %.1fs, dual done %.1fs, end %.1fs\n", t_primal, t_main, seconds_since(t0));
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
            if (xs[j] < orig_lb[j] - 1e-6 || xs[j] > orig_ub[j] + 1e-6) return false;
            if (M.isint[j] && std::fabs(xs[j] - std::round(xs[j])) > opt.int_tol) return false;
        }
        for (int i = 0; i < m_orig; ++i) {
            double s = 0.0;
            for (auto& e : rowsR[i]) s += e.second * xs[e.first];
            // tolerance 1e-6 (1 + |rhs|) in the units of the row as given: scaled row i is rowscale[i] times it, so
            // a row scaled far down could otherwise hide a large violation (enigma: 1.0 passed as 1e-6)
            double rs = i < (int)rowscale.size() ? rowscale[i] : 1.0;
            double sc = rs + std::max(std::fabs(M.rlo[i] == -INF ? 0 : M.rlo[i]), std::fabs(M.rhi[i] == INF ? 0 : M.rhi[i]));
            if (s < M.rlo[i] - 1e-6 * sc || s > M.rhi[i] + 1e-6 * sc) return false;
        }
        return true;
    }

    // prune threshold; +inf while there is no incumbent (avoids inf - inf = NaN)
    double cutoff() const {
        if (!std::isfinite(inc_obj)) return INF;
        double c = inc_obj - opt.gap * std::max(1.0, std::fabs(inc_obj));
        // integral objective: a node whose bound exceeds the incumbent minus one step holds nothing better (the margin
        // covers the LP's rounding)
        if (obj_step > 0) c = std::min(c, inc_obj - obj_step + 1e-6 * std::max(1.0, std::fabs(inc_obj)));
        return c;
    }

    // The objective takes only multiples of obj_step on integer points when every column with a cost is an integer
    // column and every cost is an integer multiple of it (costs with up to six decimals); 0 if not. A dual bound may
    // then be rounded up to the next multiple, and nodes pruned accordingly (HiGHS does the same).
    double obj_step = 0.0;
    double objective_step() const {
        bool any = false;
        for (int j = 0; j < M.n; ++j) {
            if (M.c[j] == 0.0) continue;
            if (!M.isint[j]) return 0.0;
            any = true;
        }
        if (!any) return 0.0;
        for (int k = 0; k <= 6; ++k) {
            double f = std::pow(10.0, k);
            long long g = 0;
            bool ok = true;
            for (int j = 0; j < M.n && ok; ++j) {
                if (M.c[j] == 0.0) continue;
                double v = std::fabs(M.c[j]) * f, r = std::round(v);
                if (r < 1.0 || r > 1e12 || std::fabs(v - r) > 1e-9 * std::max(1.0, v)) { ok = false; break; }
                long long a = (long long)r;
                while (a) { long long t = g % a; g = a; a = t; }
            }
            if (ok && g > 0) return (double)g / f;
        }
        return 0.0;
    }
    // a dual bound rounded up to the next multiple of the objective step
    double round_bound(double b) const {
        if (obj_step <= 0 || !std::isfinite(b)) return b;
        return std::ceil(b / obj_step - 1e-6) * obj_step;
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

    // node LP state, so that heuristics can change bounds and leave the LP exactly as they found it
    struct LPState {
        std::vector<double> lb, ub, c, w;
        std::vector<int> head;
        std::vector<int8_t> st;
        std::vector<char> art;
        double big;
    };
    LPState save_lp() const { return {S.lb, S.ub, S.c, S.w, S.head, S.st, S.art, S.big}; }
    void restore_lp(const LPState& s) {
        S.lb = s.lb; S.ub = s.ub; S.c = s.c; S.w = s.w; S.head = s.head; S.st = s.st; S.art = s.art; S.big = s.big;
        S.refactor_full();
    }

    // Complete an integer assignment xr: fix the integer columns at xr, solve the LP over the continuous columns
    // with the original costs, and offer the result as an incumbent. The LP is restored afterwards.
    Result last_completion = OPTIMAL;            // LP status of the last complete_rounding (diagnostics)
    long root_lp_iters = 0;                      // iterations of the root LP (sizes the heuristics' LP budgets)
    double root_lp_seconds = 0.0;                // ... and its time
    bool complete_rounding(const std::vector<double>& xr, const std::vector<double>& c0, bool has_cont) {
        if (!has_cont) {
            double before = inc_obj;
            try_incumbent(xr);
            return inc_obj < before;
        }
        LPState saved = save_lp();
        for (int j = 0; j < M.n; ++j) {
            S.c[j] = c0[j];
            if (M.isint[j]) { S.lb[j] = xr[j]; S.ub[j] = xr[j]; }
        }
        double before = inc_obj;
        last_completion = S.refactor_full() ? S.dual() : NUMERIC;
        if (last_completion == OPTIMAL) try_incumbent(std::vector<double>(S.x.begin(), S.x.begin() + M.n));
        restore_lp(saved);
        return inc_obj < before;
    }

    // Feasibility pump (Fischetti, Glover & Lodi 2005; objective pump of Achterberg & Berthold 2007): alternate
    // between rounding the LP point and an LP that minimises the L1 distance to the rounding (integer columns at
    // a bound of the rounding), mixed with a fading share of the original objective. A repeated rounding flips the
    // columns furthest from it; a longer cycle perturbs the rounding at random. Stops at the first solution.
    void feasibility_pump(int max_rounds, long lp_budget) {
        with_budget(lp_budget, 0.05 * opt.time_limit, [&]() { pump_body(max_rounds); });
    }

    void pump_body(int max_rounds) {
        int n = M.n;
        std::vector<double> c0(S.c.begin(), S.c.begin() + n);
        double cnorm = 0.0;
        int nint = 0;
        bool has_cont = false;
        for (int j = 0; j < n; ++j) {
            cnorm += c0[j] * c0[j];
            if (M.isint[j]) ++nint; else has_cont = true;
        }
        cnorm = std::sqrt(cnorm);
        double scale = cnorm > 0 ? std::sqrt((double)nint) / cnorm : 0.0;
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        auto round_point = [&](std::vector<double>& xr) {
            xr.assign(S.x.begin(), S.x.begin() + n);
            for (int j = 0; j < n; ++j) if (M.isint[j]) xr[j] = std::round(xr[j]);
        };
        std::vector<double> xr, xn;
        round_point(xr);
        std::vector<size_t> history;
        auto hash_of = [&](const std::vector<double>& v) {
            size_t h = 1469598103934665603ull;
            for (int j = 0; j < n; ++j) if (M.isint[j]) h = (h ^ (size_t)(long long)v[j]) * 1099511628211ull;
            return h;
        };
        double alpha = 1.0;
        for (int k = 0; k < max_rounds; ++k) {
            if (complete_rounding(xr, c0, has_cont)) return;
            alpha *= 0.9;
            for (int j = 0; j < n; ++j) {
                double dist = 0.0;
                if (M.isint[j]) {
                    if (xr[j] <= S.lb[j] + 1e-9) dist = 1.0;
                    else if (xr[j] >= S.ub[j] - 1e-9) dist = -1.0;
                }
                S.c[j] = (1.0 - alpha) * dist + alpha * scale * c0[j];
            }
            if (!S.refactor_basis() || S.primal() != OPTIMAL) return;
            round_point(xn);
            bool same = true;
            for (int j = 0; j < n && same; ++j) if (M.isint[j] && xn[j] != xr[j]) same = false;
            if (same) {                                   // short cycle: flip the T most distant columns
                std::vector<std::pair<double, int>> dev;
                for (int j = 0; j < n; ++j) if (M.isint[j]) dev.push_back({-std::fabs(S.x[j] - xr[j]), j});
                int T = 10 + (int)(unif(rng) * 20);
                std::partial_sort(dev.begin(), dev.begin() + std::min<size_t>(T, dev.size()), dev.end());
                for (int t = 0; t < T && t < (int)dev.size() && dev[t].first < 0; ++t) {
                    int j = dev[t].second;
                    double up = xr[j] + 1, dn = xr[j] - 1;
                    xn[j] = S.x[j] > xr[j] ? std::min(up, S.ub[j]) : std::max(dn, S.lb[j]);
                }
            }
            size_t h = hash_of(xn);
            if (std::find(history.begin(), history.end(), h) != history.end()) {   // longer cycle: perturb
                for (int j = 0; j < n; ++j) {
                    if (!M.isint[j]) continue;
                    double rho = unif(rng) - 0.3;
                    if (std::fabs(S.x[j] - xn[j]) + std::max(rho, 0.0) > 0.5)
                        xn[j] = xn[j] >= S.x[j] ? std::max(xn[j] - 1, S.lb[j]) : std::min(xn[j] + 1, S.ub[j]);
                }
            }
            history.push_back(h);
            if (history.size() > 5) history.erase(history.begin());
            xr.swap(xn);
        }
    }

    // Diving: from the current LP solution, repeatedly fix one fractional integer column at an integer and
    // re-solve with the dual simplex from the current basis, until the LP solution is integral (a new
    // incumbent), infeasible after one backtrack, no better than the incumbent, or the LP budget is spent.
    // Without an incumbent the column closest to an integer is rounded (fractional diving); with one, the column
    // is moved towards the incumbent's value (guided diving). The LP is restored afterwards.
    double heur_time = 0.0;                     // seconds spent in diving / feasibility pump
    // heuristics share 10% of the time; fix-and-propagate may go up to 25% while there is no incumbent on models
    // whose LPs are expensive (root LP above 1% of the time limit), where 10% does not cover a completion LP
    bool heur_allowed(double share = 0.1) const {
        return heur_time < share * opt.time_limit && Clock::now() < S.t_end;
    }

    // run a heuristic with an LP iteration budget and a deadline; the LP is restored afterwards
    template <class Fn> void with_budget(long lp_budget, double seconds, Fn fn, double share = 0.1) {
        if (!heur_allowed(share)) return;
        auto start = Clock::now();
        LPState saved = save_lp();
        long max_iter = S.opt.max_lp_iter;
        auto t_end = S.t_end;
        S.opt.max_lp_iter = S.iters + lp_budget;
        S.t_end = std::min(t_end, start + std::chrono::milliseconds((long long)(seconds * 1000)));
        fn();
        S.opt.max_lp_iter = max_iter;
        S.t_end = t_end;
        restore_lp(saved);
        heur_time += seconds_since(start);
    }

    void dive(long lp_budget) {
        with_budget(lp_budget, 0.03 * opt.time_limit, [&]() { dive_body(); });
    }

    void dive_body() {
        bool guided = std::isfinite(inc_obj), backtracked = false;
        for (int depth = 0; depth < 2 * M.n; ++depth) {
            int bj = -1; double bscore = INF, target = 0.0;
            for (int j = 0; j < M.n; ++j) {
                if (!M.isint[j]) continue;
                double v = S.x[j], fl = std::floor(v), f = v - fl;
                if (f < opt.int_tol || f > 1 - opt.int_tol) continue;
                double t = guided ? (inc[j] < v ? fl : fl + 1.0) : std::round(v);
                double score = std::fabs(v - t);
                if (score < bscore) { bscore = score; bj = j; target = t; }
            }
            if (bj < 0) {                                       // integral LP solution
                try_incumbent(std::vector<double>(S.x.begin(), S.x.begin() + M.n));
                break;
            }
            double v = S.x[bj], olb = S.lb[bj], oub = S.ub[bj];
            if (target <= v) S.ub[bj] = target; else S.lb[bj] = target;
            Result r = S.dual();
            if (r == INFEASIBLE && !backtracked) {              // one backtrack: the other side of the same column
                backtracked = true;
                S.lb[bj] = olb; S.ub[bj] = oub;
                if (target <= v) S.lb[bj] = std::ceil(v); else S.ub[bj] = std::floor(v);
                r = S.dual();
            }
            if (r != OPTIMAL || S.objective() >= cutoff()) break;
            rounding_heuristic();
        }
    }

    // Fix-and-propagate: the integer columns are fixed one by one and every fixing is propagated through the rows
    // (continuous columns included); a value that the propagation proves impossible is undone and the other side
    // of the LP value tried. With every integer column fixed, one LP over the continuous columns completes the
    // point. One propagation per fixing and a single LP, so it finds a first solution on models with thousands of
    // binaries where diving and the pump need too many LPs. Two orders: the rounded LP point, most integral
    // column first; and the strongest LP decisions first (largest LP value, rounded up once it is clearly
    // positive), which keeps the decisions a time-indexed LP spreads thinly over many periods.
    // (budgets: the completion LP starts from the node basis with the integer columns fixed; if it needs more
    // than about twice the root LP's iterations, or three times its time, it is unlikely to finish usefully)
    void fix_and_propagate(long lp_budget) {
        for (int mode = 0; mode < 2; ++mode)
            with_budget(lp_budget, std::min(0.15, std::max(0.03, 3.0 * root_lp_seconds / opt.time_limit)) *
                        opt.time_limit, [&]() { fix_propagate_body(mode); },
                        !std::isfinite(inc_obj) && root_lp_seconds > 0.01 * opt.time_limit ? 0.25 : 0.1);
    }

    void fix_propagate_body(int mode) {
        if ((int)prop_rows.size() != M.m) build_propagation();
        const int n = M.n;
        std::vector<double> L(S.lb.begin(), S.lb.begin() + n), U(S.ub.begin(), S.ub.begin() + n);
        for (int j = 0; j < n; ++j) if (S.art[j]) { L[j] = root_lb[j]; U[j] = root_ub[j]; }   // not the boxes
        std::vector<int> order;
        for (int j = 0; j < n; ++j) if (M.isint[j] && L[j] < U[j]) order.push_back(j);
        auto frac = [&](int j) { return std::fabs(S.x[j] - std::round(S.x[j])); };
        if (mode == 0)
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return frac(a) < frac(b); });
        else
            std::stable_sort(order.begin(), order.end(),
                             [&](int a, int b) { return S.x[a] - L[a] > S.x[b] - L[b]; });
        std::vector<BoundTrail> trail;
        std::vector<int> changed;
        auto t_fp = Clock::now();
        long long w_fp = fp_work;
        auto undo = [&](size_t mark) {
            while (trail.size() > mark) {
                L[trail.back().j] = trail.back().l; U[trail.back().j] = trail.back().u;
                trail.pop_back();
            }
        };
        auto fix = [&](int j, double v) {
            size_t mark = trail.size();
            trail.push_back({j, L[j], U[j]});
            L[j] = U[j] = v;
            changed.clear();
            if (propagate({j}, L, U, changed, &trail, true)) return true;
            undo(mark);
            return false;
        };
        for (size_t q = 0; q < order.size(); ++q) {
            int j = order[q];
            if (L[j] == U[j]) continue;                          // fixed by the propagation of an earlier column
            if ((q & 255) == 0 && Clock::now() > S.t_end) return;
            double x = S.x[j];
            double v = std::min(U[j], std::max(L[j], mode == 0 ? std::round(x) : std::ceil(x - 0.1)));
            if (fix(j, v)) continue;
            double w = x >= v ? v + 1.0 : v - 1.0;               // the other side of the LP value
            if (w < L[j] || w > U[j]) w = x >= v ? v - 1.0 : v + 1.0;
            if (w < L[j] || w > U[j] || !fix(j, w)) {            // both sides impossible: give up
                if (opt.verbose)
                    std::printf("  fix-and-propagate (%s): stuck after %zu of %zu columns\n",
                                mode == 0 ? "rounded" : "strongest first", q, order.size());
                return;
            }
        }
        std::vector<double> xr(S.x.begin(), S.x.begin() + n), c0(S.c.begin(), S.c.begin() + n);
        bool has_cont = false;
        for (int j = 0; j < n; ++j) {
            if (M.isint[j]) xr[j] = L[j]; else has_cont = true;
        }
        double before = inc_obj;
        double t_prop = seconds_since(t_fp);
        complete_rounding(xr, c0, has_cont);
        if (opt.verbose)
            std::printf("  fix-and-propagate (%s): all %zu columns fixed (propagation %lld entries, %.2f s), %s\n",
                        mode == 0 ? "rounded" : "strongest first", order.size(), fp_work - w_fp, t_prop,
                        inc_obj < before ? "new incumbent" : last_completion == INFEASIBLE ? "LP infeasible" :
                        last_completion == OPTIMAL ? "LP point rejected" : "LP not finished");
    }

    // RINS (Danna, Rothberg & Le Pape 2005): the integer columns on which the incumbent and the current LP solution
    // agree are fixed at that value, and the sub-MIP over the others (root cuts included) is solved by a nested
    // branch-and-cut with a node and time budget and the incumbent as cutoff. Skipped when fewer than 30% of the
    // integer columns agree (the sub-MIP would be about as hard as the model) or when all do.
    long rins_calls = 0;
    bool rins_off = false;                       // a sub-MIP that could not even branch: not worth repeating
    double rins_time = 0.0;
    const std::vector<int8_t>* warm_st = nullptr;  // starting basis for solve_scaled (set by a parent's RINS)
    const std::vector<int>* warm_head = nullptr;
    Clock::time_point rins_last;
    // Only while the gap to the bound exceeds 1% (near-proven models do not need it). In the tree the calls are
    // spaced by rins_wait: 10% of the time limit after a success or a new incumbent, doubled after each call that
    // found nothing. RINS may use 30% of the time limit, and up to 70% while its last call improved the incumbent
    // (crude scheduling: each call found a cheaper schedule until the 30% ran out).
    double rins_last_inc = INF;
    double rins_wait = 0.0;
    bool rins_last_ok = false;
    long rins_hits = 0;                          // RINS calls that improved the incumbent
    bool rins_due() {
        double wait = inc_obj < rins_last_inc - 1e-9 ? 0.1 * opt.time_limit : rins_wait;
        return seconds_since(rins_last) > wait;
    }
    void rins(double seconds, long nodes, double bound) {
        if (rins_off || !std::isfinite(inc_obj) || rins_time > (rins_last_ok ? 0.7 : 0.3) * opt.time_limit ||
            Clock::now() >= S.t_end)
            return;
        if (inc_obj - bound <= 0.01 * std::max(1.0, std::fabs(inc_obj))) return;
        rins_last = Clock::now();
        int nint = 0, agree = 0;
        for (int j = 0; j < M.n; ++j) {
            if (!M.isint[j]) continue;
            ++nint;
            if (std::fabs(S.x[j] - inc[j]) < 1e-6) ++agree;
        }
        if (agree < 0.3 * nint || agree == nint) return;
        auto start = Clock::now();
        double left = std::chrono::duration<double>(S.t_end - start).count();
        if (left < 1.0) return;
        Solver sub;
        sub.M = M;
        for (int j = 0; j < M.n; ++j) {
            sub.M.lb[j] = root_lb[j]; sub.M.ub[j] = root_ub[j];
            if (M.isint[j] && std::fabs(S.x[j] - inc[j]) < 1e-6) sub.M.lb[j] = sub.M.ub[j] = inc[j];
        }
        sub.opt = opt;
        sub.opt.time_limit = std::min(seconds, 0.5 * left);
        sub.opt.node_limit = nodes;
        sub.opt.features = opt.features & ~(128 | 8);   // a quick search: no nested RINS, no strong branching
        sub.opt.cut_rounds = 0;
        sub.opt.verbose = 0;
        sub.opt.threads = 1;
        sub.obj_step = obj_step;
        sub.inc = inc; sub.inc_obj = inc_obj;
        sub.rowscale = rowscale;                  // its rows are ours, scaled: same feasibility test
        sub.warm_st = &S.st; sub.warm_head = &S.head;
        Info sinfo;
        std::vector<double> sx;
        sub.solve_scaled(sinfo, sx);
        ++rins_calls;
        if (sinfo.nodes < 2) rins_off = true;
        double before = inc_obj;
        if (sub.inc_obj < inc_obj - 1e-9) try_incumbent(sub.inc);
        rins_wait = inc_obj < before ? 0.1 * opt.time_limit : std::max(0.2 * opt.time_limit, 2 * rins_wait);
        rins_last_ok = inc_obj < before;
        rins_hits += rins_last_ok;
        rins_last_inc = inc_obj;
        rins_time += seconds_since(start);
        if (opt.verbose)
            std::printf("  RINS: %d of %d integer columns fixed, %.0f nodes, %.2f s: %s\n", agree, nint, sinfo.nodes,
                        seconds_since(start), inc_obj < before ? "better incumbent" : "nothing better");
    }

    // Local branching (Fischetti & Lodi 2003): the model with one more row, at most k of the binary columns flipped
    // from the incumbent, solved by a nested branch-and-bound with a node and time budget and the incumbent as
    // cutoff. Where RINS fixes the columns on which the LP point agrees with the incumbent, this searches the whole
    // neighbourhood of the incumbent. It runs when a RINS call found nothing better on a model where RINS has found
    // better solutions before (neighbourhoods pay there), while the gap exceeds 1%, within 10% of the time limit,
    // and stops after two calls in a row found nothing (on models whose tree is about to finish it only takes
    // time). A neighbourhood searched to the end without a better solution is widened next time.
    long lb_calls = 0;
    double lb_time = 0.0;
    int lb_k = 10, lb_fails = 0;
    void local_branching(double seconds, long nodes, double bound) {
        if (!std::isfinite(inc_obj) || rins_hits == 0 || lb_fails >= 2 || lb_time > 0.1 * opt.time_limit ||
            Clock::now() >= S.t_end)
            return;
        if (inc_obj - bound <= 0.01 * std::max(1.0, std::fabs(inc_obj))) return;
        auto start = Clock::now();
        double left = std::chrono::duration<double>(S.t_end - start).count();
        if (left < 1.0) return;
        std::vector<std::pair<int, double>> row;
        int ones = 0;
        for (int j = 0; j < M.n; ++j) {
            if (!M.isint[j] || root_lb[j] != 0.0 || root_ub[j] != 1.0) continue;
            if (inc[j] > 0.5) { row.push_back({j, -1.0}); ++ones; }
            else row.push_back({j, 1.0});
        }
        if ((int)row.size() <= 2 * lb_k) return;
        int k_used = lb_k;
        Solver sub;
        sub.M = M;
        for (int j = 0; j < M.n; ++j) { sub.M.lb[j] = root_lb[j]; sub.M.ub[j] = root_ub[j]; }
        sub.M.add_rows({row}, {-INF}, {(double)(lb_k - ones)});
        sub.opt = opt;
        sub.opt.time_limit = std::min(seconds, 0.5 * left);
        sub.opt.node_limit = nodes;
        sub.opt.features = opt.features & ~(128 | 8 | 512);
        sub.opt.cut_rounds = 0;
        sub.opt.verbose = 0;
        sub.opt.threads = 1;
        sub.obj_step = obj_step;
        sub.inc = inc; sub.inc_obj = inc_obj;
        sub.rowscale = rowscale;
        // warm start from the node basis, the new row's logical basic
        std::vector<int8_t> st2(S.st);
        std::vector<int> head2(S.head);
        st2.push_back(BASIC);
        head2.push_back(M.n + M.m);
        sub.warm_st = &st2; sub.warm_head = &head2;
        Info sinfo;
        std::vector<double> sx;
        int status = sub.solve_scaled(sinfo, sx);
        ++lb_calls;
        double before = inc_obj;
        if (sub.inc_obj < inc_obj - 1e-9) try_incumbent(sub.inc);
        bool better = inc_obj < before;
        if (!better && (status == 0 || status == 1)) lb_k = std::min(2 * lb_k, 80);   // searched to the end
        lb_fails = better ? 0 : lb_fails + 1;
        lb_time += seconds_since(start);
        if (opt.verbose)
            std::printf("  local branching: %d binaries, k %d, %.0f nodes, %.2f s: %s\n", (int)row.size(),
                        k_used, sinfo.nodes,
                        seconds_since(start), better ? "better incumbent" : "nothing better");
    }

    // Reduced-cost fixing with the root LP: a nonbasic integer column whose reduced cost shows that moving it by
    // k units would push the LP bound past the incumbent cannot move that far in any better solution, so its
    // global bound is tightened. Returns the number of tightened bounds.
    int reduced_cost_fixing(double root_obj) {
        double slack = cutoff() - root_obj;
        if (!std::isfinite(slack) || slack < 0) return 0;
        int k = 0;
        for (int j = 0; j < M.n; ++j) {
            if (!M.isint[j] || S.st[j] == BASIC) continue;
            double d = S.d[j];
            if (S.st[j] == AT_LB && d > 1e-9 && std::isfinite(root_lb[j])) {
                double ub = root_lb[j] + std::floor(slack / d + 1e-9);
                if (ub < root_ub[j]) { root_ub[j] = ub; S.ub[j] = ub; ++k; }
            } else if (S.st[j] == AT_UB && d < -1e-9 && std::isfinite(root_ub[j])) {
                double lb = root_ub[j] - std::floor(slack / -d + 1e-9);
                if (lb > root_lb[j]) { root_lb[j] = lb; S.lb[j] = lb; ++k; }
            }
        }
        return k;
    }

    // ---- domain propagation at the nodes
    // Rows (original rows and cuts) give activity bounds over the node's column bounds; a row that cannot be
    // satisfied proves the node infeasible, otherwise the bounds of its integer columns are tightened (rounded).
    // Tightened columns queue their rows again. Work per call is capped.
    std::vector<std::vector<std::pair<int, double>>> prop_rows;   // row copy of the model with the root cuts
    std::vector<std::vector<int>> prop_cols;                       // rows of each column
    std::vector<std::vector<double>> prop_vals;                     // ... and the coefficients there
    std::vector<char> queued;                                       // work array of propagate
    long long prop_work = 0;                                        // row-entry operations of probing
    long long fp_work = 0;                                          // row entries visited by propagate()

    void build_propagation() {
        M.rows_of(prop_rows);
        prop_cols.assign(M.n, {});
        prop_vals.assign(M.n, {});
        for (int i = 0; i < M.m; ++i)
            for (auto& e : prop_rows[i]) { prop_cols[e.first].push_back(i); prop_vals[e.first].push_back(e.second); }
    }

    // Returns false if the node is infeasible. Tightened bounds are written to lb/ub and listed in `changed`; with
    // a trail, the old bounds of every tightened column are pushed there first (so the caller can undo). With
    // `continuous`, continuous columns are tightened too (only by a noticeable step, so it cannot crawl).
    struct BoundTrail { int j; double l, u; };
    bool propagate(const std::vector<int>& start, std::vector<double>& lb, std::vector<double>& ub,
                   std::vector<int>& changed, std::vector<BoundTrail>* trail = nullptr, bool continuous = false) {
        std::vector<int> queue;
        queued.assign(M.m, 0);
        for (int j : start)
            for (int i : prop_cols[j]) if (!queued[i]) { queued[i] = 1; queue.push_back(i); }
        long work = 0, cap = (continuous ? 200L : 20L) * (M.n + M.m);
        for (size_t qi = 0; qi < queue.size() && work < cap; ++qi) {
            int i = queue[qi];
            queued[i] = 0;
            const auto& row = prop_rows[i];
            work += (long)row.size();
            fp_work += (long long)row.size();
            double mn = 0, mx = 0;
            int mn_inf = 0, mx_inf = 0;
            for (auto& e : row) {
                double a = e.second, l = lb[e.first], u = ub[e.first];
                double lo = a > 0 ? l : u, hi = a > 0 ? u : l;
                if (std::isfinite(lo)) mn += a * lo; else ++mn_inf;
                if (std::isfinite(hi)) mx += a * hi; else ++mx_inf;
            }
            double rlo = M.rlo[i], rhi = M.rhi[i];
            double tol = 1e-6 * (1.0 + std::max(std::fabs(std::isfinite(rlo) ? rlo : 0.0),
                                                 std::fabs(std::isfinite(rhi) ? rhi : 0.0)));
            if ((mn_inf == 0 && mn > rhi + tol) || (mx_inf == 0 && mx < rlo - tol)) return false;
            for (auto& e : row) {
                int j = e.first;
                bool integral = M.isint[j];
                if (!integral && !continuous) continue;
                double a = e.second, l = lb[j], u = ub[j];
                double cmin = a > 0 ? a * l : a * u, cmax = a > 0 ? a * u : a * l;
                bool fmin = std::isfinite(cmin), fmax = std::isfinite(cmax);
                // rest of the row without column j
                bool rmin_ok = mn_inf - (fmin ? 0 : 1) == 0, rmax_ok = mx_inf - (fmax ? 0 : 1) == 0;
                double rmin = mn - (fmin ? cmin : 0.0), rmax = mx - (fmax ? cmax : 0.0);
                double nl = l, nu = u;
                auto down = [&](double v) { return integral ? std::floor(v + 1e-6) : v; };
                auto up = [&](double v) { return integral ? std::ceil(v - 1e-6) : v; };
                if (std::isfinite(rhi) && rmin_ok) {            // a x_j <= rhi - rmin
                    double v = (rhi - rmin) / a;
                    if (a > 0) nu = std::min(nu, down(v)); else nl = std::max(nl, up(v));
                }
                if (std::isfinite(rlo) && rmax_ok) {            // a x_j >= rlo - rmax
                    double v = (rlo - rmax) / a;
                    if (a > 0) nl = std::max(nl, up(v)); else nu = std::min(nu, down(v));
                }
                if (!integral) {
                    if (nl > nu) {
                        if (nl - nu > 1e-6 * (1.0 + std::fabs(nu))) return false;
                        nl = nu = std::max(l, std::min(u, 0.5 * (nl + nu)));
                    }
                    double step = std::isfinite(u - l) ? 1e-3 * (u - l) : 0.0;
                    if (nl - l <= std::max(step, 1e-6 * (1.0 + std::fabs(l)))) nl = l;
                    if (u - nu <= std::max(step, 1e-6 * (1.0 + std::fabs(u)))) nu = u;
                }
                if (nl > nu + 1e-9) return false;
                if (nl > l || nu < u) {
                    if (trail) trail->push_back({j, l, u});
                    lb[j] = nl; ub[j] = nu;
                    changed.push_back(j);
                    for (int k : prop_cols[j]) if (!queued[k]) { queued[k] = 1; queue.push_back(k); }
                    // the activities of row i changed: finish this row with the old values (still valid bounds)
                }
            }
        }
        return true;
    }

    // ---- probing (presolve): every binary column is tentatively fixed at 0 and at 1 and the fixing propagated
    // through the rows. If one value is infeasible the column takes the other (both: the model is infeasible), and
    // a bound that both values imply holds for every solution. Equivalences found on the way (x_k = x_j, or
    // x_k = 1 - x_j when the sign is -1) go to `equiv` as (k, j, sign). The most connected columns go first; the run
    // stops once the propagation has done `work_limit` row-entry operations (or after `seconds`, a safety net).
    // L/U are the bounds, tightened in place. Returns false if the model is infeasible.
    // Row activity bounds are kept incrementally: a bound change updates the rows of its column, and a row is
    // scanned in full only when its slack drops below the largest |a_k| (u_k - l_k) of its integer columns, the
    // only case in which it can tighten anything. On set partitioning models, where fixing one column at 1 fixes
    // thousands at 0, a probe then costs about the columns it touches instead of the whole matrix.
    bool probe(std::vector<double>& L, std::vector<double>& U, double work_limit, double seconds, long& fixed,
               long& tightened, std::vector<std::array<int, 3>>* equiv = nullptr) {
        build_propagation();
        auto stop = Clock::now() + std::chrono::milliseconds((long long)(seconds * 1000));
        prop_work = 0;
        const int m = M.m, n = M.n;
        auto contrib = [](double a, double l, double u, double& lo, double& hi) {
            lo = a > 0 ? a * l : a * u;
            hi = a > 0 ? a * u : a * l;
        };
        // activity bounds over the global bounds L/U (finite part and number of infinite terms) and the largest
        // |a_k| (u_k - l_k) over the integer columns of the row
        std::vector<double> amin(m), amax(m), arng(m);
        std::vector<int> nmin(m), nmax(m);
        auto row_stats = [&](int i) {
            double mn = 0, mx = 0, rg = 0;
            int ni = 0, nx = 0;
            for (auto& e : prop_rows[i]) {
                double lo, hi;
                contrib(e.second, L[e.first], U[e.first], lo, hi);
                if (std::isfinite(lo)) mn += lo; else ++ni;
                if (std::isfinite(hi)) mx += hi; else ++nx;
                if (M.isint[e.first]) rg = std::max(rg, std::fabs(e.second) * (U[e.first] - L[e.first]));
            }
            amin[i] = mn; amax[i] = mx; nmin[i] = ni; nmax[i] = nx; arng[i] = rg;
        };
        for (int i = 0; i < m; ++i) row_stats(i);
        auto tolerance = [&](int i) {
            double rlo = M.rlo[i], rhi = M.rhi[i];
            return 1e-6 * (1.0 + std::max(std::fabs(std::isfinite(rlo) ? rlo : 0.0),
                                          std::fabs(std::isfinite(rhi) ? rhi : 0.0)));
        };
        // the same during a probe, for the rows it has touched
        std::vector<double> pmin(m), pmax(m);
        std::vector<int> pnmin(m), pnmax(m), touched, queue;
        std::vector<char> is_touched(m, 0), in_queue(m, 0);
        std::vector<double> WL = L, WU = U;
        // column k moves from [l, u] to [nl, nu] in the probe and the activities of its rows follow; false if a row
        // can no longer be satisfied. Rows that can now tighten something are queued for a full scan.
        auto update = [&](int k, double l, double u, double nl, double nu) {
            const auto& rs = prop_cols[k];
            const auto& vs = prop_vals[k];
            prop_work += (long long)rs.size();
            for (size_t t = 0; t < rs.size(); ++t) {
                int i = rs[t];
                double a = vs[t], olo, ohi, nlo, nhi;
                if (!is_touched[i]) {
                    is_touched[i] = 1; touched.push_back(i);
                    pmin[i] = amin[i]; pmax[i] = amax[i]; pnmin[i] = nmin[i]; pnmax[i] = nmax[i];
                }
                contrib(a, l, u, olo, ohi);
                contrib(a, nl, nu, nlo, nhi);
                if (std::isfinite(olo)) pmin[i] -= olo; else --pnmin[i];
                if (std::isfinite(nlo)) pmin[i] += nlo; else ++pnmin[i];
                if (std::isfinite(ohi)) pmax[i] -= ohi; else --pnmax[i];
                if (std::isfinite(nhi)) pmax[i] += nhi; else ++pnmax[i];
                double rlo = M.rlo[i], rhi = M.rhi[i], tol = tolerance(i);
                if ((pnmin[i] == 0 && pmin[i] > rhi + tol) || (pnmax[i] == 0 && pmax[i] < rlo - tol)) return false;
                bool live = (std::isfinite(rhi) && (pnmin[i] == 1 || (pnmin[i] == 0 && rhi - pmin[i] < arng[i]))) ||
                            (std::isfinite(rlo) && (pnmax[i] == 1 || (pnmax[i] == 0 && pmax[i] - rlo < arng[i])));
                if (live && !in_queue[i]) { in_queue[i] = 1; queue.push_back(i); }
            }
            return true;
        };
        // full scan of a queued row: bounds of its integer columns from the rest of the row
        auto scan = [&](int i, std::vector<int>& changes) {
            double rlo = M.rlo[i], rhi = M.rhi[i];
            const auto& row = prop_rows[i];
            prop_work += (long long)row.size();
            for (auto& e : row) {
                int k = e.first;
                if (!M.isint[k]) continue;
                double a = e.second, l = WL[k], u = WU[k], lo, hi;
                contrib(a, l, u, lo, hi);
                bool flo = std::isfinite(lo), fhi = std::isfinite(hi);
                bool rmin_ok = pnmin[i] - (flo ? 0 : 1) == 0, rmax_ok = pnmax[i] - (fhi ? 0 : 1) == 0;
                double rmin = pmin[i] - (flo ? lo : 0.0), rmax = pmax[i] - (fhi ? hi : 0.0);
                double nl = l, nu = u;
                if (std::isfinite(rhi) && rmin_ok) {            // a x_k <= rhi - rmin
                    double v = (rhi - rmin) / a;
                    if (a > 0) nu = std::min(nu, std::floor(v + 1e-6)); else nl = std::max(nl, std::ceil(v - 1e-6));
                }
                if (std::isfinite(rlo) && rmax_ok) {            // a x_k >= rlo - rmax
                    double v = (rlo - rmax) / a;
                    if (a > 0) nl = std::max(nl, std::ceil(v - 1e-6)); else nu = std::min(nu, std::floor(v + 1e-6));
                }
                if (nl > nu + 1e-9) return false;
                if (nl > l || nu < u) {
                    WL[k] = nl; WU[k] = nu;
                    changes.push_back(k);
                    if (!update(k, l, u, nl, nu)) return false;
                }
            }
            return true;
        };
        // x_j = v and everything it implies (changes: the columns tightened, their bounds left in WL/WU)
        long long cap = 2LL * (n + m);
        auto side = [&](int j, double v, std::vector<int>& changes) {
            changes.clear();
            long long w0 = prop_work;
            WL[j] = WU[j] = v;
            bool ok = update(j, L[j], U[j], v, v);
            for (size_t qi = 0; ok && qi < queue.size() && prop_work - w0 <= cap; ++qi) {
                int i = queue[qi];
                in_queue[i] = 0;
                ok = scan(i, changes);
            }
            for (int i : queue) in_queue[i] = 0;
            queue.clear();
            for (int i : touched) is_touched[i] = 0;
            touched.clear();
            return ok;
        };
        std::vector<int> order;
        for (int j = 0; j < n; ++j) if (M.isint[j] && L[j] == 0.0 && U[j] == 1.0) order.push_back(j);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return prop_cols[a].size() > prop_cols[b].size(); });
        std::vector<double> l0(n), u0(n), l1(n), u1(n);
        std::vector<char> seen0(n, 0), dirty(m, 0), subst(n, 0);    // subst: already expressed by another column
        std::vector<int> ch0, ch1, dirty_rows;
        auto restore = [&](int j, std::vector<int>& changes) {
            WL[j] = L[j]; WU[j] = U[j];
            for (int k : changes) { WL[k] = L[k]; WU[k] = U[k]; }
        };
        auto tighten = [&](int k, double nl, double nu) {            // global bound change, kept in WL/WU too
            bool t = false;
            if (nl > L[k]) { L[k] = WL[k] = nl; ++tightened; t = true; }
            if (nu < U[k]) { U[k] = WU[k] = nu; ++tightened; t = true; }
            if (t) for (int i : prop_cols[k]) if (!dirty[i]) { dirty[i] = 1; dirty_rows.push_back(i); }
        };
        for (size_t q = 0; q < order.size(); ++q) {
            if ((double)prop_work > work_limit || ((q & 63) == 0 && Clock::now() > stop)) break;
            for (int i : dirty_rows) { row_stats(i); dirty[i] = 0; }
            dirty_rows.clear();
            int j = order[q];
            if (L[j] != 0.0 || U[j] != 1.0 || subst[j]) continue;    // fixed, or equal to another binary
            bool ok0 = side(j, 0.0, ch0);
            for (int k : ch0) { seen0[k] = 1; l0[k] = WL[k]; u0[k] = WU[k]; }
            restore(j, ch0);
            bool ok1 = side(j, 1.0, ch1);
            for (int k : ch1) { l1[k] = WL[k]; u1[k] = WU[k]; }
            restore(j, ch1);
            if (!ok0 && !ok1) return false;
            if (!ok0 || !ok1) {                 // one value is impossible: fix the other, with what it implies
                double v = ok0 ? 0.0 : 1.0;
                tighten(j, v, v);
                ++fixed;
                if (ok0) { for (int k : ch0) tighten(k, l0[k], u0[k]); }
                else { for (int k : ch1) tighten(k, l1[k], u1[k]); }
            } else {
                for (int k : ch1) {
                    if (!seen0[k]) continue;
                    // a binary that follows x_j in both directions is x_j (or its complement)
                    if (equiv && k != j && !subst[k] && L[k] == 0.0 && U[k] == 1.0) {
                        if (u0[k] == 0.0 && l1[k] == 1.0) { equiv->push_back({k, j, 1}); subst[k] = 1; continue; }
                        if (l0[k] == 1.0 && u1[k] == 0.0) { equiv->push_back({k, j, -1}); subst[k] = 1; continue; }
                    }
                    tighten(k, std::min(l0[k], l1[k]), std::max(u0[k], u1[k]));
                }
            }
            for (int k : ch0) seen0[k] = 0;
        }
        return true;
    }

    // ---- reliability branching
    // Candidates are ranked by pseudocost score; a candidate whose pseudocosts rest on fewer than 4 observations in
    // either direction is strong-branched: both children are solved with a few dual simplex iterations from the
    // node basis (the objective of a dual feasible basis is a valid lower bound) and the gains seed its
    // pseudocosts. Strong branching stops after 8 candidates without improvement, and its total work is kept below
    // half of all LP iterations.
    long sb_iters = 0;

    int reliability_branch(double node_obj) {
        double avg[2] = {1.0, 1.0};
        for (int s = 0; s < 2; ++s) {
            double sum = 0; int cnt = 0;
            for (int j = 0; j < M.n; ++j) if (pc_cnt[s][j]) { sum += pc_sum[s][j] / pc_cnt[s][j]; ++cnt; }
            if (cnt) avg[s] = sum / cnt;
        }
        std::vector<std::pair<double, int>> cand;
        for (int j = 0; j < M.n; ++j) {
            if (!M.isint[j]) continue;
            double v = S.x[j], f = v - std::floor(v);
            if (f < opt.int_tol || f > 1 - opt.int_tol) continue;
            double pd = pc_cnt[0][j] ? pc_sum[0][j] / pc_cnt[0][j] : avg[0];
            double pu = pc_cnt[1][j] ? pc_sum[1][j] / pc_cnt[1][j] : avg[1];
            cand.push_back({std::max(pd * f, 1e-6) * std::max(pu * (1 - f), 1e-6), j});
        }
        if (cand.empty()) return -1;
        std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        int best = cand[0].second;
        bool sb_ok = sb_iters < S.iters / 2 + 1000 && Clock::now() < S.t_end;
        if (!sb_ok) return best;
        std::vector<double> xnode(S.x.begin(), S.x.begin() + M.n);
        LPState saved = save_lp();
        long max_iter = S.opt.max_lp_iter;
        double best_score = -1.0;
        int since_best = 0, done = 0;
        auto gain = [&](int j, bool up) {
            double v = xnode[j];
            if (up) S.lb[j] = std::ceil(v); else S.ub[j] = std::floor(v);
            long it0 = S.iters;
            S.opt.max_lp_iter = S.iters + 25;
            Result r = S.dual();
            sb_iters += S.iters - it0;
            double g = (r == INFEASIBLE || (r != NUMERIC && S.objective() >= cutoff())) ? 1e30
                       : (r == OPTIMAL || r == ITER_LIMIT) ? std::max(0.0, S.objective() - node_obj) : -1.0;
            S.opt.max_lp_iter = max_iter;
            restore_lp(saved);
            return g;
        };
        for (auto& c : cand) {
            if (since_best >= 8 || done >= 20 || Clock::now() > S.t_end) break;
            int j = c.second;
            double score = c.first;
            if (pc_cnt[0][j] < 4 || pc_cnt[1][j] < 4) {
                double f = xnode[j] - std::floor(xnode[j]);
                double gd = gain(j, false), gu = gain(j, true);
                ++done;
                if (gd < 0 || gu < 0) continue;                      // numerical trouble: keep the estimate
                if (gd < 1e29) { pc_sum[0][j] += gd / std::max(f, 1e-6); pc_cnt[0][j]++; }
                if (gu < 1e29) { pc_sum[1][j] += gu / std::max(1 - f, 1e-6); pc_cnt[1][j]++; }
                score = std::max(std::min(gd, 1e15), 1e-6) * std::max(std::min(gu, 1e15), 1e-6);
            }
            if (score > best_score) { best_score = score; best = j; since_best = 0; }
            else ++since_best;
        }
        return best;
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

    // ---- knapsack cover cuts from the original rows
    // Each side of a row  sum a_j x_j <= b  (>= rows are negated) is relaxed to a 0-1 knapsack over its binary
    // columns: continuous and general-integer terms are replaced by their smallest value over the root bounds,
    // binaries with a negative coefficient are complemented (y = 1 - x). A cover C (total weight > capacity) is
    // built greedily by (1 - y*_j) / a_j, extended by every item at least as heavy as the heaviest in C, and added
    // as  sum_{E(C)} y_j <= |C| - 1  if the LP point violates it.
    int add_cover_round() {
        auto is_bin = [&](int j) { return M.isint[j] && root_lb[j] == 0.0 && root_ub[j] == 1.0; };
        std::vector<std::vector<std::pair<int, double>>> cuts;
        std::vector<double> lo, hi;
        struct Item { int j; double a; bool comp; double y; };
        std::vector<Item> items;
        for (int i = 0; i < m_orig && (int)cuts.size() < opt.max_cuts_per_round; ++i) {
            for (int side = 0; side < 2; ++side) {               // side 0: a x <= rhi, side 1: -a x <= -rlo
                double b = side == 0 ? M.rhi[i] : -M.rlo[i];
                if (!std::isfinite(b)) continue;
                double sgn = side == 0 ? 1.0 : -1.0;
                items.clear();
                bool ok = true;
                for (auto& e : rowsR[i]) {
                    int j = e.first;
                    double a = sgn * e.second;
                    if (a == 0.0) continue;
                    if (is_bin(j)) {
                        if (a > 0) items.push_back({j, a, false, S.x[j]});
                        else { items.push_back({j, -a, true, 1.0 - S.x[j]}); b -= a; }
                    } else {
                        double v = a > 0 ? root_lb[j] : root_ub[j];
                        if (!std::isfinite(v)) { ok = false; break; }
                        b -= a * v;
                    }
                }
                if (!ok || items.size() < 2 || b <= 1e-9) continue;
                double total = 0.0;
                for (auto& it : items) total += it.a;
                if (total <= b + 1e-9) continue;                 // no cover exists
                std::sort(items.begin(), items.end(), [](const Item& p, const Item& q) {
                    return (1.0 - p.y) / p.a < (1.0 - q.y) / q.a;
                });
                double w = 0.0, amax = 0.0;
                size_t k = 0;
                while (k < items.size() && w <= b + 1e-9) { w += items[k].a; amax = std::max(amax, items[k].a); ++k; }
                if (w <= b + 1e-9) continue;
                double lhs = 0.0;
                std::vector<std::pair<int, double>> row;
                double rhs = (double)k - 1.0;
                for (size_t t = 0; t < items.size(); ++t) {
                    if (t >= k && items[t].a < amax) continue;   // extended cover
                    lhs += items[t].y;
                    double coef = items[t].comp ? -1.0 : 1.0;    // y = 1 - x  ->  -x, rhs - 1
                    if (items[t].comp) rhs -= 1.0;
                    row.push_back({items[t].j, coef});
                }
                if (lhs - ((double)k - 1.0) < 1e-4 * std::sqrt((double)row.size())) continue;   // not violated
                cuts.push_back(row); lo.push_back(-INF); hi.push_back(rhs);
            }
        }
        if (cuts.empty()) return 0;
        M.add_rows(cuts, lo, hi);
        S.resize_to_model();
        S.refactor_full();
        return (int)cuts.size();
    }

    // ---- simple generalized flow cover cuts (Van Roy & Wolsey 1986) from the original rows
    // Each side  sum a_j z_j <= b  of a row is relaxed to a single-node flow set
    //     sum_{N1} x_j - sum_{N2} x_j <= b',   0 <= x_j <= u_j y_j,   y_j binary
    // with one arc per term (N1 for a > 0, N2 for a < 0): a continuous column with a variable upper bound z <= v y
    // and lower bound 0 gives the flow |a| z with capacity |a| v and switch y; a binary column y gives the flow |a| y
    // with capacity |a|; any other column with a finite lower bound l gives the always-open arc |a| (z - l) (switch
    // 1). Arcs may share a switch: the row's feasible set lies inside the flow set with one switch per arc. For a
    // flow cover C1 in N1 with lambda = sum_{C1} u_j - b' > 0 and any L2 in N2,
    //     sum_{C1} [x_j + max(u_j - lambda, 0) (1 - y_j)] - lambda sum_{L2} y_j - sum_{N2 \ L2} x_j <= b'
    // holds: if every arc j of C1 with u_j > lambda is open, the flow balance bounds the left side unless an arc of
    // L2 is open, and then the capacities do (sum_{C1} x_j <= b' + lambda); if one of them is closed, the
    // capacities give sum_{C1} x_j <= sum u_j over the open ones. C1 is built from the arcs that are most open at
    // the LP point, per unit of capacity; L2 holds the arcs with lambda y*_j < x*_j.
    // The flow cover cut of the relation  sum rel_z z <= b  over structural columns (z < n) and row activities
    // (z = n + i, bounds rlo_i and rhi_i, value act_i; a row activity can only be an always-open arc). Returns the
    // efficacy (violation over the norm; 0 if the cut is not violated enough or too badly scaled) and fills cut /
    // cut_rhs over the structural columns (row activities expanded into their rows).
    std::vector<double> fc_dense;
    std::vector<char> fc_mark;
    double flowcover_from_relation(const std::vector<std::pair<int, double>>& rel, double b, const std::vector<double>& act,
                                   std::vector<std::pair<int, double>>& cut, double& cut_rhs) {
        int n = M.n;
        auto is_bin = [&](int j) { return M.isint[j] && root_lb[j] == 0.0 && root_ub[j] == 1.0; };
        struct Arc { int z; int y; double a; double shift; double u; double xv; double yv; bool out; };
        std::vector<Arc> arcs;
        bool has_switch = false;
        for (auto& e : rel) {
            int j = e.first;
            double a = e.second, aa = std::fabs(a);
            if (a == 0.0) continue;
            if (j < n && is_bin(j)) {
                arcs.push_back({j, j, a, 0.0, aa, aa * S.x[j], S.x[j], a < 0});
                has_switch = true;
            } else if (j < n && !M.isint[j] && vub[j].y >= 0 && root_lb[j] == 0.0) {
                double v = std::min(vub[j].u, root_ub[j]);           // z <= min(v, ub) y
                arcs.push_back({j, vub[j].y, a, 0.0, aa * v, aa * S.x[j], S.x[vub[j].y], a < 0});
                has_switch = true;
            } else {
                double l = j < n ? root_lb[j] : M.rlo[j - n], u = j < n ? root_ub[j] : M.rhi[j - n];
                double x = j < n ? S.x[j] : act[j - n];
                if (!std::isfinite(l)) return 0.0;
                b -= a * l;
                double cap = std::isfinite(u) ? aa * (u - l) : INF;
                arcs.push_back({j, -1, a, l, cap, aa * std::max(x - l, 0.0), 1.0, a < 0});
            }
        }
        if (!has_switch || arcs.size() < 2) return 0.0;
        // flow cover from the inflow arcs: most open first, per unit of capacity
        std::vector<int> c1;
        for (int t = 0; t < (int)arcs.size(); ++t)
            if (!arcs[t].out && std::isfinite(arcs[t].u) && arcs[t].u > 1e-9) c1.push_back(t);
        std::sort(c1.begin(), c1.end(), [&](int p, int q) {
            double rp = (1.0 - arcs[p].yv) / arcs[p].u, rq = (1.0 - arcs[q].yv) / arcs[q].u;
            return rp < rq || (rp == rq && arcs[p].u > arcs[q].u);
        });
        double cap = 0.0, tol = 1e-9 * (1.0 + std::fabs(b));
        size_t k = 0;
        while (k < c1.size() && cap <= b + tol) cap += arcs[c1[k++]].u;
        if (cap <= b + tol) return 0.0;
        c1.resize(k);
        double lambda = cap - b;
        if (lambda <= 1e-6 * (1.0 + std::fabs(b))) return 0.0;
        // the cut over the arcs, then over the original columns:  sum coef z <= rhs
        double rhs = b;
        int nz_all = n + m_orig;
        if ((int)fc_dense.size() < nz_all) { fc_dense.assign(nz_all, 0.0); fc_mark.assign(nz_all, 0); }
        std::vector<int> nz;
        auto add = [&](int j, double c) {
            if (!fc_mark[j]) { fc_mark[j] = 1; nz.push_back(j); fc_dense[j] = 0.0; }
            fc_dense[j] += c;
        };
        auto add_flow = [&](const Arc& r, double c) {           // c * x_arc
            double aa = std::fabs(r.a);
            add(r.z, c * aa);                                    // x_arc = |a| (z - shift)
            rhs += c * aa * r.shift;
        };
        std::vector<char> in_c1(arcs.size(), 0);
        for (int t : c1) in_c1[t] = 1;
        for (int t = 0; t < (int)arcs.size(); ++t) {
            const Arc& r = arcs[t];
            if (!r.out) {
                if (!in_c1[t]) continue;                         // inflow outside the cover: coefficient 0
                add_flow(r, 1.0);
                double d = std::max(r.u - lambda, 0.0);
                if (d > 0.0 && r.y >= 0) { add(r.y, -d); rhs -= d; }   // d (1 - y)
            } else if (r.y >= 0 && lambda * r.yv < r.xv) {
                add(r.y, -lambda);                               // L2: - lambda y
            } else if (r.y < 0 && lambda < r.xv) {
                rhs += lambda;                                   // L2 with switch 1: - lambda
            } else {
                add_flow(r, -1.0);                               // - x
            }
        }
        // row activities into their rows
        std::vector<int> rows_in;
        for (int j : nz) if (j >= n) rows_in.push_back(j);
        for (int z : rows_in) {
            double c = fc_dense[z];
            fc_mark[z] = 0;
            if (c == 0.0) continue;
            for (auto& e : rowsR[z - n]) add(e.first, c * e.second);
        }
        cut.clear();
        double lhs = 0.0, nrm = 0.0, amax = 0.0, amin = INF;
        for (int j : nz) {
            if (j >= n) continue;
            fc_mark[j] = 0;
            double c = fc_dense[j];
            if (std::fabs(c) < 1e-12) continue;
            cut.push_back({j, c});
            lhs += c * S.x[j]; nrm += c * c;
            amax = std::max(amax, std::fabs(c)); amin = std::min(amin, std::fabs(c));
        }
        cut_rhs = rhs;
        if (cut.empty() || amax / amin > 1e6) return 0.0;
        double eff = (lhs - rhs) / std::sqrt(nrm);
        if (eff < 1e-4 * (1.0 + std::fabs(rhs) / std::sqrt(nrm))) return 0.0;
        return eff;
    }

    int add_flowcover_round() {
        int n = M.n;
        if ((int)vub.size() != n) prepare_mir();
        std::vector<std::vector<std::pair<int, double>>> cuts;
        std::vector<double> lo, hi, act;
        std::vector<std::pair<int, double>> rel, cut;
        for (int i = 0; i < m_orig && (int)cuts.size() < opt.max_cuts_per_round; ++i) {
            if (Clock::now() > S.t_end) break;
            for (int side = 0; side < 2; ++side) {
                double b = side == 0 ? M.rhi[i] : -M.rlo[i];
                if (!std::isfinite(b)) continue;
                double sgn = side == 0 ? 1.0 : -1.0;
                rel.clear();
                for (auto& e : rowsR[i]) rel.push_back({e.first, sgn * e.second});
                double rhs = 0.0;
                if (flowcover_from_relation(rel, b, act, cut, rhs) > 0.0) {
                    cuts.push_back(cut); lo.push_back(-INF); hi.push_back(rhs);
                }
            }
        }
        if (cuts.empty()) return 0;
        M.add_rows(cuts, lo, hi);
        S.resize_to_model();
        S.refactor_full();
        return (int)cuts.size();
    }

    // ---- cut-set inequalities of fixed-charge networks
    // Flow balance rows: equalities whose every column is an arc, a continuous column with lower bound 0 that appears
    // in exactly two of them with coefficients +c and -c (in the model's own units). Summing the balances of a node
    // set T cancels the arcs inside it:  sum_{arcs into T} c x - sum_{arcs out of T} c x = b(T).  If b(T) > 0 some
    // arc into T carries flow, so its switch (x <= u y) is open:  sum_{switches of the arcs into T} y >= 1.
    // An arc whose other end is a row outside the node set comes from (or goes to) one 'outside' node, whose balance is
    // unknown: it counts as a supply. Separation: for every node with b > 0, a minimum cut on the LP values of the
    // switches from a super source joined to the outside node and to every node with b < 0 (so T, the far side,
    // holds no supply and b(T) > 0); a cut of value below 1 is violated.
    struct NetArc { int col, from, to, y; };
    bool net_ready = false;
    std::vector<NetArc> net_arcs;
    std::vector<double> net_b;
    void prepare_network() {
        net_ready = true;
        net_arcs.clear(); net_b.clear();
        int n = M.n;
        if (colscale.size() != (size_t)n || rowscale.size() < (size_t)m_orig) return;    // (nested sub-MIPs)
        if ((int)vub.size() != n) prepare_mir();
        std::vector<char> cand(m_orig, 0);
        for (int i = 0; i < m_orig; ++i) {
            if (!(M.rlo[i] == M.rhi[i]) || !std::isfinite(M.rlo[i]) || rowsR[i].size() < 2) continue;
            bool ok = true;
            for (auto& e : rowsR[i]) if (M.isint[e.first] || root_lb[e.first] != 0.0) { ok = false; break; }
            cand[i] = ok;
        }
        // columns with exactly two entries in candidate rows, opposite and of equal size in model units
        std::vector<int> cnt(n, 0), r1(n, -1), r2(n, -1);
        std::vector<double> a1(n, 0.0), a2(n, 0.0);
        for (int i = 0; i < m_orig; ++i) {
            if (!cand[i]) continue;
            for (auto& e : rowsR[i]) {
                int j = e.first;
                double a = e.second / (rowscale[i] * colscale[j]);
                if (cnt[j] == 0) { r1[j] = i; a1[j] = a; } else if (cnt[j] == 1) { r2[j] = i; a2[j] = a; }
                cnt[j]++;
            }
        }
        std::vector<char> arc(n, 0);
        for (int j = 0; j < n; ++j)
            arc[j] = cnt[j] == 2 && a1[j] * a2[j] < 0 && std::fabs(a1[j] + a2[j]) <= 1e-9 * std::fabs(a1[j]);
        // node rows: candidates made of arcs only
        std::vector<int> node(m_orig, -1);
        int V = 0;
        for (int i = 0; i < m_orig; ++i) {
            if (!cand[i]) continue;
            bool ok = true;
            for (auto& e : rowsR[i]) if (!arc[e.first]) { ok = false; break; }
            if (ok) node[i] = V++;
        }
        if (V < 2) return;
        net_b.assign(V, 0.0);
        for (int i = 0; i < m_orig; ++i) if (node[i] >= 0) net_b[node[i]] = M.rlo[i] / rowscale[i];
        bool demand = false;
        for (double b : net_b) demand = demand || b > 0;
        if (!demand) return;                        // (the outside node can supply)
        for (int j = 0; j < n; ++j) {
            if (!arc[j] || (node[r1[j]] < 0 && node[r2[j]] < 0)) continue;
            int n1 = node[r1[j]] >= 0 ? node[r1[j]] : V, n2 = node[r2[j]] >= 0 ? node[r2[j]] : V;   // V: outside
            int to = a1[j] > 0 ? n1 : n2, from = a1[j] > 0 ? n2 : n1;
            net_arcs.push_back({j, from, to, vub[j].y});
        }
        if (opt.verbose && !net_arcs.empty())
            std::printf("  network: %d flow balance rows, %zu arcs\n", V, net_arcs.size());
    }

    int add_network_round() {
        if (!net_ready) prepare_network();
        if (net_arcs.empty()) return 0;
        int V = (int)net_b.size(), outside = V, src = V + 1;
        // residual graph (Dinic): edges in pairs, forward then backward
        struct Edge { int to; double cap; };
        std::vector<Edge> E;
        std::vector<std::vector<int>> adj(V + 2);
        std::vector<double> cap0;
        auto edge = [&](int u, int v, double c) {
            adj[u].push_back((int)E.size()); E.push_back({v, c}); cap0.push_back(c);
            adj[v].push_back((int)E.size()); E.push_back({u, 0.0}); cap0.push_back(0.0);
        };
        const double BIG = 1e9;
        for (auto& a : net_arcs) edge(a.from, a.to, a.y >= 0 ? std::max(S.x[a.y], 0.0) : BIG);
        for (int v = 0; v < V; ++v) if (net_b[v] < 0) edge(src, v, BIG);
        edge(src, outside, BIG);
        std::vector<int> level(V + 2), it(V + 2), q;
        auto bfs = [&](int t) {
            std::fill(level.begin(), level.end(), -1);
            q.assign(1, src); level[src] = 0;
            for (size_t h = 0; h < q.size(); ++h)
                for (int e : adj[q[h]]) if (E[e].cap > 1e-12 && level[E[e].to] < 0) { level[E[e].to] = level[q[h]] + 1; q.push_back(E[e].to); }
            return level[t] >= 0;
        };
        std::function<double(int, int, double)> dfs = [&](int u, int t, double f) -> double {
            if (u == t) return f;
            for (int& k = it[u]; k < (int)adj[u].size(); ++k) {
                int e = adj[u][k];
                if (E[e].cap > 1e-12 && level[E[e].to] == level[u] + 1) {
                    double d = dfs(E[e].to, t, std::min(f, E[e].cap));
                    if (d > 0) { E[e].cap -= d; E[e ^ 1].cap += d; return d; }
                }
            }
            return 0.0;
        };
        std::vector<std::vector<std::pair<int, double>>> cuts;
        std::vector<double> lo, hi;
        std::set<std::vector<int>> seen;
        std::vector<char> mark(M.n, 0);
        for (int t = 0; t < V && (int)cuts.size() < opt.max_cuts_per_round; ++t) {
            if (net_b[t] <= 0) continue;
            if (Clock::now() > S.t_end) break;
            for (size_t e = 0; e < E.size(); ++e) E[e].cap = cap0[e];
            double flow = 0.0;
            while (flow < 1.0 && bfs(t)) {
                std::fill(it.begin(), it.end(), 0);
                double f;
                while ((f = dfs(src, t, INF)) > 0) { flow += f; if (flow >= 1.0) break; }
            }
            if (flow >= 1.0 - 1e-6) continue;
            // two minimum cuts: the one next to the source (T: what the source cannot reach in the residual graph)
            // and the one next to the sink (T: what can still reach the sink), usually the stronger; neither T holds
            // a supply node or the outside node (the source would then reach the sink)
            std::vector<char> inT(V + 2, 0);
            for (int side = 0; side < 2; ++side) {
                if (side == 0) {
                    bfs(t);
                    for (int v = 0; v < V + 2; ++v) inT[v] = level[v] < 0;
                } else {
                    std::fill(inT.begin(), inT.end(), 0);
                    q.assign(1, t); inT[t] = 1;
                    for (size_t h = 0; h < q.size(); ++h)
                        for (int e : adj[q[h]]) {                    // edge e^1 runs E[e].to -> q[h]
                            int u = E[e].to;
                            if (!inT[u] && E[e ^ 1].cap > 1e-12) { inT[u] = 1; q.push_back(u); }
                        }
                }
                std::vector<int> sw;
                bool open = false;
                for (auto& a : net_arcs) {
                    if (inT[a.from] || !inT[a.to]) continue;                 // arcs into T
                    if (a.y < 0) { open = true; break; }                       // an arc without a switch: always open
                    if (!mark[a.y]) { mark[a.y] = 1; sw.push_back(a.y); }
                }
                for (int y : sw) mark[y] = 0;
                if (open || sw.empty()) continue;
                std::sort(sw.begin(), sw.end());
                if (!seen.insert(sw).second) continue;
                std::vector<std::pair<int, double>> cut;
                for (int y : sw) cut.push_back({y, -1.0});                   // - sum y <= -1
                cuts.push_back(cut); lo.push_back(-INF); hi.push_back(-1.0);
            }
        }
        if (cuts.empty()) return 0;
        M.add_rows(cuts, lo, hi);
        S.resize_to_model();
        S.refactor_full();
        return (int)cuts.size();
    }

    // ---- complemented mixed-integer rounding (c-MIR) cuts with aggregation (Marchand & Wolsey 2001)
    // Every original row is the equation  a_i x - r_i = 0  with a continuous row activity r_i in [rlo_i, rhi_i],
    // so any combination of rows is a valid relation. Starting from one row, the relation is tried as it is and
    // after eliminating up to 5 continuous columns that lie strictly inside their bounds, each time by adding a
    // row that contains the column. For a relation  sum e_k z_k <= b  (both signs are tried):
    //   * a continuous column with a variable upper bound  x <= u y  (y binary) is replaced by u y - xbar when that
    //     bound is the nearer one, otherwise it is shifted to its nearer simple bound; row activities likewise;
    //   * integer columns are shifted to their nearer bound (complemented at the upper bound);
    //   * the relation is divided by delta in {|e_k| of fractional integer terms} x {1, 1/2, 1/4, 1/8} and the MIR
    //     inequality of the scaled relation is formed; the most violated one (relative to its norm) is kept
    // and mapped back to the original columns (r_i = a_i x).
    struct VUB { int y; double u; };          // x_j <= u * y  with y binary (y = -1: none)
    std::vector<VUB> vub;
    std::vector<std::vector<int>> rows_of_col;  // original rows containing each column

    void prepare_mir() {
        vub.assign(M.n, {-1, 0.0});
        rows_of_col.assign(M.n, {});
        auto is_bin = [&](int k) { return M.isint[k] && root_lb[k] == 0.0 && root_ub[k] == 1.0; };
        for (int i = 0; i < m_orig; ++i) {
            for (auto& e : rowsR[i]) rows_of_col[e.first].push_back(i);
            if (rowsR[i].size() != 2) continue;
            for (int s = 0; s < 2; ++s) {
                int j = rowsR[i][s].first, k = rowsR[i][1 - s].first;
                double aj = rowsR[i][s].second, ak = rowsR[i][1 - s].second;
                if (M.isint[j] || !is_bin(k) || root_lb[j] != 0.0) continue;
                double u = INF;
                if (M.rhi[i] == 0.0 && aj > 0 && -ak / aj > 0) u = -ak / aj;          // aj x + ak y <= 0
                if (M.rlo[i] == 0.0 && aj < 0 && ak / -aj > 0) u = std::min(u, ak / -aj);  // aj x + ak y >= 0
                if (std::isfinite(u) && (vub[j].y < 0 || u < vub[j].u)) vub[j] = {k, u};
            }
        }
    }

    // MIR cut from the relation  sum rel[k].second * z_k <= b  over structural columns (z < n) and row activities
    // (z = n + i). Returns the efficacy (0 if no violated cut) and fills cut / cut_rhs over structural columns.
    double mir_from_relation(const std::vector<std::pair<int, double>>& rel, double b, const std::vector<double>& act,
                             std::vector<double>& dense, std::vector<std::pair<int, double>>& cut, double& cut_rhs) {
        int n = M.n;
        // merge into a work map: original columns get their coefficients (with VUB substitution applied first)
        struct Cont { int z; double a; double shift; double sign; int vub_y; double vub_u; double v; };
        struct Int { int j; double a; double shift; double sign; double v; };
        std::vector<Cont> cont;
        std::vector<std::pair<int, double>> ints;                    // (column, coefficient on x) before shifting
        std::vector<char>& mark = mir_mark;
        std::vector<int> touched;
        auto add_int = [&](int j, double a) {
            if (!mark[j]) { mark[j] = 1; touched.push_back(j); dense[j] = 0.0; }
            dense[j] += a;
        };
        for (auto& e : rel) {
            int z = e.first;
            double a = e.second;
            if (a == 0.0) continue;
            if (z < n && M.isint[z]) { add_int(z, a); continue; }
            double l, u, x;
            if (z < n) { l = root_lb[z]; u = root_ub[z]; x = S.x[z]; }
            else { int i = z - n; l = M.rlo[i]; u = M.rhi[i]; x = act[i]; }
            if (z < n && vub[z].y >= 0) {                               // x <= u y: use it if it is nearer
                int y = vub[z].y;
                double gap_vub = vub[z].u * S.x[y] - x, gap_lb = std::isfinite(l) ? x - l : INF;
                if (gap_vub <= gap_lb) {
                    add_int(y, a * vub[z].u);                           // a x = a u y - a xbar
                    cont.push_back({z, -a, 0.0, 0.0, y, vub[z].u, std::max(gap_vub, 0.0)});
                    continue;
                }
            }
            bool use_lb = std::isfinite(l) && (!std::isfinite(u) || x - l <= u - x);
            if (!use_lb && !std::isfinite(u)) { for (int j : touched) mark[j] = 0; return 0.0; }   // free: no cut
            double shift = use_lb ? l : u, s = use_lb ? 1.0 : -1.0;
            b -= a * shift;
            cont.push_back({z, a * s, shift, s, -1, 0.0, std::max(s * (x - shift), 0.0)});
        }
        for (int j : touched) mark[j] = 0;
        std::vector<Int> it;
        for (int j : touched) {
            double a = dense[j];
            if (std::fabs(a) < 1e-12) continue;
            double l = root_lb[j], u = root_ub[j], x = S.x[j];
            bool use_lb = std::isfinite(l) && (!std::isfinite(u) || x - l <= u - x);
            if (!use_lb && !std::isfinite(u)) return 0.0;
            double shift = use_lb ? l : u, s = use_lb ? 1.0 : -1.0;
            b -= a * shift;
            it.push_back({j, a * s, shift, s, std::max(s * (x - shift), 0.0)});
        }
        std::vector<double> deltas;
        for (auto& q : it) {
            double f = S.x[q.j] - std::floor(S.x[q.j]);
            if (f > 0.01 && f < 0.99 && std::fabs(q.a) > 1e-9) deltas.push_back(std::fabs(q.a));
        }
        if (deltas.empty()) return 0.0;
        std::sort(deltas.begin(), deltas.end());
        deltas.erase(std::unique(deltas.begin(), deltas.end()), deltas.end());
        if (deltas.size() > 6) deltas.resize(6);
        double best_eff = 1e-4, best_delta = 0.0;
        for (double d0 : deltas)
            for (double div : {1.0, 2.0, 4.0, 8.0}) {
                double delta = d0 / div, beta = b / delta, f0 = beta - std::floor(beta);
                if (f0 < 0.05 || f0 > 0.95) continue;
                double lhs = 0.0, nrm = 0.0;
                for (auto& q : it) {
                    double alpha = q.a / delta, fj = alpha - std::floor(alpha);
                    double c = std::floor(alpha) + std::max(0.0, fj - f0) / (1.0 - f0);
                    lhs += c * q.v; nrm += c * c;
                }
                for (auto& q : cont) {
                    double c = std::min(0.0, q.a / delta) / (1.0 - f0);
                    lhs += c * q.v; nrm += c * c;
                }
                if (nrm < 1e-12) continue;
                double eff = (lhs - std::floor(beta)) / std::sqrt(nrm);
                if (eff > best_eff) { best_eff = eff; best_delta = delta; }
            }
        if (best_delta == 0.0) return 0.0;
        // the cut over the transformed variables, mapped back to structural columns
        double delta = best_delta, beta = b / delta, f0 = beta - std::floor(beta);
        double rhs = std::floor(beta);
        std::vector<int> nz;
        auto add = [&](int j, double c) {
            if (!mark[j]) { mark[j] = 1; nz.push_back(j); dense[j] = 0.0; }
            dense[j] += c;
        };
        for (auto& q : it) {
            double alpha = q.a / delta, fj = alpha - std::floor(alpha);
            double c = std::floor(alpha) + std::max(0.0, fj - f0) / (1.0 - f0);
            if (c == 0.0) continue;
            add(q.j, c * q.sign);                                     // x' = sign (x - shift)
            rhs += c * q.sign * q.shift;
        }
        for (auto& q : cont) {
            double c = std::min(0.0, q.a / delta) / (1.0 - f0);
            if (c == 0.0) continue;
            if (q.vub_y >= 0) {                                       // xbar = u y - x
                add(q.vub_y, c * q.vub_u);
                add(q.z, -c);
            } else if (q.z < n) {
                add(q.z, c * q.sign);
                rhs += c * q.sign * q.shift;
            } else {                                                  // row activity r_i = a_i x
                for (auto& e : rowsR[q.z - n]) add(e.first, c * q.sign * e.second);
                rhs += c * q.sign * q.shift;
            }
        }
        cut.clear();
        double amax = 0.0, amin = INF;
        for (int j : nz) {
            mark[j] = 0;
            double c = dense[j];
            if (std::fabs(c) < 1e-12) continue;
            cut.push_back({j, c});
            amax = std::max(amax, std::fabs(c)); amin = std::min(amin, std::fabs(c));
        }
        if (cut.empty() || amax / amin > 1e6) return 0.0;
        // recheck the violation on the original columns
        double lhs = 0.0, nrm = 0.0;
        for (auto& e : cut) { lhs += e.second * S.x[e.first]; nrm += e.second * e.second; }
        double eff = (lhs - rhs) / std::sqrt(nrm);
        if (eff < 1e-4) return 0.0;
        cut_rhs = rhs;
        return eff;
    }
    std::vector<char> mir_mark;

    int add_mir_round() {
        int n = M.n;
        if ((int)vub.size() != n) prepare_mir();
        mir_mark.assign(n, 0);
        std::vector<double> dense(n, 0.0), act(m_orig, 0.0);
        for (int i = 0; i < m_orig; ++i) for (auto& e : rowsR[i]) act[i] += e.second * S.x[e.first];
        std::vector<std::vector<std::pair<int, double>>> cuts;
        std::vector<double> lo, hi;
        std::vector<std::pair<int, double>> cut, rel, neg;
        std::vector<double> relc(n + m_orig, 0.0);
        std::vector<char> inrel(n + m_orig, 0);
        std::vector<int> relz;
        double cut_rhs = 0.0;
        for (int i0 = 0; i0 < m_orig && (int)cuts.size() < opt.max_cuts_per_round; ++i0) {
            if (Clock::now() > S.t_end) break;
            // relation  a_i0 x - r_i0 = 0
            for (int z : relz) { relc[z] = 0.0; inrel[z] = 0; }
            relz.clear();
            auto radd = [&](int z, double a) {
                if (!inrel[z]) { inrel[z] = 1; relz.push_back(z); }
                relc[z] += a;
            };
            for (auto& e : rowsR[i0]) radd(e.first, e.second);
            radd(n + i0, -1.0);
            std::vector<int> used = {i0};
            for (int agg = 0; agg <= 5; ++agg) {
                rel.clear(); neg.clear();
                for (int z : relz) if (std::fabs(relc[z]) > 1e-12) { rel.push_back({z, relc[z]}); neg.push_back({z, -relc[z]}); }
                double best = 0.0, rhs_best = 0.0;
                std::vector<std::pair<int, double>> best_cut;
                for (auto* R : {&rel, &neg}) {
                    double eff = mir_from_relation(*R, 0.0, act, dense, cut, cut_rhs);
                    if (eff > best) { best = eff; best_cut = cut; rhs_best = cut_rhs; }
                    // and the flow cover of the aggregated relation (path aggregation in network models: summing
                    // flow balances cancels the inner flows, and the cover then counts the arcs into the set)
                    if (agg > 0 && (opt.features & 1024)) {
                        eff = flowcover_from_relation(*R, 0.0, act, cut, cut_rhs);
                        if (eff > best) { best = eff; best_cut = cut; rhs_best = cut_rhs; }
                    }
                }
                if (best > 0.0) {
                    cuts.push_back(best_cut); lo.push_back(-INF); hi.push_back(rhs_best);
                    break;
                }
                // eliminate the continuous column farthest from its bounds
                int jb = -1; double far = 1e-6;
                for (int z : relz) {
                    if (z >= n || M.isint[z] || std::fabs(relc[z]) < 1e-12) continue;
                    double x = S.x[z], d = std::min(x - root_lb[z], root_ub[z] - x);
                    if (d > far) { far = d; jb = z; }
                }
                if (jb < 0) break;
                int kb = -1; double slack_best = INF;
                for (int k : rows_of_col[jb]) {
                    if (std::find(used.begin(), used.end(), k) != used.end()) continue;
                    double s = std::min(act[k] - M.rlo[k], M.rhi[k] - act[k]);
                    if (s < slack_best) { slack_best = s; kb = k; }
                }
                if (kb < 0) break;
                double akj = 0.0;
                for (auto& e : rowsR[kb]) if (e.first == jb) akj = e.second;
                if (std::fabs(akj) < 1e-9) break;
                double lam = -relc[jb] / akj;
                for (auto& e : rowsR[kb]) radd(e.first, lam * e.second);
                radd(n + kb, -lam);
                relc[jb] = 0.0;
                used.push_back(kb);
            }
        }
        if (cuts.empty()) return 0;
        M.add_rows(cuts, lo, hi);
        S.resize_to_model();
        S.refactor_full();
        return (int)cuts.size();
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

    // Root LPs of a MILP (the first one and those of the cut rounds) on a fully scaled copy of M, with the root
    // bounds. Integer columns keep the scale factor 1 in M (integrality stays unchanged), which can slow the dual
    // simplex down badly (crude scheduling L: no root LP within 120 s, against 8 s fully scaled). A positive column
    // scaling does not change which columns are basic, so the copy starts from S's basis, and its optimal basis is
    // loaded into S, whose own solve then has (almost) nothing left to do. Returns the iterations.
    long lp_on_scaled_copy() {
        Solver aux;
        aux.M = M;
        std::fill(aux.M.isint.begin(), aux.M.isint.end(), 0);
        aux.scale_model();
        aux.S.M = &aux.M; aux.S.opt = opt; aux.S.t_end = S.t_end;
        aux.S.init_slack_basis(aux.M.lb, aux.M.ub);
        if ((int)S.st.size() == aux.S.N && (int)S.head.size() == aux.S.m) {
            aux.S.st = S.st; aux.S.head = S.head;
            if (!aux.S.refactor_full()) aux.S.init_slack_basis(aux.M.lb, aux.M.ub);
        }
        Result r = aux.S.solve();
        if (r == OPTIMAL) {
            S.st = aux.S.st; S.head = aux.S.head;
            if (!S.refactor_full()) S.init_slack_basis(root_lb, root_ub);
        }
        return aux.S.iters;
    }

    // A root LP of a MILP: in S as usual, and on the fully scaled copy (from where S stopped) once it has taken
    // 1.5 (m + n) iterations; then the cut rounds' LPs go to the copy too. The root LPs of all MIPLIB 3 models and
    // MRPL-shaped models but one need at most 1.3 (m + n) iterations and so keep exactly their path; the crude
    // scheduling L root needs 12 (m + n) in M against 0.9 (m + n) fully scaled. Iterations, not seconds, decide,
    // so that runs stay repeatable.
    bool root_on_copy = false;
    Result solve_root_lp(long& copy_iters) {
        if (root_on_copy) {
            copy_iters += lp_on_scaled_copy();
            return S.solve();
        }
        long max_iter = S.opt.max_lp_iter;
        S.opt.max_lp_iter = std::min(max_iter, S.iters + std::max(1000L, (long)(1.5 * (M.m + M.n))));
        Result r = S.solve();
        S.opt.max_lp_iter = max_iter;
        if (r != ITER_LIMIT) return r;
        root_on_copy = true;
        if (opt.verbose)
            std::printf("  root LP: %ld iterations, continuing on a fully scaled copy\n", S.iters);
        copy_iters += lp_on_scaled_copy();
        return S.solve();
    }

    int solve_scaled(Info& info, std::vector<double>& xout) {
        t0 = Clock::now();
        S.M = &M; S.opt = opt;
        S.t_end = t0 + std::chrono::milliseconds((long long)(opt.time_limit * 1000));
        root_lb = M.lb; root_ub = M.ub;
        orig_lb = M.lb; orig_ub = M.ub;
        m_orig = M.m;
        M.rows_of(rowsR);
        for (int s = 0; s < 2; ++s) { pc_sum[s].assign(M.n, 0.0); pc_cnt[s].assign(M.n, 0); }
        bool has_int = false;
        for (int j = 0; j < M.n; ++j) if (M.isint[j]) has_int = true;

        S.init_slack_basis(root_lb, root_ub);
        if (warm_st) {                         // start from a given basis (RINS sub-MIP: the parent's node basis)
            S.st = *warm_st; S.head = *warm_head;
            if (!S.refactor_full()) S.init_slack_basis(root_lb, root_ub);
        }
        long copy_iters = 0;
        Result r = has_int && !warm_st ? solve_root_lp(copy_iters) : S.solve();
        info.lp_iters = (double)(S.iters + copy_iters);
        root_lp_iters = S.iters + copy_iters;
        root_lp_seconds = seconds_since(t0);
        if (r != OPTIMAL) {
            info.time = seconds_since(t0);
            return r == TIME_LIMIT ? 3 : r == INFEASIBLE ? 1 : r == UNBOUNDED ? 2 : 5;
        }
        info.root_bound = S.objective();
        if (!has_int) {
            // polish: re-solve from the optimal basis with tolerances 100x tighter, and confirm the result on a
            // fresh factorisation (the primal values then come from one solve, not from the updates); keep the
            // first answer if the tight pass does not finish cleanly (ill-conditioned models)
            xout.assign(S.x.begin(), S.x.begin() + M.n);
            double obj = S.objective();
            S.opt.tol_p = S.opt.tol_d = 1e-9;
            if (S.refactor_full() && S.dual() == OPTIMAL && (S.since_refactor == 0 || S.solve() == OPTIMAL)) {
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
        // cutting gets at most 40% of the time limit; a round whose LP does not finish by then is rolled back
        auto cut_end = t0 + std::chrono::milliseconds((long long)(0.4 * opt.time_limit * 1000));
        // network cut-set inequalities first, round after round while they are violated and the bound moves: they
        // alone can close a fixed-charge network's gap, and among the other families they would get a few rounds
        for (int nr = 0; (opt.features & 1024) && opt.cut_rounds > 0 && nr < 200 && Clock::now() < cut_end; ++nr) {
            int m_before = M.m;
            LPState before = save_lp();
            int added = add_network_round();
            if (!added) break;
            auto t_end = S.t_end;
            S.t_end = std::min(t_end, cut_end);
            r = solve_root_lp(copy_iters);
            S.t_end = t_end;
            if (r != OPTIMAL) {
                M.truncate_rows(m_before);
                S.shrink_to_model();
                restore_lp(before);
                r = OPTIMAL;
                break;
            }
            info.cuts += added;
            double now = S.objective();
            if (opt.verbose) std::printf("  network round %d: +%d cut-set cuts, bound %.10g\n", nr + 1, added, now);
            bool stalled = now - prev < 1e-6 * (1.0 + std::fabs(now));
            prev = now;
            if (stalled && nr >= 4) break;
        }
        for (int round = 0; round < opt.cut_rounds && Clock::now() < cut_end; ++round) {
            rounding_heuristic();
            int m_before = M.m;
            LPState before = save_lp();
            int n_gom = add_gomory_round();
            int n_cov = (opt.features & 4) ? add_cover_round() : 0;
            int n_mir = (opt.features & 32) ? add_mir_round() : 0;
            int n_flow = (opt.features & 256) ? add_flowcover_round() : 0;
            int n_net = (opt.features & 1024) ? add_network_round() : 0;
            int added = n_gom + n_cov + n_mir + n_flow + n_net;
            if (!added) break;
            auto t_end = S.t_end;
            S.t_end = std::min(t_end, cut_end);
            r = solve_root_lp(copy_iters);
            S.t_end = t_end;
            if (r != OPTIMAL) {
                // the LP with this round's cuts did not solve (time limit or numerical trouble): drop them and go on
                // from the last root LP that did
                M.truncate_rows(m_before);
                S.shrink_to_model();
                restore_lp(before);
                r = OPTIMAL;
                if (opt.verbose) std::printf("  cut round %d rolled back\n", round + 1);
                break;
            }
            info.cuts += added;
            double now = S.objective();
            if (opt.verbose)
                std::printf("  cut round %d: +%d cuts (Gomory %d, cover %d, c-MIR %d, flow cover %d, network %d), "
                            "bound %.10g\n", round + 1, added, n_gom, n_cov, n_mir, n_flow, n_net, now);
            if (now - prev < 1e-4 * (1.0 + std::fabs(now))) { prev = now; break; }
            prev = now;
        }
        info.root_bound_cuts = S.objective();
        if (opt.features & 16) build_propagation();

        // root heuristics: feasibility pump and a dive; with an incumbent, reduced-cost fixing and a guided dive
        long dive_budget = 1000 + 2L * S.m;
        if (opt.threads > 1 && (opt.features & 2048)) root_heuristics_parallel(dive_budget);
        else {
        if ((opt.features & 1) && !std::isfinite(inc_obj)) feasibility_pump(100, 10000 + 20L * S.m);
        if (opt.features & 2) dive(dive_budget);
        // when the cheap heuristics found nothing (a first incumbent that is merely feasible would stop the pump)
        if ((opt.features & 64) && !std::isfinite(inc_obj)) fix_and_propagate(std::max(1000L, 2 * root_lp_iters));
        if (std::isfinite(inc_obj)) {
            int fixed = reduced_cost_fixing(S.objective());
            if (opt.verbose && fixed) std::printf("  reduced-cost fixing: %d bounds tightened\n", fixed);
            if (opt.features & 2) dive(dive_budget);
            if (opt.features & 128) rins(0.1 * opt.time_limit, 2000, S.objective());
            if ((opt.features & 512) && !rins_last_ok) local_branching(0.05 * opt.time_limit, 1000, S.objective());
        }
        }
        if (opt.threads > 1) {
            long nodes = 0, iters = 0;
            double gb = INF;
            int status = parallel_tree(nodes, iters, gb);
            return finish_tree(info, xout, status, nodes, iters, gb);
        }

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
        // domain propagation after a bound change on column bv: tightenings become branching records of the node;
        // recompute: refresh the basic values (nonbasic columns may have moved to a new bound)
        std::vector<int> prop_changed;
        auto run_propagation = [&](int bv, bool recompute) {
            if (!(opt.features & 16) || bv < 0) return true;
            prop_changed.clear();
            if (!propagate({bv}, cur_lb, cur_ub, prop_changed)) return false;
            std::sort(prop_changed.begin(), prop_changed.end());
            prop_changed.erase(std::unique(prop_changed.begin(), prop_changed.end()), prop_changed.end());
            bool moved = false;
            for (int j : prop_changed) {
                tree.push_back({path, j, cur_lb[j], cur_ub[j]});
                path = (int)tree.size() - 1;
                if (!S.art[j] || std::isfinite(root_lb[j])) S.lb[j] = cur_lb[j];
                if (!S.art[j] || std::isfinite(root_ub[j])) S.ub[j] = cur_ub[j];
                if (S.st[j] != BASIC) moved = true;
            }
            if (moved && recompute) S.compute_xB();
            return true;
        };
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
                if (!run_propagation(pc_var, false)) { ++nodes; have_node = false; continue; }
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
            int bv = (opt.features & 8) ? reliability_branch(cur_bound) : most_fractional_or_pc(cur_bound);
            if (bv < 0) {
                std::vector<double> xs(S.x.begin(), S.x.begin() + M.n);
                try_incumbent(xs);
                have_node = false;
                continue;
            }
            if ((nodes & 15) == 0) rounding_heuristic();
            if ((opt.features & 2) && nodes % (std::isfinite(inc_obj) ? 2000 : 250) == 125) dive(500 + S.m);
            if ((opt.features & 128) && std::isfinite(inc_obj) && (nodes & 31) == 0 && rins_due())
            {
                rins(0.08 * opt.time_limit, 2000, global_bound());
                if ((opt.features & 512) && !rins_last_ok)
                    local_branching(0.05 * opt.time_limit, 1000, global_bound());
            }
            if ((opt.features & 64) && nodes % (std::isfinite(inc_obj) ? 500 : 250) == 60)
                fix_and_propagate(std::max(1000L, 2 * root_lp_iters));
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
            if (!run_propagation(bv, true)) { ++nodes; have_node = false; continue; }
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
        return finish_tree(info, xout, status, nodes, S.iters, gb);
    }

    // result and statistics of the tree search; gb: best bound of the nodes still open (inf if none)
    int finish_tree(Info& info, std::vector<double>& xout, int status, long nodes, long iters, double gb) {
        info.nodes = (double)nodes;
        info.lp_iters = (double)iters;
        info.time = seconds_since(t0);
        info.obj = inc_obj;
        info.bound = std::min(round_bound(gb), inc_obj);
        info.gap = std::isfinite(inc_obj) ? (inc_obj - info.bound) / std::max(1.0, std::fabs(inc_obj)) : INF;
        if (std::isfinite(inc_obj)) xout = inc;
        if (status == 5) return 5;
        if (status == 3) return std::isfinite(inc_obj) ? 3 : 6;
        if (status == 4) return std::isfinite(inc_obj) ? 4 : 6;
        return std::isfinite(inc_obj) ? 0 : 1;
    }

    // a copy of the solver for another thread (its simplex points to the copy's own model)
    std::unique_ptr<Solver> clone() const {
        std::unique_ptr<Solver> s(new Solver(*this));
        s->S.M = &s->M;
        s->warm_st = nullptr; s->warm_head = nullptr;
        s->opt.verbose = 0;
        return s;
    }

    // With several threads the root heuristics run side by side, each on its own copy of the solver at the root LP,
    // and their incumbents are merged in a fixed order, so the result does not depend on which finishes first.
    // Without an incumbent: feasibility pump, dive and fix-and-propagate; with one, after reduced-cost fixing: a
    // guided dive, RINS and local branching. (On one thread they run one after the other, fix-and-propagate only
    // when the others found nothing and local branching only when RINS found nothing.) Each keeps its own budget;
    // the time they take together counts once against the heuristics' share of the time limit.
    void root_heuristics_parallel(long dive_budget) {
        using Task = std::function<void(Solver&)>;
        auto run = [&](const std::vector<Task>& tasks, const char* what) {
            std::vector<std::unique_ptr<Solver>> C;
            if (tasks.empty()) return C;
            for (size_t t = 0; t < tasks.size(); ++t) C.push_back(clone());
            int nt = std::max(1, std::min(opt.threads, (int)tasks.size()));
            Crew crew(nt);
            crew.run([&](int w) { for (size_t t = w; t < tasks.size(); t += nt) tasks[t](*C[t]); });
            double h0 = heur_time, dh = 0.0;
            for (size_t t = 0; t < tasks.size(); ++t) {
                dh = std::max(dh, C[t]->heur_time - h0);
                if (C[t]->inc_obj < inc_obj - 1e-9) {
                    inc_obj = C[t]->inc_obj; inc = C[t]->inc;
                    if (opt.verbose)
                        std::printf("  incumbent %.10g  (root %s %zu, %.1fs)\n", inc_obj, what, t, seconds_since(t0));
                }
            }
            heur_time = h0 + dh;
            return C;
        };
        if (!std::isfinite(inc_obj)) {
            std::vector<Task> a;
            if (opt.features & 1) a.push_back([&](Solver& c) { c.feasibility_pump(100, 10000 + 20L * c.S.m); });
            if (opt.features & 2) a.push_back([&](Solver& c) { c.dive(dive_budget); });
            if (opt.features & 64)
                a.push_back([&](Solver& c) { c.fix_and_propagate(std::max(1000L, 2 * c.root_lp_iters)); });
            run(a, "heuristic");
        }
        if (!std::isfinite(inc_obj)) return;
        int fixed = reduced_cost_fixing(S.objective());
        if (opt.verbose && fixed) std::printf("  reduced-cost fixing: %d bounds tightened\n", fixed);
        std::vector<Task> b;
        int t_rins = -1, t_lb = -1;
        if (opt.features & 2) b.push_back([&](Solver& c) { c.dive(dive_budget); });
        if (opt.features & 128) {
            t_rins = (int)b.size();
            b.push_back([&](Solver& c) { c.rins(0.1 * opt.time_limit, 2000, c.S.objective()); });
        }
        if (opt.features & 512) {
            t_lb = (int)b.size();
            b.push_back([&](Solver& c) { c.local_branching(0.05 * opt.time_limit, 1000, c.S.objective()); });
        }
        auto C = run(b, "improvement heuristic");
        // keep RINS's and local branching's state (their spacing and budgets) from the copies that ran them
        if (t_rins >= 0) {
            const Solver& r = *C[t_rins];
            rins_calls = r.rins_calls; rins_off = r.rins_off; rins_time = r.rins_time; rins_last = r.rins_last;
            rins_last_inc = r.rins_last_inc; rins_wait = r.rins_wait; rins_last_ok = r.rins_last_ok;
            rins_hits = r.rins_hits;
        }
        if (t_lb >= 0) {
            lb_calls = C[t_lb]->lb_calls; lb_time = C[t_lb]->lb_time; lb_k = C[t_lb]->lb_k; lb_fails = C[t_lb]->lb_fails;
        }
    }

    // ---- parallel tree search (opt.threads > 1)
    // The search runs in rounds. At the start of a round the best open nodes are dealt out to the threads, each of
    // which works on its own copy of the solver (model with the root cuts, simplex, pseudocosts, incumbent,
    // heuristics) and processes its nodes as the serial search does, best bound first and plunging into one child,
    // until it has done a fixed amount of LP work; the node of the plunge it is in stays with it for the next round.
    // At the end of the round the open nodes, incumbents and pseudocost observations of all threads are merged in
    // thread order. A round ends after a number of simplex iterations, not after a time, so the search does not
    // depend on how the threads are scheduled: a model gives the same tree on every run (up to the time-based
    // heuristic budgets that the serial search has too).
    struct TreeWork {
        NodeHeap heap;                              // nodes dealt to this thread, and the children it creates
        long nodes = 0, budget = 0, next_id = 0;
        double bound = -INF;                        // global bound at the start of the round (RINS's gap test)
        bool plunging = false;                      // the simplex holds a solved node, to go on with next round
        double cur_bound = INF;                     // ... its bound
        std::shared_ptr<const BoundBlock> chain;    // ... and its bound changes
        bool time_up = false, numeric = false;
        std::vector<double> cur_lb, cur_ub;
        std::vector<char> seen;
        std::vector<int> touched, changed;
    };

    // one thread's share of a round (runs on a worker copy of the solver)
    void tree_round(TreeWork& R) {
        const long it0 = S.iters + S.work_extra, nodes0 = R.nodes;
        auto work = [&]() { return S.iters + S.work_extra - it0 + 10 * (R.nodes - nodes0); };
        auto set_bound = [&](int j) {
            if (!S.art[j] || std::isfinite(root_lb[j])) S.lb[j] = R.cur_lb[j];
            if (!S.art[j] || std::isfinite(root_ub[j])) S.ub[j] = R.cur_ub[j];
        };
        auto extend = [&](std::vector<BoundChange> ch) {
            auto b = std::make_shared<BoundBlock>();
            b->parent = R.chain;
            b->ch = std::move(ch);
            return std::shared_ptr<const BoundBlock>(std::move(b));
        };
        // domain propagation after a bound change on bv; the tightened bounds become a block of the node's chain
        auto propagate_node = [&](int bv, bool recompute) {
            if (!(opt.features & 16) || bv < 0) return true;
            R.changed.clear();
            if (!propagate({bv}, R.cur_lb, R.cur_ub, R.changed)) return false;
            std::sort(R.changed.begin(), R.changed.end());
            R.changed.erase(std::unique(R.changed.begin(), R.changed.end()), R.changed.end());
            if (R.changed.empty()) return true;
            std::vector<BoundChange> ch;
            bool moved = false;
            for (int j : R.changed) {
                ch.push_back({j, R.cur_lb[j], R.cur_ub[j]});
                set_bound(j);
                if (S.st[j] != BASIC) moved = true;
            }
            R.chain = extend(std::move(ch));
            if (moved && recompute) S.compute_xB();
            return true;
        };
        bool have_node = R.plunging;
        double cur_bound = R.cur_bound;
        for (;;) {
            if (!have_node) {
                if (work() >= R.budget) break;
                Node* nd = nullptr;
                while (!R.heap.empty()) {
                    nd = R.heap.top(); R.heap.pop();
                    if (nd->bound < cutoff()) break;
                    delete nd; nd = nullptr;
                }
                if (!nd) break;
                double popped_bound = nd->bound;
                // the node's bounds: the root bounds, then the changes of its chain, deepest first
                R.chain = nd->chain;
                R.cur_lb = root_lb; R.cur_ub = root_ub;
                for (const BoundBlock* b = R.chain.get(); b; b = b->parent.get())
                    for (auto e = b->ch.rbegin(); e != b->ch.rend(); ++e) {
                        if (R.seen[e->var]) continue;
                        R.seen[e->var] = 1; R.touched.push_back(e->var);
                        R.cur_lb[e->var] = e->lo; R.cur_ub[e->var] = e->hi;
                    }
                for (int j : R.touched) R.seen[j] = 0;
                R.touched.clear();
                for (int j = 0; j < M.n; ++j) set_bound(j);
                nd->load(S.st, S.head);
                S.w.assign(S.m, S.m <= 1000 ? -1.0 : 1.0);
                int pc_var = nd->branch_var, pc_dir = nd->dir;
                double pc_frac = nd->frac, pc_parent = nd->parent_obj;
                delete nd;
                if (!propagate_node(pc_var, false)) { ++R.nodes; continue; }
                if (!S.refactor_full()) { R.numeric = true; break; }
                Result rr = S.dual();
                ++R.nodes;
                if (rr == TIME_LIMIT) { R.time_up = true; cur_bound = popped_bound; have_node = true; break; }
                if (rr != OPTIMAL) continue;
                cur_bound = S.objective();
                have_node = true;
                if (pc_var >= 0) {
                    int s = pc_dir < 0 ? 0 : 1;
                    double dist = pc_dir < 0 ? pc_frac : 1 - pc_frac;
                    pc_sum[s][pc_var] += std::max(0.0, cur_bound - pc_parent) / std::max(dist, 1e-6);
                    pc_cnt[s][pc_var]++;
                }
            }
            // the current node is solved and held in S
            if (Clock::now() > S.t_end) { R.time_up = true; break; }
            if (cur_bound >= cutoff()) { have_node = false; continue; }
            if (work() >= R.budget) break;                   // the plunge goes on in the next round
            int bv = (opt.features & 8) ? reliability_branch(cur_bound) : most_fractional_or_pc(cur_bound);
            if (bv < 0) {
                try_incumbent(std::vector<double>(S.x.begin(), S.x.begin() + M.n));
                have_node = false;
                continue;
            }
            const long nodes = R.nodes;
            if ((nodes & 15) == 0) rounding_heuristic();
            if ((opt.features & 2) && nodes % (std::isfinite(inc_obj) ? 2000 : 250) == 125) dive(500 + S.m);
            if ((opt.features & 128) && std::isfinite(inc_obj) && (nodes & 31) == 0 && rins_due())
            {
                rins(0.08 * opt.time_limit, 2000, std::min(R.bound, cur_bound));
                if ((opt.features & 512) && !rins_last_ok)
                    local_branching(0.05 * opt.time_limit, 1000, std::min(R.bound, cur_bound));
            }
            if ((opt.features & 64) && nodes % (std::isfinite(inc_obj) ? 500 : 250) == 60)
                fix_and_propagate(std::max(1000L, 2 * root_lp_iters));
            double v = S.x[bv], f = v - std::floor(v);
            double dn = std::floor(v), up = std::ceil(v);
            // plunge into the child with the smaller estimated degradation, keep the other one
            double pd = pc_cnt[0][bv] ? pc_sum[0][bv] / pc_cnt[0][bv] : 1.0;
            double pu = pc_cnt[1][bv] ? pc_sum[1][bv] / pc_cnt[1][bv] : 1.0;
            int first = (pd * f < pu * (1 - f)) ? -1 : +1;
            double olb = R.cur_lb[bv], oub = R.cur_ub[bv];
            Node* other = new Node();
            other->bound = cur_bound; other->id = R.next_id++;
            other->branch_var = bv; other->dir = -first; other->frac = f; other->parent_obj = cur_bound;
            other->chain = first < 0 ? extend({{bv, up, oub}}) : extend({{bv, olb, dn}});
            other->save(S.st);
            R.heap.push(other);
            if (first < 0) { R.cur_ub[bv] = dn; S.ub[bv] = dn; R.chain = extend({{bv, olb, dn}}); }
            else { R.cur_lb[bv] = up; S.lb[bv] = up; R.chain = extend({{bv, up, oub}}); }
            if (!propagate_node(bv, true)) { ++R.nodes; have_node = false; continue; }
            double parent = cur_bound;
            Result rr = S.dual();
            ++R.nodes;
            if (rr == TIME_LIMIT) { R.time_up = true; cur_bound = parent; break; }
            if (rr != OPTIMAL) { have_node = false; continue; }
            cur_bound = S.objective();
            int s = first < 0 ? 0 : 1;
            pc_sum[s][bv] += std::max(0.0, cur_bound - parent) / std::max(first < 0 ? f : 1 - f, 1e-6);
            pc_cnt[s][bv]++;
        }
        R.plunging = have_node && !R.numeric;
        R.cur_bound = cur_bound;
        if (!R.plunging) R.chain.reset();
    }

    int parallel_tree(long& nodes_out, long& iters_out, double& gb_out) {
        const int K = std::max(2, std::min(opt.threads, 64));
        NodeHeap pool;                                  // open nodes between rounds, best bound first
        long next_id = 0;
        Node* root = new Node();                        // the root, solved, as the first open node
        root->bound = S.objective(); root->id = next_id++;
        root->branch_var = -1; root->dir = 0; root->frac = 0.0; root->parent_obj = root->bound;
        root->save(S.st);
        pool.push(root);
        std::vector<std::unique_ptr<Solver>> W;
        std::vector<TreeWork> T(K);
        for (int w = 0; w < K; ++w) {
            W.emplace_back(new Solver(*this));
            Solver& s = *W.back();
            s.S.M = &s.M;
            s.warm_st = nullptr; s.warm_head = nullptr;
            s.opt.verbose = 0;
            T[w].cur_lb = root_lb; T[w].cur_ub = root_ub;
            T[w].seen.assign(M.n, 0);
        }
        const long base_iters = S.iters;
        const long B = 4000;           // work per thread and round: simplex iterations (and equivalents) + 10 per node
        auto global_bound = [&]() {
            double gb = pool.empty() ? INF : pool.top()->bound;
            for (auto& R : T) {
                if (!R.heap.empty()) gb = std::min(gb, R.heap.top()->bound);
                if (R.plunging) gb = std::min(gb, R.cur_bound);
            }
            return gb;
        };
        Crew crew(K);
        int status = 0, rounds = 0;
        long nodes = 0, last_print = 0;
        double busy = 0.0, wall = 0.0;                  // thread-seconds of work, seconds of rounds
        std::vector<double> busy_w(K, 0.0);
        for (int round = 0;; ++round) {
            // deal out the best open nodes, up to eight per thread, in turn
            double cut = cutoff();
            bool dealt_all = false;
            for (int k = 0; k < 8 && !dealt_all; ++k)
                for (int w = 0; w < K; ++w) {
                    Node* nd = nullptr;
                    while (!pool.empty()) {
                        nd = pool.top(); pool.pop();
                        if (nd->bound < cut) break;
                        delete nd; nd = nullptr;
                    }
                    if (!nd) { dealt_all = true; break; }
                    T[w].heap.push(nd);
                }
            bool work_left = false;
            for (auto& R : T) work_left = work_left || !R.heap.empty() || R.plunging;
            if (!work_left) break;                      // tree exhausted
            double gb = std::min(global_bound(), inc_obj);
            for (int w = 0; w < K; ++w) {
                Solver& s = *W[w];
                if (inc_obj < s.inc_obj) { s.inc_obj = inc_obj; s.inc = inc; }
                for (int d = 0; d < 2; ++d) { s.pc_sum[d] = pc_sum[d]; s.pc_cnt[d] = pc_cnt[d]; }
                T[w].budget = std::min(B, 250L << std::min(round, 8));   // short first rounds: spread the tree
                T[w].bound = gb;
                T[w].next_id = next_id;
            }
            auto r0 = Clock::now();
            crew.run([&](int w) {
                auto w0 = Clock::now();
                W[w]->tree_round(T[w]);
                busy_w[w] = seconds_since(w0);
            });
            wall += seconds_since(r0);
            for (double b : busy_w) busy += b;
            ++rounds;
            // merge in thread order: open nodes, incumbents, pseudocost observations
            nodes = 0;
            bool time_up = false, numeric = false;
            for (int w = 0; w < K; ++w) {
                Solver& s = *W[w];
                for (Node* nd : T[w].heap.take()) { nd->id = next_id++; pool.push(nd); }
                if (s.inc_obj < inc_obj - 1e-9) {
                    inc_obj = s.inc_obj; inc = s.inc;
                    if (opt.verbose)
                        std::printf("  incumbent %.10g  (thread %d, %.1fs)\n", inc_obj, w, seconds_since(t0));
                }
                nodes += T[w].nodes;
                time_up = time_up || T[w].time_up;
                numeric = numeric || T[w].numeric;
            }
            for (int d = 0; d < 2; ++d)
                for (int j = 0; j < M.n; ++j) {
                    double sum = pc_sum[d][j];
                    int cnt = pc_cnt[d][j];
                    for (int w = 0; w < K; ++w) {
                        sum += W[w]->pc_sum[d][j] - pc_sum[d][j];
                        cnt += W[w]->pc_cnt[d][j] - pc_cnt[d][j];
                    }
                    pc_sum[d][j] = sum; pc_cnt[d][j] = cnt;
                }
            if (numeric) { status = 5; break; }
            if (time_up || Clock::now() > S.t_end) { status = 3; break; }
            if (nodes >= opt.node_limit) { status = 4; break; }
            if (opt.verbose && nodes - last_print >= 2000) {
                last_print = nodes;
                size_t open = pool.size();
                for (auto& R : T) open += R.plunging;
                std::printf("  nodes %8ld  open %7zu  incumbent %.10g  bound %.10g  %.1fs  (%d threads)\n", nodes,
                            open, inc_obj, std::min(global_bound(), inc_obj), seconds_since(t0), K);
            }
        }
        long iters = S.iters;
        for (auto& s : W) iters += s->S.iters - base_iters;
        if (opt.verbose)
            std::printf("  parallel tree: %d threads, %d rounds, %.2f s, threads busy %.0f%% of it\n", K, rounds, wall,
                        wall > 0 ? 100.0 * busy / (K * wall) : 0.0);
        gb_out = status == 0 ? INF : global_bound();
        // free the open nodes on all threads (millions of them on some models: a second on one thread)
        std::vector<Node*> left = pool.take();
        for (auto& R : T) for (Node* nd : R.heap.take()) left.push_back(nd);
        crew.run([&](int w) { for (size_t i = w; i < left.size(); i += K) delete left[i]; });
        nodes_out = nodes;
        iters_out = iters;
        return status;
    }
};

}  // namespace sm

// ---------------------------------------------------------------------------------------------- C ABI
// opts: [time_limit, node_limit, cut_rounds, verbose, gap, features (see Options), threads]
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
    s.opt.features = (int)opts[5];
    s.opt.threads = (int)opts[6];
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
                        const double* y0, const double* opts, double* x_out, double* info_out) {
    sm::Solver s;
    load_model(s, n, m, colptr, rowidx, vals, c, lb, ub, rlo, rhi);
    s.opt.time_limit = opts[0];
    s.opt.verbose = (int)opts[1];
    if (opts[2] > 0) s.opt.crash_tol = opts[2];
    sm::Info info;
    std::vector<double> x, xs(x0, x0 + n), ys;
    if (y0) ys.assign(y0, y0 + m);
    int status = s.crossover(xs, ys, info, x);
    for (int j = 0; j < n; ++j) x_out[j] = j < (int)x.size() ? x[j] : 0.0;
    store_info(info, info_out);
    return status;
}

// Probing for presolve: the model rlo <= A x <= rhi (CSC) with integer columns isint and bounds lb/ub, tightened
// in place by probing its binary columns until the propagation has visited `work_limit` row entries (or after
// `seconds`, a safety net). Equivalent binaries are written to eq as triples (k, j, sign): x_k = x_j (sign 1) or
// x_k = 1 - x_j (sign -1); eq has room for n triples. Returns 1 if the model is infeasible, else 0;
// stats = {columns fixed by a probe, bounds tightened, equivalences, work}.
SM_API int sm_probe(int n, int m, const int* colptr, const int* rowidx, const double* vals, const double* rlo,
                    const double* rhi, const char* isint, double* lb, double* ub, double work_limit, double seconds,
                    double* stats, int* eq) {
    sm::Solver s;
    std::vector<double> zero(n, 0.0);
    load_model(s, n, m, colptr, rowidx, vals, zero.data(), lb, ub, rlo, rhi);
    s.M.isint.assign(isint, isint + n);
    std::vector<double> L(lb, lb + n), U(ub, ub + n);
    long fixed = 0, tightened = 0;
    std::vector<std::array<int, 3>> equiv;
    bool ok = s.probe(L, U, work_limit, seconds, fixed, tightened, &equiv);
    std::memcpy(lb, L.data(), sizeof(double) * n);
    std::memcpy(ub, U.data(), sizeof(double) * n);
    for (size_t t = 0; t < equiv.size() && t < (size_t)n; ++t)
        for (int c = 0; c < 3; ++c) eq[3 * t + c] = equiv[t][c];
    stats[0] = (double)fixed; stats[1] = (double)tightened; stats[2] = (double)std::min(equiv.size(), (size_t)n);
    stats[3] = (double)s.prop_work;
    return ok ? 0 : 1;
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

// Test hook for the sparse LU (tests/test_samadhan.py). Factorises the m x m CSC matrix B, then replaces the
// basis columns rs[0..k-1] one after another by the dense columns newcols[t*m .. t*m+m-1] through Forrest-Tomlin
// updates, refactorising whenever an update fails or asks for it (as the simplex does), and solves B x = b and
// B' y = d in place. Returns the rank of the original B (the solves are skipped when it is singular) and the
// number of refactorisations in counts[0], of updates refused as unstable in counts[1].
SM_API int sm_lu_check(int m, const int* colptr, const int* rowidx, const double* vals, int k, const int* rs,
                       const double* newcols, double* b, double* d, int* counts) {
    sm::Factor F;
    std::vector<int> bs(colptr, colptr + m + 1), bi(rowidx, rowidx + colptr[m]);
    std::vector<double> bv(vals, vals + colptr[m]);
    counts[0] = counts[1] = 0;
    if (!F.factor(m, bs, bi, bv)) return F.rank;
    std::vector<std::vector<std::pair<int, double>>> cols(m);
    for (int c = 0; c < m; ++c)
        for (int t = bs[c]; t < bs[c + 1]; ++t) cols[c].push_back({bi[t], bv[t]});
    for (int t = 0; t < k; ++t) {
        const double* a = newcols + (size_t)t * m;
        int r = rs[t];
        cols[r].clear();
        for (int i = 0; i < m; ++i) if (a[i] != 0.0) cols[r].push_back({i, a[i]});
        std::vector<double> alpha(a, a + m);
        F.ftran(alpha, true);
        if (!F.replace_column(r, alpha[r])) counts[1]++;
        if (F.stale || F.refresh) {
            bs.assign(1, 0); bi.clear(); bv.clear();
            for (int c = 0; c < m; ++c) {
                for (auto& e : cols[c]) { bi.push_back(e.first); bv.push_back(e.second); }
                bs.push_back((int)bi.size());
            }
            if (!F.factor(m, bs, bi, bv)) return -1;
            counts[0]++;
        }
    }
    std::vector<double> x(b, b + m), y(d, d + m);
    F.ftran(x);
    F.btran(y);
    std::memcpy(b, x.data(), sizeof(double) * m);
    std::memcpy(d, y.data(), sizeof(double) * m);
    return F.rank;
}

// ---------------------------------------------------------------------------------------------- presolve
// The presolve of samadhan/presolve.py in C++: the same reductions in the same order with the same arithmetic, so
// that both give the same reduced model (presolve.py stays as the reference implementation; the tests compare the
// two). The comments there explain each reduction; here only what the port needs.
namespace ps {

constexpr double FEAS_TOL = 1e-9, INT_TOL = 1e-6, PROP_WORK = 5e7;
constexpr double INF = std::numeric_limits<double>::infinity();

static double fin0(double v) { return std::isfinite(v) ? v : 0.0; }
static double pfloor(double v) { return std::floor(v + INT_TOL * std::max(1.0, std::fabs(fin0(v)))); }
static double pceil(double v) { return std::ceil(v - INT_TOL * std::max(1.0, std::fabs(fin0(v)))); }

struct Sub {                       // x_k = (b - coefs . x[cols]) / a, recovered in postsolve
    int k;
    double b;
    std::vector<int> cols;
    std::vector<double> coefs;
    double a;
};

enum Stat { PASSES, FIXED, EMPTY_ROWS, SINGLETON_ROWS, REDUNDANT, FORCING, TIGHTENED, FREE_SINGLETONS, DUAL_FIXED,
            EMPTY_COLS, PROBE_FIXED, PROBE_TIGHTENED, PROBE_EQUIV, PARALLEL, DOUBLETONS, NSTAT };

struct Presolver {
    int m = 0, n = 0;
    std::vector<int> ap, aj;                   // A as CSR, columns sorted within each row (inactive rows emptied
    std::vector<double> ax;                    // at the next substitution)
    std::vector<double> lo, hi, l, u, c, pl, pu, x_fixed;
    std::vector<char> isint, row_on, col_on;
    double cnst = 0.0;
    std::vector<Sub> stack;
    long st[NSTAT] = {0};
    int status = 0;                            // 0 reduced, 1 infeasible, 2 unbounded
    // the active submatrix S of the current pass: rows `rows`, columns `cols` (local indices in sp/si/sv, CSR)
    std::vector<int> rows, cols, colloc, sp, si;
    std::vector<double> sv;
    // work arrays
    std::vector<double> wval;
    std::vector<char> wflag;

    bool cols_dirty = true;                    // col_on changed since the last build_S
    void build_S() {
        if (cols_dirty) {
            cols.clear();
            colloc.assign(n, -1);
            for (int j = 0; j < n; ++j) if (col_on[j]) { colloc[j] = (int)cols.size(); cols.push_back(j); }
            cols_dirty = false;
        }
        rows.clear();
        sp.resize(m + 1); si.resize(aj.size()); sv.resize(aj.size());
        int r = 0, nz = 0;
        sp[0] = 0;
        const int* cl = colloc.data();
        for (int i = 0; i < m; ++i) {
            if (!row_on[i]) continue;
            rows.push_back(i);
            for (int t = ap[i]; t < ap[i + 1]; ++t) {
                int jl = cl[aj[t]];
                if (jl >= 0) { si[nz] = jl; sv[nz] = ax[t]; ++nz; }
            }
            sp[++r] = nz;
        }
        sp.resize(r + 1); si.resize(nz); sv.resize(nz);
    }

    // fix columns fc at vals: objective constant and the rows' sides absorb them
    void fix(const std::vector<int>& fc, const std::vector<double>& vals) {
        double dot = 0.0;
        for (size_t t = 0; t < fc.size(); ++t) {
            x_fixed[fc[t]] = vals[t]; col_on[fc[t]] = 0; cols_dirty = true;
            dot += c[fc[t]] * vals[t];
            wflag[fc[t]] = 1; wval[fc[t]] = vals[t];
        }
        cnst += dot;
        for (int i = 0; i < m; ++i) {
            if (!row_on[i]) continue;
            double s = 0.0;
            bool any = false;
            for (int t = ap[i]; t < ap[i + 1]; ++t)
                if (wflag[aj[t]]) { s += ax[t] * wval[aj[t]]; any = true; }
            if (any) { lo[i] -= s; hi[i] -= s; }
        }
        for (int j : fc) wflag[j] = 0;
    }

    // row activity bounds of S over bl <= x <= bu (local columns): finite parts and numbers of infinite terms
    void activity(const std::vector<double>& bl, const std::vector<double>& bu, std::vector<double>& mn,
                  std::vector<double>& mn_inf, std::vector<double>& mx, std::vector<double>& mx_inf) const {
        int ms = (int)rows.size();
        mn.assign(ms, 0.0); mn_inf.assign(ms, 0.0); mx.assign(ms, 0.0); mx_inf.assign(ms, 0.0);
        for (int r = 0; r < ms; ++r) {
            double pmn = 0.0, nmn = 0.0, pmx = 0.0, nmx = 0.0, imn = 0.0, imx = 0.0;
            for (int t = sp[r]; t < sp[r + 1]; ++t) {
                double a = sv[t], lj = bl[si[t]], uj = bu[si[t]];
                if (a > 0) {
                    pmn += a * fin0(lj); pmx += a * fin0(uj);
                    imn += !std::isfinite(lj); imx += !std::isfinite(uj);
                } else if (a < 0) {
                    nmn += a * fin0(uj); nmx += a * fin0(lj);
                    imn += !std::isfinite(uj); imx += !std::isfinite(lj);
                }
            }
            mn[r] = pmn + nmn; mx[r] = pmx + nmx; mn_inf[r] = imn; mx_inf[r] = imx;
        }
    }

    // bound propagation (presolve.py: _propagate); returns the number of tightened integer bounds
    long propagate(const std::vector<double>& rlo, const std::vector<double>& rhi) {
        long nnz = (long)si.size();
        if (!nnz) return 0;
        int ms = (int)rows.size(), ns = (int)cols.size();
        long sweeps = std::max(1L, (long)std::floor(PROP_WORK / (double)nnz));
        std::vector<double> bl(ns), bu(ns), mn, mn_inf, mx, mx_inf, cand_u(n), cand_l(n);
        std::vector<char> touched(n, 0);
        long count = 0;
        for (long sw = 0; sw < std::min(sweeps, 500L); ++sw) {
            bool bad = false;
            for (int k = 0; k < ns; ++k) {
                int j = cols[k];
                bl[k] = isint[j] ? l[j] : std::max(l[j], pl[j]);
                bu[k] = isint[j] ? u[j] : std::min(u[j], pu[j]);
                if (bl[k] > bu[k] + FEAS_TOL * (1.0 + std::fabs(fin0(bl[k])))) bad = true;
            }
            if (bad) break;
            activity(bl, bu, mn, mn_inf, mx, mx_inf);
            std::fill(cand_u.begin(), cand_u.end(), INF);
            std::fill(cand_l.begin(), cand_l.end(), -INF);
            for (int r = 0; r < ms; ++r)
                for (int t = sp[r]; t < sp[r + 1]; ++t) {
                    double a = sv[t], lj = bl[si[t]], uj = bu[si[t]];
                    int j = cols[si[t]];
                    double cmin = a > 0 ? a * lj : a * uj, cmax = a > 0 ? a * uj : a * lj;
                    bool inf_min = !std::isfinite(cmin), inf_max = !std::isfinite(cmax);
                    double rest_min = (mn_inf[r] - inf_min == 0) ? mn[r] - (inf_min ? 0.0 : cmin) : -INF;
                    double rest_max = (mx_inf[r] - inf_max == 0) ? mx[r] - (inf_max ? 0.0 : cmax) : INF;
                    double from_hi = (rhi[r] - rest_min) / a, from_lo = (rlo[r] - rest_max) / a;
                    double nu = a > 0 ? from_hi : from_lo, nl = a > 0 ? from_lo : from_hi;
                    if (std::isnan(nu)) nu = INF;
                    if (std::isnan(nl)) nl = -INF;
                    if (isint[j]) { nu = pfloor(nu); nl = pceil(nl); }
                    cand_u[j] = std::min(cand_u[j], nu);
                    cand_l[j] = std::max(cand_l[j], nl);
                }
            bool imp = false;
            std::vector<int> iu, il, cu, cl;
            for (int j = 0; j < n; ++j) {
                double cur_u = isint[j] ? u[j] : std::min(u[j], pu[j]);
                double cur_l = isint[j] ? l[j] : std::max(l[j], pl[j]);
                double ref_u = std::fabs(std::isfinite(cur_u) ? cur_u : fin0(cand_u[j]));
                double ref_l = std::fabs(std::isfinite(cur_l) ? cur_l : fin0(cand_l[j]));
                bool imp_u = cand_u[j] < cur_u - (isint[j] ? 0.5 : 1e-3 * std::max(1.0, ref_u));
                bool imp_l = cand_l[j] > cur_l + (isint[j] ? 0.5 : 1e-3 * std::max(1.0, ref_l));
                if (imp_u) { imp = true; (isint[j] ? iu : cu).push_back(j); }
                if (imp_l) { imp = true; (isint[j] ? il : cl).push_back(j); }
            }
            if (!imp) break;
            for (int j : iu) u[j] = cand_u[j];
            for (int j : il) l[j] = cand_l[j];
            count += (long)(iu.size() + il.size());
            for (int j : cu) pu[j] = cand_u[j];
            for (int j : cl) pl[j] = cand_l[j];
        }
        return count;
    }

    // x_k = b0_k + t_k x_{target_k} for every k in ks: A <- A T, c <- T'c, the rows' sides absorb A b0
    // (presolve.py builds the matrix T for doubletons and for probing equivalences)
    void substitute(const std::vector<int>& ks, const std::vector<int>& target, const std::vector<double>& t,
                    const std::vector<double>& b0) {
        std::vector<int> kidx(n, -1);
        for (size_t q = 0; q < ks.size(); ++q) kidx[ks[q]] = (int)q;
        // shift = A b0 (sequential over each row's entries; only substituted columns have b0 != 0)
        for (int i = 0; i < m; ++i) {
            if (!row_on[i]) continue;
            double s = 0.0;
            bool any = false;
            for (int e = ap[i]; e < ap[i + 1]; ++e) {
                int q = kidx[aj[e]];
                if (q >= 0 && b0[q] != 0.0) { s += ax[e] * b0[q]; any = true; }
            }
            if (any) { lo[i] -= s; hi[i] -= s; }
        }
        double dot = 0.0;
        for (size_t q = 0; q < ks.size(); ++q) dot += c[ks[q]] * b0[q];
        cnst += dot;
        // c <- T'c: the column j of T has 1 at row j (kept columns) and t_k at the rows k with target j, summed in
        // row order
        std::vector<std::array<double, 3>> into;          // (target, row of T, value), sorted
        into.reserve(2 * ks.size());
        for (size_t q = 0; q < ks.size(); ++q) into.push_back({(double)target[q], (double)ks[q], c[ks[q]] * t[q]});
        std::vector<char> tgt(n, 0);
        for (size_t q = 0; q < ks.size(); ++q) tgt[target[q]] = 1;
        for (int j = 0; j < n; ++j) if (tgt[j] && kidx[j] < 0) into.push_back({(double)j, (double)j, c[j] * 1.0});
        std::sort(into.begin(), into.end(), [](const std::array<double, 3>& a, const std::array<double, 3>& b) {
            return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
        });
        std::vector<double> cn(c);
        for (size_t q = 0; q < ks.size(); ++q) cn[ks[q]] = 0.0;
        for (size_t g = 0; g < into.size();) {
            size_t e = g;
            int j = (int)into[g][0];
            double s = 0.0;
            while (e < into.size() && (int)into[e][0] == j) s += into[e++][2];
            if (kidx[j] < 0) cn[j] = s;
            g = e;
        }
        c.swap(cn);
        // A <- A T, row by row in entry order (scipy's sparse product), exact zeros dropped, columns sorted; the
        // rows no longer active are dropped
        std::vector<double> acc(n, 0.0);
        std::vector<char> seen(n, 0);
        std::vector<int> touched, nap(m + 1, 0), naj;
        std::vector<double> nax;
        naj.reserve(aj.size()); nax.reserve(ax.size());
        for (int i = 0; i < m; ++i) {
            if (row_on[i]) {
                bool hit = false;
                for (int e = ap[i]; e < ap[i + 1]; ++e) if (kidx[aj[e]] >= 0) { hit = true; break; }
                if (!hit) {
                    naj.insert(naj.end(), aj.begin() + ap[i], aj.begin() + ap[i + 1]);
                    nax.insert(nax.end(), ax.begin() + ap[i], ax.begin() + ap[i + 1]);
                } else {
                    touched.clear();
                    for (int e = ap[i]; e < ap[i + 1]; ++e) {
                        int j = aj[e], q = kidx[j];
                        int to = q >= 0 ? target[q] : j;
                        double v = q >= 0 ? ax[e] * t[q] : ax[e] * 1.0;
                        if (!seen[to]) { seen[to] = 1; acc[to] = 0.0; touched.push_back(to); }
                        acc[to] += v;
                    }
                    std::sort(touched.begin(), touched.end());
                    for (int j : touched) {
                        if (acc[j] != 0.0) { naj.push_back(j); nax.push_back(acc[j]); }
                        seen[j] = 0;
                    }
                }
            }
            nap[i + 1] = (int)naj.size();
        }
        ap.swap(nap); aj.swap(naj); ax.swap(nax);
    }

    // parallel rows of S (two or more entries): pairs (i, k, lam), row k = lam row i, i the first row of its group
    std::vector<std::tuple<int, int, double>> parallel_rows() const {
        int ms = (int)rows.size();
        std::vector<int> cand;
        std::vector<double> first(ms, 0.0);
        for (int r = 0; r < ms; ++r) {
            int cnt = sp[r + 1] - sp[r];
            if (cnt > 0) first[r] = sv[sp[r]];
            if (cnt > 1) cand.push_back(r);
        }
        std::vector<std::tuple<int, int, double>> pairs;
        if (cand.size() < 2) return pairs;
        auto scaled = [&](int r, int t) { return sv[t] / first[r]; };
        // order by count, pattern and scaled values; equal rows end up next to each other, lowest row first
        auto less = [&](int a, int b) {
            int ca = sp[a + 1] - sp[a], cb = sp[b + 1] - sp[b];
            if (ca != cb) return ca < cb;
            for (int t = 0; t < ca; ++t) {
                int ja = si[sp[a] + t], jb = si[sp[b] + t];
                if (ja != jb) return ja < jb;
            }
            for (int t = 0; t < ca; ++t) {
                double va = scaled(a, sp[a] + t), vb = scaled(b, sp[b] + t);
                if (va != vb) return va < vb;              // exact order: a valid sort; equal-within-1e-12 rows
            }                                              // still end up next to each other
            return a < b;
        };
        std::vector<int> order(cand);
        std::stable_sort(order.begin(), order.end(), less);
        auto same = [&](int head, int k) {                 // np.allclose(kv, hv, rtol=1e-12, atol=0)
            int ch = sp[head + 1] - sp[head], ck = sp[k + 1] - sp[k];
            if (ch != ck) return false;
            for (int t = 0; t < ch; ++t) if (si[sp[head] + t] != si[sp[k] + t]) return false;
            for (int t = 0; t < ch; ++t) {
                double kv = scaled(k, sp[k] + t), hv = scaled(head, sp[head] + t);
                if (!(std::fabs(kv - hv) <= 1e-12 * std::fabs(hv))) return false;
            }
            return true;
        };
        for (size_t g = 0; g < order.size();) {
            size_t e = g + 1;
            while (e < order.size() && same(order[g], order[e])) ++e;
            if (e - g > 1) {
                std::vector<int> grp(order.begin() + g, order.begin() + e);
                std::sort(grp.begin(), grp.end());
                for (size_t q = 1; q < grp.size(); ++q)
                    if (same(grp[0], grp[q])) pairs.push_back({grp[0], grp[q], first[grp[q]] / first[grp[0]]});
            }
            g = e;
        }
        return pairs;
    }

    void run(int max_passes, double probe_work, double probe_seconds) {
        for (int j = 0; j < n; ++j) if (isint[j]) { l[j] = pceil(l[j]); u[j] = pfloor(u[j]); }
        row_on.assign(m, 1); col_on.assign(n, 1);
        pl = l; pu = u;
        x_fixed.assign(n, 0.0);
        wval.assign(n, 0.0); wflag.assign(n, 0);
        int probes_left = 2, parallel_left = 2;
        std::vector<double> mn, mn_inf, mx, mx_inf;
        for (int pass = 0; pass < max_passes; ++pass) {
            st[PASSES]++;
            bool changed = false;
            for (int j = 0; j < n; ++j) if (l[j] > u[j] + FEAS_TOL) { status = 1; return; }
            // fixed columns
            {
                std::vector<int> fc; std::vector<double> fv;
                for (int j = 0; j < n; ++j) if (col_on[j] && u[j] - l[j] <= FEAS_TOL) { fc.push_back(j); fv.push_back(l[j]); }
                if (!fc.empty()) { fix(fc, fv); st[FIXED] += (long)fc.size(); changed = true; }
            }
            build_S();
            int ms = (int)rows.size(), ns = (int)cols.size();
            std::vector<int> cnt(ms);
            for (int r = 0; r < ms; ++r) cnt[r] = sp[r + 1] - sp[r];
            // empty rows
            {
                bool any = false;
                for (int r = 0; r < ms; ++r) {
                    if (cnt[r]) continue;
                    int i = rows[r];
                    if (lo[i] > FEAS_TOL || hi[i] < -FEAS_TOL) { status = 1; return; }
                }
                for (int r = 0; r < ms; ++r) if (!cnt[r]) { row_on[rows[r]] = 0; st[EMPTY_ROWS]++; any = true; }
                if (any) changed = true;
            }
            // singleton rows -> column bounds
            {
                bool any = false;
                for (int r = 0; r < ms; ++r) {
                    if (cnt[r] != 1) continue;
                    any = true;
                    int i = rows[r], j = cols[si[sp[r]]];
                    double a = sv[sp[r]], b1 = lo[i] / a, b2 = hi[i] / a;
                    double nl = a > 0 ? b1 : b2, nu = a > 0 ? b2 : b1;
                    if (isint[j]) { nl = pceil(nl); nu = pfloor(nu); }
                    l[j] = std::max(l[j], nl); u[j] = std::min(u[j], nu);
                }
                if (any) {
                    for (int r = 0; r < ms; ++r) if (cnt[r] == 1) { row_on[rows[r]] = 0; st[SINGLETON_ROWS]++; }
                    continue;
                }
            }
            // activity bounds: redundant, infeasible and forcing rows
            std::vector<double> Sl(ns), Su(ns), rlo(ms), rhi(ms), tol(ms), rmn(ms), rmx(ms);
            for (int k = 0; k < ns; ++k) { Sl[k] = l[cols[k]]; Su[k] = u[cols[k]]; }
            activity(Sl, Su, mn, mn_inf, mx, mx_inf);
            std::vector<char> red(ms, 0);
            bool any_red = false;
            for (int r = 0; r < ms; ++r) {
                rmn[r] = mn_inf[r] > 0 ? -INF : mn[r];
                rmx[r] = mx_inf[r] > 0 ? INF : mx[r];
                rlo[r] = lo[rows[r]]; rhi[r] = hi[rows[r]];
                tol[r] = FEAS_TOL * (1.0 + std::fabs(fin0(rlo[r])) + std::fabs(fin0(rhi[r])));
                if (rmn[r] > rhi[r] + 1e3 * tol[r] || rmx[r] < rlo[r] - 1e3 * tol[r]) { status = 1; return; }
            }
            for (int r = 0; r < ms; ++r) {
                red[r] = rmn[r] >= rlo[r] - tol[r] && rmx[r] <= rhi[r] + tol[r];
                if (red[r]) { row_on[rows[r]] = 0; st[REDUNDANT]++; any_red = true; }
            }
            if (any_red) changed = true;
            // forcing rows
            {
                std::vector<char> f_up(ms, 0), f_dn(ms, 0);
                bool any = false;
                for (int r = 0; r < ms; ++r) {
                    f_up[r] = !red[r] && std::isfinite(rmx[r]) && std::fabs(rmx[r] - rlo[r]) <= tol[r];
                    f_dn[r] = !red[r] && !f_up[r] && std::isfinite(rmn[r]) && std::fabs(rmn[r] - rhi[r]) <= tol[r];
                    any = any || f_up[r] || f_dn[r];
                }
                if (any) {
                    for (int pass2 = 0; pass2 < 2; ++pass2) {
                        const std::vector<char>& mask = pass2 == 0 ? f_up : f_dn;
                        bool at_max = pass2 == 0;
                        std::vector<int> js; std::vector<double> vals;
                        for (int r = 0; r < ms; ++r) {
                            if (!mask[r]) continue;
                            for (int t = sp[r]; t < sp[r + 1]; ++t) {
                                int j = cols[si[t]];
                                js.push_back(j); vals.push_back((sv[t] > 0) == at_max ? u[j] : l[j]);
                            }
                        }
                        std::vector<char> done(n, 0);           // np.unique(..., return_index): first occurrence
                        for (size_t q = 0; q < js.size(); ++q)
                            if (!done[js[q]]) { done[js[q]] = 1; l[js[q]] = u[js[q]] = vals[q]; }
                    }
                    for (int r = 0; r < ms; ++r) st[FORCING] += f_up[r] + f_dn[r];
                    continue;
                }
            }
            // domain propagation
            {
                bool anyint = false;
                for (int j : cols) if (isint[j]) { anyint = true; break; }
                if (anyint) {
                    long k = propagate(rlo, rhi);
                    if (k) { st[TIGHTENED] += k; continue; }
                }
            }
            // free column singletons
            std::vector<int> ccnt(ns, 0), cfirst(ns, -1), cfirst_row(ns, -1);
            for (int r = 0; r < ms; ++r)
                for (int t = sp[r]; t < sp[r + 1]; ++t) {
                    int k = si[t];
                    if (!ccnt[k]) { cfirst[k] = t; cfirst_row[k] = r; }
                    ccnt[k]++;
                }
            {
                std::vector<int> sel;                         // candidates t (local columns), first per row
                std::vector<char> row_taken(ms, 0);
                for (int kc = 0; kc < ns; ++kc) {
                    int k = cols[kc];
                    if (ccnt[kc] != 1 || isint[k]) continue;
                    int il = cfirst_row[kc], i = rows[il];
                    double a = sv[cfirst[kc]], lk = l[k], uk = u[k];
                    double cmin = a > 0 ? a * lk : a * uk, cmax = a > 0 ? a * uk : a * lk;
                    bool fmin = std::isfinite(cmin), fmax = std::isfinite(cmax);
                    double rest_min = (mn_inf[il] - !fmin == 0) ? mn[il] - (fmin ? cmin : 0.0) : -INF;
                    double rest_max = (mx_inf[il] - !fmax == 0) ? mx[il] - (fmax ? cmax : 0.0) : INF;
                    double b = lo[i];
                    double lo_imp = (b - rest_max) / a, hi_imp = (b - rest_min) / a;
                    if (a < 0) std::swap(lo_imp, hi_imp);
                    double tl = 1e-9 * (1.0 + std::fabs(b));
                    bool ok = std::isfinite(hi[i]) && row_on[i] && std::fabs(a) >= 1e-9 && lo_imp >= lk - tl &&
                              hi_imp <= uk + tl;
                    if (ok && !row_taken[il]) { row_taken[il] = 1; sel.push_back(kc); }
                }
                for (int kc : sel) {
                    int k = cols[kc], il = cfirst_row[kc], i = rows[il];
                    double a = sv[cfirst[kc]], b = lo[i];
                    Sub s{k, b, {}, {}, a};
                    for (int t = sp[il]; t < sp[il + 1]; ++t)
                        if (si[t] != kc) { s.cols.push_back(cols[si[t]]); s.coefs.push_back(sv[t]); }
                    if (c[k] != 0.0) {
                        for (size_t q = 0; q < s.cols.size(); ++q) c[s.cols[q]] -= c[k] * s.coefs[q] / a;
                        cnst += c[k] * b / a;
                    }
                    stack.push_back(std::move(s));
                    col_on[k] = 0; row_on[i] = 0; cols_dirty = true;
                }
                if (!sel.empty()) { st[FREE_SINGLETONS] += (long)sel.size(); continue; }
            }
            // doubleton equations
            {
                std::vector<int> ks, js, rows_done;
                std::vector<double> bs, ajs, aks;
                std::vector<char> used(n, 0);
                for (int r = 0; r < ms; ++r) {
                    int i = rows[r];
                    if (cnt[r] != 2 || !std::isfinite(hi[i])) continue;
                    int j1 = cols[si[sp[r]]], j2 = cols[si[sp[r] + 1]];
                    double v1 = sv[sp[r]], v2 = sv[sp[r] + 1];
                    if (used[j1] || used[j2] || v1 == 0.0 || v2 == 0.0) continue;
                    // x_k: a continuous column, the one with the larger coefficient (the first of equals)
                    int k, j; double ak, aj;
                    bool c2 = !isint[j2], c1 = !isint[j1];
                    if (!c1 && !c2) continue;
                    if (c2 && (!c1 || std::fabs(v2) >= std::fabs(v1))) { k = j2; ak = v2; j = j1; aj = v1; }
                    else { k = j1; ak = v1; j = j2; aj = v2; }
                    if (std::fabs(aj / ak) > 1e3) continue;
                    double b = lo[i];
                    double e1 = b - ak * u[k], e2 = b - ak * l[k];
                    double ylo = ak > 0 ? e1 : e2, yhi = ak > 0 ? e2 : e1;
                    double nl = aj > 0 ? ylo / aj : yhi / aj, nu = aj > 0 ? yhi / aj : ylo / aj;
                    if (std::isnan(nl)) nl = -INF;
                    if (std::isnan(nu)) nu = INF;
                    if (isint[j]) {
                        if (std::isfinite(nl)) nl = std::ceil(nl - INT_TOL * std::max(1.0, std::fabs(nl)));
                        if (std::isfinite(nu)) nu = std::floor(nu + INT_TOL * std::max(1.0, std::fabs(nu)));
                    }
                    nl = l[j] >= nl ? l[j] : nl;               // Python's max(L[j], nl) and min(U[j], nu)
                    nu = u[j] <= nu ? u[j] : nu;
                    if (std::isfinite(nl) && std::isfinite(nu) && nl - nu > FEAS_TOL * (1.0 + std::fabs(nl))) {
                        status = 1; return;
                    }
                    l[j] = nl; u[j] = nl >= nu ? nl : nu;
                    used[j] = used[k] = 1;
                    ks.push_back(k); js.push_back(j); bs.push_back(b); ajs.push_back(aj); aks.push_back(ak);
                    rows_done.push_back(i);
                }
                for (int i : rows_done) row_on[i] = 0;
                if (!ks.empty()) {
                    std::vector<double> t(ks.size()), b0(ks.size());
                    for (size_t q = 0; q < ks.size(); ++q) { t[q] = -ajs[q] / aks[q]; b0[q] = bs[q] / aks[q]; }
                    substitute(ks, js, t, b0);
                    for (size_t q = 0; q < ks.size(); ++q) {
                        col_on[ks[q]] = 0; cols_dirty = true;
                        stack.push_back({ks[q], bs[q], {js[q]}, {ajs[q]}, aks[q]});
                    }
                    st[DOUBLETONS] += (long)ks.size();
                    continue;
                }
            }
            // dual fixing
            {
                std::vector<double> down(ns, 0.0), up(ns, 0.0);
                for (int r = 0; r < ms; ++r) {
                    bool eqr = std::isfinite(hi[rows[r]]);
                    for (int t = sp[r]; t < sp[r + 1]; ++t) {
                        down[si[t]] += (eqr || sv[t] > 0) ? 1.0 : 0.0;
                        up[si[t]] += (eqr || sv[t] < 0) ? 1.0 : 0.0;
                    }
                }
                std::vector<int> jl, ju;
                for (int k = 0; k < ns; ++k) {
                    int j = cols[k];
                    bool at_l = c[j] >= 0 && down[k] == 0 && std::isfinite(l[j]);
                    bool at_u = !at_l && c[j] <= 0 && up[k] == 0 && std::isfinite(u[j]);
                    if (at_l) jl.push_back(j);
                    if (at_u) ju.push_back(j);
                }
                if (!jl.empty() || !ju.empty()) {
                    std::vector<int> fc(jl); fc.insert(fc.end(), ju.begin(), ju.end());
                    std::vector<double> fv;
                    for (int j : jl) fv.push_back(l[j]);
                    for (int j : ju) fv.push_back(u[j]);
                    fix(fc, fv);
                    st[DUAL_FIXED] += (long)fc.size();
                    continue;
                }
            }
            // empty columns: fix at the bound the cost prefers
            {
                std::vector<int> ec; std::vector<double> ev;
                for (int k = 0; k < ns; ++k) {
                    if (ccnt[k]) continue;
                    int j = cols[k];
                    double v = c[j] > 0 ? l[j] : c[j] < 0 ? u[j] : std::min(std::max(0.0, l[j]), u[j]);
                    ec.push_back(j); ev.push_back(v);
                }
                if (!ec.empty()) {
                    for (double v : ev) if (!std::isfinite(v)) { status = 2; return; }
                    fix(ec, ev); st[EMPTY_COLS] += (long)ec.size(); changed = true;
                }
            }
            // parallel rows
            long merged = 0;
            if (!changed && parallel_left) {
                for (auto& pr : parallel_rows()) {
                    int i = rows[std::get<0>(pr)], k = rows[std::get<1>(pr)];
                    double lam = std::get<2>(pr);
                    if (!(row_on[i] && row_on[k])) continue;
                    double k_lo = lam > 0 ? lo[k] / lam : hi[k] / lam, k_hi = lam > 0 ? hi[k] / lam : lo[k] / lam;
                    double new_lo = lo[i] >= k_lo ? lo[i] : k_lo, new_hi = hi[i] <= k_hi ? hi[i] : k_hi;
                    double tl = FEAS_TOL * (std::isfinite(new_lo) ? 1.0 + std::fabs(new_lo) : 1.0);
                    if (new_lo > new_hi + 1e3 * tl) { status = 1; return; }
                    if (std::isfinite(new_hi)) {
                        if (new_hi - new_lo > tl) continue;
                        double v = std::isfinite(hi[i]) ? lo[i] : (std::isfinite(hi[k]) ? lo[k] / lam : new_lo);
                        lo[i] = hi[i] = v;
                    } else {
                        lo[i] = new_lo;
                    }
                    row_on[k] = 0;
                    merged++;
                }
            }
            if (!changed && parallel_left) parallel_left--;
            if (merged) { st[PARALLEL] += merged; continue; }
            // probing
            bool anyint = false;
            for (int j : cols) if (isint[j]) { anyint = true; break; }
            if (!changed && probes_left && probe_work > 0 && anyint) {
                probes_left--;
                // S as CSC (rows sorted within each column), as core.probe hands it to sm_probe
                std::vector<int> cp(ns + 1, 0), ri(si.size());
                std::vector<double> cv(si.size());
                for (int k : si) cp[k + 1]++;
                for (int k = 0; k < ns; ++k) cp[k + 1] += cp[k];
                std::vector<int> pos(cp.begin(), cp.end() - 1);
                for (int r = 0; r < ms; ++r)
                    for (int t = sp[r]; t < sp[r + 1]; ++t) { int p = pos[si[t]]++; ri[p] = r; cv[p] = sv[t]; }
                std::vector<double> zero(ns, 0.0), L(ns), U(ns);
                std::vector<char> ints(ns);
                for (int k = 0; k < ns; ++k) { L[k] = l[cols[k]]; U[k] = u[cols[k]]; ints[k] = isint[cols[k]]; }
                sm::Solver s;
                s.M.n = ns; s.M.m = ms;
                s.M.cp = cp; s.M.ri = ri; s.M.cv = cv;
                s.M.c = zero; s.M.lb = L; s.M.ub = U;
                s.M.rlo = rlo; s.M.rhi = rhi;
                s.M.isint.assign(ints.begin(), ints.end());
                long fx = 0, tg = 0;
                std::vector<std::array<int, 3>> eqv;
                if (!s.probe(L, U, probe_work, probe_seconds, fx, tg, &eqv)) { status = 1; return; }
                if (fx || tg) {
                    for (int k = 0; k < ns; ++k) { l[cols[k]] = L[k]; u[cols[k]] = U[k]; }
                    st[PROBE_FIXED] += fx; st[PROBE_TIGHTENED] += tg; changed = true;
                    parallel_left = std::max(parallel_left, 1);
                }
                if (eqv.size() > (size_t)ns) eqv.resize(ns);
                if (!eqv.empty()) {
                    // x_k = x_r (sign 1) or 1 - x_r (sign -1), r found by following the chain k -> j -> ...
                    std::vector<int> keys;
                    std::vector<int> lj(n, -1), lsg(n, 0);
                    for (auto& e : eqv) {
                        int k = cols[e[0]];
                        if (lj[k] < 0) keys.push_back(k);
                        lj[k] = cols[e[1]]; lsg[k] = e[2];
                    }
                    std::vector<int> ks2, rs2; std::vector<double> sg2, b02;
                    for (int k : keys) {
                        int r = k, sg = 1;
                        for (size_t guard = 0; lj[r] >= 0 && guard <= keys.size(); ++guard) { sg *= lsg[r]; r = lj[r]; }
                        ks2.push_back(k); rs2.push_back(r); sg2.push_back((double)sg); b02.push_back(sg < 0 ? 1.0 : 0.0);
                    }
                    substitute(ks2, rs2, sg2, b02);
                    for (size_t q = 0; q < ks2.size(); ++q) {
                        int k = ks2[q], r = rs2[q];
                        if (sg2[q] > 0) { l[r] = std::max(l[r], l[k]); u[r] = std::min(u[r], u[k]); }
                        else { l[r] = std::max(l[r], 1.0 - u[k]); u[r] = std::min(u[r], 1.0 - l[k]); }
                        col_on[k] = 0; cols_dirty = true;
                        stack.push_back({k, sg2[q] < 0 ? 1.0 : 0.0, {r}, {sg2[q] < 0 ? 1.0 : -1.0}, 1.0});
                    }
                    st[PROBE_EQUIV] += (long)ks2.size(); changed = true;
                }
            }
            if (!changed) break;
        }
    }
};

}  // namespace ps

// Presolve (see ps::Presolver): the model rlo <= A x (<= rhi for the first n_eq rows, which are equalities),
// A as sorted CSR without duplicates or zeros. Returns a handle; sizes = {status (0 reduced, 1 infeasible,
// 2 unbounded), rows kept, columns kept, nonzeros kept, equality rows kept, substitutions, their total columns}.
SM_API void* sm_presolve(int m, int n, int n_eq, const int* indptr, const int* indices, const double* data,
                         const double* q, const double* c, const double* l, const double* u, const char* isint,
                         double obj_const, int max_passes, double probe_work, double probe_seconds, long* sizes) {
    auto* P = new ps::Presolver();
    P->m = m; P->n = n;
    P->ap.assign(indptr, indptr + m + 1);
    P->aj.assign(indices, indices + indptr[m]);
    P->ax.assign(data, data + indptr[m]);
    P->lo.assign(q, q + m);
    P->hi.assign(m, ps::INF);
    for (int i = 0; i < n_eq; ++i) P->hi[i] = q[i];
    P->c.assign(c, c + n); P->l.assign(l, l + n); P->u.assign(u, u + n);
    P->isint.assign(isint, isint + n);
    P->cnst = obj_const;
    P->run(max_passes, probe_work, probe_seconds);
    long kr = 0, kc = 0, nz = 0, ne = 0, sc = 0;
    if (P->status == 0) {
        for (int i = 0; i < m; ++i) if (P->row_on[i]) {
            ++kr; ne += std::isfinite(P->hi[i]);
            for (int e = P->ap[i]; e < P->ap[i + 1]; ++e) nz += P->col_on[P->aj[e]];
        }
        for (int j = 0; j < n; ++j) kc += P->col_on[j];
    }
    for (auto& s : P->stack) sc += (long)s.cols.size();
    sizes[0] = P->status; sizes[1] = kr; sizes[2] = kc; sizes[3] = nz; sizes[4] = ne;
    sizes[5] = (long)P->stack.size(); sizes[6] = sc;
    return P;
}

// The presolved model (status 0): kept rows (equalities first, original indices) and columns, the reduced CSR
// matrix and vectors, x_fixed (n), stats (ps::NSTAT), the objective constant and the substitution stack
// (k, b, a and the columns / coefficients of each, concatenated with offsets st_ptr).
SM_API void sm_presolve_get(void* handle, int* rows, int* cols, int* Kp, int* Ki, double* Kx, double* q, double* c,
                            double* l, double* u, char* isint, double* x_fixed, double* stats, double* cnst,
                            int* st_k, double* st_b, double* st_a, int* st_ptr, int* st_cols, double* st_coefs) {
    auto* P = static_cast<ps::Presolver*>(handle);
    bool ok = P->status == 0;                       // otherwise only x_fixed, stats, the constant and the stack
    std::vector<int> colloc(P->n, -1);
    int kc = 0;
    for (int j = 0; j < P->n && ok; ++j) if (P->col_on[j]) { colloc[j] = kc; cols[kc++] = j; }
    int r = 0, t = 0;
    Kp[0] = 0;
    for (int eq = 1; eq >= 0 && ok; --eq)
        for (int i = 0; i < P->m; ++i) {
            if (!P->row_on[i] || std::isfinite(P->hi[i]) != (bool)eq) continue;
            rows[r] = i; q[r] = P->lo[i];
            for (int e = P->ap[i]; e < P->ap[i + 1]; ++e) {
                int jl = colloc[P->aj[e]];
                if (jl >= 0) { Ki[t] = jl; Kx[t] = P->ax[e]; ++t; }
            }
            Kp[++r] = t;
        }
    for (int j = 0, k = 0; j < P->n; ++j) {
        x_fixed[j] = P->x_fixed[j];
        if (!ok || !P->col_on[j]) continue;
        c[k] = P->c[j]; l[k] = P->l[j]; u[k] = P->u[j]; isint[k] = P->isint[j]; ++k;
    }
    for (int s = 0; s < ps::NSTAT; ++s) stats[s] = (double)P->st[s];
    *cnst = P->cnst;
    int off = 0;
    st_ptr[0] = 0;
    for (size_t s = 0; s < P->stack.size(); ++s) {
        auto& e = P->stack[s];
        st_k[s] = e.k; st_b[s] = e.b; st_a[s] = e.a;
        for (size_t q2 = 0; q2 < e.cols.size(); ++q2) { st_cols[off] = e.cols[q2]; st_coefs[off] = e.coefs[q2]; ++off; }
        st_ptr[s + 1] = off;
    }
}

SM_API void sm_presolve_free(void* handle) { delete static_cast<ps::Presolver*>(handle); }

// Scaling of the GPU engine (samadhan/pdlp.py: _scale) in place on a CSR matrix: iters passes of Ruiz equilibration
// (row and column max norms), then one with the 1-norms. The same arithmetic in the same order as the numpy
// version, so the same result.
SM_API void sm_scale(int m, int n, const int* indptr, const int* indices, double* data, double* dr, double* dc,
                     int iters) {
    std::vector<double> rmax(m), cmax(n), rs(m), cs(n);
    for (int i = 0; i < m; ++i) dr[i] = 1.0;
    for (int j = 0; j < n; ++j) dc[j] = 1.0;
    auto apply = [&]() {
        for (int i = 0; i < m; ++i)
            for (int t = indptr[i]; t < indptr[i + 1]; ++t) { data[t] = data[t] * rs[i]; data[t] = data[t] * cs[indices[t]]; }
    };
    for (int it = 0; it < iters; ++it) {
        std::fill(rmax.begin(), rmax.end(), 0.0);
        std::fill(cmax.begin(), cmax.end(), 0.0);
        for (int i = 0; i < m; ++i)
            for (int t = indptr[i]; t < indptr[i + 1]; ++t) {
                double a = std::fabs(data[t]);
                rmax[i] = std::max(rmax[i], a);
                cmax[indices[t]] = std::max(cmax[indices[t]], a);
            }
        for (int i = 0; i < m; ++i) rs[i] = 1.0 / std::sqrt(rmax[i] > 0 ? rmax[i] : 1.0);
        for (int j = 0; j < n; ++j) cs[j] = 1.0 / std::sqrt(cmax[j] > 0 ? cmax[j] : 1.0);
        apply();
        for (int i = 0; i < m; ++i) dr[i] *= rs[i];
        for (int j = 0; j < n; ++j) dc[j] *= cs[j];
    }
    std::fill(rmax.begin(), rmax.end(), 0.0);       // now the 1-norms
    std::fill(cmax.begin(), cmax.end(), 0.0);
    for (int i = 0; i < m; ++i)
        for (int t = indptr[i]; t < indptr[i + 1]; ++t) {
            double a = std::fabs(data[t]);
            rmax[i] += a;
            cmax[indices[t]] += a;
        }
    for (int i = 0; i < m; ++i) rs[i] = 1.0 / std::sqrt(rmax[i] > 0 ? rmax[i] : 1.0);
    for (int j = 0; j < n; ++j) cs[j] = 1.0 / std::sqrt(cmax[j] > 0 ? cmax[j] : 1.0);
    apply();
    for (int i = 0; i < m; ++i) dr[i] *= rs[i];
    for (int j = 0; j < n; ++j) dc[j] *= cs[j];
}
