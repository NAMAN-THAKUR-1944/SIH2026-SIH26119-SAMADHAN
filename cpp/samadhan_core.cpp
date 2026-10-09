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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <queue>
#include <random>
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
    int features = 251;          // branch-and-cut: 1 feasibility pump, 2 diving, 4 cover cuts, 8 reliability branching,
                                 // 16 node domain propagation, 32 c-MIR cuts, 64 fix-and-propagate, 128 RINS
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
        wp.assign(N, 1.0);
        for (;;) {
            if (want_refactor() && !refactor_basis()) return NUMERIC;
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
        for (size_t q = 0; q < superbasic.size(); ++q) {
            int j = superbasic[q];
            // (refactorise while column j is still held at its value by its fixed bounds)
            if (want_refactor() && !refactor_basis()) return -1;
            double v = x[j];
            lb[j] = sb_lo[j]; ub[j] = sb_hi[j];
            if (Clock::now() > t_end) {                     // out of time: the rest goes to a bound
                place_nonbasic(j);
                continue;
            }
            col_alpha(j);
            double s0 = d[j] < -opt.tol_d ? 1.0 : d[j] > opt.tol_d ? -1.0 : (ub[j] - v <= v - lb[j] ? 1.0 : -1.0);
            bool done = false;
            for (int pass = 0; pass < 2 && !done; ++pass) {
                double s = pass == 0 ? s0 : -s0;
                double dist = s > 0 ? ub[j] - v : v - lb[j];
                // Harris two-pass ratio test over the basic variables (x_B moves by -t s alpha)
                double tmax = INF;
                for (int k = 0; k < m; ++k) {
                    double a = s * acol[k];
                    if (std::fabs(a) < opt.tol_piv) continue;
                    int jb = head[k];
                    double room = a > 0 ? x[jb] - lb[jb] : ub[jb] - x[jb];
                    if (std::isfinite(room)) tmax = std::min(tmax, (std::max(room, 0.0) + opt.tol_p) / std::fabs(a));
                }
                int r = -1; double t = 0.0, amax = 0.0;
                if (tmax < INF) {
                    for (int k = 0; k < m; ++k) {
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
                    for (int k = 0; k < m; ++k) x[head[k]] -= dist * s * acol[k];
                    st[j] = s > 0 ? AT_UB : AT_LB;
                    x[j] = s > 0 ? ub[j] : lb[j];
                    done = true;
                } else if (r >= 0) {                                         // a basic variable blocks: swap
                    int jl = head[r];
                    bool leave_lb = s * acol[r] > 0;
                    for (int k = 0; k < m; ++k) x[head[k]] -= t * s * acol[k];
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
                for (int k = 0; k < m; ++k) x[head[k]] -= dx * acol[k];
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
        long it_push = S.push_superbasics();
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
    // found nothing.
    double rins_last_inc = INF;
    double rins_wait = 0.0;
    bool rins_due() {
        double wait = inc_obj < rins_last_inc - 1e-9 ? 0.1 * opt.time_limit : rins_wait;
        return seconds_since(rins_last) > wait;
    }
    void rins(double seconds, long nodes, double bound) {
        if (rins_off || !std::isfinite(inc_obj) || rins_time > 0.3 * opt.time_limit || Clock::now() >= S.t_end) return;
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
        sub.inc = inc; sub.inc_obj = inc_obj;
        sub.warm_st = &S.st; sub.warm_head = &S.head;
        Info sinfo;
        std::vector<double> sx;
        sub.solve_scaled(sinfo, sx);
        ++rins_calls;
        if (sinfo.nodes < 2) rins_off = true;
        double before = inc_obj;
        if (sub.inc_obj < inc_obj - 1e-9) try_incumbent(sub.inc);
        rins_wait = inc_obj < before ? 0.1 * opt.time_limit : std::max(0.2 * opt.time_limit, 2 * rins_wait);
        rins_last_inc = inc_obj;
        rins_time += seconds_since(start);
        if (opt.verbose)
            std::printf("  RINS: %d of %d integer columns fixed, %.0f nodes, %.2f s: %s\n", agree, nint, sinfo.nodes,
                        seconds_since(start), inc_obj < before ? "better incumbent" : "nothing better");
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
        for (int round = 0; round < opt.cut_rounds && Clock::now() < cut_end; ++round) {
            rounding_heuristic();
            int m_before = M.m;
            LPState before = save_lp();
            int added = add_gomory_round();
            if (opt.features & 4) added += add_cover_round();
            if (opt.features & 32) added += add_mir_round();
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
            if (opt.verbose) std::printf("  cut round %d: +%d cuts, bound %.10g\n", round + 1, added, now);
            if (now - prev < 1e-4 * (1.0 + std::fabs(now))) { prev = now; break; }
            prev = now;
        }
        info.root_bound_cuts = S.objective();
        if (opt.features & 16) build_propagation();

        // root heuristics: feasibility pump and a dive; with an incumbent, reduced-cost fixing and a guided dive
        long dive_budget = 1000 + 2L * S.m;
        if ((opt.features & 1) && !std::isfinite(inc_obj)) feasibility_pump(100, 10000 + 20L * S.m);
        if (opt.features & 2) dive(dive_budget);
        // when the cheap heuristics found nothing (a first incumbent that is merely feasible would stop the pump)
        if ((opt.features & 64) && !std::isfinite(inc_obj)) fix_and_propagate(std::max(1000L, 2 * root_lp_iters));
        if (std::isfinite(inc_obj)) {
            int fixed = reduced_cost_fixing(S.objective());
            if (opt.verbose && fixed) std::printf("  reduced-cost fixing: %d bounds tightened\n", fixed);
            if (opt.features & 2) dive(dive_budget);
            if (opt.features & 128) rins(0.1 * opt.time_limit, 2000, S.objective());
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
                rins(0.08 * opt.time_limit, 2000, global_bound());
            if ((opt.features & 64) && !std::isfinite(inc_obj) && nodes % 250 == 60)
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
// opts: [time_limit, node_limit, cut_rounds, verbose, gap, features (see Options)]
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
