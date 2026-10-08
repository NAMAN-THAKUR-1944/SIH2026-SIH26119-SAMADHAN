"""From-scratch reader for free- and fixed-format MPS files (Netlib, MIPLIB, Mittelmann, the MPS writers of Pyomo and
other modelling tools). OBJSENSE MAX is read as the minimisation of the negated objective (LP.sense = -1)."""
import gzip

import numpy as np
import scipy.sparse as sp

from .lp import LP


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")


class _NeedFixed(Exception):
    pass


NO_VALUE_BOUNDS = ("FR", "MI", "PL", "BV")


def _fixed_fields(line):
    """Fixed-format MPS columns: 2-3, 5-12, 15-22, 25-36, 40-47, 50-61 (names may contain spaces)."""
    return [line[1:3].strip(), line[4:12].strip(), line[14:22].strip(), line[24:36].strip(),
            line[39:47].strip(), line[49:61].strip()]


def _records(path, fixed):
    """Yield (section, record) with records normalised to the same shape in both formats."""
    section = None
    with _open(path) as f:
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line.strip() or line[0] == "*":
                continue
            if line[0] not in " \t":
                tok = line.split()
                section = tok[0].upper()
                yield "HEADER", tok
                if section == "ENDATA":
                    return
                continue
            if "'MARKER'" in line.upper():
                yield "MARKER", "'INTORG'" in line.upper()
                continue
            if fixed and section == "OBJSENSE":
                yield section, line.split()
                continue
            if fixed:
                f1, f2, f3, f4, f5, f6 = _fixed_fields(line.expandtabs(8))
                if section == "ROWS":
                    yield section, (f1.upper() or f2, f2)
                elif section in ("COLUMNS", "RHS", "RANGES"):
                    pairs = [(f3, f4)] + ([(f5, f6)] if f5 else [])
                    yield section, (f2, pairs)
                elif section == "BOUNDS":
                    yield section, (f1.upper(), f3, f4)
                continue
            tok = line.split()
            if section == "OBJSENSE":
                yield section, tok
                continue
            if section == "ROWS":
                if len(tok) != 2:
                    raise _NeedFixed
                yield section, (tok[0].upper(), tok[1])
            elif section == "COLUMNS":
                if len(tok) not in (3, 5):
                    raise _NeedFixed
                yield section, (tok[0], [(tok[k], tok[k + 1]) for k in range(1, len(tok), 2)])
            elif section in ("RHS", "RANGES"):
                if len(tok) not in (2, 3, 4, 5):
                    raise _NeedFixed
                body = tok[1:] if len(tok) % 2 == 1 else tok  # odd count = leading set name
                yield section, ("", [(body[k], body[k + 1]) for k in range(0, len(body), 2)])
            elif section == "BOUNDS":
                bt = tok[0].upper()
                if bt in NO_VALUE_BOUNDS:
                    if len(tok) not in (2, 3):
                        raise _NeedFixed
                    yield section, (bt, tok[-1], "0")
                else:
                    if len(tok) not in (3, 4):
                        raise _NeedFixed
                    yield section, (bt, tok[-2], tok[-1])  # set name optional


def read_mps(path):
    try:
        return _read(path, fixed=False)
    except (_NeedFixed, ValueError, KeyError):
        return _read(path, fixed=True)


def _read(path, fixed):
    rows = {}          # name -> type (N, E, L, G)
    row_order = []
    obj_row = None
    free_rows = set()  # extra N rows: free rows, ignored (only the first N row is the objective)
    cols = {}          # name -> index
    entries = []       # (row_name, col_idx, val)
    rhs = {}
    ranges = {}
    bounds = []        # (type, col_idx, val)
    integer = []
    in_int = False
    name = str(path)
    maximize = False

    for section, rec in _records(path, fixed):
        if section == "HEADER":
            if rec[0].upper() == "NAME" and len(rec) > 1:
                name = rec[1]
            elif rec[0].upper() == "OBJSENSE" and len(rec) > 1:     # OBJSENSE MAX on one line
                maximize = rec[1].upper().startswith("MAX")
        elif section == "OBJSENSE":
            maximize = rec[0].upper().startswith("MAX")
        elif section == "MARKER":
            in_int = rec
        elif section == "ROWS":
            t, r = rec
            if t == "N":
                if obj_row is None:
                    obj_row = r
                else:
                    free_rows.add(r)
                continue
            if t not in ("E", "L", "G"):
                raise ValueError(f"bad row type {t}")
            rows[r] = t
            row_order.append(r)
        elif section == "COLUMNS":
            cname, pairs = rec
            if cname not in cols:
                cols[cname] = len(cols)
                integer.append(in_int)
            j = cols[cname]
            for r, v in pairs:
                if r in free_rows:
                    continue
                if r != obj_row and r not in rows:
                    raise KeyError(r)
                entries.append((r, j, float(v)))
        elif section in ("RHS", "RANGES"):
            target = rhs if section == "RHS" else ranges
            for r, v in rec[1]:
                if r in free_rows:
                    continue
                if r != obj_row and r not in rows:
                    if not fixed:
                        raise KeyError(r)      # maybe a fixed-format file: retry that way
                    continue                    # unknown row: ignored, as HiGHS does
                target[r] = float(v)
        elif section == "BOUNDS":
            bt, cname, v = rec
            if cname not in cols:
                if not fixed:
                    raise KeyError(cname)
                continue
            bounds.append((bt, cols[cname], float(v) if v else 0.0))

    n = len(cols)
    c = np.zeros(n)
    l = np.zeros(n)
    u = np.full(n, np.inf)
    integer = np.array(integer, bool)
    # MPS convention (MIPLIB, HiGHS, CPLEX): an integer column from a MARKER block with no BOUNDS record at all is
    # binary; once any bound is given the others keep the usual defaults (0 and +inf)
    unbounded_int = integer.copy()
    for _, j, _ in bounds:
        unbounded_int[j] = False
    u[unbounded_int] = 1.0

    ridx = {r: i for i, r in enumerate(row_order)}
    rr, cc, vv = [], [], []
    for r, j, v in entries:
        if r == obj_row:
            c[j] += v
        elif r in ridx:
            rr.append(ridx[r]); cc.append(j); vv.append(v)
    A = sp.csr_matrix((vv, (rr, cc)), shape=(len(row_order), n))
    obj_const = -rhs.get(obj_row, 0.0) if obj_row else 0.0

    for bt, j, v in bounds:
        if bt == "UP":
            u[j] = v
            if v < 0 and l[j] == 0:
                l[j] = -np.inf
        elif bt == "LO": l[j] = v
        elif bt == "FX": l[j] = u[j] = v
        elif bt == "FR": l[j], u[j] = -np.inf, np.inf
        elif bt == "MI": l[j] = -np.inf
        elif bt == "PL": u[j] = np.inf
        elif bt == "BV": l[j], u[j] = 0.0, 1.0; integer[j] = True
        elif bt == "LI": l[j] = v; integer[j] = True
        elif bt == "UI": u[j] = v; integer[j] = True

    # Every row becomes row_lo <= a'x <= row_hi, then split into = and >= rows.
    eq_idx, eq_rhs, ge_idx, ge_sign, ge_rhs = [], [], [], [], []
    for r in row_order:
        i, t, b = ridx[r], rows[r], rhs.get(r, 0.0)
        R = ranges.get(r)
        if t == "E" and (R is None or R == 0):
            eq_idx.append(i); eq_rhs.append(b); continue
        if t == "E":
            lo, hi = (b, b + R) if R > 0 else (b + R, b)
        elif t == "G":
            lo, hi = b, (b + abs(R) if R is not None else np.inf)
        else:  # L
            lo, hi = (b - abs(R) if R is not None else -np.inf), b
        if np.isfinite(lo):
            ge_idx.append(i); ge_sign.append(1.0); ge_rhs.append(lo)
        if np.isfinite(hi):
            ge_idx.append(i); ge_sign.append(-1.0); ge_rhs.append(-hi)

    Keq = A[eq_idx]
    Kge = sp.diags(ge_sign) @ A[ge_idx] if ge_idx else sp.csr_matrix((0, n))
    K = sp.vstack([Keq, Kge]).tocsr()
    q = np.array(eq_rhs + ge_rhs, float)
    sense = -1 if maximize else 1
    return LP(sense * c, K, q, len(eq_idx), l, u, obj_const=sense * obj_const, name=name,
              col_names=list(cols), integer=integer if integer.any() else None, sense=sense)
