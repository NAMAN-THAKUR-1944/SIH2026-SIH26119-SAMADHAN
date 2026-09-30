"""Reference solve with HiGHS (open-source), used ONLY as the benchmark comparator."""
import time

import numpy as np
import highspy

from .lp import LP


def solve_highs(lp: LP, solver="choose", time_limit=600.0, threads=0):
    m, n = lp.K.shape
    A = lp.K.tocsc()
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", solver)
    h.setOptionValue("time_limit", float(time_limit))
    if threads:
        h.setOptionValue("threads", threads)
    model = highspy.HighsLp()
    model.num_col_, model.num_row_ = n, m
    model.col_cost_ = lp.c
    model.col_lower_ = np.where(np.isfinite(lp.l), lp.l, -highspy.kHighsInf)
    model.col_upper_ = np.where(np.isfinite(lp.u), lp.u, highspy.kHighsInf)
    lo = lp.q.copy()
    hi = np.full(m, highspy.kHighsInf)
    hi[:lp.n_eq] = lp.q[:lp.n_eq]
    model.row_lower_, model.row_upper_ = lo, hi
    model.offset_ = lp.obj_const
    model.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    model.a_matrix_.start_ = A.indptr
    model.a_matrix_.index_ = A.indices
    model.a_matrix_.value_ = A.data
    if lp.integer is not None and lp.integer.any():
        model.integrality_ = [highspy.HighsVarType.kInteger if f else highspy.HighsVarType.kContinuous
                              for f in lp.integer]
    h.passModel(model)
    t0 = time.perf_counter()
    h.run()
    elapsed = time.perf_counter() - t0
    status = h.modelStatusToString(h.getModelStatus())
    obj = h.getInfo().objective_function_value
    return dict(status=status, obj=obj, time=elapsed)
