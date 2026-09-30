"""Netlib LP benchmark: SAMADHAN (own MPS reader + PDLP) vs HiGHS reading the same file itself.

Runs each model in a worker process (1 torch thread each) so several small models solve in parallel.
Usage: python bench_netlib.py [--device cpu] [--tol 1e-4] [--time-limit 60] [--workers 6]
"""
import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_one(path, device, tol, time_limit):
    import torch
    torch.set_num_threads(1)
    import highspy
    from samadhan.mps import read_mps
    from samadhan.pdlp import solve
    from samadhan.baseline import solve_highs

    name = os.path.basename(path).split(".")[0]
    out = dict(name=name)
    try:
        # HiGHS reads the raw file itself: independent check of our parser
        import gzip, shutil, tempfile
        tmp = os.path.join(tempfile.gettempdir(), f"samadhan_{name}.mps")
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("threads", 1)
        h.readModel(tmp)
        t0 = time.perf_counter(); h.run(); out["highs_time"] = time.perf_counter() - t0
        out["highs_status"] = h.modelStatusToString(h.getModelStatus())
        out["ref_obj"] = h.getInfo().objective_function_value

        lp = read_mps(path)
        m, n = lp.K.shape
        out.update(n=n, m=m, nnz=int(lp.K.nnz))
        hp = solve_highs(lp, threads=1)  # HiGHS on OUR parsed model
        out["parser_obj_diff"] = abs(hp["obj"] - out["ref_obj"]) / (1 + abs(out["ref_obj"]))

        r = solve(lp, device=device, tol=tol, time_limit=time_limit, max_iter=2_000_000)
        out.update(status=r.status, obj=r.primal_obj, iters=r.iterations, time=r.solve_time,
                   rel_gap=r.rel_gap, rel_p=r.rel_primal_res, rel_d=r.rel_dual_res,
                   obj_rel_diff=abs(r.primal_obj - out["ref_obj"]) / (1 + abs(out["ref_obj"])))
    except Exception as e:  # keep going, report the failure
        out["error"] = repr(e)[:300]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    files = sorted(glob.glob("data/netlib/*.mps.gz"))
    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_one, f, a.device, a.tol, a.time_limit): f for f in files}
        for fu in as_completed(futs):
            r = fu.result()
            results.append(r)
            print(f"{r['name']:10} {r.get('status', 'ERR'):>15}  n={r.get('n', 0):6}  "
                  f"objdiff={r.get('obj_rel_diff', float('nan')):.1e}  parser={r.get('parser_obj_diff', float('nan')):.0e}  "
                  f"t={r.get('time', 0):6.1f}s  {r.get('error', '')}", flush=True)
            json.dump(sorted(results, key=lambda x: x["name"]),
                      open(f"results/netlib_{a.device}_tol{a.tol:g}.json", "w"), indent=1)
    ok = [r for r in results if r.get("status") == "optimal"]
    print(f"\nsolved {len(ok)}/{len(results)} to relative KKT {a.tol:g}")


if __name__ == "__main__":
    main()
