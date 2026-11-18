import numpy as np, pandas as pd, time, grid as g, ipm, ref, vqls
from model import Model
rows = []
for F in (3, 4):
    fl = g.FLEETS[:F]; sc = g.sample_scenarios(50, 47273, fl); m = Model(fl, sc)
    x0, Y0, _ = ref.phase1(m); B = ipm.Barrier(m); rec = []
    def solver(S, r, ctx):
        rec.append((S.copy(), r.copy())); return np.linalg.solve(S, r), {}
    ipm.solve(m, x0, Y0, reduced_solver=solver, tau0=B.ncons / (0.1 * abs(m.objective(x0, Y0))), mu=20.0,
              gap_tol=1e-3, newton_tol=1e-7, max_total_newton=12)
    S, r = rec[-1]; xe = np.linalg.solve(S, r)
    for L in (14, 21, 28):
        errs = []; ts = []
        for s in range(3):
            V = vqls.VQLS(layers=L, seed=500 + s, warm=False, maxiter=5000)
            t0 = time.perf_counter(); dx, inf = V.solve(S, r); ts.append(time.perf_counter() - t0)
            errs.append(np.linalg.norm(dx - xe) / np.linalg.norm(xe))
        rows.append(dict(F=F, L=L, npar=V.ans.npar, err_mean=np.mean(errs), err_max=np.max(errs), time=np.mean(ts)))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/depth7.csv', index=False)
