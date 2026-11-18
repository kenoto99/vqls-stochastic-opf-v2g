"""End-to-end cost and resource scaling with the number of fleets F and scenarios N."""
import numpy as np, pickle, time, pandas as pd
import grid as g, ipm, ref, vqls
from model import Model
SEED = 47273
rows = []
for F in (1, 2, 3, 4):
    for N in (10, 50, 200):
        fl = g.FLEETS[:F]
        sc = g.sample_scenarios(N, SEED, fl)
        t0 = time.perf_counter(); m = Model(fl, sc); t_build = time.perf_counter() - t0
        t0 = time.perf_counter(); x0, Y0, _ = ref.phase1(m); t_p1 = time.perf_counter() - t0
        B = ipm.Barrier(m)
        rec = []
        def solver(S, r, ctx):
            t1 = time.perf_counter(); x = np.linalg.solve(S, r); tl = time.perf_counter() - t1
            rec.append((S.copy(), r.copy(), tl))
            return x, {}
        tau0 = B.ncons / (1e-1 * abs(m.objective(x0, Y0)))
        log = []
        t0 = time.perf_counter()
        ipm.solve(m, x0, Y0, reduced_solver=solver, tau0=tau0, mu=20.0, gap_tol=1e-3, newton_tol=1e-7,
                  log=log, max_total_newton=12)
        t_ipm = time.perf_counter() - t0
        S, r, tl = rec[-1]
        nq = int(np.ceil(np.log2(len(r)))); L = 2 * nq
        V = vqls.VQLS(layers=L, seed=SEED, warm=False)
        t0 = time.perf_counter(); dx, inf = V.solve(S, r); t_v = time.perf_counter() - t0
        xe = np.linalg.solve(S, r)
        P = V.prepare(S, r)
        termsM = vqls.pauli_decompose(P['M'], tol=1e-10)
        grp = vqls.qwc_groups([t[0] for t in termsM])
        nsteps = len(log)
        tc = np.mean([l.get('t_cls', np.nan) for l in log]) if log else np.nan
        row = dict(F=F, N=N, nx=m.nx, ny=m.ny, n_full=m.nx + N * m.ny, n_ineq=B.ncons,
                   red_dim=len(r), qubits=nq, layers=L, npar=V.ans.npar, n_cnot=V.ans.n_cnot, depth=V.ans.depth,
                   n_pauli_M=len(termsM), n_groups=len(grp), cond_S=np.linalg.cond(S),
                   t_build=t_build, t_phase1=t_p1, t_newton_total=t_ipm / nsteps, t_lu=np.mean([x[2] for x in rec]),
                   t_vqls=t_v, vqls_nfev=inf['vqls_nfev'], vqls_err=np.linalg.norm(dx - xe) / np.linalg.norm(xe),
                   vqls_cost=inf['vqls_cost'])
        rows.append(row)
        print(row, flush=True)
pd.DataFrame(rows).to_csv('results/scaling.csv', index=False)
print('done')
