"""Read-out error mitigation: snapshot study and complete OPF with REM (+ZNE)."""
import pickle, time, numpy as np, pandas as pd, grid as g, core, vqls
d = pickle.load(open('results/snapshots.pkl', 'rb')); rec = d['rec']; nrec = len(rec)
picks = {'early': 3, 'middle': nrec // 2, 'late': nrec - 1}
rows = []
for tag, k in picks.items():
    S, r, tau, nw = rec[k]; xe = np.linalg.solve(S, r)
    V = vqls.VQLS(layers=12, seed=7, warm=False, restarts=2); V.solve(S, r); theta = V.theta.copy()
    for p2 in (0.0, 1e-4, 1e-3, 3e-3, 1e-2):
        for zne in (False, True):
            errs = []
            for rep in range(5):
                W = vqls.VQLS(layers=12, seed=4000 + rep, mode='noisy', shots=10 ** 6, p1=p2 / 10, p2=p2, p_ro=0.01, zne=zne, rem=True)
                W.ans = V.ans; W.theta = theta.copy(); W.maxiter = 0
                dx, inf = W.solve(S, r); errs.append(np.linalg.norm(dx - xe) / np.linalg.norm(xe))
            rows.append(dict(snapshot=tag, p2=p2, p1=p2 / 10, p_ro=0.01, zne=zne, rem=True, rel_err_mean=np.mean(errs), rel_err_sd=np.std(errs, ddof=1)))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/rem_snap.csv', index=False)
g.configure('33', 24)
FL = g.FLEETS[:2]; sc = g.sample_scenarios(50, 47273, FL)
out = {}
for name, kw in [('noisy p2=1e-3, ro=1%, 1e6, ZNE+REM', dict(p1=1e-4, p2=1e-3)), ('noisy p2=1e-2, ro=1%, 1e6, ZNE+REM', dict(p1=1e-3, p2=1e-2))]:
    V = vqls.VQLS(layers=12, seed=47273, mode='noisy', shots=10 ** 6, p_ro=0.01, zne=True, rem=True, **kw)
    log = []; t0 = time.perf_counter()
    r = core.solve_case(FL, sc, reduced_solver=lambda S, rr, ctx: V.solve(S, rr, ctx), log=log)
    dd = pd.DataFrame(log)
    out[name] = dict(res=dict(ok=True, wall=time.perf_counter() - t0, newton=len(dd), fallback=int(dd.fallback.sum()),
                              err_med=float(dd.rel_err.median()), err_p95=float(dd.rel_err.quantile(.95)), err_max=float(dd.rel_err.max()),
                              obj=r['hist'][-1]['obj'], viol=core.violations(r['ac'])), log=log)
    print(name, {k: v for k, v in out[name]['res'].items()}, flush=True)
    pickle.dump(out, open('results/rem_loop.pkl', 'wb'))
print('done')
