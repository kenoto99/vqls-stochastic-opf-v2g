"""Impact of finite shots and NISQ noise on the complete OPF when VQLS runs inside the IPM.
Training of theta is emulated noise-free; read-out (energy <A^2>, overlap, amplitudes) is shot
and/or noise limited.  N = 50 base case."""
import numpy as np, pickle, time, pandas as pd, grid as g, core, vqls
SEED = 47273; FL = g.FLEETS[:2]
sc = g.sample_scenarios(50, SEED, FL)
cases = [('shots 1e5', dict(mode='shots', shots=10 ** 5)),
         ('shots 1e6', dict(mode='shots', shots=10 ** 6)),
         ('noisy p2=1e-3, ro=1%, 1e6', dict(mode='noisy', shots=10 ** 6, p1=1e-4, p2=1e-3, p_ro=0.01, zne=False)),
         ('noisy p2=1e-3, ro=1%, 1e6, ZNE', dict(mode='noisy', shots=10 ** 6, p1=1e-4, p2=1e-3, p_ro=0.01, zne=True)),
         ('noisy p2=1e-2, ro=1%, 1e6', dict(mode='noisy', shots=10 ** 6, p1=1e-3, p2=1e-2, p_ro=0.01, zne=False)),
         ('noisy p2=1e-2, ro=1%, 1e6, ZNE', dict(mode='noisy', shots=10 ** 6, p1=1e-3, p2=1e-2, p_ro=0.01, zne=True))]
out = {}
for name, kw in cases:
    V = vqls.VQLS(layers=12, seed=SEED, **kw)
    log = []
    t0 = time.perf_counter()
    try:
        r = core.solve_case(FL, sc, reduced_solver=lambda S, rr, ctx: V.solve(S, rr, ctx), log=log)
        ok = True
    except Exception as e:
        print(name, 'FAILED', repr(e), flush=True); ok = False; r = None
    wall = time.perf_counter() - t0
    d = pd.DataFrame(log)
    res = dict(ok=ok, wall=wall, newton=len(d), fallback=int(d.fallback.sum()) if len(d) else None,
               err_med=float(d.rel_err.median()) if len(d) else None, err_p95=float(d.rel_err.quantile(.95)) if len(d) else None,
               err_max=float(d.rel_err.max()) if len(d) else None)
    if ok:
        res.update(obj=r['hist'][-1]['obj'], x=r['x'], viol=core.violations(r['ac']), hist=r['hist'])
    out[name] = dict(res=res, log=log)
    print(name, {k: v for k, v in res.items() if k not in ('x', 'hist')}, flush=True)
    pickle.dump(out, open('results/noise_loop.pkl', 'wb'))
print('done')
