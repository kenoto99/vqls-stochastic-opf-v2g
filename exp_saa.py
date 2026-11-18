"""SAA convergence: in-sample and out-of-sample objective vs number of scenarios (LU-IPM)."""
import numpy as np, pickle, time, pandas as pd, grid as g, core, ref
SEED = 47273; FL = g.FLEETS[:2]
test = g.sample_scenarios(300, SEED + 1, FL)
rc = core.Recourse(FL)
res = []
xs = {}
for N in (10, 25, 50, 100, 200):
    sc = g.sample_scenarios(N, SEED, FL)
    t0 = time.perf_counter(); r = core.solve_case(FL, sc); ts = time.perf_counter() - t0
    xs[N] = r['x']
    xr, Yr, pr = ref.reference(r['model']); fr = r['model'].objective(xr, Yr)
    rows, _, _ = rc.evaluate(r['x'], test)
    d = pd.DataFrame(rows)
    res.append(dict(N=N, in_sample=r['hist'][-1]['obj'], oos_mean=d.obj.mean(), oos_ci95=1.96 * d.obj.std(ddof=1) / np.sqrt(len(d)),
                    oos_soft=d.soft.max(), oos_vmin=d.vmin.min(), solve_time=ts, rel_clarabel=abs(r['hist'][-1]['obj'] - fr) / fr, newton=sum(h['newton'] for h in r['hist'])))
    print(res[-1], flush=True)
for N in xs:
    res[[q['N'] for q in res].index(N)]['dx_vs_200'] = float(np.abs(xs[N] - xs[200]).max())
pd.DataFrame(res).to_csv('results/saa.csv', index=False)
pickle.dump(xs, open('results/saa_x.pkl', 'wb'))
print('done')
