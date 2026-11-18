"""Weight sensitivity: degradation price sweep (Pareto front) and voltage-deviation weight."""
import numpy as np, pickle, pandas as pd, grid as g, core, ref
SEED = 47273; FL = g.FLEETS[:2]
sc = g.sample_scenarios(50, SEED, FL)
rows = []
for cdeg in (0.0, 10.0, 20.0, 27.0, 40.0, 60.0, 100.0, 200.0):
    r = core.solve_case(FL, sc, c_deg=cdeg, inner='clarabel')
    m = r['model']; bd = m.breakdown(r['x'], r['Y'])
    relc = 0.0
    thr = float(bd['thr_mwh'])
    rows.append(dict(sweep='c_deg', value=cdeg, energy_cost=bd['energy'], throughput_mwh=thr,
                     deg_cost_at_27=27.0 * thr, deg_cost=bd['deg'], vdev=bd['vdev'], loss_mwh=bd['loss_mwh'], obj=r['hist'][-1]['obj'], rel_clarabel=relc,
                     vmin=r['hist'][-1]['vmin']))
    print(rows[-1], flush=True)
for wv in (0.0, 5.0, 50.0, 500.0):
    r = core.solve_case(FL, sc, w_v=wv, inner='clarabel')
    m = r['model']; bd = m.breakdown(r['x'], r['Y'])
    relc = 0.0
    thr = float(bd['thr_mwh'])
    v = r['ac']['v'][..., 1:]
    rows.append(dict(sweep='w_v', value=wv, energy_cost=bd['energy'], throughput_mwh=thr, deg_cost_at_27=27.0 * thr, deg_cost=bd['deg'],
                     vdev_pu2=float(np.mean(np.sum((v - 1) ** 2, axis=-1).sum(-1))), loss_mwh=bd['loss_mwh'],
                     obj=r['hist'][-1]['obj'], rel_clarabel=relc, vmin=r['hist'][-1]['vmin'], v0_mean=float(r['ac']['V0'].mean())))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/pareto.csv', index=False)
print('done')
