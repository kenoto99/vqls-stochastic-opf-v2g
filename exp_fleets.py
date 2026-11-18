"""Sensitivity to fleet location and fleet type (33-bus, hourly, N=50, LU-IPM)."""
import pickle, numpy as np, pandas as pd, grid as g, core, ref
g.configure('33', 24)
base = {f['name']: f for f in g.FLEETS}
def fl(name, bus, kind, ecap, pmax, smax, n):
    return dict(name=name, bus=bus - 1, ecap=ecap, pmax=pmax, smax=smax, kind=kind, n_ev=n)
R18 = fl('F18', 18, 'residential', 3.0, .50, .55, 50); W33 = fl('F33', 33, 'workplace', 2.4, .40, .44, 40)
cases = {
    'A: F18 residential, F33 workplace (base)': [R18, W33],
    'B: F18 workplace, F33 residential': [fl('F18', 18, 'workplace', 3.0, .50, .55, 50), fl('F33', 33, 'residential', 2.4, .40, .44, 40)],
    'C: both residential': [R18, fl('F33', 33, 'residential', 2.4, .40, .44, 40)],
    'D: both workplace': [fl('F18', 18, 'workplace', 3.0, .50, .55, 50), W33],
    'E: F18 residential, F25 workplace': [R18, fl('F25', 25, 'workplace', 2.4, .40, .44, 40)],
    'F: F18 residential, F6 workplace': [R18, fl('F6', 6, 'workplace', 2.4, .40, .44, 40)],
    'G: single fleet at Bus 18 (5.4 MWh, 0.9 MW)': [fl('F18', 18, 'residential', 5.4, .90, .99, 90)],
}
rows = []; sched = {}
for name, F in cases.items():
    sc = g.sample_scenarios(50, 47273, F)
    r = core.solve_case(F, sc); m = r['model']; bd = m.breakdown(r['x'], r['Y'])
    xr, Yr, pr = ref.reference(m)
    ac = r['ac']; vi = core.violations(ac)
    psub = ac['P'].sum(-1) + ac['loss']
    rows.append(dict(case=name, obj=r['hist'][-1]['obj'], rel_clarabel=abs(r['hist'][-1]['obj'] - m.objective(xr, Yr)) / m.objective(xr, Yr),
                     energy=bd['energy'], deg=bd['deg'], thr=bd['thr_mwh'], loss=bd['loss_mwh'], peak=float(psub.max(1).mean()),
                     vmin=vi['vmin'], v0=float(ac['V0'].mean()), util=vi['max_util'], viol=vi['freq_v'] + vi['freq_therm'],
                     qmean=float(np.abs([m.Qf @ np.concatenate([r['x'], y]) for y in r['Y']]).mean())))
    sched[name] = r['x'].reshape(len(F), g.T)
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/fleets.csv', index=False)
pickle.dump(sched, open('results/fleets_sched.pkl', 'wb'))
print('done')
