"""Extension cases: sub-hourly 33-bus (T=48) and IEEE 69-bus (T=24); LU and VQLS-in-the-loop,
Clarabel check, AC validation and out-of-sample evaluation."""
import sys, time, pickle, numpy as np, pandas as pd
import grid as g, core, ipm, ref, vqls
net, steps, L = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tag = f'{net}bus_T{steps}'
g.configure(net, steps)
FL = g.FLEETS[:2]
sc = g.sample_scenarios(50, 47273, FL)
out = dict(net=net, steps=steps, fleets=FL)
for name in ('LU', 'VQLS'):
    if name == 'LU':
        solver = ipm.lu_solver
    else:
        V = vqls.VQLS(layers=L, seed=47273)
        solver = lambda S, r, ctx: V.solve(S, r, ctx)
    log = []; t0 = time.perf_counter()
    r = core.solve_case(FL, sc, reduced_solver=solver, log=log)
    wall = time.perf_counter() - t0
    m = r['model']
    out[name] = dict(x=r['x'], hist=r['hist'], log=log, wall=wall, breakdown=m.breakdown(r['x'], r['Y']),
                     viol=core.violations(r['ac']), v0=r['ac']['V0'], vmin_t=r['ac']['v'][..., 1:].min(-1))
    if name == 'VQLS':
        out[name]['records'] = V.records
    print(tag, name, r['hist'][-1]['obj'], len(log), round(wall), out[name]['viol'], flush=True)
    if name == 'LU':
        t0 = time.perf_counter(); xr, Yr, pr = ref.reference(m)
        out['CLARABEL'] = dict(obj=m.objective(xr, Yr), status=pr.status, wall=time.perf_counter() - t0)
        print(tag, 'CLARABEL', out['CLARABEL'], flush=True)
    pickle.dump(out, open(f'results/ext_{tag}.pkl', 'wb'))
test = g.sample_scenarios(300, 47274, FL)
rc = core.Recourse(FL)
rows, Yt, arr = rc.evaluate(out['VQLS']['x'], test)
d = pd.DataFrame(rows)
soc = np.array([a['soc'] for a in arr]); ps = np.array([a['psub'] for a in arr])
out['OOS'] = dict(obj=d.obj.mean(), loss=d.loss_mwh.mean(), peak=ps.max(1).mean(), vmin=d.vmin.min(), vmax=d.vmax.max(),
                  util=d.max_util.max(), socmin=soc.min(), soft=d.soft.max(), res=d.max_res.max())
print(tag, 'OOS', out['OOS'], flush=True)
pickle.dump(out, open(f'results/ext_{tag}.pkl', 'wb'))
print('done')
