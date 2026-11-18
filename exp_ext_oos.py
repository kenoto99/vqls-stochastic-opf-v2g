"""Out-of-sample evaluation for the extension cases (300 test scenarios)."""
import sys, pickle, numpy as np, pandas as pd
import grid as g, core
net, steps = sys.argv[1], int(sys.argv[2]); tag = f'{net}bus_T{steps}'
g.configure(net, steps)
out = pickle.load(open(f'results/ext_{tag}.pkl', 'rb')); FL = out['fleets']
test = g.sample_scenarios(300, 47274, FL)
rc = core.Recourse(FL, const=True); rows, arr = [], []
for k in range(0, 300, 25):   # batches keep memory below the container limit at T=48
    sub = {kk: (vv[k:k + 25] if hasattr(vv, 'shape') and vv.shape[:1] == (300,) else vv) for kk, vv in test.items()} if isinstance(test, dict) else test[k:k + 25]
    sub['n'] = len(sub['pv']); r_, _, a_ = rc.evaluate(out['VQLS']['x'], sub); rows += r_; arr += a_
d = pd.DataFrame(rows)
soc = np.array([a['soc'] for a in arr]); ps = np.array([a['psub'] for a in arr])
out['OOS'] = dict(obj=d.obj.mean(), loss=d.loss_mwh.mean(), peak=ps.max(1).mean(), vmin=d.vmin.min(), vmax=d.vmax.max(),
                  util=d.max_util.max(), socmin=soc.min(), soft=d.soft.max(), res=d.max_res.max())
print(tag, 'OOS', out['OOS'], flush=True)
pickle.dump(out, open(f'results/ext_{tag}.pkl', 'wb')); print('done')
