"""Power-flow-only evaluation ([38]-type operation): schedules applied without OPF recourse
(V0 = 1.00 pu, no reactive support, availability clipping only), nonlinear AC sweep."""
import numpy as np, pickle, pandas as pd, grid as g
FL = g.FLEETS[:2]; T = g.T
test = g.sample_scenarios(300, 47273 + 1, FL)
oos = pickle.load(open('results/base_oos.pkl', 'rb'))['oos']
pols = {k: v['x'] for k, v in oos.items()}
pols['No V2G (fleets idle)'] = np.zeros(2 * T)
rows = []
ecap = np.array([f['ecap'] for f in FL]); pmax = np.array([f['pmax'] for f in FL])
for name, x in pols.items():
    for v0 in (1.00, 1.03):
        P = test['p_load'].copy(); Q = test['q_load'].copy()
        u = np.zeros((test['n'], 2, T))
        for f in range(2):
            lim = test['avail'][:, f] * pmax[f]
            u[:, f] = np.clip(x[f * T:(f + 1) * T][None], -lim, lim)
            P[:, :, FL[f]['bus']] -= u[:, f]
        B = test['n'] * T
        v, s, loss, res, it = g.bfs(P.reshape(B, -1), Q.reshape(B, -1), np.full(B, v0))
        v = v.reshape(test['n'], T, -1)[..., 1:]
        de = np.where(u > 0, u / g.ETA, u * g.ETA)
        e = test['e0'][:, :, None] - np.cumsum(de, 2) - np.cumsum(test['drain'], 2)
        soc = e / ecap[None, :, None]
        psub = P.sum(-1) + loss.reshape(test['n'], T)
        util = (np.abs(s) / g.SMAX_LINE).max(1).reshape(test['n'], T)
        rows.append(dict(policy=name, v0=v0, vmin=v.min(), freq_under=float((v.min(-1) < 0.95).mean()),
                         max_under=float(max(0, 0.95 - v.min())), energy_cost=float(np.mean((psub * g.TARIFF).sum(1))),
                         loss_mwh=float(loss.sum() / test['n']), peak_mw=float(np.mean(psub.max(1))),
                         soc_min=float(soc.min()), soc_end_short=float(np.mean(np.maximum(0.5 - soc[:, :, -1], 0).max(1) > 1e-9)),
                         max_util=float(util.max()), freq_therm=float((util > 1).mean()), max_res=float(res.max())))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/pfonly.csv', index=False)
