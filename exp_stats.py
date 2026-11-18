"""VQLS statistics on recorded Schur snapshots: layers, optimizers, random initialisations,
finite shots and gate/readout noise with and without ZNE."""
import numpy as np, pickle, time, pandas as pd, vqls
d = pickle.load(open('results/snapshots.pkl', 'rb'))
rec = d['rec']
nrec = len(rec)
picks = {'early': 3, 'middle': nrec // 2, 'late': nrec - 1}
rows = []; shot_rows = []; noise_rows = []; info_rows = []
for tag, k in picks.items():
    S, r, tau, nw = rec[k]
    xe = np.linalg.solve(S, r)
    wD = np.sqrt(np.diag(S))
    werr = lambda dx: np.linalg.norm(wD * (dx - xe)) / np.linalg.norm(wD * xe)
    dj = 1 / np.sqrt(np.diag(S)); Sh = S * dj[:, None] * dj[None]
    P = vqls.VQLS().prepare(S, r)
    terms = vqls.pauli_decompose(P['A']); termsM = vqls.pauli_decompose(P['M'])
    grp = vqls.qwc_groups([t[0] for t in termsM])
    info_rows.append(dict(snapshot=tag, newton=nw, tau=tau, cond_S=np.linalg.cond(S), cond_scaled=np.linalg.cond(P['A']),
                          n_pauli_A=len(terms), n_pauli_M=len(termsM), n_groups_M=len(grp)))
    print(info_rows[-1], flush=True)
    for L in (4, 8, 12, 16):
        for s in range(20):
            V = vqls.VQLS(layers=L, seed=1000 + s, warm=False)
            t0 = time.perf_counter(); dx, inf = V.solve(S, r); tt = time.perf_counter() - t0
            rows.append(dict(snapshot=tag, optimizer='L-BFGS-B (parameter-shift gradient)', layers=L, seed=s,
                             cost=inf['vqls_cost'], rel_err=np.linalg.norm(dx - xe) / np.linalg.norm(xe), wrel_err=werr(dx),
                             nfev=inf['vqls_nfev'], time=tt, npar=inf['npar'],
                             bound=np.linalg.cond(P['A']) * np.sqrt(max(inf['vqls_cost'], 0))))
        print(tag, L, pd.DataFrame(rows).query('snapshot==@tag and layers==@L').rel_err.describe()[['mean', 'max']].values, flush=True)
    for s in range(5):
        V = vqls.VQLS(layers=12, seed=2000 + s, warm=False, optimizer='COBYLA', maxiter=10000, tol=1e-10)
        t0 = time.perf_counter(); dx, inf = V.solve(S, r); tt = time.perf_counter() - t0
        rows.append(dict(snapshot=tag, optimizer='COBYLA', layers=12, seed=s, cost=inf['vqls_cost'],
                         rel_err=np.linalg.norm(dx - xe) / np.linalg.norm(xe), wrel_err=werr(dx), nfev=inf['vqls_nfev'], time=tt,
                         npar=inf['npar'], bound=np.linalg.cond(P['A']) * np.sqrt(max(inf['vqls_cost'], 0))))
    print(tag, 'COBYLA', pd.DataFrame(rows).query('snapshot==@tag and optimizer=="COBYLA"').rel_err.mean(), flush=True)
    # trained parameters (best of L=12) for read-out studies
    V = vqls.VQLS(layers=12, seed=7, warm=False, restarts=2)
    dx0, inf0 = V.solve(S, r)
    theta = V.theta.copy()
    for shots in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7):
        for rep in range(20):
            W = vqls.VQLS(layers=12, seed=3000 + rep, mode='shots', shots=shots)
            W.ans = V.ans; W.theta = theta.copy(); W.maxiter = 0
            dx, inf = W.solve(S, r)
            shot_rows.append(dict(snapshot=tag, shots=shots, rep=rep, rel_err=np.linalg.norm(dx - xe) / np.linalg.norm(xe)))
    print(tag, 'shots done', flush=True)
    for p2 in (0.0, 1e-4, 1e-3, 3e-3, 1e-2):
        for pro in (0.0, 0.01):
            for zne in (False, True):
                errs = []
                for rep in range(5):
                    W = vqls.VQLS(layers=12, seed=4000 + rep, mode='noisy', shots=10 ** 6, p1=p2 / 10, p2=p2, p_ro=pro, zne=zne)
                    W.ans = V.ans; W.theta = theta.copy(); W.maxiter = 0
                    dx, inf = W.solve(S, r)
                    errs.append(np.linalg.norm(dx - xe) / np.linalg.norm(xe))
                noise_rows.append(dict(snapshot=tag, p2=p2, p1=p2 / 10, p_ro=pro, zne=zne, rel_err_mean=np.mean(errs),
                                       rel_err_sd=np.std(errs, ddof=1)))
        print(tag, 'noise p2', p2, noise_rows[-1], flush=True)
pickle.dump(dict(rows=rows, shot_rows=shot_rows, noise_rows=noise_rows, info_rows=info_rows), open('results/vqls_stats.pkl', 'wb'))
print('done')
