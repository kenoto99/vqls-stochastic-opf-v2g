import pickle, numpy as np, pandas as pd
from figstyle import *
import grid as g
T = g.T; H = np.arange(1, T + 1)
d = pickle.load(open('results/base_runs.pkl', 'rb')); runs = d['runs']; sc = d['sc']
oos = pickle.load(open('results/base_oos.pkl', 'rb'))['oos']
FL = g.FLEETS[:2]

def band(ax, x, Y, col, label, ls='-', q=(5, 95)):
    lo, hi = np.percentile(Y, q, axis=0)
    ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
    ax.plot(x, Y.mean(0), color=col, ls=ls, label=label)

# ---------- Fig. 3 scenario inputs
fig, ax = plt.subplots(1, 3, figsize=(TWO, 1.9))
L = g.LOAD_PROFILE[None] * sc['load_mult']
band(ax[0], H, L, C[0], 'Mean')
ax[0].plot(H, g.LOAD_PROFILE, 'k:', lw=0.8, label='Nominal profile')
ax[0].set(xlabel='Hour', ylabel='Load factor (p.u. of peak)', xlim=(1, 24)); ax[0].legend(loc='lower right')
band(ax[1], H, sc['pv'], C[4], 'Mean')
ax[1].set(xlabel='Hour', ylabel='PV output (MW)', xlim=(1, 24))
band(ax[2], H, sc['avail'][:, 0], C[0], 'F18 (residential)')
band(ax[2], H, sc['avail'][:, 1], C[1], 'F33 (workplace)', ls='--')
ax[2].set(xlabel='Hour', ylabel='Fleet availability $\\xi$', xlim=(1, 24), ylim=(0, 1.05)); ax[2].legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='none')
for k, a in enumerate(ax): a.text(0.02, 0.96, f'({"abc"[k]})', transform=a.transAxes, va='top')
fig.tight_layout(w_pad=1.0); save(fig, 'fig03_scenarios')

# ---------- Fig. 4 schedule, SoC, V0
x = runs['VQLS']['x'].reshape(2, T); xl = runs['LU']['x'].reshape(2, T)
arr = oos['Proposed (VQLS-IPM)']['arrays']
fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.0))
w = 0.4
ax[0].bar(H - w / 2, x[0] * 1000, w, color=C[0], label='F18')
ax[0].bar(H + w / 2, x[1] * 1000, w, color=C[1], hatch='////', edgecolor='w', lw=0, label='F33')
ax[0].axhline(0, color='k', lw=0.5)
ax[0].set(xlabel='Hour', ylabel='Scheduled $p_{f,t}$ (kW)', xlim=(0.4, 24.6), ylim=(1000 * x.min() - 40, 1000 * x.max() + 160)); ax[0].legend(loc='upper left', ncol=2, fontsize=6.3)
soc = np.array([a['soc'] for a in arr])  # (n,F,T+1)
hh = np.arange(0, T + 1)
band(ax[1], hh, 100 * soc[:, 0], C[0], 'F18')
band(ax[1], hh, 100 * soc[:, 1], C[1], 'F33', ls='--')
ax[1].axhline(20, color='k', ls=':', lw=0.8); ax[1].axhline(90, color='k', ls=':', lw=0.8)
ax[1].axhline(50, color='#777777', ls='-.', lw=0.6)
ax[1].set(xlabel='Hour', ylabel='SoC (%)', xlim=(0, 24), ylim=(10, 100)); ax[1].legend(loc='lower left', bbox_to_anchor=(0.28, 0.25), ncol=2)
v0 = np.array([a['v0'] for a in arr])
band(ax[2], H, v0, C[2], 'OLTC set-point $V_0$')
qq = np.array([a['q'] for a in arr])
ax2 = ax[2].twinx()
ax2.plot(H, 1000 * qq[:, :, 0].mean(0), color=C[0], ls='--', label='$q$ F18')
ax2.plot(H, 1000 * qq[:, :, 1].mean(0), color=C[1], ls='-.', label='$q$ F33')
ax2.set_ylabel('Mean reactive support (kvar)'); ax2.tick_params(direction='in')
ax[2].set(xlabel='Hour', ylabel='$V_0$ (p.u.)', xlim=(1, 24), ylim=(1.015, 1.05)); ax2.set_ylim(-150, 420)
h1, l1 = ax[2].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax[2].legend(h1 + h2, l1 + l2, loc='lower left', ncol=1, fontsize=6.3)
for k, a in enumerate(ax): a.text(0.97, 0.96, f'({"abc"[k]})', transform=a.transAxes, ha='right', va='top')
fig.tight_layout(w_pad=0.8); save(fig, 'fig04_schedule')

# ---------- Fig. 5 voltages and feeder demand (out-of-sample)
test = g.sample_scenarios(300, 47273 + 1, FL)
def pf_only(xv, v0=1.03):
    P = test['p_load'].copy(); Q = test['q_load'].copy()
    pm = np.array([f['pmax'] for f in FL])
    for f in range(2):
        lim = test['avail'][:, f] * pm[f]
        P[:, :, FL[f]['bus']] -= np.clip(xv[f * T:(f + 1) * T][None], -lim, lim)
    B = test['n'] * T
    v, s, loss, res, it = g.bfs(P.reshape(B, -1), Q.reshape(B, -1), np.full(B, v0))
    return v.reshape(test['n'], T, -1)[..., 1:].min(-1), (P.sum(-1) + loss.reshape(test['n'], T)), loss.reshape(test['n'], T)
fig, ax = plt.subplots(1, 3, figsize=(TWO, 2.15))
vm = np.array([a['vmin_t'] for a in arr])
band(ax[0], H, vm, C[0], 'Two-stage OPF (with recourse)')
for k, (nm, xv) in enumerate([('Proposed schedule, PF only', oos['Proposed (VQLS-IPM)']['x']),
                              ('[38]-type, PF only', oos['[38]-type prescribed']['x']),
                              ('No V2G, PF only', np.zeros(2 * T))]):
    v_, ps, ls_ = pf_only(xv)
    ax[0].plot(H, np.percentile(v_, 5, axis=0), color=C[k + 1], ls=LS[k + 1], label=nm)
ax[0].axhline(0.95, color='k', ls=':', lw=0.8)
ax[0].set(xlabel='Hour', ylabel='Minimum bus voltage (p.u.)', xlim=(1, 24), ylim=(0.92, 0.995)); ax[0].legend(loc='lower left', fontsize=6.0)
v_, ps0, ls0 = pf_only(np.zeros(2 * T))
keys = [('Proposed', 'Proposed (VQLS-IPM)'), ('[38]-type', '[38]-type prescribed'), ('Uncoordinated', 'Uncoordinated charging')]
for k, (nm, key) in enumerate(keys):
    a2 = oos[key]['arrays']
    ps = np.array([a['psub'] for a in a2]); ls_ = np.array([a['loss'] for a in a2])
    ax[1].plot(H, ps.mean(0), color=C[k], ls=LS[k], marker=MK[k], ms=2.2, label=nm)
    ax[2].plot(H, 1000 * ls_.mean(0), color=C[k], ls=LS[k], marker=MK[k], ms=2.2, label=nm)
ax[1].plot(H, ps0.mean(0), color='#666666', ls=':', label='No V2G (PF only)')
ax[2].plot(H, 1000 * ls0.mean(0), color='#666666', ls=':', label='No V2G (PF only)')
ax[1].set(xlabel='Hour', ylabel='Mean substation demand (MW)', xlim=(1, 24), ylim=(1.5, 4.2)); ax[1].legend(loc='upper left', fontsize=6.0)
ax[2].set(xlabel='Hour', ylabel='Mean active loss (kW)', xlim=(1, 24), ylim=(0, 230)); ax[2].legend(loc='upper left', fontsize=6.0)
for k, a in enumerate(ax): a.text(0.97, 0.04, f'({"abc"[k]})', transform=a.transAxes, ha='right')
fig.tight_layout(w_pad=0.8); save(fig, 'fig05_voltage_demand')

# ---------- Fig. 6 IPM convergence with LU and VQLS
lu = pd.DataFrame(runs['LU']['log']); vq = pd.DataFrame(runs['VQLS']['log'])
fig, ax = plt.subplots(1, 2, figsize=(TWO, 2.0))
fstar = runs['CLARABEL']['obj']
ax[0].semilogy(lu.index + 1, np.maximum(lu.lam2 / 2, 1e-14), color=C[0], label='LU')
ax[0].semilogy(vq.index + 1, np.maximum(vq.lam2 / 2, 1e-14), color=C[1], ls='--', label='VQLS')
for o in (1, 2):
    nb = (lu.outer < o).sum(); ax[0].axvline(nb + 0.5, color='k', lw=0.5, ls=':')
ax[0].set(xlabel='Cumulative Newton step', ylabel='Newton decrement $\\lambda^2/2$', ylim=(1e-14, 1e15)); ax[0].legend(loc='upper center', ncol=2, fontsize=6.3, frameon=True, framealpha=1, edgecolor='none')
ax[1].semilogy(vq.index + 1, vq.rel_err, color=C[1], marker='.', ms=2, lw=0.5, label='Relative error of $\\Delta\\mathbf{x}$ (VQLS vs. LU)')
ax[1].semilogy(vq.index + 1, vq.vqls_cost, color=C[2], ls=':', lw=0.7, label='VQLS cost $C_G(\\boldsymbol{\\theta}^*)$')
ax[1].set(xlabel='Cumulative Newton step', ylabel='Error / cost', ylim=(1e-13, 1e2)); ax[1].legend(loc='upper center', frameon=True, framealpha=1, edgecolor='none', ncol=1)
for k, a in enumerate(ax): a.text(0.02, 0.04, f'({"ab"[k]})', transform=a.transAxes)
fig.tight_layout(w_pad=1.0); save(fig, 'fig06_convergence')
print('ok')
