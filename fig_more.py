import pickle, sys, numpy as np, pandas as pd, os
from figstyle import *
which = sys.argv[1:] or ['stats', 'saa', 'pareto', 'scaling', 'noise']

if 'stats' in which and os.path.exists('results/vqls_stats.pkl'):
    d = pickle.load(open('results/vqls_stats.pkl', 'rb'))
    rows = pd.DataFrame(d['rows']); sh = pd.DataFrame(d['shot_rows']); nz = pd.DataFrame(d['noise_rows'])
    info = pd.DataFrame(d['info_rows']).set_index('snapshot')
    fig, ax = plt.subplots(1, 2, figsize=(TWO, 2.2))
    tags = ['early', 'middle', 'late']
    lb = rows[rows.optimizer != 'COBYLA']
    Ls = sorted(lb.layers.unique())
    for k, tg in enumerate(tags):
        sub = lb[lb.snapshot == tg]
        data = [sub[sub.layers == L].wrel_err.values for L in Ls]
        pos = np.arange(len(Ls)) + (k - 1) * 0.25
        bp = ax[0].boxplot(data, positions=pos, widths=0.2, patch_artist=True, showfliers=True,
                           flierprops=dict(marker=MK[k], ms=2, mec=C[k]), medianprops=dict(color='k', lw=0.8),
                           boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        for b in bp['boxes']:
            b.set_facecolor(C[k]); b.set_alpha(0.55)
        cob = rows[(rows.optimizer == 'COBYLA') & (rows.snapshot == tg)]
        ax[0].plot([len(Ls) + (k - 1) * 0.25] * len(cob), cob.wrel_err, MK[k], color=C[k], ms=3, mfc='none')
        ax[0].plot([], [], 's', color=C[k], alpha=0.6, label=f'{tg} step ($\\kappa(\\mathbf{{A}})$={info.loc[tg, "cond_scaled"]:.1f})')
    ax[0].set_yscale('log')
    ax[0].set_xticks(list(range(len(Ls))) + [len(Ls)])
    ax[0].set_xticklabels([f'L-BFGS-B\n$L$={L}' for L in Ls] + ['COBYLA\n$L$=12'], fontsize=6.5)
    ax[0].set_ylabel('Relative error of $\\Delta\\mathbf{x}$'); ax[0].legend(loc='lower left', fontsize=6.0)
    c = np.logspace(-14, 0, 50)
    for k, tg in enumerate(tags):
        sub = rows[rows.snapshot == tg]
        ax[1].loglog(sub.cost.clip(1e-16), sub.wrel_err, MK[k], color=C[k], ms=2.5, mfc='none', mew=0.6, label=f'{tg} step')
        ax[1].loglog(c, info.loc[tg, 'cond_scaled'] * np.sqrt(c), color=C[k], ls=LS[k + 1], lw=0.7)
    ax[1].plot([], [], 'k--', lw=0.7, label='Bound $\\kappa(\\mathbf{A})\\sqrt{C_G}$')
    ax[1].set(xlabel='VQLS cost $C_G$', ylabel='Relative error ($\\mathbf{D}^{-1}$-norm)', xlim=(1e-14, 1), ylim=(1e-8, 10))
    ax[1].legend(loc='upper left', fontsize=6.3)
    for k, a in enumerate(ax): a.text(0.98, 0.04, f'({"ab"[k]})', transform=a.transAxes, ha='right')
    fig.tight_layout(w_pad=1.0); save(fig, 'fig07_vqls_stats')
    # Fig 8: shots and noise
    fig, ax = plt.subplots(1, 2, figsize=(TWO, 2.1))
    for k, tg in enumerate(tags):
        g_ = sh[sh.snapshot == tg].groupby('shots').rel_err
        m_, s_ = g_.mean(), g_.std()
        ax[0].errorbar(m_.index, m_.values, yerr=s_.values, fmt=MK[k] + LS[0], color=C[k], ms=3, capsize=2, lw=0.8, label=f'{tg} step')
    xs = np.logspace(3, 7, 10); ax[0].loglog(xs, 3 * xs ** -0.5, 'k:', lw=0.7, label='$\\propto N_s^{-1/2}$')
    ax[0].set(xscale='log', yscale='log', xlabel='Shots per measurement setting $N_s$', ylabel='Relative error of $\\Delta\\mathbf{x}$')
    ax[0].legend(loc='upper right', fontsize=6.3)
    sub = nz[nz.snapshot == 'late']
    for k, (pro, zne) in enumerate([(0.0, False), (0.0, True), (0.01, False), (0.01, True)]):
        q = sub[(sub.p_ro == pro) & (sub.zne == zne)].sort_values('p2')
        xv = q.p2.replace(0, 3e-5)
        ax[1].errorbar(xv, q.rel_err_mean, yerr=q.rel_err_sd, fmt=MK[k], ls=LS[k], color=C[k], ms=3, capsize=2, lw=0.8,
                       label=f'$p_{{ro}}$={pro:.2f}' + (', ZNE' if zne else ''))
    if os.path.exists('results/rem_snap.csv'):
        rm = pd.read_csv('results/rem_snap.csv'); rm = rm[rm.snapshot == 'late']
        for k, zne in enumerate((False, True)):
            q = rm[rm.zne == zne].sort_values('p2'); xv = q.p2.replace(0, 3e-5)
            ax[1].errorbar(xv, q.rel_err_mean, yerr=q.rel_err_sd, fmt=MK[4 + k], ls=LS[4 + k], color=C[4 + k], ms=3, capsize=2, lw=0.8,
                           label='$p_{ro}$=0.01, REM' + (' + ZNE' if zne else ''))
    ax[1].set(xscale='log', yscale='log', xlabel='Two-qubit depolarizing probability $p_2$ ($p_1=p_2/10$)', ylabel='Relative error of $\\Delta\\mathbf{x}$')
    ax[1].set_xticks([3e-5, 1e-4, 1e-3, 1e-2]); ax[1].set_xticklabels(['0', '$10^{-4}$', '$10^{-3}$', '$10^{-2}$'])
    ax[1].legend(loc='lower right', fontsize=5.8, ncol=2); ax[1].set_ylim(1e-4, 3)
    ax[0].text(0.02, 0.04, '(a)', transform=ax[0].transAxes); ax[1].text(0.02, 0.96, '(b)', transform=ax[1].transAxes, va='top')
    fig.tight_layout(w_pad=1.0); save(fig, 'fig08_shots_noise')

if 'saa' in which and os.path.exists('results/saa.csv'):
    s = pd.read_csv('results/saa.csv')
    fig, ax = plt.subplots(figsize=(ONE, 2.2))
    ax.plot(s.N, s.in_sample, 'o-', color=C[0], ms=3, label='In-sample objective')
    ax.plot(s.N, s.oos_mean, 's--', color=C[1], ms=3, label='Out-of-sample mean (300 test scenarios)')
    ax.set(xscale='log', xlabel='Number of scenarios $N_\\omega$', ylabel='Expected daily cost (USD)')
    ax2 = ax.twinx(); ax2.plot(s.N, 1000 * s.dx_vs_200, '^:', color=C[2], ms=3, label='$\\max|\\mathbf{x}_{N}-\\mathbf{x}_{200}|$')
    ax2.set_ylabel('Schedule change (kW)'); ax2.tick_params(direction='in')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=6.0)
    fig.tight_layout(); save(fig, 'fig09_saa')

if 'pareto' in which and os.path.exists('results/pareto.csv'):
    p = pd.read_csv('results/pareto.csv')
    c = p[p.sweep == 'c_deg'].sort_values('value')
    fig, ax = plt.subplots(figsize=(ONE, 2.3))
    ax.plot(c.throughput_mwh, c.energy_cost, 'o-', color=C[0], ms=3)
    groups = []
    for _, r in c.iterrows():
        for gr in groups:
            if abs(gr[0] - r.throughput_mwh) < 0.15 and abs(gr[1] - r.energy_cost) < 8:
                gr[2].append(f"{r.value:g}"); break
        else:
            groups.append([r.throughput_mwh, r.energy_cost, [f"{r.value:g}"]])
    for x_, y_, labs in groups:
        dy = -10 if 4.2 < x_ < 4.6 else 4
        ax.annotate(', '.join(labs), (x_, y_), textcoords='offset points', xytext=(4, dy), fontsize=6)
    base = pickle.load(open('results/base_runs.pkl', 'rb'))['runs']['VQLS']['breakdown']
    ax.plot(base['thr_mwh'], base['energy'], '*', color=C[1], ms=8, label='VQLS-assisted solution ($c^{\\rm deg}$=27)')
    ax.set(xlabel='Expected fleet throughput (MWh/day)', ylabel='Expected energy cost (USD/day)', xlim=(0.3, 6.4))
    ax.legend(loc='upper right', fontsize=6.3)
    ax.text(0.03, 0.04, 'labels: $c^{\\rm deg}$ (USD/MWh)', transform=ax.transAxes, fontsize=6)
    fig.tight_layout(); save(fig, 'fig10_tradeoff')

if 'scaling' in which and os.path.exists('results/scaling.csv'):
    s = pd.read_csv('results/scaling.csv')
    fig, ax = plt.subplots(1, 2, figsize=(TWO, 2.2))
    for k, N in enumerate(sorted(s.N.unique())):
        q = s[s.N == N].sort_values('F')
        ax[0].semilogy(q.F, q.t_newton_total * 1e3, MK[k] + '-', color=C[k], ms=3, label=f'Newton step, $N_\\omega$={N}')
    q = s[s.N == 50].sort_values('F')
    ax[0].semilogy(q.F, q.t_lu * 1e3, 'x:', color='k', ms=3, label='LU of $\\mathbf{S}$')
    ax[0].semilogy(q.F, q.t_vqls * 1e3, 'D--', color=C[3], ms=3, label='VQLS of $\\mathbf{S}$ (emulated, cold start)')
    ax[0].set(xlabel='Number of fleets $F$', ylabel='Wall-clock time (ms)', xticks=[1, 2, 3, 4], ylim=(5e-3, 1e8)); ax[0].legend(loc='upper left', fontsize=6.0, ncol=2)
    q = s[s.N == 50].sort_values('F')
    ax[1].semilogy(q.F, q.n_pauli_M, 'o-', color=C[0], ms=3, label='Pauli terms of $\\mathbf{A}^2$')
    ax[1].semilogy(q.F, q.n_groups, 's--', color=C[1], ms=3, label='QWC measurement groups')
    ax[1].semilogy(q.F, q.npar, '^-.', color=C[2], ms=3, label='Ansatz parameters')
    ax[1].semilogy(q.F, q.n_full, 'v:', color=C[3], ms=3, label='Full Newton system size')
    for _, r in q.iterrows():
        ax[1].annotate(f"{int(r.qubits)} qubits", (r.F, r.npar), textcoords='offset points', xytext=(-10, 6), fontsize=6)
    ax[1].set(xlabel='Number of fleets $F$', ylabel='Count', xticks=[1, 2, 3, 4], ylim=(20, 3e5), xlim=(0.7, 4.3)); ax[1].legend(loc='upper left', fontsize=6.0, ncol=2)
    for k, a in enumerate(ax): a.text(0.98, 0.04, f'({"ab"[k]})', transform=a.transAxes, ha='right')
    fig.tight_layout(w_pad=1.0); save(fig, 'fig11_scaling')
print('ok')
