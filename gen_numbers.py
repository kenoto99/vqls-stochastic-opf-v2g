"""Generate LaTeX number macros and tables directly from the result files, so that every
number in the manuscript and the response letter is traceable to a result file."""
import json, os, pickle, numpy as np, pandas as pd
import grid as g

P = os.environ.get('PAPER_DIR', '../paper/')
N = {}


def sci(v, d=1):
    if v == 0:
        return '0'
    e = int(np.floor(np.log10(abs(v)))); m = v / 10 ** e
    if round(m, d) >= 10:
        m /= 10; e += 1
    return f'{m:.{d}f}\\times10^{{{e}}}'


def texname(k):
    w = dict(zip('0123456789', ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']))
    return ''.join(w.get(c, c) for c in k)


def fmt3(v):
    return f'{v:,.0f}' if v >= 100 else f'{v:.3g}'


def put(k, v):
    N[k] = v


# ---------------- base case
d = pickle.load(open('results/base_runs.pkl', 'rb')); runs = d['runs']
lu, vq = runs['LU'], runs['VQLS']
fref = runs['CLARABEL']['obj']
put('NScen', '50'); put('NTest', '300')
put('ObjOpt', f"{fref:,.2f}"); put('ObjOptFour', f"{runs['VQLS']['hist'][-1]['obj']:,.4f}")
rd = max(abs(runs[k]['hist'][-1]['obj'] - fref) / fref for k in ('LU', 'CG', 'GMRES', 'VQLS'))
put('ObjRelDiff', f"${sci(rd)}$")
put('ObjDiffVQLSLU', f"${sci(abs(vq['hist'][-1]['obj'] - lu['hist'][-1]['obj']) / lu['hist'][-1]['obj'])}$")
put('XDiffVQLSLU', f"${sci(1000 * np.abs(vq['x'] - lu['x']).max())}$")
put('XDiffRefLU', f"${sci(1000 * np.abs(runs['CLARABEL']['x'] - lu['x']).max())}$")
lg = pd.DataFrame(vq['log']); lgl = pd.DataFrame(lu['log'])
put('NewtonTot', str(len(lg))); put('NewtonLU', str(len(lgl)))
put('OuterIt', str(len(vq['hist'])))
put('NewtonPerOuter', ', '.join(str(h['newton']) for h in vq['hist']))
put('DvFirst', f"${sci(vq['hist'][0]['dv'])}$"); put('DvFinal', f"${sci(vq['hist'][-1]['dv'])}$")
put('StepErrMed', f"${sci(lg.rel_err.median())}$"); put('StepErrPninetyfive', f"${sci(lg.rel_err.quantile(.95))}$")
put('StepErrMax', f"${sci(lg.rel_err.max())}$"); put('StepErrMean', f"${sci(lg.rel_err.mean())}$")
put('CostMed', f"${sci(lg.vqls_cost.median())}$"); put('CostMax', f"${sci(lg.vqls_cost.max())}$")
put('Fallback', str(int(lg.fallback.sum())))
rec = pd.DataFrame(vq['vqls_records'])
put('NfevMed', f"{rec.vqls_nfev.median():.0f}"); put('NfevMax', f"{rec.vqls_nfev.max():.0f}")
put('PadLeakMax', f"${sci(rec.pad_leak.max())}$")
put('CondSMin', f"{lg.cond.min():.0f}"); put('CondSMax', f"{lg.cond.max():.0f}")
put('TimeLU', f"{lu['wall']:.0f}"); put('TimeCG', f"{runs['CG']['wall']:.0f}"); put('TimeGMRES', f"{runs['GMRES']['wall']:.0f}")
put('TimeVQLS', f"{vq['wall']:.0f}"); put('TimeClarabel', f"{runs['CLARABEL']['wall']:.1f}")
tcl = sum(h['t_classical'] for h in vq['hist']); tred = sum(h['t_reduced'] for h in vq['hist'])
tred_lu = sum(h['t_reduced'] for h in lu['hist'])
put('TClassicalVQLS', f"{tcl:.0f}"); put('TReducedVQLS', f"{tred:.0f}")
put('TLUPerSolveUs', f"{1e6 * tred_lu / len(lgl):.0f}"); put('TVQLSPerSolve', f"{tred / len(lg):.2f}")
ratio = (tred / len(lg)) / (tred_lu / len(lgl))
put('SpeedRatio', f"${sci(ratio, 0)}$".replace('1\\times', '').replace('$10^', '$10^') if False else f"${sci(ratio, 0)}$")
put('ShareReduced', f"{100 * tred / (tred + tcl):.0f}")
bd = vq['breakdown']
for k, n in (('energy', 'CostEnergy'), ('deg', 'CostDeg'), ('vdev', 'CostVdev'), ('qpen', 'CostQ'), ('dpen', 'CostD')):
    put(n, f"{bd[k]:,.2f}")
put('LossMWh', f"{bd['loss_mwh']:.3f}"); put('ThrMWh', f"{bd['thr_mwh']:.2f}")
vi = vq['viol']
put('VminIn', f"{vi['vmin']:.4f}"); put('VmaxIn', f"{vi['vmax']:.4f}"); put('UtilMaxIn', f"{100 * vi['max_util']:.1f}")
put('ResMaxIn', f"${sci(vi['max_res'])}$")
put('NVar', str(48 + 50 * 120))
from model import Model
import ipm
mm = Model(g.FLEETS[:2], g.sample_scenarios(50, 47273, g.FLEETS[:2]))
B = ipm.Barrier(mm)
put('NIneq', f"{B.ncons:,}")
put('NIneqScen', f"{mm.G.shape[0] + 2 * mm.T * mm.F:,}")

# per-variable step errors (VQLS vs LU, per Newton step)
def q3(x): return np.percentile(x, 50), np.percentile(x, 95), np.max(x)
rows = []
for col, lab, scale, unit in (('err_p', 'First-stage schedule $\\Delta p$', 1000, 'kW'),
                              ('err_v0', 'OLTC set-point $\\Delta V_0$', 1, 'p.u.'),
                              ('err_q', 'Reactive power $\\Delta q$', 1000, 'kvar'),
                              ('err_d', 'Deviation $\\Delta\\delta$', 1000, 'kW'),
                              ('err_soc', 'SoC trajectory', 100, '\\%')):
    a, b, c = q3(lg[col].values * scale)
    rows.append(f"{lab} & {unit} & ${sci(a)}$ & ${sci(b)}$ & ${sci(c)}$ \\\\")
open(P + 'tab_stepvars.tex', 'w').write('\n'.join(rows))
ux = (np.abs(vq['x'] - lu['x']).max())
put('FinalXDiffKW', f"${sci(1000 * ux)}$")

# solver table
rows = []
for k, lab in (('CLARABEL', 'Clarabel'), ('LU', 'IPM-LU'), ('CG', 'IPM-CG'),
               ('GMRES', 'IPM-GMRES'), ('VQLS', 'IPM-VQLS')):
    r = runs[k]
    if k == 'CLARABEL':
        rows.append(f"{lab} & {r['obj']:,.4f} & -- & -- & {r['wall']:.1f}$^{{\\rm a}}$ & -- \\\\")
    else:
        L = pd.DataFrame(r['log'])
        h = r['hist']; tr = sum(x['t_reduced'] for x in h)
        rows.append(f"{lab} & {h[-1]['obj']:,.4f} & ${sci(abs(h[-1]['obj'] - fref) / fref)}$ & {len(L)} & {r['wall']:.0f} & {fmt3(1e3 * tr / len(L))} \\\\")
open(P + 'tab_solvers.tex', 'w').write('\n'.join(rows))

# ---------------- out-of-sample
o = pickle.load(open('results/base_oos.pkl', 'rb'))['oos']
rows = []
names = {'Proposed (VQLS-IPM)': 'Proposed (VQLS-assisted)', 'Proposed (LU-IPM)': 'Proposed (LU)',
         'Deterministic EV': 'Deterministic (expected value)', '[38]-type prescribed': 'Prescribed, type of \\cite{ref38}',
         'Uncoordinated charging': 'Uncoordinated charging'}
oo = {}
for k, v in o.items():
    dd = pd.DataFrame(v['rows']); a = v['arrays']
    soc = np.array([x['soc'] for x in a]); ps = np.array([x['psub'] for x in a])
    oo[k] = dict(obj=dd.obj.mean(), ci=1.96 * dd.obj.std(ddof=1) / np.sqrt(len(dd)), energy=dd.energy.mean(), loss=dd.loss_mwh.mean(),
                 peak=ps.max(1).mean(), vmin=dd.vmin.min(), vmax=dd.vmax.max(), util=dd.max_util.max(), socmin=soc.min(),
                 socend=soc[:, :, -1].min(), soft=dd.soft.max(), res=dd.max_res.max())
    r = oo[k]
    rows.append(f"{names[k]} & {r['obj']:,.2f} & {r['loss']:.3f} & {r['peak']:.3f} & {r['vmin']:.4f} & {100 * r['util']:.1f} & {100 * r['socmin']:.1f} & ${sci(r['soft'])}$ \\\\")
open(P + 'tab_oos.tex', 'w').write('\n'.join(rows))
p_ = oo['Proposed (VQLS-IPM)']
put('OosProp', f"{p_['obj']:,.2f}"); put('OosCI', f"{p_['ci']:.2f}")
put('OosDet', f"{oo['Deterministic EV']['obj']:,.2f}"); put('OosPres', f"{oo['[38]-type prescribed']['obj']:,.2f}")
put('OosUnc', f"{oo['Uncoordinated charging']['obj']:,.2f}")
put('VSS', f"{oo['Deterministic EV']['obj'] - p_['obj']:.2f}")
put('GainPres', f"{oo['[38]-type prescribed']['obj'] - p_['obj']:.2f}")
put('GainUnc', f"{oo['Uncoordinated charging']['obj'] - p_['obj']:.2f}")
put('PeakProp', f"{p_['peak']:.3f}"); put('PeakUnc', f"{oo['Uncoordinated charging']['peak']:.3f}")
put('PeakPres', f"{oo['[38]-type prescribed']['peak']:.3f}")
put('OosVmin', f"{min(v['vmin'] for v in oo.values()):.4f}"); put('OosSocMin', f"{100 * p_['socmin']:.1f}")
put('OosResMax', f"${sci(max(v['res'] for v in oo.values()))}$")
put('OosSoftMax', f"${sci(max(v['soft'] for v in oo.values()))}$")
put('OosObjDiffVQLSLU', f"${sci(abs(p_['obj'] - oo['Proposed (LU-IPM)']['obj']))}$")
pf = pd.read_csv('results/pfonly.csv')
def pfv(pol, v0, col): return float(pf[(pf.policy == pol) & (np.isclose(pf.v0, v0))][col].iloc[0])
for key, nm in (('Proposed (VQLS-IPM)', 'Prop'), ('[38]-type prescribed', 'Pres'), ('Uncoordinated charging', 'Unc'), ('No V2G (fleets idle)', 'NoV')):
    put(f'PF{nm}FreqA', f"{100 * pfv(key, 1.03, 'freq_under'):.1f}"); put(f'PF{nm}VminA', f"{pfv(key, 1.03, 'vmin'):.4f}")
    put(f'PF{nm}FreqB', f"{100 * pfv(key, 1.00, 'freq_under'):.1f}"); put(f'PF{nm}VminB', f"{pfv(key, 1.00, 'vmin'):.4f}")
    put(f'PF{nm}SocMin', f"{100 * pfv(key, 1.03, 'soc_min'):.1f}"); put(f'PF{nm}Peak', f"{pfv(key, 1.03, 'peak_mw'):.3f}")
rows = []
for key, lab in (('Proposed (VQLS-IPM)', 'Proposed schedule'), ('[38]-type prescribed', 'Prescribed, type of \\cite{ref38}'),
                 ('Uncoordinated charging', 'Uncoordinated charging'), ('No V2G (fleets idle)', 'Idle fleets (no V2G)')):
    rows.append(f"{lab} & {pfv(key, 1.00, 'vmin'):.4f} & {100 * pfv(key, 1.00, 'freq_under'):.1f} & {pfv(key, 1.03, 'vmin'):.4f} & {100 * pfv(key, 1.03, 'freq_under'):.1f} & {100 * pfv(key, 1.03, 'soc_min'):.1f} \\\\")
open(P + 'tab_pfonly.tex', 'w').write('\n'.join(rows))

# ---------------- VQLS statistics
if os.path.exists('results/vqls_stats.pkl'):
    s = pickle.load(open('results/vqls_stats.pkl', 'rb'))
    rr = pd.DataFrame(s['rows']); info = pd.DataFrame(s['info_rows']).set_index('snapshot')
    sh = pd.DataFrame(s['shot_rows']); nz = pd.DataFrame(s['noise_rows'])
    put('NPauliA', str(int(info.n_pauli_A.iloc[0]))); put('NPauliM', str(int(info.n_pauli_M.iloc[0])))
    put('NGroupsM', str(int(info.n_groups_M.iloc[0])))
    for tg in ('early', 'middle', 'late'):
        T_ = tg.capitalize()
        put(f'Cond{T_}', f"{info.loc[tg, 'cond_scaled']:.1f}"); put(f'CondRaw{T_}', f"{info.loc[tg, 'cond_S']:.1f}")
        put(f'Newton{T_}', str(int(info.loc[tg, 'newton'])))
    rows = []
    for (opt, L), q in rr.groupby(['optimizer', 'layers']):
        lab = 'COBYLA' if opt == 'COBYLA' else 'L-BFGS-B'
        cells = []
        for tg in ('early', 'middle', 'late'):
            z = q[q.snapshot == tg]
            cells.append(f"${sci(z.wrel_err.mean())}$ & ${sci(z.wrel_err.max())}$")
        z = q
        rows.append((0 if lab == 'L-BFGS-B' else 1, L, f"{lab} & {L} & {int(z.npar.iloc[0])} & {len(q[q.snapshot=='late'])} & " + ' & '.join(cells) + f" & {z.nfev.mean():.0f} & {z.time.mean():.2f} \\\\"))
    rows.sort()
    open(P + 'tab_vqlsstats.tex', 'w').write('\n'.join(r[2] for r in rows))
    lb = rr[(rr.optimizer != 'COBYLA') & (rr.layers == 12)]
    put('ErrLtwelveMax', f"${sci(lb.wrel_err.max())}$")
    put('ErrLfourMean', f"${sci(rr[(rr.optimizer != 'COBYLA') & (rr.layers == 4)].wrel_err.mean())}$")
    put('ErrCobylaMean', f"${sci(rr[rr.optimizer == 'COBYLA'].wrel_err.mean())}$")
    put('ErrCobylaMin', f"${sci(rr[rr.optimizer == 'COBYLA'].wrel_err.min())}$")
    put('TimeCobyla', f"{rr[rr.optimizer == 'COBYLA'].time.mean():.0f}")
    put('TimeLbfgsTwelve', f"{lb.time.mean():.2f}")
    put('BoundViol', str(int((rr.wrel_err > rr.bound * (1 + 1e-9)).sum())))
    for ns in (1000, 10000, 100000, 1000000, 10000000):
        z = sh[sh.shots == ns].groupby('snapshot').rel_err.mean()
        put(f'ShotErr{int(np.log10(ns))}', f"${sci(z.min())}$--${sci(z.max())}$")
    def nzv(tg, p2, pro, zne):
        z = nz[(nz.snapshot == tg) & np.isclose(nz.p2, p2) & np.isclose(nz.p_ro, pro) & (nz.zne == zne)]
        return float(z.rel_err_mean.iloc[0])
    for p2, nm in ((1e-4, 'A'), (1e-3, 'B'), (1e-2, 'C')):
        put(f'Noise{nm}', f"{100 * nzv('late', p2, 0, False):.1f}"); put(f'Noise{nm}Zne', f"{100 * nzv('late', p2, 0, True):.1f}")
        put(f'Noise{nm}Ro', f"{100 * nzv('late', p2, 0.01, False):.1f}"); put(f'Noise{nm}RoZne', f"{100 * nzv('late', p2, 0.01, True):.1f}")
    put('NoiseZero', f"{100 * nzv('late', 0, 0, False):.2f}"); put('NoiseRoOnly', f"{100 * nzv('late', 0, 0.01, False):.1f}")

# ---------------- noise in the loop
if os.path.exists('results/noise_loop.pkl'):
    nl = pickle.load(open('results/noise_loop.pkl', 'rb'))
    rows = []
    pretty = {'shots 1e5': 'Shots, $N_s=10^5$', 'shots 1e6': 'Shots, $N_s=10^6$',
              'noisy p2=1e-3, ro=1%, 1e6': 'Noisy, $p_2=10^{-3}$, $p_{\\rm ro}=1\\%$',
              'noisy p2=1e-3, ro=1%, 1e6, ZNE': 'Noisy, $p_2=10^{-3}$, $p_{\\rm ro}=1\\%$, ZNE',
              'noisy p2=1e-2, ro=1%, 1e6': 'Noisy, $p_2=10^{-2}$, $p_{\\rm ro}=1\\%$',
              'noisy p2=1e-2, ro=1%, 1e6, ZNE': 'Noisy, $p_2=10^{-2}$, $p_{\\rm ro}=1\\%$, ZNE'}
    for k0, v in nl.items():
        k = pretty.get(k0, k0)
        r = v['res']
        if r['ok']:
            vi = r['viol']
            rows.append(f"{k} & {r['obj']:,.4f} & ${sci(abs(r['obj'] - fref) / fref)}$ & {r['newton']} & {r['fallback']} & ${sci(r['err_med'])}$ & ${sci(r['err_max'])}$ & {vi['vmin']:.4f} & {100 * vi['max_util']:.1f} \\\\")
        else:
            rows.append(f"{k} & \\multicolumn{{8}}{{l}}{{did not converge}} \\\\")
    open(P + 'tab_noiseloop.tex', 'w').write('\n'.join(rows))
    N['_noise_loop'] = {k: {kk: (vv if not isinstance(vv, np.ndarray) else None) for kk, vv in v['res'].items() if kk not in ('x', 'hist')} for k, v in nl.items()}

# ---------------- SAA
if os.path.exists('results/saa.csv'):
    s = pd.read_csv('results/saa.csv')
    for _, r in s.iterrows():
        put(f"SaaIn{int(r.N)}", f"{r.in_sample:,.2f}"); put(f"SaaOut{int(r.N)}", f"{r.oos_mean:,.2f}")
        put(f"SaaDx{int(r.N)}", f"{1000 * r.dx_vs_200:.1f}"); put(f"SaaTime{int(r.N)}", f"{r.solve_time:.0f}")
    put('SaaCI', f"{s.oos_ci95.mean():.1f}")
# ---------------- Pareto
if os.path.exists('results/pareto.csv'):
    p = pd.read_csv('results/pareto.csv')
    c = p[p.sweep == 'c_deg'].sort_values('value')
    put('ParetoThrMax', f"{c.throughput_mwh.max():.2f}"); put('ParetoThrMin', f"{c.throughput_mwh.min():.2f}")
    put('ParetoEnergyMin', f"{c.energy_cost.min():,.2f}"); put('ParetoEnergyMax', f"{c.energy_cost.max():,.2f}")
    w = p[p.sweep == 'w_v'].sort_values('value')
    rows = []
    for _, r in w.iterrows():
        rows.append(f"{r.value:g} & {r.energy_cost:,.2f} & {r.throughput_mwh:.2f} & {r.vdev_pu2:.4f} & {r.vmin:.4f} & {r.v0_mean:.4f} \\\\")
    open(P + 'tab_wv.tex', 'w').write('\n'.join(rows))
    put('_pareto', None)
# ---------------- scaling
if os.path.exists('results/scaling.csv'):
    s = pd.read_csv('results/scaling.csv')
    rows = []
    for _, r in s.sort_values(['F', 'N']).iterrows():
        rows.append(f"{int(r.F)} & {int(r.N)} & {int(r.n_full):,} & {int(r.red_dim)} & {int(r.qubits)} & {int(r.npar)} & {int(r.n_cnot)} & {int(r.n_pauli_M):,} & {int(r.n_groups):,} & {1e3 * r.t_newton_total:.0f} & {1e6 * r.t_lu:.0f} & {r.t_vqls:.2f} & ${sci(r.vqls_err)}$ \\\\")
    open(P + 'tab_scaling.tex', 'w').write('\n'.join(rows))

with open(P + 'numbers.tex', 'w') as fh:
    for k, v in N.items():
        if k.startswith('_') or v is None:
            continue
        fh.write(f"\\newcommand{{\\{texname(k)}}}{{{v}\\xspace}}\n")
json.dump({k: v for k, v in N.items()}, open(P + 'numbers.json', 'w'), indent=1, default=str)
print(len(N), 'numbers')

# ---------------- extension studies (added in the second revision round)
def _w(name, rows):
    open(P + name, 'w').write('\n'.join(rows))

if os.path.exists('results/fleets.csv'):
    fdf = pd.read_csv('results/fleets.csv')
    rows = []
    for _, r in fdf.iterrows():
        lab = r.case.split(': ', 1)[1].replace('(base)', '(base case)')
        rows.append(f"{r.case.split(':')[0]} & {lab} & {r.obj:,.2f} & {r.thr:.2f} & {r.loss:.3f} & {r.peak:.3f} & {r.vmin:.4f} & {r.v0:.4f} & {1000 * r.qmean:.0f} \\\\")
    _w('tab_fleets.tex', rows)
    put('FleetRelMax', f"${sci(fdf.rel_clarabel.max())}$")
    put('FleetObjMin', f"{fdf.obj.min():,.2f}"); put('FleetObjMax', f"{fdf.obj.max():,.2f}")
    put('FleetSpread', f"{fdf.obj.max() - fdf.obj.min():.2f}")

ext = {}
for tag in ('33bus_T48', '69bus_T24'):
    fn = f'results/ext_{tag}.pkl'
    if os.path.exists(fn):
        ext[tag] = pickle.load(open(fn, 'rb'))
if ext:
    def extrow(lab, net, T, L, runs_lu, runs_vq, clar, oos, nx, ny, N=50):
        lg_ = pd.DataFrame(runs_vq['log']); rec_ = pd.DataFrame(runs_vq['records']) if 'records' in runs_vq else pd.DataFrame(runs_vq['vqls_records'])
        nq = int(np.ceil(np.log2(nx)))
        o = runs_vq['hist'][-1]['obj']
        return (f"{lab} & {nx + N * ny:,} & {nx} & {nq} & {L} & {o:,.2f} & ${sci(abs(o - clar) / clar)}$ & {len(lg_)} & {int(lg_.fallback.sum())} & "
                f"${sci(lg_.rel_err.median())}$ & ${sci(lg_.rel_err.max())}$ & {rec_.vqls_time.median():.2f} & {oos['vmin']:.4f} & {100 * oos['util']:.1f} \\\\")
    rows = [extrow('33-bus, 1 h', '33', 24, 12, runs['LU'], runs['VQLS'], fref, oo['Proposed (VQLS-IPM)'], 48, 120)]
    for tag, lab, L, nx, ny in (('33bus_T48', '33-bus, 30 min', 21, 96, 240), ('69bus_T24', '69-bus, 1 h', 12, 48, 120)):
        if tag in ext and 'OOS' in ext[tag]:
            e = ext[tag]
            rows.append(extrow(lab, None, None, L, e['LU'], e['VQLS'], e['CLARABEL']['obj'], e['OOS'], nx, ny))
            k = 'SubH' if '48' in tag else 'Big'
            put(f'{k}Obj', f"{e['VQLS']['hist'][-1]['obj']:,.2f}")
            put(f'{k}Rel', f"${sci(abs(e['VQLS']['hist'][-1]['obj'] - e['CLARABEL']['obj']) / e['CLARABEL']['obj'])}$")
            put(f'{k}OosObj', f"{e['OOS']['obj']:,.2f}"); put(f'{k}OosVmin', f"{e['OOS']['vmin']:.4f}")
            put(f'{k}Newton', str(len(e['VQLS']['log']))); put(f'{k}Fallback', str(int(pd.DataFrame(e['VQLS']['log']).fallback.sum())))
            put(f'{k}ErrMed', f"${sci(pd.DataFrame(e['VQLS']['log']).rel_err.median())}$")
            put(f'{k}ErrMax', f"${sci(pd.DataFrame(e['VQLS']['log']).rel_err.max())}$")
            put(f'{k}VqlsTime', f"{pd.DataFrame(e['VQLS']['records']).vqls_time.median():.2f}")
            put(f'{k}LuWall', f"{e['LU']['wall']:.0f}"); put(f'{k}VqWall', f"{e['VQLS']['wall']:.0f}")
            put(f'{k}LossMWh', f"{e['VQLS']['breakdown']['loss_mwh']:.3f}")
    _w('tab_ext.tex', rows)

if os.path.exists('results/rem_snap.csv'):
    rm = pd.read_csv('results/rem_snap.csv')
    def rv(tg, p2, zne):
        z = rm[(rm.snapshot == tg) & np.isclose(rm.p2, p2) & (rm.zne == zne)]
        return 100 * float(z.rel_err_mean.iloc[0])
    put('RemRoOnly', f"{rv('late', 0, False):.2f}")
    put('RemB', f"{rv('late', 1e-3, False):.1f}"); put('RemBZne', f"{rv('late', 1e-3, True):.2f}")
    put('RemC', f"{rv('late', 1e-2, False):.1f}"); put('RemCZne', f"{rv('late', 1e-2, True):.1f}")
if os.path.exists('results/rem_loop.pkl'):
    rl = pickle.load(open('results/rem_loop.pkl', 'rb'))
    rows = open(P + 'tab_noiseloop.tex').read().strip().split('\n') if os.path.exists(P + 'tab_noiseloop.tex') else []
    pretty2 = {'noisy p2=1e-3, ro=1%, 1e6, ZNE+REM': 'Noisy, $p_2=10^{-3}$, $p_{\\rm ro}=1\\%$, ZNE+REM',
               'noisy p2=1e-2, ro=1%, 1e6, ZNE+REM': 'Noisy, $p_2=10^{-2}$, $p_{\\rm ro}=1\\%$, ZNE+REM'}
    for k0, v in rl.items():
        r = v['res']; vi = r['viol']
        rows.append(f"{pretty2[k0]} & {r['obj']:,.4f} & ${sci(abs(r['obj'] - fref) / fref)}$ & {r['newton']} & {r['fallback']} & ${sci(r['err_med'])}$ & ${sci(r['err_max'])}$ & {vi['vmin']:.4f} & {100 * vi['max_util']:.1f} \\\\")
        k = 'RemLoopB' if '1e-3' in k0 else 'RemLoopC'
        put(k + 'Med', f"${sci(r['err_med'])}$"); put(k + 'Newton', str(r['newton']))
    _w('tab_noiseloop.tex', rows)

with open(P + 'numbers.tex', 'w') as fh:
    for k, v in N.items():
        if k.startswith('_') or v is None:
            continue
        fh.write(f"\\newcommand{{\\{texname(k)}}}{{{v}\\xspace}}\n")
json.dump({k: v for k, v in N.items()}, open(P + 'numbers.json', 'w'), indent=1, default=str)
print('ext numbers', len(N))
