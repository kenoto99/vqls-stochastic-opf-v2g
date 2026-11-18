"""Successive-linearisation driver, AC validation and out-of-sample recourse evaluation."""
import numpy as np
import cvxpy as cp
import grid as g
from model import Model, EPS_ABS
import ipm
import ref

V_MARGIN = 0.0


def injections(m, x, Y):
    """Net loads (N,T,33) P,Q after realised fleet power, and V0 (N,T)."""
    P = m.sc['p_load'].copy(); Q = m.sc['q_load'].copy()
    V0 = np.empty((m.N, m.T))
    for w in range(m.N):
        z = np.concatenate([x, Y[w]])
        u = (m.U @ z).reshape(m.T, m.F); q = (m.Qf @ z).reshape(m.T, m.F)
        for f in range(m.F):
            P[w, :, m.bus[f]] -= u[:, f]
            Q[w, :, m.bus[f]] -= q[:, f]
        V0[w] = (m.V0sel @ z)
    return P, Q, V0


def ac_eval(m, x, Y):
    P, Q, V0 = injections(m, x, Y)
    B = m.N * m.T
    v, s, loss, res, it = g.bfs(P.reshape(B, -1), Q.reshape(B, -1), V0.reshape(-1))
    vlin = g.lindist_voltage(P.reshape(B, -1), Q.reshape(B, -1), V0.reshape(-1))
    Pf = P.reshape(B, -1) @ g.D.T; Qf = Q.reshape(B, -1) @ g.D.T
    loss_quad = (g.R[None] * (Pf ** 2 + Qf ** 2)).sum(1)
    out = dict(v=v.reshape(m.N, m.T, -1), vlin=vlin.reshape(m.N, m.T, -1),
               s=s.reshape(m.N, m.T, -1), loss=loss.reshape(m.N, m.T),
               loss_quad=loss_quad.reshape(m.N, m.T), res=res.reshape(m.N, m.T), iters=it,
               P=P, Q=Q, V0=V0)
    out['vcorr'] = out['v'] - out['vlin']
    out['losscorr'] = out['loss'] - out['loss_quad']
    out['fcorr'] = out['s'] - (Pf + 1j * Qf).reshape(m.N, m.T, -1)
    return out


def violations(ac, smax=None):
    smax = g.SMAX_LINE if smax is None else smax
    v = ac['v'][..., 1:]
    under = np.maximum(g.V_MIN - v, 0); over = np.maximum(v - g.V_MAX, 0)
    util = np.abs(ac['s']) / smax[None, None]
    return dict(vmin=float(v.min()), vmax=float(v.max()), max_under=float(under.max()),
                max_over=float(over.max()), freq_v=float(((under > 1e-9) | (over > 1e-9)).any(-1).mean()),
                max_util=float(util.max()), freq_therm=float((util > 1 + 1e-9).any(-1).mean()),
                max_res=float(ac['res'].max()))


def solve_case(fleets, scen, reduced_solver=ipm.lu_solver, outer=5, tol_v=1e-5, verbose=False,
               w_v=None, c_deg=None, log=None, tau0=None, max_newton=60, inner='ipm'):
    kw = {}
    if w_v is not None: kw['w_v'] = w_v
    if c_deg is not None: kw['c_deg'] = c_deg
    vcorr = None; losscorr = None; fcorr = None
    hist = []
    x = Y = None
    for k in range(outer):
        m = Model(fleets, scen, vcorr=vcorr, losscorr=losscorr, fcorr=fcorr, **kw)
        if inner == 'clarabel':
            x, Y, pr = ref.reference(m)
            info = dict(obj=m.objective(x, Y), newton=0, t_classical=0.0, t_reduced=0.0)
            ac = ac_eval(m, x, Y)
            dv = np.abs(ac['vcorr'] - m.vcorr).max()
            hist.append(dict(outer=k, obj=info['obj'], newton=0, dv=dv, **violations(ac), t_classical=0.0, t_reduced=0.0))
            vcorr, losscorr, fcorr = ac['vcorr'], ac['losscorr'], ac['fcorr']
            if dv < tol_v and hist[-1]['freq_therm'] == 0:
                break
            continue
        B = ipm.Barrier(m)
        if x is None or not B.slacks(x, Y):
            x0, Y0, _ = ref.phase1(m)
        else:
            x0, Y0 = x, Y
        t0 = tau0 if tau0 is not None else B.ncons / (1e-1 * max(abs(m.objective(x0, Y0)), 1.0))
        lg = [] if log is not None else None
        x, Y, info = ipm.solve(m, x0, Y0, reduced_solver=reduced_solver, tau0=t0, mu=20.0,
                               gap_tol=1e-3, newton_tol=1e-7, verbose=verbose, log=lg, max_newton=max_newton)
        if log is not None:
            for r in lg:
                r['outer'] = k
            log.extend(lg)
        ac = ac_eval(m, x, Y)
        dv = np.abs(ac['vcorr'] - m.vcorr).max()
        hist.append(dict(outer=k, obj=info['obj'], newton=info['newton'], dv=dv, **violations(ac),
                         t_classical=info['t_classical'], t_reduced=info['t_reduced']))
        if verbose:
            print('outer', hist[-1])
        vcorr, losscorr, fcorr = ac['vcorr'], ac['losscorr'], ac['fcorr']
        if dv < tol_v and hist[-1]['freq_therm'] == 0:
            break
    # final model consistent with last AC point (for reporting)
    return dict(x=x, Y=Y, model=m, ac=ac, hist=hist, info=info)


# -------------------------------------------------------------------------
class Recourse:
    """Second-stage problem for one scenario with the first stage x fixed.  Voltage limits are
    softened with a large penalty so that a violation, if any, is measured instead of hidden."""
    def __init__(self, fleets, const=False):
        sc1 = g.sample_scenarios(1, 0, fleets)
        self.m = m = Model(fleets, sc1)
        self.const = const
        if const:
            return
        nx, ny = m.nx, m.ny
        self.y = y = cp.Variable(ny); self.sv = cp.Variable(m.G.shape[0], nonneg=True)
        self.xp = cp.Parameter(nx); self.hp = cp.Parameter(m.G.shape[0]); self.rp = cp.Parameter(m.T * m.F, nonneg=True)
        self.socp = cp.Parameter(m.F * m.T); self.fp = cp.Parameter(ny)
        Gx, Gy = m.G[:, :nx], m.G[:, nx:]
        nv = 2 * m.T * (g.NB - 1)
        soft = np.zeros(m.G.shape[0]); soft[:nv] = 1.0
        u = m.U[:, :nx] @ self.xp + m.U[:, nx:] @ y; q = m.Qf[:, nx:] @ y
        Cum = ref.soc_lower_expr(m, None)
        M = np.vstack([np.sqrt(2 * m.rhoR)[:, None] * m.AP, np.sqrt(2 * m.rhoR)[:, None] * m.AQ,
                       np.sqrt(2 * m.w_v * g.DT) * m.Vmat, np.sqrt(2 * m.w_q * g.DT) * m.Qf, np.sqrt(2 * m.beta * g.DT) * m.Dd])
        su = cp.Variable(m.T * m.F)
        obj = 0.5 * cp.sum_squares(M[:, :nx] @ self.xp + M[:, nx:] @ y) + self.fp @ y + 1e5 * cp.sum(self.sv[:nv]) \
            + m.c_deg * g.DT * cp.sum(su)
        cons = [Gx @ self.xp + Gy @ y <= self.hp + cp.multiply(soft, self.sv), self.sv[nv:] == 0,
                cp.square(u) + cp.square(q) <= self.rp,
                cp.norm(cp.vstack([u, EPS_ABS * np.ones(m.T * m.F)]), 2, axis=0) <= su,
                Cum @ (u + g.KAPPA * su) <= self.socp]
        self.prob = cp.Problem(cp.Minimize(obj), cons)

    def solve_const(self, mm, x):
        """Same problem as self.prob, built with numeric data (sparse) instead of cvxpy parameters.
        Used for long horizons, where the parametric canonicalization exceeds the memory limit."""
        import scipy.sparse as sp
        m = mm; nx, ny = m.nx, m.ny
        y = cp.Variable(ny); sv = cp.Variable(m.G.shape[0], nonneg=True)
        nv = 2 * m.T * (g.NB - 1); soft = np.zeros(m.G.shape[0]); soft[:nv] = 1.0
        S = lambda A: sp.csr_matrix(A)
        u = S(m.U[:, nx:]) @ y + m.U[:, :nx] @ x; q = S(m.Qf[:, nx:]) @ y
        Cum = S(ref.soc_lower_expr(m, None))
        M = np.vstack([np.sqrt(2 * m.rhoR)[:, None] * m.AP, np.sqrt(2 * m.rhoR)[:, None] * m.AQ,
                       np.sqrt(2 * m.w_v * g.DT) * m.Vmat, np.sqrt(2 * m.w_q * g.DT) * m.Qf, np.sqrt(2 * m.beta * g.DT) * m.Dd])
        su = cp.Variable(m.T * m.F)
        obj = 0.5 * cp.sum_squares(S(M[:, nx:]) @ y + M[:, :nx] @ x) + m.f[0][nx:] @ y + 1e5 * cp.sum(sv[:nv]) \
            + m.c_deg * g.DT * cp.sum(su)
        cons = [S(m.G[:, nx:]) @ y <= m.h[0] - m.G[:, :nx] @ x + cp.multiply(soft, sv), sv[nv:] == 0,
                cp.square(u) + cp.square(q) <= m.rdisk[0],
                cp.norm(cp.vstack([u, EPS_ABS * np.ones(m.T * m.F)]), 2, axis=0) <= su,
                Cum @ (u + g.KAPPA * su) <= m.soclo_c[0]]
        cp.Problem(cp.Minimize(obj), cons).solve(solver='CLARABEL')
        return y.value, sv.value

    def evaluate(self, x, scen, iters=4):
        """Returns per-scenario dict with AC results after successive linearisation."""
        fleets = self.m.fl
        rows = []
        Yall = []
        arrays = []
        for w in range(scen['n']):
            one = {k: (v[w:w + 1] if isinstance(v, np.ndarray) else v) for k, v in scen.items()}
            one['n'] = 1
            vcorr = None; losscorr = None; fcorr = None
            for k in range(iters):
                mm = Model(fleets, one, vcorr=vcorr, losscorr=losscorr, fcorr=fcorr)
                if self.const:
                    y, svv = self.solve_const(mm, x)
                else:
                    self.xp.value = x; self.hp.value = mm.h[0]; self.rp.value = mm.rdisk[0]
                    self.socp.value = mm.soclo_c[0]; self.fp.value = mm.f[0][mm.nx:]
                    self.prob.solve(solver='CLARABEL')
                    y = self.y.value; svv = self.sv.value
                ac = ac_eval(mm, x, y[None])
                dv = np.abs(ac['vcorr'] - mm.vcorr).max()
                vcorr, losscorr, fcorr = ac['vcorr'], ac['losscorr'], ac['fcorr']
                if dv < 1e-5:
                    break
            Yall.append(y)
            viol = violations(ac)
            z = np.concatenate([x, y])
            u = (mm.U @ z).reshape(mm.T, mm.F); qq = (mm.Qf @ z).reshape(mm.T, mm.F)
            ecap = np.array([f['ecap'] for f in fleets])
            de = g.DT * np.where(u.T > 0, u.T / g.ETA, u.T * g.ETA)   # exact asymmetric efficiencies
            e = one['e0'][0][:, None] - np.cumsum(de, axis=1) - np.cumsum(one['drain'][0], axis=1)
            soc = np.hstack([one['e0'][0][:, None], e]) / ecap[:, None]
            psub = ac['P'][0].sum(1) + ac['loss'][0]
            arrays.append(dict(vmin_t=ac['v'][0, :, 1:].min(1), vmax_t=ac['v'][0, :, 1:].max(1),
                               v18=ac['v'][0, :, 17], v33=ac['v'][0, :, 32], psub=psub, loss=ac['loss'][0],
                               soc=soc, v0=ac['V0'][0], u=u, q=qq, util=(np.abs(ac['s'][0]) / g.SMAX_LINE).max(1)))
            obj = mm.objective(x, y[None]) - mm.c_deg * 0  # expected over the single scenario
            bd = mm.breakdown(x, y[None])
            rows.append(dict(scenario=w, obj=obj, energy=bd['energy'], loss_mwh=bd['loss_mwh'],
                             soft=float(svv[:2 * mm.T * (g.NB - 1)].max()),
                             dmax=float(np.abs(mm.Dd @ np.concatenate([x, y])).max()), **viol))
        return rows, np.array(Yall), arrays
