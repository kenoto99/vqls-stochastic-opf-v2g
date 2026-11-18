"""Two-stage scenario-based multi-period OPF model (sample average approximation).

First stage  x : day-ahead active-power schedule p_{f,t} of every V2G fleet (non-anticipative,
                 shared by all scenarios).
Second stage y_w: per scenario and hour, substation voltage set-point V0, fleet reactive power q_f
                 and availability-driven deviation d_f (realised power u = p - d).
Voltage and flows use a LinDistFlow model with AC fixed-point corrections (successive
linearisation); every solution is re-validated with the nonlinear AC sweep.
"""
import numpy as np
import grid as g

EPS_ABS = 1e-3      # MW, smoothing of |u|
W_V = 50.0          # USD per pu^2 per bus-hour (voltage-deviation weight)
W_Q = 2.0           # USD per Mvar^2-h (inverter reactive-power usage)
BETA = 200.0        # USD per MW^2-h (deviation from day-ahead schedule)
THERM_DIRS = np.linspace(0, 2 * np.pi, 8, endpoint=False)
V_MARGIN = 5e-4     # pu security margin on voltage limits in the optimisation model
S_MARGIN = 0.995    # thermal security factor in the optimisation model


def sabs(u):
    return np.sqrt(u * u + EPS_ABS ** 2)


class Model:
    def __init__(self, fleets, scen, w_v=W_V, c_deg=g.C_DEG, beta=BETA, w_q=W_Q,
                 vcorr=None, losscorr=None, fcorr=None):
        self.fl = fleets
        self.F = F = len(fleets)
        self.T = T = g.T
        self.N = N = scen['n']
        self.sc = scen
        self.w_v, self.c_deg, self.beta, self.w_q = w_v, c_deg, beta, w_q
        self.nx = F * T
        self.k = 1 + 2 * F
        self.ny = self.k * T
        self.n = self.nx + self.ny
        self.pi = np.full(N, 1.0 / N)
        bus = np.array([f['bus'] for f in fleets])
        self.bus = bus
        self.vcorr = np.zeros((N, T, g.NB)) if vcorr is None else vcorr
        self.losscorr = np.zeros((N, T)) if losscorr is None else losscorr
        self.fcorr = np.zeros((N, T, g.NL), complex) if fcorr is None else fcorr
        n, nx = self.n, self.nx

        def ip(f, t): return f * T + t
        def iy(t, k): return nx + t * self.k + k
        self.ip, self.iy = ip, iy
        # u_{f,t} and q_{f,t} selectors (rows t*F+f)
        U = np.zeros((T * F, n)); Qf = np.zeros((T * F, n)); V0 = np.zeros((T, n)); Dd = np.zeros((T * F, n))
        for t in range(T):
            V0[t, iy(t, 0)] = 1
            for f in range(F):
                U[t * F + f, ip(f, t)] = 1; U[t * F + f, iy(t, 1 + F + f)] = -1
                Qf[t * F + f, iy(t, 1 + f)] = 1
                Dd[t * F + f, iy(t, 1 + F + f)] = 1
        self.U, self.Qf, self.V0sel, self.Dd = U, Qf, V0, Dd
        # voltage map: V[t,i] (i = 1..32) = Vmat z + Vb
        nbus = g.NB - 1
        Vmat = np.zeros((T * nbus, n))
        for t in range(T):
            rows = slice(t * nbus, (t + 1) * nbus)
            Vmat[rows] += V0[t][None, :]
            for f in range(F):
                Vmat[rows] += np.outer(g.Rp[1:, bus[f]], U[t * F + f]) + np.outer(g.Xp[1:, bus[f]], Qf[t * F + f])
        self.Vmat = Vmat
        # branch flows: P_l = Pc - AP z, Q_l = Qc - AQ z
        AP = np.zeros((T * g.NL, n)); AQ = np.zeros((T * g.NL, n))
        for t in range(T):
            rows = slice(t * g.NL, (t + 1) * g.NL)
            for f in range(F):
                AP[rows] += np.outer(g.D[:, bus[f]], U[t * F + f])
                AQ[rows] += np.outer(g.D[:, bus[f]], Qf[t * F + f])
        self.AP, self.AQ = AP, AQ
        rho = np.repeat(g.TARIFF, g.NL) * g.DT
        Rl = np.tile(g.R, T)
        self.rhoR = rho * Rl
        # constant quadratic Hessian of the expected objective (identical for every scenario)
        H = 2 * (AP.T * self.rhoR) @ AP + 2 * (AQ.T * self.rhoR) @ AQ
        H += 2 * w_v * g.DT * Vmat.T @ Vmat
        H += 2 * w_q * g.DT * Qf.T @ Qf + 2 * beta * g.DT * Dd.T @ Dd
        self.Hq = H
        # ---- linear inequality rows G z <= h_w (G common, h scenario dependent)
        rows, kinds = [], []
        # voltage limits
        rows.append(-Vmat); kinds.append(('vlo', T * nbus))
        rows.append(Vmat); kinds.append(('vhi', T * nbus))
        # thermal polygon only for branches carrying fleet power
        lmask = (g.D[:, bus].sum(axis=1) > 0)
        self.lsel = np.where(lmask)[0]
        th = []
        for t in range(T):
            for l in self.lsel:
                for a in THERM_DIRS:
                    th.append(-(np.cos(a) * AP[t * g.NL + l] + np.sin(a) * AQ[t * g.NL + l]))
        rows.append(np.array(th)); kinds.append(('therm', len(th)))
        # V0 bounds
        rows.append(V0); kinds.append(('v0hi', T)); rows.append(-V0); kinds.append(('v0lo', T))
        # |u| <= xi Pmax
        rows.append(U); kinds.append(('uhi', T * F)); rows.append(-U); kinds.append(('ulo', T * F))
        # |d| <= Pmax
        rows.append(Dd); kinds.append(('dhi', T * F)); rows.append(-Dd); kinds.append(('dlo', T * F))
        # SoC upper (linear, conservative): E0 - cumsum(u) - cumdrain <= Emax  ->  -cumsum(u) <= Emax - E0 + cumdrain
        Cs = np.zeros((F * T, n))
        for f in range(F):
            for t in range(T):
                for tau in range(t + 1):
                    Cs[f * T + t] += g.DT * U[tau * F + f]
        self.Cs = Cs
        rows.append(-Cs); kinds.append(('socup', F * T))
        self.G = np.vstack(rows)
        self.kinds = kinds
        # first-stage bounds |p| <= Pmax
        Gx = np.vstack([np.eye(self.nx), -np.eye(self.nx)])
        self.Gx = Gx
        self.hx = np.concatenate([np.repeat([f['pmax'] for f in fleets], T)] * 2)
        self.update_scenario_data()

    # ------------------------------------------------------------------
    def update_scenario_data(self):
        sc, T, F, N = self.sc, self.T, self.F, self.N
        nbus = g.NB - 1
        P, Qn = sc['p_load'], sc['q_load']           # (N,T,33)
        self.Vb = np.empty((N, T * nbus))
        self.Pc = np.empty((N, T * g.NL)); self.Qc = np.empty((N, T * g.NL))
        for t in range(T):
            vlin = -P[:, t] @ g.Rp.T - Qn[:, t] @ g.Xp.T + self.vcorr[:, t]
            self.Vb[:, t * nbus:(t + 1) * nbus] = vlin[:, 1:]
            self.Pc[:, t * g.NL:(t + 1) * g.NL] = P[:, t] @ g.D.T
            self.Qc[:, t * g.NL:(t + 1) * g.NL] = Qn[:, t] @ g.D.T
        h = []
        h.append(self.Vb - (g.V_MIN + V_MARGIN))                       # -V <= -0.95  -> -(Vmat z) <= Vb - 0.95
        h.append((g.V_MAX - V_MARGIN) - self.Vb)
        th = np.empty((N, T * len(self.lsel) * len(THERM_DIRS)))
        k = 0
        smax = S_MARGIN * g.SMAX_LINE * np.cos(np.pi / 8)
        for t in range(T):
            for l in self.lsel:
                for a in THERM_DIRS:
                    pc = self.Pc[:, t * g.NL + l] + self.fcorr[:, t, l].real
                    qc = self.Qc[:, t * g.NL + l] + self.fcorr[:, t, l].imag
                    th[:, k] = smax[l] - (np.cos(a) * pc + np.sin(a) * qc)
                    k += 1
        h.append(th)
        h.append(np.full((N, T), g.V0_MAX)); h.append(np.full((N, T), -g.V0_MIN))
        pmax = np.array([f['pmax'] for f in self.fl]); smaxf = np.array([f['smax'] for f in self.fl])
        xi = sc['avail']                                   # (N,F,T)
        ulim = (xi * pmax[None, :, None]).transpose(0, 2, 1).reshape(N, T * F)
        h.append(ulim); h.append(ulim)
        h.append(np.tile(np.repeat(pmax[None], T, 0).reshape(-1), (N, 1)))
        h.append(np.tile(np.repeat(pmax[None], T, 0).reshape(-1), (N, 1)))
        ecap = np.array([f['ecap'] for f in self.fl])
        cumdrain = np.cumsum(sc['drain'], axis=2)          # (N,F,T)
        socup = g.SOC_MAX * ecap[None, :, None] - sc['e0'][:, :, None] + cumdrain
        h.append(socup.reshape(N, F * T))
        self.h = np.hstack(h)
        self.rdisk = (xi * smaxf[None, :, None]).transpose(0, 2, 1).reshape(N, T * F) ** 2
        # SoC lower (convex): g = E0 - cumdrain - Emin_t - Cs z - kappa*cumsum(sabs(u)) >= 0
        emin = np.repeat((g.SOC_MIN * ecap)[:, None], T, 1)
        emin[:, -1] = g.SOC_REQ * ecap
        self.soclo_c = (sc['e0'][:, :, None] - cumdrain - emin[None]).reshape(N, F * T)
        # linear objective terms per scenario
        f = np.zeros((N, self.n)); c0 = np.zeros(N)
        rho_u = np.zeros(self.n)
        for t in range(T):
            for ff in range(F):
                rho_u -= g.TARIFF[t] * g.DT * self.U[t * F + ff]
        for w in range(N):
            f[w] = rho_u.copy()
            f[w] -= 2 * self.AP.T @ (self.rhoR * self.Pc[w]) + 2 * self.AQ.T @ (self.rhoR * self.Qc[w])
            f[w] += 2 * self.w_v * g.DT * self.Vmat.T @ (self.Vb[w] - 1.0)
            c0[w] = (g.DT * np.sum(g.TARIFF * (P[w].sum(axis=1) + self.losscorr[w]))
                     + np.sum(self.rhoR * (self.Pc[w] ** 2 + self.Qc[w] ** 2))
                     + self.w_v * g.DT * np.sum((self.Vb[w] - 1) ** 2))
        self.f, self.c0 = f, c0

    # ------------------------------------------------------------------
    def split(self, z):
        return z[:self.nx], z[self.nx:]

    def objective(self, x, Y):
        """Expected objective (USD/day) for first stage x and recourse Y (N, ny)."""
        J = 0.0
        for w in range(self.N):
            z = np.concatenate([x, Y[w]])
            J += self.pi[w] * (0.5 * z @ self.Hq @ z + self.f[w] @ z + self.c0[w]
                               + self.c_deg * np.sum(sabs(self.U @ z)) * g.DT)
        return J

    def breakdown(self, x, Y):
        """Energy cost, loss cost, degradation, penalties (expected)."""
        out = dict(energy=0, deg=0, thr_mwh=0, vdev=0, qpen=0, dpen=0, loss_mwh=0)
        for w in range(self.N):
            z = np.concatenate([x, Y[w]])
            P = self.Pc[w] - self.AP @ z; Q = self.Qc[w] - self.AQ @ z
            loss = (g.R[None] * (P.reshape(self.T, -1) ** 2 + Q.reshape(self.T, -1) ** 2)).sum(1) + self.losscorr[w]
            u = (self.U @ z).reshape(self.T, self.F)
            psub = self.sc['p_load'][w].sum(1) - u.sum(1) + loss
            out['energy'] += self.pi[w] * g.DT * np.sum(g.TARIFF * psub)
            out['loss_mwh'] += self.pi[w] * g.DT * loss.sum()
            V = self.Vmat @ z + self.Vb[w]
            out['vdev'] += self.pi[w] * self.w_v * g.DT * np.sum((V - 1) ** 2)
            out['qpen'] += self.pi[w] * self.w_q * g.DT * np.sum((self.Qf @ z) ** 2)
            out['dpen'] += self.pi[w] * self.beta * g.DT * np.sum((self.Dd @ z) ** 2)
            out['thr_mwh'] += self.pi[w] * np.sum(np.abs(self.U @ z)) * g.DT
            out['deg'] += self.pi[w] * self.c_deg * np.sum(sabs(self.U @ z)) * g.DT
        return out
