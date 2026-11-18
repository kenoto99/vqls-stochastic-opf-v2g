"""Primal log-barrier interior-point method for the two-stage model with a Schur-complement
reduction onto the non-anticipative first-stage variables.  The dense reduced Newton system
S dx = r is passed to a pluggable solver (LU, CG, GMRES or VQLS)."""
import time
import numpy as np
import scipy.sparse as sp
import grid as g
from model import sabs, EPS_ABS


def _sabs_d(u):
    s = sabs(u)
    return u / s, EPS_ABS ** 2 / s ** 3


class Barrier:
    def __init__(self, m):
        self.m = m
        F, T = m.F, m.T
        self.Gs = sp.csr_matrix(m.G)
        self.Us = sp.csr_matrix(m.U)
        self.Qs = sp.csr_matrix(m.Qf)
        # cumulative operator from u (t*F+f order) to (f*T+t)
        Cum = np.zeros((F * T, T * F))
        for f in range(F):
            for t in range(T):
                for tau in range(t + 1):
                    Cum[f * T + t, tau * F + f] = g.DT
        self.Cum = Cum
        self.ncons = m.G.shape[0] * m.N + 2 * m.nx + 2 * T * F * m.N
        self.nx = m.nx

    # ---------------------------------------------------------------
    def slacks(self, x, Y):
        m = self.m
        ok = True
        sx = m.hx - m.Gx @ x
        if np.any(sx <= 0):
            return False
        for w in range(m.N):
            z = np.concatenate([x, Y[w]])
            if np.any(m.h[w] - self.Gs @ z <= 0):
                return False
            u = self.Us @ z; q = self.Qs @ z
            if np.any(m.rdisk[w] - u * u - q * q <= 0):
                return False
            if np.any(m.soclo_c[w] - self.Cum @ (u + g.KAPPA * sabs(u)) <= 0):
                return False
        return ok

    def nl_slacks(self, x, Y):
        m = self.m
        out = []
        for w in range(m.N):
            z = np.concatenate([x, Y[w]])
            u = self.Us @ z; q = self.Qs @ z
            out.append(np.concatenate([m.rdisk[w] - u * u - q * q,
                                       m.soclo_c[w] - self.Cum @ (u + g.KAPPA * sabs(u))]))
        return np.array(out)

    def value(self, x, Y, tau):
        m = self.m
        sx = m.hx - m.Gx @ x
        if np.any(sx <= 0):
            return np.inf
        val = -np.sum(np.log(sx))
        for w in range(m.N):
            z = np.concatenate([x, Y[w]])
            s = m.h[w] - self.Gs @ z
            u = self.Us @ z; q = self.Qs @ z
            sd = m.rdisk[w] - u * u - q * q
            sl = m.soclo_c[w] - self.Cum @ (u + g.KAPPA * sabs(u))
            if np.any(s <= 0) or np.any(sd <= 0) or np.any(sl <= 0):
                return np.inf
            val += tau * m.pi[w] * (0.5 * z @ m.Hq @ z + m.f[w] @ z + m.c0[w] + m.c_deg * np.sum(sabs(u)) * g.DT)
            val -= np.sum(np.log(s)) + np.sum(np.log(sd)) + np.sum(np.log(sl))
        return val

    def scenario_derivs(self, w, x, y, tau):
        m = self.m
        z = np.concatenate([x, y])
        s = m.h[w] - self.Gs @ z
        grad = tau * m.pi[w] * (m.Hq @ z + m.f[w]) + self.Gs.T @ (1 / s)
        H = tau * m.pi[w] * m.Hq + (self.Gs.T @ sp.diags(1 / s ** 2) @ self.Gs).toarray()
        u = self.Us @ z; q = self.Qs @ z
        # disk
        sd = m.rdisk[w] - u * u - q * q
        gu = 2 * u / sd; gq = 2 * q / sd
        grad += self.Us.T @ gu + self.Qs.T @ gq
        Huu = sp.diags(4 * u * u / sd ** 2 + 2 / sd); Hqq = sp.diags(4 * q * q / sd ** 2 + 2 / sd)
        Huq = sp.diags(4 * u * q / sd ** 2)
        H += (self.Us.T @ Huu @ self.Us + self.Qs.T @ Hqq @ self.Qs
              + self.Us.T @ Huq @ self.Qs + self.Qs.T @ Huq @ self.Us).toarray()
        d1, d2 = _sabs_d(u)
        # degradation cost on realised throughput
        grad += self.Us.T @ (tau * m.pi[w] * m.c_deg * g.DT * d1)
        H += (self.Us.T @ sp.diags(tau * m.pi[w] * m.c_deg * g.DT * d2) @ self.Us).toarray()
        # SoC lower (convex)
        sl = m.soclo_c[w] - self.Cum @ (u + g.KAPPA * sabs(u))
        A = self.Cum * (1 + g.KAPPA * d1)[None, :]
        gu2 = A.T @ (1 / sl)
        grad += self.Us.T @ gu2
        Hu2 = (A.T * (1 / sl ** 2)) @ A + np.diag((self.Cum.T @ (1 / sl)) * g.KAPPA * d2)
        H += (self.Us.T @ sp.csr_matrix(Hu2) @ self.Us).toarray()
        return grad, H

    def first_stage_derivs(self, x, tau):
        m = self.m
        sx = m.hx - m.Gx @ x
        grad = m.Gx.T @ (1 / sx)
        H = (m.Gx.T * (1 / sx ** 2)) @ m.Gx
        return grad, H


def lu_solver(S, r, ctx):
    return np.linalg.solve(S, r), {}


def cg_solver(S, r, ctx):
    from scipy.sparse.linalg import cg
    it = [0]
    x, info = cg(S, r, rtol=1e-10, atol=0, maxiter=10 * len(r), callback=lambda xk: it.__setitem__(0, it[0] + 1))
    return x, dict(iters=it[0])


def gmres_solver(S, r, ctx):
    from scipy.sparse.linalg import gmres
    it = [0]
    x, info = gmres(S, r, rtol=1e-10, atol=0, restart=len(r), maxiter=10 * len(r),
                    callback=lambda rk: it.__setitem__(0, it[0] + 1), callback_type='pr_norm')
    return x, dict(iters=it[0])


def solve(m, x0, Y0, reduced_solver=lu_solver, tau0=1.0, mu=10.0, gap_tol=1e-6,
          newton_tol=1e-9, max_newton=60, verbose=False, log=None, max_total_newton=None):
    """Returns x, Y, info.  `log` (list) receives one dict per Newton step."""
    B = Barrier(m)
    x, Y = x0.copy(), Y0.copy()
    if not B.slacks(x, Y):
        raise ValueError('initial point is not strictly feasible')
    tau = tau0
    nx = m.nx
    total_newton = 0
    t_classical = t_reduced = 0.0
    while True:
        for it in range(max_newton):
            tc = time.perf_counter()
            gx, Hxx = B.first_stage_derivs(x, tau)
            r = -gx.copy()
            S = Hxx.copy()
            facts = []
            for w in range(m.N):
                gw, Hw = B.scenario_derivs(w, x, Y[w], tau)
                Axx, Axy, Ayy = Hw[:nx, :nx], Hw[:nx, nx:], Hw[nx:, nx:]
                gxw, gyw = gw[:nx], gw[nx:]
                try:
                    L = np.linalg.cholesky(Ayy)
                except np.linalg.LinAlgError:
                    reg = 1e-12 * np.trace(Ayy) / len(Ayy)
                    while True:
                        try:
                            L = np.linalg.cholesky(Ayy + reg * np.eye(len(Ayy))); break
                        except np.linalg.LinAlgError:
                            reg *= 10
                Wy = np.linalg.solve(L, Axy.T)          # L^{-1} Ayx
                wg = np.linalg.solve(L, gyw)
                S += Axx - Wy.T @ Wy
                r += -gxw + Wy.T @ wg
                facts.append((L, Axy, gyw, gxw))
            S = 0.5 * (S + S.T)
            t_classical += time.perf_counter() - tc
            tr = time.perf_counter()
            if not (np.all(np.isfinite(S)) and np.all(np.isfinite(r))):
                import pickle as _pk; _pk.dump(dict(x=x, Y=Y, tau=tau, S=S, r=r, newton=total_newton), open('/tmp/nan_state.pkl', 'wb'))
                raise FloatingPointError('non-finite reduced system at Newton %d' % total_newton)
            try:
                dx, rinfo = reduced_solver(S, r, dict(tau=tau, newton=total_newton))
            except np.linalg.LinAlgError:
                dx, rinfo = np.full_like(r, np.nan), {}   # treated as a failed inexact solve (LU fallback below)
            t_reduced += time.perf_counter() - tr
            tc = time.perf_counter()
            dY = np.empty_like(Y)
            gfull = gx.copy()
            for w, (L, Axy, gyw, gxw) in enumerate(facts):
                rhs = -(gyw + Axy.T @ dx)
                dY[w] = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
                gfull += gxw
            # directional derivative (Newton decrement estimate)
            gdir = gfull @ dx + sum(facts[w][2] @ dY[w] for w in range(m.N))
            exact_dx = np.linalg.solve(S, r)
            rel_err = np.linalg.norm(dx - exact_dx) / max(np.linalg.norm(exact_dx), 1e-300)
            if not np.isfinite(gdir) or not np.all(np.isfinite(dx)) or gdir >= 0:
                # not a (finite) descent direction (possible with an inexact reduced solve)
                dx = exact_dx
                for w, (L, Axy, gyw, gxw) in enumerate(facts):
                    dY[w] = np.linalg.solve(L.T, np.linalg.solve(L, -(gyw + Axy.T @ dx)))
                gdir = gfull @ dx + sum(facts[w][2] @ dY[w] for w in range(m.N))
                fallback = True
            else:
                fallback = False
            lam2 = -gdir
            # per-variable-type error of the inexact step (diagnostics only)
            dY_ex = np.empty_like(Y)
            for w, (L, Axy, gyw, gxw) in enumerate(facts):
                dY_ex[w] = np.linalg.solve(L.T, np.linalg.solve(L, -(gyw + Axy.T @ exact_dx)))
            k = m.k; F = m.F
            ey = (dY - dY_ex).reshape(m.N, m.T, k)
            diag = dict(err_p=np.abs(dx - exact_dx).max(), err_v0=np.abs(ey[..., 0]).max(),
                        err_q=np.abs(ey[..., 1:1 + F]).max(), err_d=np.abs(ey[..., 1 + F:]).max(),
                        nrm_p=np.abs(exact_dx).max(), nrm_v0=np.abs(dY_ex.reshape(m.N, m.T, k)[..., 0]).max())
            ue = (dx - exact_dx).reshape(F, m.T)[None] - ey[..., 1 + F:].transpose(0, 2, 1)
            ecap = np.array([f['ecap'] for f in m.fl])
            diag['err_soc'] = np.abs(np.cumsum(ue, axis=2) / ecap[None, :, None]).max()
            f0 = B.value(x, Y, tau)
            # fraction-to-boundary for the linear rows, then backtracking
            a = 1.0
            sx = m.hx - m.Gx @ x; gdx = m.Gx @ dx
            msk = gdx > 0
            if np.any(msk):
                a = min(a, 0.99 * np.min(sx[msk] / gdx[msk]))
            for w in range(m.N):
                z = np.concatenate([x, Y[w]]); dz = np.concatenate([dx, dY[w]])
                s_ = m.h[w] - B.Gs @ z; gd = B.Gs @ dz
                msk = gd > 0
                if np.any(msk):
                    a = min(a, 0.99 * np.min(s_[msk] / gd[msk]))
            quad = lam2 < 0.1   # quadratic-convergence region: accept any feasible decrease
            nl0 = B.nl_slacks(x, Y)
            while a > 1e-14:
                xn, Yn = x + a * dx, Y + a * dY
                fn = B.value(xn, Yn, tau)
                # fraction-to-boundary for the nonlinear constraints: keep at least 1 % of each slack
                if np.isfinite(fn) and np.all(B.nl_slacks(xn, Yn) >= 0.01 * nl0) and \
                        (fn <= f0 + 0.25 * a * gdir or (quad and fn <= f0 + 1e-12 * abs(f0))):
                    break
                a *= 0.5
            if a > 1e-14:
                x, Y = xn, Yn
            t_classical += time.perf_counter() - tc
            total_newton += 1
            rec = dict(newton=total_newton, tau=tau, lam2=lam2, step=a, rel_err=rel_err,
                       fallback=fallback, cond=np.linalg.cond(S), obj=m.objective(x, Y))
            rec.update({kk: vv for kk, vv in rinfo.items() if np.isscalar(vv)})
            rec.update(diag)
            if log is not None:
                log.append(rec)
            if verbose:
                print(f"  N{total_newton:3d} tau={tau:.1e} lam2={lam2:.2e} a={a:.3f} err={rel_err:.2e} obj={rec['obj']:.4f}")
            if lam2 / 2 < newton_tol:
                break
            if max_total_newton is not None and total_newton >= max_total_newton:
                break
        if B.ncons / tau < gap_tol:
            break
        if max_total_newton is not None and total_newton >= max_total_newton:
            break
        tau *= mu
    info = dict(newton=total_newton, obj=m.objective(x, Y), gap=B.ncons / tau,
                t_classical=t_classical, t_reduced=t_reduced)
    return x, Y, info
