"""Reference classical OPF (cvxpy + Clarabel interior-point) and Phase-I feasibility point."""
import numpy as np
import cvxpy as cp
import grid as g
from model import EPS_ABS


def _common(m, x, Y):
    N, nx = m.N, m.nx
    Gxp, Gyp = m.G[:, :nx], m.G[:, nx:]
    ones = np.ones((N, 1))
    def lift(A, v):  # rows of A applied to x, broadcast over scenarios
        return ones @ cp.reshape(A @ v, (1, A.shape[0]), order='C')
    Ux, Uy = m.U[:, :nx], m.U[:, nx:]
    Qy = m.Qf[:, nx:]
    u = lift(Ux, x) + Y @ Uy.T                 # (N, T*F)
    q = Y @ Qy.T
    Gz = lift(Gxp, x) + Y @ Gyp.T
    return u, q, Gz


def soc_lower_expr(m, u):
    """returns expression soclo_c - Cum(u + kappa*|u|_eps) per scenario (N, F*T)."""
    F, T = m.F, m.T
    Cum = np.zeros((F * T, T * F))
    for f in range(F):
        for t in range(T):
            Cum[f * T + t, [tau * F + f for tau in range(t + 1)]] = g.DT
    return Cum


def reference(m, solver='CLARABEL', verbose=False, tol=1e-9):
    N, nx, ny = m.N, m.nx, m.ny
    x = cp.Variable(nx); Y = cp.Variable((N, ny))
    u, q, Gz = _common(m, x, Y)
    Cum = soc_lower_expr(m, None)
    # epigraph variables for smooth |.|: s >= sqrt(v^2 + eps^2)
    su = cp.Variable((N, m.T * m.F))
    cons = [Gz <= m.h, m.Gx @ x <= m.hx,
            cp.square(u) + cp.square(q) <= m.rdisk]
    for w in range(N):
        cons.append(cp.norm(cp.vstack([u[w], EPS_ABS * np.ones(m.T * m.F)]), 2, axis=0) <= su[w])
        cons.append(Cum @ (u[w] + g.KAPPA * su[w]) <= m.soclo_c[w])
    # objective: 0.5||M z||^2 + f'z + c0
    M = np.vstack([np.sqrt(2 * m.rhoR)[:, None] * m.AP, np.sqrt(2 * m.rhoR)[:, None] * m.AQ,
                   np.sqrt(2 * m.w_v * g.DT) * m.Vmat, np.sqrt(2 * m.w_q * g.DT) * m.Qf, np.sqrt(2 * m.beta * g.DT) * m.Dd])
    Mx, My = M[:, :nx], M[:, nx:]
    ones = np.ones((N, 1))
    MZ = ones @ cp.reshape(Mx @ x, (1, M.shape[0]), order='C') + Y @ My.T
    obj = 0.5 * cp.sum_squares(MZ) / N + cp.sum(m.f[:, :nx] @ x) / N + cp.sum(cp.multiply(m.f[:, nx:], Y)) / N \
        + np.mean(m.c0) + m.c_deg * g.DT * cp.sum(su) / N
    prob = cp.Problem(cp.Minimize(obj), cons)
    kw = dict(tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol) if solver == 'CLARABEL' else {}
    prob.solve(solver=solver, verbose=verbose, **kw)
    return x.value, Y.value, prob


def phase1(m, margin_frac=0.5, solver='CLARABEL'):
    """Classical Phase I: maximise the smallest normalised slack of every inequality
    (Chebyshev-centre type point); used only to start the barrier method."""
    N, nx, ny = m.N, m.nx, m.ny
    x = cp.Variable(nx); Y = cp.Variable((N, ny)); t = cp.Variable()
    u, q, Gz = _common(m, x, Y)
    Cum = soc_lower_expr(m, None)
    su = cp.Variable((N, m.T * m.F))
    rn = np.maximum(np.linalg.norm(m.G, axis=1), 1e-9)
    cons = [Gz + t * rn[None] <= m.h, m.Gx @ x + t <= m.hx,
            cp.square(u) + cp.square(q) <= m.rdisk - t * 1e-2,
            t <= 0.02]
    for w in range(N):
        cons.append(cp.norm(cp.vstack([u[w], EPS_ABS * np.ones(m.T * m.F)]), 2, axis=0) <= su[w])
        cons.append(Cum @ (u[w] + g.KAPPA * su[w]) + t <= m.soclo_c[w])
    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver=solver)
    if t.value is None or t.value <= 0:
        raise RuntimeError(f'Phase I failed: t={t.value}, status={prob.status}')
    # pull slightly towards the centre to obtain a well-centred start
    return x.value, Y.value, t.value
