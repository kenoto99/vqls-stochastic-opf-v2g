"""Variational quantum linear solver (VQLS) for the reduced Schur system.

* Reduced system is Jacobi-scaled, zero-padded to 2^n and normalised.
* A is decomposed into Pauli strings (only strings with an even number of Y appear for a
  real symmetric matrix).  The VQLS global cost (Bravo-Prieto et al.)
      C_G = 1 - |<b|A|psi>|^2 / <psi|A^2|psi>
  is evaluated from (i) the Pauli expansion of M = A^2, measured in qubit-wise-commuting groups,
  and (ii) a Hadamard test for the overlap o = <v^|psi>, v = A b.
* Ansatz: hardware-efficient real-amplitude circuit (R_y layers + linear CNOT chain).
* Modes: 'exact' (state-vector expectation values, equivalent to infinite shots),
         'shots' (multinomial sampling of every measurement group + binomial Hadamard test),
         'noisy' (density-matrix simulation with depolarising gate noise and readout error,
                  optionally with zero-noise extrapolation by unitary gate folding).
"""
import time
import numpy as np
from scipy.optimize import minimize

# ---------------------------------------------------------------- Pauli algebra
_I = np.eye(2); _X = np.array([[0, 1], [1, 0]]); _Y = np.array([[0, -1j], [1j, 0]]); _Z = np.diag([1., -1.])


def pauli_decompose(A, tol=1e-12):
    """Return list of (label, coeff) with A = sum c_P P (A real symmetric, size 2^n)."""
    n = int(np.log2(A.shape[0])); d = 2 ** n
    idx = np.arange(d)
    terms = []
    for xm in range(d):            # X-part bitmask
        cols = idx ^ xm
        a = A[idx, cols]           # A[i, i^xm]
        for zm in range(d):        # Z-part bitmask
            # P = i^{#Y} X^xm Z^zm ; entry P[i^xm, i] ... use Tr(P A)/d
            ny = bin(xm & zm).count('1')
            if ny % 2:             # imaginary Paulis vanish for real symmetric A
                continue
            sign = (-1) ** np.array([bin(i & zm).count('1') for i in idx])
            # P = (i)^{ny} X^xm Z^zm has elements P[j, i] = i^ny (-1)^{popcount(i&zm)} for j=i^xm
            ph = (1j) ** ny
            c = (ph * np.sum(sign * a)).real / d   # Tr(P A) = sum_i P[i^xm,i]... A symmetric
            if abs(c) > tol:
                lab = ''.join('I' if not ((xm >> q) & 1) and not ((zm >> q) & 1) else
                              'X' if ((xm >> q) & 1) and not ((zm >> q) & 1) else
                              'Z' if not ((xm >> q) & 1) and ((zm >> q) & 1) else 'Y'
                              for q in range(n))
                terms.append((lab, c, xm, zm))
    return terms


def pauli_matrix(lab):
    M = np.array([[1.]])
    for ch in lab[::-1]:  # qubit 0 is least significant
        M = np.kron(M, {'I': _I, 'X': _X, 'Y': _Y, 'Z': _Z}[ch])
    return M


def qwc_groups(labels):
    """Greedy qubit-wise-commuting grouping (bit-mask implementation).
    Returns list of (basis string, member indices)."""
    n = len(labels[0])
    xs = np.array([sum(1 << q for q, c in enumerate(l) if c in 'XY') for l in labels])
    zs = np.array([sum(1 << q for q, c in enumerate(l) if c in 'ZY') for l in labels])
    order = np.argsort([-sum(c != 'I' for c in l) for l in labels])  # heaviest first
    gx, gz, members = [], [], []
    for i in order:
        x, z = xs[i], zs[i]
        sup = x | z
        placed = False
        if gx:
            GX = np.array(gx); GZ = np.array(gz)
            gs = GX | GZ
            conflict = (sup & gs) & ((x ^ GX) | (z ^ GZ))
            ok = np.where(conflict == 0)[0]
            if len(ok):
                k = ok[0]
                gx[k] |= x; gz[k] |= z; members[k].append(int(i)); placed = True
        if not placed:
            gx.append(x); gz.append(z); members.append([int(i)])
    out = []
    for X_, Z_, mem in zip(gx, gz, members):
        lab = ''.join('Y' if (X_ >> q) & 1 and (Z_ >> q) & 1 else 'X' if (X_ >> q) & 1 else 'Z' if (Z_ >> q) & 1 else 'I'
                      for q in range(n))
        out.append((lab, mem))
    return out


# ---------------------------------------------------------------- ansatz (state vector)
class Ansatz:
    def __init__(self, n, layers):
        self.n, self.L = n, layers
        self.npar = n * (layers + 1)
        d = 2 ** n
        idx = np.arange(d)
        perm = idx.copy()
        for q in range(n - 1):     # CNOT(q -> q+1) chain applied in order
            c = (perm >> q) & 1
            perm = perm ^ (c << (q + 1))
        # state_out[perm[i]] = state_in[i]
        self.perm = perm
        self.inv = np.argsort(perm)
        self.n_cnot = (n - 1) * layers
        self.n_ry = self.npar
        self.depth = layers * n + 1  # Ry layer (1) + CNOT chain (n-1) per layer, plus final Ry layer

    def _ry_layer(self, psi, th):
        n = self.n
        s = psi.reshape([2] * n)
        for q in range(n):
            c, sn = np.cos(th[q] / 2), np.sin(th[q] / 2)
            ax = n - 1 - q
            s0 = np.take(s, 0, axis=ax); s1 = np.take(s, 1, axis=ax)
            s = np.stack([c * s0 - sn * s1, sn * s0 + c * s1], axis=ax)
        return s.reshape(-1)

    def state(self, theta):
        n = self.n
        psi = np.zeros(2 ** n); psi[0] = 1.0
        th = theta.reshape(self.L + 1, n)
        for l in range(self.L):
            psi = self._ry_layer(psi, th[l])
            out = np.empty_like(psi); out[self.perm] = psi; psi = out
        psi = self._ry_layer(psi, th[self.L])
        return psi

    def state_and_grad(self, theta, gpsi_fn):
        """Return f and gradient via adjoint differentiation (identical to parameter-shift)."""
        n = self.n
        th = theta.reshape(self.L + 1, n)
        states = []
        psi = np.zeros(2 ** n); psi[0] = 1.0
        for l in range(self.L + 1):
            # within an Ry layer apply qubit by qubit, store intermediate
            for q in range(n):
                states.append(psi)
                psi = self._ry_one(psi, q, th[l, q])
            if l < self.L:
                out = np.empty_like(psi); out[self.perm] = psi; psi = out
        f, lam = gpsi_fn(psi)
        grad = np.zeros_like(th)
        k = len(states) - 1
        for l in range(self.L, -1, -1):
            if l < self.L:
                lam = lam[self.perm]   # adjoint of permutation
            for q in range(n - 1, -1, -1):
                pin = states[k]; k -= 1
                dpsi = self._dry_one(pin, q, th[l, q])
                grad[l, q] = lam @ dpsi
                lam = self._ry_one(lam, q, -th[l, q])
        return f, grad.reshape(-1), psi

    def _ry_one(self, psi, q, a):
        n = self.n
        s = psi.reshape([2] * n); ax = n - 1 - q
        c, sn = np.cos(a / 2), np.sin(a / 2)
        s0 = np.take(s, 0, axis=ax); s1 = np.take(s, 1, axis=ax)
        return np.stack([c * s0 - sn * s1, sn * s0 + c * s1], axis=ax).reshape(-1)

    def _dry_one(self, psi, q, a):
        n = self.n
        s = psi.reshape([2] * n); ax = n - 1 - q
        c, sn = -0.5 * np.sin(a / 2), 0.5 * np.cos(a / 2)
        s0 = np.take(s, 0, axis=ax); s1 = np.take(s, 1, axis=ax)
        return np.stack([c * s0 - sn * s1, sn * s0 + c * s1], axis=ax).reshape(-1)

    # ------------- density-matrix simulation with depolarising noise (tensor form)
    def density(self, theta, p1, p2, fold=1):
        """Noisy density matrix.  After every R_y gate a single-qubit depolarising channel
        rho -> (1-p1) rho + p1/3 sum_{P in X,Y,Z} P rho P; after every CNOT a two-qubit channel
        rho -> (1-p2) rho + p2 Tr_ab(rho) (x) I/4.  fold = 1,3,5 applies unitary folding
        G (G^dag G)^((fold-1)/2) to every gate (each copy followed by its noise channel)."""
        n = self.n; d = 2 ** n
        rho = np.zeros((d, d)); rho[0, 0] = 1.0
        th = theta.reshape(self.L + 1, n)
        idx = np.arange(d)
        cperm = [idx ^ (((idx >> q) & 1) << (q + 1)) for q in range(n - 1)]
        for l in range(self.L + 1):
            for q in range(n):
                for k in range(fold):
                    a = th[l, q] if k % 2 == 0 else -th[l, q]
                    rho = _apply_1q(rho, n, q, _ry(a))
                    rho = _dep1(rho, n, q, p1)
            if l < self.L:
                for q in range(n - 1):
                    for k in range(fold):
                        pm = cperm[q]
                        rho = rho[np.ix_(pm, pm)]
                        rho = _dep2(rho, n, q, q + 1, p2)
        return rho


def _apply_1q(rho, n, q, U):
    t = rho.reshape([2] * (2 * n))
    ax = n - 1 - q
    t = np.moveaxis(np.tensordot(U, t, axes=([1], [ax])), 0, ax)
    t = np.moveaxis(np.tensordot(U.conj(), t, axes=([1], [n + ax])), 0, n + ax)
    return t.reshape(2 ** n, 2 ** n)


def _replace_mixed(rho, n, qs):
    """Tr_qs(rho) (x) I/2^|qs| placed back on qubits qs."""
    t = rho.reshape([2] * (2 * n))
    axes = [n - 1 - q for q in qs]
    out = np.zeros_like(t)
    tr = t
    for ax in sorted(axes, reverse=True):
        nn = tr.ndim // 2
        tr = np.trace(tr, axis1=ax, axis2=ax + nn)
    # rebuild with identity on the traced qubits
    import itertools
    for bits in itertools.product((0, 1), repeat=len(axes)):
        sl = [slice(None)] * (2 * n)
        for ax, b in zip(axes, bits):
            sl[ax] = b; sl[n + ax] = b
        out[tuple(sl)] = tr / (2 ** len(axes))
    return out.reshape(2 ** n, 2 ** n)


def _dep1(rho, n, q, p):
    if p <= 0:
        return rho
    # (1-p) rho + p/3 sum PρP = (1-4p/3) rho + (4p/3) Tr_q(rho) (x) I/2
    return (1 - 4 * p / 3) * rho + (4 * p / 3) * _replace_mixed(rho, n, [q])


def _dep2(rho, n, q1, q2, p):
    if p <= 0:
        return rho
    return (1 - p) * rho + p * _replace_mixed(rho, n, [q1, q2])


def _ry(a):
    c, s = np.cos(a / 2), np.sin(a / 2)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------- solver
class VQLS:
    def __init__(self, layers=None, optimizer='L-BFGS-B', maxiter=3000, tol=1e-12, seed=0,
                 warm=True, mode='exact', shots=8192, p1=0.0, p2=0.0, p_ro=0.0, zne=False,
                 restarts=0, rem=False):
        self.layers, self.optimizer, self.maxiter, self.tol = layers, optimizer, maxiter, tol
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.warm, self.mode, self.shots = warm, mode, shots
        self.p1, self.p2, self.p_ro, self.zne = p1, p2, p_ro, zne
        self.shots = shots
        self.rem = rem
        self.theta = None
        self.restarts = restarts
        self.records = []

    # -- problem preparation
    def prepare(self, S, r):
        m = len(r)
        n = int(np.ceil(np.log2(m)))
        d = 2 ** n
        dg = np.abs(np.diag(S))
        dj = 1 / np.sqrt(np.maximum(dg, 1e-14 * dg.max()))   # guard against round-off near the end of a centering step
        Sh = S * dj[:, None] * dj[None, :]
        A = np.eye(d); A[:m, :m] = Sh
        nrm = np.linalg.norm(A, 2)
        A = A / nrm
        rp = np.zeros(d); rp[:m] = r * dj
        b = rp / np.linalg.norm(rp)
        return dict(n=n, d=d, m=m, A=A, b=b, rnorm=np.linalg.norm(rp), nrm=nrm, dj=dj,
                    v=A @ b, M=A @ A)

    def solve(self, S, r, ctx=None):
        P = self.prepare(S, r)
        n = P['n']
        L = self.layers if self.layers is not None else 2 * n
        if not hasattr(self, 'ans') or self.ans.n != n or self.ans.L != L:
            self.ans = Ansatz(n, L); self.theta = None
        ans = self.ans
        A, b, v, M = P['A'], P['b'], P['v'], P['M']

        def fg(psi):
            mv = M @ psi; e_m = psi @ mv; o = v @ psi
            f = 1 - o * o / e_m
            gpsi = -(2 * o * v * e_m - o * o * 2 * mv) / e_m ** 2
            return f, gpsi

        def fun(th):
            f, gr, _ = ans.state_and_grad(th, fg)
            return f, gr

        t0 = time.perf_counter()
        nfev = [0]
        def fun_c(th):
            nfev[0] += 1
            return fg(ans.state(th))[0]
        starts = []
        if self.warm and self.theta is not None:
            starts.append(self.theta.copy())
        while len(starts) < 1 + self.restarts:
            starts.append(self.rng.uniform(-np.pi, np.pi, ans.npar) * 0.5)
        best = None
        if self.maxiter == 0 and self.theta is not None:   # read-out only (parameters already trained)
            from types import SimpleNamespace
            best = SimpleNamespace(x=self.theta.copy(), fun=fg(ans.state(self.theta))[0])
            starts = []
        for th0 in starts:
            if self.optimizer == 'COBYLA':
                res = minimize(fun_c, th0, method='COBYLA', options=dict(maxiter=self.maxiter, rhobeg=0.3, tol=self.tol))
            else:
                res = minimize(fun, th0, jac=True, method='L-BFGS-B',
                               options=dict(maxiter=self.maxiter, ftol=self.tol, gtol=1e-12, maxcor=30))
                nfev[0] += res.nfev
            if best is None or res.fun < best.fun:
                best = res
        if not np.all(np.isfinite(best.x)) or not np.isfinite(best.fun):
            # recover from a diverged warm start with a fresh random initialisation
            th0 = self.rng.uniform(-np.pi, np.pi, ans.npar) * 0.5
            best = minimize(fun, th0, jac=True, method='L-BFGS-B',
                            options=dict(maxiter=self.maxiter, ftol=self.tol, gtol=1e-12, maxcor=30))
            nfev[0] += best.nfev
        self.theta = best.x
        psi = ans.state(best.x)
        cost_exact = fg(psi)[0]
        # ----- read-out of the solution according to mode
        e_m, o, psi_hat = self.readout(P, best.x, psi)
        gamma = np.linalg.norm(v) * o / e_m
        y = P['rnorm'] * gamma * psi_hat / P['nrm']
        dx = P['dj'] * y[:P['m']]
        cost_est = 1 - (np.linalg.norm(v) * o) ** 2 / e_m
        info = dict(vqls_cost=cost_exact, vqls_cost_est=cost_est, vqls_nfev=nfev[0],
                    vqls_time=time.perf_counter() - t0, qubits=n, layers=L, npar=ans.npar,
                    gamma=gamma, pad_leak=float(np.linalg.norm(psi[P['m']:])))
        self.records.append(info)
        return dx, info

    def readout(self, P, theta, psi):
        """Returns <M>, signed overlap o = <v^|psi>, and the estimated solution state."""
        vh = P['v'] / np.linalg.norm(P['v'])
        n, d = P['n'], P['d']
        if self.mode == 'exact':
            return psi @ P['M'] @ psi, vh @ psi, psi
        sgn = np.sign(psi); sgn[sgn == 0] = 1.0
        if self.mode == 'shots':
            e_m = measure_pauli_sum(P, psi, self.shots, self.rng)
            k = self.rng.binomial(self.shots, (1 + vh @ psi) / 2)      # Hadamard test
            o = 2 * k / self.shots - 1
            cnt = self.rng.multinomial(self.shots, psi ** 2 / np.sum(psi ** 2))
            psi_hat = sgn * np.sqrt(cnt / self.shots)
            return e_m, o, psi_hat / np.linalg.norm(psi_hat)
        if self.mode == 'noisy':
            if 'terms_M' not in P:
                P['terms_M'] = pauli_decompose(P['M'], tol=1e-10)
            labels = [t[0] for t in P['terms_M']]; coefs = np.array([t[1] for t in P['terms_M']])
            wts = np.array([sum(ch != 'I' for ch in lab) for lab in labels])
            ro = (1 - 2 * self.p_ro) ** wts
            # Householder reflection mapping v^ to |0>, used for the compute-uncompute overlap test
            e0 = np.zeros(d); e0[0] = 1.0
            u = vh - e0; Hh = np.eye(d) - 2 * np.outer(u, u) / max(u @ u, 1e-300) if np.linalg.norm(u) > 1e-14 else np.eye(d)
            Cro = np.array([[1 - self.p_ro, self.p_ro], [self.p_ro, 1 - self.p_ro]])
            Cn = np.array([[1.]]); Cinv = np.array([[1.]])
            Cro_inv = np.linalg.inv(Cro)
            for _ in range(n):
                Cn = np.kron(Cn, Cro); Cinv = np.kron(Cinv, Cro_inv)
            folds = (1, 3, 5) if self.zne else (1,)
            vals_m, vals_o, vals_p = [], [], []
            for fo in folds:
                rho = self.ans.density(theta, self.p1, self.p2, fold=fo)
                cP = pauli_coeffs_of(rho, P['terms_M'])
                em_meas = np.sum(coefs * ro * cP * d)
                q0 = Cn @ np.diag(Hh @ rho @ Hh.T)
                if self.rem:
                    # read-out error mitigation with a calibrated symmetric confusion matrix:
                    # Pauli expectations are rescaled by (1-2p_ro)^-w, probabilities by C^{-1}
                    em_meas = np.sum(coefs * ro * cP * d / ro)
                    q0 = Cinv @ q0
                vals_m.append(em_meas)
                vals_o.append(q0[0])
                vals_p.append(Cn @ np.diag(rho))
            lam = np.array(folds, float)
            if self.zne:
                e_m = np.polyfit(lam, vals_m, 2)[-1]
                o2 = np.polyfit(lam, vals_o, 2)[-1]
                pr = np.polyfit(lam, np.array(vals_p), 2)[-1]
            else:
                e_m, o2, pr = vals_m[0], vals_o[0], vals_p[0]
            pr = np.clip(pr, 0, None); pr = pr / pr.sum()
            cnt = self.rng.multinomial(self.shots, pr)
            emp = cnt / self.shots
            if self.rem:
                emp = np.clip(Cinv @ emp, 0, None); emp = emp / emp.sum()
            psi_hat = sgn * np.sqrt(emp)
            o = np.sign(vh @ psi) * np.sqrt(max(o2, 0.0))
            return e_m, o, psi_hat / np.linalg.norm(psi_hat)
        raise ValueError(self.mode)


def pauli_coeffs_of(rho, terms):
    """Tr(P rho)/d for the given Pauli terms (rho real symmetric)."""
    d = rho.shape[0]; idx = np.arange(d)
    out = np.empty(len(terms))
    for k, (lab, c, xm, zm) in enumerate(terms):
        a = rho[idx, idx ^ xm]
        sign = 1 - 2 * (np.vectorize(lambda i: bin(i & zm).count('1') & 1)(idx))
        ny = bin(xm & zm).count('1')
        out[k] = ((1j) ** ny * np.sum(sign * a)).real / d
    return out


_GROUP_CACHE = {}


def measure_pauli_sum(P, psi, shots, rng):
    """Shot-based estimate of <psi|M|psi> with M Pauli-expanded and measured in QWC groups."""
    key = id(P['M'])
    M = P['M']; n = P['n']; d = P['d']
    terms = pauli_decompose(M, tol=1e-10)
    labels = [t[0] for t in terms]; coefs = np.array([t[1] for t in terms])
    groups = qwc_groups(labels)
    P['n_terms_M'] = len(terms); P['n_groups_M'] = len(groups)
    est = 0.0
    idx = np.arange(d)
    for basis, members in groups:
        # rotate into the measurement basis: X -> H, Y -> S^dag H
        U = np.array([[1.]])
        for ch in basis[::-1]:
            if ch == 'X':
                Uq = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            elif ch == 'Y':
                Uq = np.array([[1, -1j], [1, 1j]]) / np.sqrt(2)
            else:
                Uq = np.eye(2)
            U = np.kron(U, Uq)
        prob = np.abs(U @ psi) ** 2
        prob = prob / prob.sum()
        counts = rng.multinomial(shots, prob)
        for k in members:
            lab = labels[k]
            mask = sum(1 << q for q, ch in enumerate(lab) if ch != 'I')
            par = np.array([(-1) ** bin(i & mask).count('1') for i in idx])
            est += coefs[k] * (counts @ par) / shots
    return est
