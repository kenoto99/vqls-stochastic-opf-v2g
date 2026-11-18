"""Modified IEEE 33-bus feeder, AC backward-forward sweep, LinDistFlow model and
Monte Carlo scenario generator.  All network data are the standard Baran-Wu data
(no rescaling).  Per-unit base: S_base = 1 MVA, V_base = 12.66 kV.
"""
import numpy as np

SBASE_MVA = 1.0
VBASE_KV = 12.66
ZBASE = VBASE_KV ** 2 / SBASE_MVA  # ohm

# from, to, R (ohm), X (ohm)  (Baran and Wu, 1989)
BRANCH = np.array([
    [1, 2, .0922, .0470], [2, 3, .4930, .2511], [3, 4, .3660, .1864], [4, 5, .3811, .1941],
    [5, 6, .8190, .7070], [6, 7, .1872, .6188], [7, 8, 1.7114, 1.2351], [8, 9, 1.0300, .7400],
    [9, 10, 1.0440, .7400], [10, 11, .1966, .0650], [11, 12, .3744, .1238], [12, 13, 1.4680, 1.1550],
    [13, 14, .5416, .7129], [14, 15, .5910, .5260], [15, 16, .7463, .5450], [16, 17, 1.2890, 1.7210],
    [17, 18, .7320, .5740], [2, 19, .1640, .1565], [19, 20, 1.5042, 1.3554], [20, 21, .4095, .4784],
    [21, 22, .7089, .9373], [3, 23, .4512, .3083], [23, 24, .8980, .7091], [24, 25, .8960, .7011],
    [6, 26, .2030, .1034], [26, 27, .2842, .1447], [27, 28, 1.0590, .9337], [28, 29, .8042, .7006],
    [29, 30, .5075, .2585], [30, 31, .9744, .9630], [31, 32, .3105, .3619], [32, 33, .3410, .5302]])
NB = 33
NL = 32
FROM = BRANCH[:, 0].astype(int) - 1
TO = BRANCH[:, 1].astype(int) - 1
R = BRANCH[:, 2] / ZBASE
X = BRANCH[:, 3] / ZBASE

_LOAD_KW = {2: (100, 60), 3: (90, 40), 4: (120, 80), 5: (60, 30), 6: (60, 20), 7: (200, 100),
            8: (200, 100), 9: (60, 20), 10: (60, 20), 11: (45, 30), 12: (60, 35), 13: (60, 35),
            14: (120, 80), 15: (60, 10), 16: (60, 20), 17: (60, 20), 18: (90, 40), 19: (90, 40),
            20: (90, 40), 21: (90, 40), 22: (90, 40), 23: (90, 50), 24: (420, 200), 25: (420, 200),
            26: (60, 25), 27: (60, 25), 28: (60, 20), 29: (120, 70), 30: (200, 600), 31: (150, 70),
            32: (210, 100), 33: (60, 40)}
PLOAD = np.zeros(NB)
QLOAD = np.zeros(NB)
for b, (p, q) in _LOAD_KW.items():
    PLOAD[b - 1] = p / 1000.0
    QLOAD[b - 1] = q / 1000.0

# downstream incidence D[l, j] = 1 if bus j is downstream of (or equal to) the receiving end of branch l
D = np.zeros((NL, NB))
children = {i: [] for i in range(NB)}
for l in range(NL):
    children[FROM[l]].append(TO[l])


def _subtree(j):
    out = [j]
    for c in children[j]:
        out += _subtree(c)
    return out


for l in range(NL):
    D[l, _subtree(TO[l])] = 1.0
# path matrices: Rp[i, j] = sum of R over branches common to paths 1->i and 1->j
Rp = D.T @ np.diag(R) @ D
Xp = D.T @ np.diag(X) @ D


def bfs(p_inj_load, q_inj_load, v0, tol=1e-10, max_iter=100):
    """Vectorised backward-forward sweep.
    p_inj_load, q_inj_load : (B, 33) net LOAD (consumption positive) in pu.
    v0 : (B,) slack voltage magnitude in pu.
    Returns |V| (B,33), branch complex power at sending end S_l (B,32), loss (B,) pu,
    max nodal power-balance residual (B,) pu and iteration count.
    """
    p = np.atleast_2d(p_inj_load)
    q = np.atleast_2d(q_inj_load)
    B = p.shape[0]
    v0 = np.broadcast_to(np.asarray(v0, float), (B,)).astype(complex)
    s = p + 1j * q
    z = R + 1j * X
    v = np.repeat(v0[:, None], NB, axis=1)
    order = np.arange(NL)
    for it in range(max_iter):
        inj = np.conj(s / v)  # load current
        acc = inj.copy()
        ib = np.zeros((B, NL), complex)
        for l in order[::-1]:
            ib[:, l] = acc[:, TO[l]]
            acc[:, FROM[l]] += ib[:, l]
        vnew = np.empty_like(v)
        vnew[:, 0] = v0
        for l in order:
            vnew[:, TO[l]] = vnew[:, FROM[l]] - z[l] * ib[:, l]
        err = np.max(np.abs(vnew - v), axis=1)
        v = vnew
        if np.all(err < tol):
            break
    # final currents and flows at the converged voltages
    inj = np.conj(s / v)
    acc = inj.copy()
    ib = np.zeros((B, NL), complex)
    for l in order[::-1]:
        ib[:, l] = acc[:, TO[l]]
        acc[:, FROM[l]] += ib[:, l]
    s_send = v[:, FROM] * np.conj(ib)
    loss = np.sum(np.abs(ib) ** 2 * R, axis=1)
    # nodal power balance residual: S_in - S_out_children - S_load - losses on incoming branch
    s_recv = v[:, TO] * np.conj(ib)
    bal = np.zeros((B, NB), complex)
    np.add.at(bal, (slice(None), TO), s_recv)
    np.add.at(bal, (slice(None), FROM), -s_send)
    bal -= s
    bal[:, 0] = 0.0
    resid = np.max(np.abs(bal), axis=1)
    return np.abs(v), s_send, loss, resid, it + 1


def lindist_voltage(p_load, q_load, v0):
    """LinDistFlow voltage magnitude (B,33)."""
    p_load = np.atleast_2d(p_load)
    q_load = np.atleast_2d(q_load)
    return np.asarray(v0)[:, None] - p_load @ Rp.T - q_load @ Xp.T


# ---------------------------------------------------------------------------
# Case-study data (assumed study parameters; disclosed in Table II of the paper)
# ---------------------------------------------------------------------------
T = 24
DT = 1.0
LOAD_PROFILE = np.array([.55, .52, .50, .49, .50, .55, .64, .73, .80, .85, .88, .90,
                         .89, .88, .87, .88, .92, .97, 1.00, .98, .93, .84, .73, .63])
TARIFF = np.array([60.] * 8 + [90.] * 8 + [140.] * 5 + [90.] + [60.] * 2)  # USD/MWh
PV_BUS = 5            # bus 6 (0-based index 5)
PV_CAP = 1.0          # MWp
PV_SHAPE = np.clip(np.sin(np.pi * (np.arange(T) + 0.5 - 6.0) / 12.0), 0, None)

FLEETS = [
    # bus index (0-based), usable capacity MWh, Pmax MW, Smax MVA, kind
    dict(name='F18', bus=17, ecap=3.0, pmax=0.50, smax=0.55, kind='residential', n_ev=50),
    dict(name='F33', bus=32, ecap=2.4, pmax=0.40, smax=0.44, kind='workplace', n_ev=40),
    dict(name='F25', bus=24, ecap=2.4, pmax=0.40, smax=0.44, kind='residential', n_ev=40),
    dict(name='F14', bus=13, ecap=2.4, pmax=0.40, smax=0.44, kind='workplace', n_ev=40),
]
SOC_MIN, SOC_MAX, SOC_REQ = 0.20, 0.90, 0.50
ETA = 0.95
KAPPA = (1.0 / ETA - ETA) / 2.0     # symmetric conversion-loss coefficient
C_DEG = 27.0                         # USD/MWh throughput (130 USD/kWh / (2*3000*0.8))
V_MIN, V_MAX = 0.95, 1.05
V0_MIN, V0_MAX = 1.00, 1.05

# mean availability profiles (fraction of the fleet plugged in)
AVAIL_MEAN = {
    'residential': np.array([.92, .93, .93, .93, .92, .88, .75, .50, .35, .30, .30, .30,
                             .30, .30, .32, .38, .50, .65, .78, .85, .89, .91, .92, .92]),
    'workplace': np.array([.15, .12, .10, .10, .10, .12, .25, .55, .80, .88, .90, .90,
                           .88, .88, .88, .85, .70, .45, .25, .18, .15, .15, .15, .15]),
}


def thermal_limits():
    """Branch apparent-power ratings (pu): 1.5 x the base-case peak flow, at least 0.5 MVA."""
    v, s, _, _, _ = bfs(PLOAD[None], QLOAD[None], np.array([1.0]))
    return np.maximum(1.5 * np.abs(s[0]), 0.5)


SMAX_LINE = thermal_limits()


def sample_scenarios(n, seed, fleets):
    """Monte Carlo scenarios. Returns dict of arrays."""
    rng = np.random.default_rng(seed)
    # common load multiplier: AR(1), truncated to +/-12 %
    eps = np.zeros((n, T))
    eps[:, 0] = rng.normal(0, .05, n)
    for t in range(1, T):
        eps[:, t] = AR_PHI * eps[:, t - 1] + rng.normal(0, AR_SD, n)
    lm = 1 + np.clip(eps, -.12, .12)
    # bus-level idiosyncratic multiplier
    bus_noise = 1 + np.clip(rng.normal(0, .02, (n, T, NB)), -.06, .06)
    # PV: daily clearness Beta(4,1.5) with hourly perturbation
    kbar = rng.beta(4.0, 1.5, n)
    kpv = np.clip(kbar[:, None] + rng.normal(0, .08, (n, T)), 0, 1)
    pv = PV_CAP * PV_SHAPE[None] * kpv
    out = dict(load_mult=lm, bus_noise=bus_noise, pv=pv, avail=[], e0=[], drain=[])
    for f in fleets:
        am = AVAIL_MEAN[f['kind']]
        xi = np.clip(am[None] + rng.normal(0, .07, (n, T)), .05, 1.0)
        soc0 = np.clip(rng.normal(.50, .05, n), .40, .60)
        # daily driving energy: 7 kWh/EV mean, CV 15 %, removed while vehicles are away
        dday = np.clip(rng.normal(7e-3 * f['n_ev'], .15 * 7e-3 * f['n_ev'], n), 0, None)
        w = (1 - am) / (1 - am).sum()
        out['avail'].append(xi)
        out['e0'].append(soc0 * f['ecap'])
        out['drain'].append(dday[:, None] * w[None])
    for k in ('avail', 'e0', 'drain'):
        out[k] = np.stack(out[k], axis=1)  # (n, F, ...)
    out['p_load'] = PLOAD[None, None] * (LOAD_PROFILE[None, :, None] * lm[:, :, None]) * bus_noise
    out['q_load'] = QLOAD[None, None] * (LOAD_PROFILE[None, :, None] * lm[:, :, None]) * bus_noise
    out['p_load'][:, :, PV_BUS] -= pv  # PV as negative load (unity power factor)
    out['n'] = n
    return out


# ---------------------------------------------------------------------------
# Network / time-resolution configuration (base case = 33-bus, hourly)
# ---------------------------------------------------------------------------
_H_LOAD = LOAD_PROFILE.copy(); _H_TARIFF = TARIFF.copy()
_H_AVAIL = {k: v.copy() for k, v in AVAIL_MEAN.items()}
_FLEETS33 = [dict(f) for f in FLEETS]
_FLEETS69 = [
    dict(name='F65', bus=64, ecap=3.0, pmax=0.50, smax=0.55, kind='residential', n_ev=50),
    dict(name='F27', bus=26, ecap=2.4, pmax=0.40, smax=0.44, kind='workplace', n_ev=40),
]
NETWORK = '33'
AR_PHI, AR_SD = 0.8, 0.03


def _build_tree():
    global D, children, Rp, Xp
    D = np.zeros((NL, NB))
    children = {i: [] for i in range(NB)}
    for l in range(NL):
        children[FROM[l]].append(TO[l])
    for l in range(NL):
        D[l, _subtree(TO[l])] = 1.0
    Rp = D.T @ np.diag(R) @ D
    Xp = D.T @ np.diag(X) @ D


def configure(network='33', steps=24):
    """Select the feeder ('33' or '69', Baran-Wu data) and the number of periods per day."""
    global NETWORK, BRANCH, NB, NL, FROM, TO, R, X, PLOAD, QLOAD, T, DT, LOAD_PROFILE, TARIFF
    global PV_BUS, PV_SHAPE, AVAIL_MEAN, SMAX_LINE, FLEETS, AR_PHI, AR_SD
    import os, sys
    NETWORK = network
    if network == '33':
        br = np.array(_BR33); NB = 33
        PLOAD = np.zeros(NB); QLOAD = np.zeros(NB)
        for b, (p, q) in _LOAD_KW.items():
            PLOAD[b - 1] = p / 1000.0; QLOAD[b - 1] = q / 1000.0
        PV_BUS = 5; FLEETS = [dict(f) for f in _FLEETS33]
    elif network == '69':
        dat = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ieee69.npz'))
        br = dat['br']; bus = dat['bus']; NB = len(bus)
        PLOAD = bus[:, 1] / 1000.0; QLOAD = bus[:, 2] / 1000.0
        PV_BUS = 10; FLEETS = [dict(f) for f in _FLEETS69]
    else:
        raise ValueError(network)
    BRANCH = br; NL = len(br)
    FROM = br[:, 0].astype(int) - 1; TO = br[:, 1].astype(int) - 1
    R = br[:, 2] / ZBASE; X = br[:, 3] / ZBASE
    _build_tree()
    k = steps // 24
    T = steps; DT = 24.0 / steps
    LOAD_PROFILE = np.repeat(_H_LOAD, k); TARIFF = np.repeat(_H_TARIFF, k)
    AVAIL_MEAN = {kk: np.repeat(v, k) for kk, v in _H_AVAIL.items()}
    PV_SHAPE = np.clip(np.sin(np.pi * ((np.arange(T) + 0.5) * DT - 6.0) / 12.0), 0, None)
    # AR(1) load deviation with the same stationary variance and hourly autocorrelation
    AR_PHI = 0.8 ** DT
    AR_SD = 0.03 * np.sqrt((1 - AR_PHI ** 2) / (1 - 0.8 ** 2))
    SMAX_LINE = thermal_limits()


_BR33 = BRANCH.copy()
