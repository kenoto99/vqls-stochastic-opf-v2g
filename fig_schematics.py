import numpy as np
from figstyle import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon

# ---------------- Fig. 1 network
pos = {}
for i in range(1, 19): pos[i] = (i - 1, 0.0)
for k, b in enumerate(range(19, 23)): pos[b] = (1 + k + 0.0, -1.3)
for k, b in enumerate(range(23, 26)): pos[b] = (2 + k + 0.0, 1.3)
for k, b in enumerate(range(26, 34)): pos[b] = (5 + k + 0.0, -2.6)
edges = [(i, i + 1) for i in range(1, 18)] + [(2, 19), (19, 20), (20, 21), (21, 22), (3, 23), (23, 24), (24, 25), (6, 26)] + \
        [(b, b + 1) for b in range(26, 33)]
fig, ax = plt.subplots(figsize=(TWO, 2.55))
for a, b in edges:
    (x1, y1), (x2, y2) = pos[a], pos[b]
    if y1 != y2 and x1 != x2:
        ax.plot([x1, x1, x2], [y1, y2, y2], color='k', lw=0.8)
    else:
        ax.plot([x1, x2], [y1, y2], color='k', lw=0.8)
for b, (x, y) in pos.items():
    special = b in (1, 6, 18, 33)
    ax.plot(x, y, 'o', ms=4.2 if special else 3.2, mfc='w' if not special else '#333333', mec='k', mew=0.6, zorder=3)
    dy = 0.33 if y >= 0 else -0.38
    if b in (19, 23, 26):
        dy = 0.33 if y > 0 else -0.38
    xo = 0.22 if b in (2, 3, 6) else 0.0
    if b == 33: dy = 0.33
    ax.text(x + xo, y + dy, str(b), ha='center', va='center', fontsize=6.5)
# substation
ax.add_patch(Rectangle((-1.35, -0.35), 1.0, 0.7, fc='w', ec='k', lw=0.8))
ax.text(-0.85, 0, 'OLTC', ha='center', va='center', fontsize=6)
ax.plot([-0.35, 0], [0, 0], 'k', lw=0.8)
ax.text(-0.8, 0.6, 'Substation\n12.66 kV', ha='center', va='bottom', fontsize=6.5)
def tag(b, text, color, up=True, dx=0.0):
    x, y = pos[b]
    yy = y + (0.95 if up else -0.95)
    ax.annotate('', xy=(x, y + (0.12 if up else -0.12)), xytext=(x + dx, yy), arrowprops=dict(arrowstyle='<->', lw=0.7, color=color))
    ax.text(x + dx, yy + (0.08 if up else -0.08), text, ha='center', va='bottom' if up else 'top', fontsize=6.5, color=color,
            bbox=dict(boxstyle='round,pad=0.25', fc='w', ec=color, lw=0.6))
tag(6, 'PV, 1.0 MWp', C[4], up=True, dx=0.6)
tag(18, 'Fleet F18 (residential)\n50 EVs, 3.0 MWh, 0.50 MW', C[0], up=True, dx=-1.2)
tag(33, 'Fleet F33 (workplace)\n40 EVs, 2.4 MWh, 0.40 MW', C[1], up=False, dx=0.0)
ax.text(16.6, -2.55, 'Extension cases:\nF25 at bus 25, F14 at bus 14', fontsize=6.3, ha='center', va='center', color='#444444')
ax.set_xlim(-1.6, 17.8); ax.set_ylim(-4.0, 2.6); ax.axis('off')
save(fig, 'fig01_network')

# ---------------- Fig. 2 workflow
fig, ax = plt.subplots(figsize=(TWO, 3.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')
def box(x, y, w, h, text, ec='k', fc='w', fs=6.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4,rounding_size=1.2', fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal', wrap=True)
def arr(x1, y1, x2, y2, txt=None, ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=7, lw=0.8, color='k', linestyle=ls))
    if txt:
        ax.text((x1 + x2) / 2 + 0.8, (y1 + y2) / 2, txt, fontsize=6, va='center')
cls = '#F2F2F2'; qc = '#DCEBF7'
# left column: classical preparation
box(1, 50, 26, 8, 'Inputs: IEEE 33-bus data, fleets,\ntariff, Monte Carlo scenarios $\\Omega$ (seeded)', fc=cls)
box(1, 37, 26, 9, 'Two-stage SAA model (Sec. II)\nfirst stage $\\mathbf{x}$: shared V2G schedule\nrecourse $\\mathbf{y}_\\omega$: $V_0$, $q$, $\\delta$', fc=cls)
box(1, 25, 26, 9, 'Successive linearization\nLinDistFlow + AC corrections\n$\\mathbf{c}_{\\omega,t}^{(k)}$ from the AC sweep', fc=cls)
box(1, 3, 26, 12, 'Nonlinear AC validation\n(backward-forward sweep):\npower-balance residuals,\nvoltage, thermal, SoC checks', fc=cls)
arr(14, 50, 14, 46.4); arr(14, 37, 14, 34.4)
# middle: IPM
box(33, 45, 30, 13, 'Barrier interior-point Newton step\nper scenario (parallel, classical):\n$\\mathbf{H}_{\\omega}$, $\\mathbf{g}_{\\omega}$, Cholesky of $\\mathbf{H}_{yy,\\omega}$', fc=cls)
box(33, 28, 30, 12, 'Schur complement onto $\\mathbf{x}$\n$\\mathbf{S}=\\mathbf{H}_{xx}-\\sum_\\omega \\mathbf{H}_{xy,\\omega}\\mathbf{H}_{yy,\\omega}^{-1}\\mathbf{H}_{yx,\\omega}$\ndimension $24F$, independent of $N_\\omega$', fc=cls)
box(33, 8, 30, 14, 'Back-substitution $\\Delta\\mathbf{y}_\\omega$, descent check,\nfraction-to-boundary line search;\nnext Newton step until $\\lambda^2/2<\\epsilon_N$,\nthen barrier update $\\tau\\leftarrow\\mu\\tau$', fc=cls)
arr(27, 29.5, 33, 50)
arr(48, 45, 48, 40.4)
# right: VQLS
box(69, 24, 30, 34, 'VQLS module (Sec. III-C)\n\nJacobi scaling, zero padding to $2^{n}$,\nnormalization $\\Rightarrow \\mathbf{A}, |b\\rangle$\n\nPauli expansion of $\\mathbf{A}^2$ (QWC groups)\nHadamard test for $\\langle\\hat v|\\psi(\\theta)\\rangle$\n\nCost $C_G(\\theta)$, optimizer update of $\\theta$\n(warm start between Newton steps)\n\nRead-out and rescaling $\\gamma$\n$\\Rightarrow \\Delta\\mathbf{x}$', fc=qc, fs=6.4)
arr(63, 34, 69, 34, None)
ax.text(64.0, 35.8, '$\\mathbf{S}, \\mathbf{r}$', fontsize=6.5)
arr(69, 26, 63, 17)
ax.text(64.5, 23, '$\\Delta\\mathbf{x}$', fontsize=6.5)

arr(33, 11, 27, 9); ax.text(27.4, 0.4, 'converged ($m/\\tau<\\epsilon_{gap}$)', fontsize=5.8)
ax.add_patch(FancyArrowPatch((14, 15.6), (14, 24.4), arrowstyle='-|>', mutation_scale=7, lw=0.8, color='k', linestyle='--'))
ax.text(15, 18.2, 'update corrections\nuntil $\\max|\\Delta \\mathbf{c}|<10^{-5}$', fontsize=5.8)
ax.text(64, 1.5, 'Classical layer', fontsize=6.2); ax.add_patch(Rectangle((60, 1.2), 3, 2, fc=cls, ec='k', lw=0.5))
ax.text(84, 1.5, 'Quantum (emulated)', fontsize=6.2); ax.add_patch(Rectangle((80, 1.2), 3, 2, fc=qc, ec='k', lw=0.5))
save(fig, 'fig02_workflow')
print('ok')
