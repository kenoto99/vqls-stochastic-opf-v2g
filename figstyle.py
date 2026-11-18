import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': 0.6,
    'lines.linewidth': 1.0,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'legend.frameon': False,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
})
ONE = 3.5
TWO = 7.16
# colour-blind-safe palette (Okabe-Ito)
C = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']
LS = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
MK = ['o', 's', '^', 'D', 'v', 'x']


def save(fig, name):
    fig.savefig(f'figs/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'figs/{name}.png', bbox_inches='tight', dpi=600)
    plt.close(fig)
