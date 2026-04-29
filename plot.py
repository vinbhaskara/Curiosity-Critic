"""
Plot results from curiosity experiment.

Usage:
    python plot.py --input-dir results/ --output-dir figures/
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.family':        'serif',
    'font.size':          13,
    'axes.titlesize':     15,
    'axes.labelsize':     14,
    'legend.fontsize':    14,
    'xtick.labelsize':    12,
    'ytick.labelsize':    12,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.linewidth':     0.8,
    'grid.alpha':         0.25,
    'grid.linewidth':     0.5,
    'lines.linewidth':    2.4,
    'savefig.dpi':        300,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
})

METHOD_STYLE = {
    # (label, color, linestyle, zorder, linewidth)
    'random':                         ('Random',                         '#888888', ':',    5, 1.8),
    'curiosity_v1':                   ('Curiosity V1',                   '#d62728', '-',   10, 1.8),
    'curiosity_v2':                   ('Curiosity V2',                   '#ff7f0e', '-',   20, 1.8),
    'visitation_count':               ('Visitation Count',               '#9467bd', '-',   15, 1.8),
    'rnd_state':                      ('RND (State)',                    '#8c564b', '-',   18, 1.8),
    'rnd_observation':                ('RND (Observation)',              '#e377c2', '-',   19, 1.8),
    'curiosity_critic_ours_tabular_critic': ('Ours (Tabular Critic)',    '#6baed6', '-',   30, 1.8),
    'curiosity_critic_ours_nnet':     ('Ours (Neural Critic Model)',     '#1f77b4', '-',   35, 1.8),
    'curiosity_critic_ours_ideal':    ('Ours Oracle (Ground-Truth Critic)', '#2ca02c', ':', 25, 1.8),
}

METHODS_TO_PLOT = [
    'random', 'curiosity_v1', 'curiosity_v2', 'visitation_count',
    'rnd_state', 'rnd_observation',
    'curiosity_critic_ours_tabular_critic', 'curiosity_critic_ours_nnet',
    'curiosity_critic_ours_ideal',
]


def _fmt(x, _):
    return f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'


def load_results(input_dir):
    grouped = {}
    log_interval = None
    for path in sorted(Path(input_dir).glob('*.pkl')):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        method = data['method']
        if method == 'curiosity_critic_ours':
            method = 'curiosity_critic_ours_tabular_critic'
        if log_interval is None and 'config' in data:
            log_interval = data['config'].get('log_interval', 100)
        if method not in grouped:
            grouped[method] = {
                'det_error_trace': [], 'det_visit_frac_trace': [], 'seeds': [],
                'nnet_critic_det_trace': [], 'nnet_critic_stoch_trace': [],
            }
        grouped[method]['det_error_trace'].append(data['det_error_trace'])
        grouped[method]['det_visit_frac_trace'].append(data['det_visit_frac_trace'])
        grouped[method]['nnet_critic_det_trace'].append(data.get('nnet_critic_det_trace', []))
        grouped[method]['nnet_critic_stoch_trace'].append(data.get('nnet_critic_stoch_trace', []))
        grouped[method]['seeds'].append(data['seed'])
    return grouped, log_interval or 100


def _prep(grouped, methods, key, log_interval):
    out = []
    for m in methods:
        if m not in grouped:
            continue
        label, color, ls, zo, lw = METHOD_STYLE[m]
        traces  = grouped[m][key]
        min_len = min(len(t) for t in traces)
        raw     = np.array([t[:min_len] for t in traces])
        mean    = raw.mean(axis=0)
        std     = raw.std(axis=0)
        n       = raw.shape[0]
        x       = np.arange(1, min_len + 1) * log_interval
        out.append((m, label, color, ls, zo, lw, x, mean, std, n))
    return out


def save_error(grouped, methods, output_dir, log_interval):
    data = _prep(grouped, methods, 'det_error_trace', log_interval)
    fig, ax = plt.subplots(figsize=(14, 6))

    for m, label, color, ls, zo, lw, x, mean, std, n in data:
        final = mean[-1]
        tag = f'{label}  [{final:.3f} \u00b1 {std[-1]:.3f}]' if n > 1 else f'{label}  [{final:.3f}]'
        ax.plot(x, mean, label=tag, color=color, linestyle=ls, linewidth=lw, zorder=zo)
        if n > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, zorder=zo - 1)

    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Mean L2 prediction error (deterministic cells)')
    ax.set_title('World-Model Quality in the Learnable Region', fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax.grid(True, axis='y')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
              loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, borderpad=0.6, handlelength=2.5, borderaxespad=0)
    fig.subplots_adjust(bottom=0.28)

    out = os.path.join(output_dir, 'error.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


_ZOOM_EXCLUDED = {'curiosity_v1', 'visitation_count', 'rnd_observation'}


def save_zoomed_error(grouped, methods, output_dir, log_interval, zoom_steps=10_000):
    data = _prep(grouped, methods, 'det_error_trace', log_interval)
    if not data:
        return

    si = max(0, len(data[0][6]) - zoom_steps // log_interval)

    fig, ax = plt.subplots(figsize=(14, 6))

    for m, label, color, ls, zo, lw, x, mean, std, n in data:
        if m in _ZOOM_EXCLUDED:
            continue
        final = mean[-1]
        tag = f'{label}  [{final:.3f} \u00b1 {std[-1]:.3f}]' if n > 1 else f'{label}  [{final:.3f}]'
        ax.plot(x[si:], mean[si:], label=tag, color=color, linestyle=ls, linewidth=lw, zorder=zo)
        if n > 1:
            ax.fill_between(x[si:], (mean - std)[si:], (mean + std)[si:],
                            color=color, alpha=0.10, zorder=zo - 1)

    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Mean L2 prediction error (deterministic cells)')
    ax.set_title('World-Model Quality — Last 10k Steps (excl Curiosity V1, Visitation, RND(Obs))', fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax.grid(True, axis='y')
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
              loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, borderpad=0.6, handlelength=2.5, borderaxespad=0)
    fig.subplots_adjust(bottom=0.28)

    out = os.path.join(output_dir, 'zoomed_error.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


def save_error_with_zoom(grouped, methods, output_dir, log_interval, zoom_steps=10_000):
    data = _prep(grouped, methods, 'det_error_trace', log_interval)
    if not data:
        return

    si = max(0, len(data[0][6]) - zoom_steps // log_interval)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(18, 6))

    handles, labels_list = [], []
    for m, label, color, ls, zo, lw, x, mean, std, n in data:
        final = mean[-1]
        tag = f'{label}  [{final:.3f} \u00b1 {std[-1]:.3f}]' if n > 1 else f'{label}  [{final:.3f}]'

        # full plot — all methods
        h, = ax_full.plot(x, mean, label=tag, color=color, linestyle=ls, linewidth=lw, zorder=zo)
        if n > 1:
            ax_full.fill_between(x, mean - std, mean + std, color=color, alpha=0.10, zorder=zo - 1)
        handles.append(h)
        labels_list.append(tag)

        # zoomed plot — exclude methods that stay high or don't resolve well
        if m in _ZOOM_EXCLUDED:
            continue
        ax_zoom.plot(x[si:], mean[si:], color=color, linestyle=ls, linewidth=lw, zorder=zo)
        if n > 1:
            ax_zoom.fill_between(x[si:], (mean - std)[si:], (mean + std)[si:],
                                 color=color, alpha=0.10, zorder=zo - 1)

    ax_full.set_xlabel('Environment steps')
    ax_full.set_ylabel('Mean L2 prediction error (deterministic cells)')
    ax_full.set_title('Full Training Run', fontweight='bold')
    ax_full.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax_full.grid(True, axis='y')

    ax_zoom.set_xlabel('Environment steps')
    ax_zoom.set_ylabel('Mean L2 prediction error (deterministic cells)')
    ax_zoom.set_title('Last 10k Steps (excl Curiosity V1, Visitation, RND(Obs))', fontweight='bold')
    ax_zoom.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax_zoom.grid(True, axis='y')

    fig.legend(handles, labels_list,
               loc='upper center', bbox_to_anchor=(0.5, 0.15),
               ncol=3, frameon=True, fancybox=False, edgecolor='#cccccc',
               borderpad=0.6, handlelength=2.5)
    fig.subplots_adjust(bottom=0.28, wspace=0.3)

    out = os.path.join(output_dir, 'error_w_zoom.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


def save_latex_table(grouped, methods, output_dir):
    """Write a LaTeX booktabs table of final det_error per method x seed."""
    all_seeds = sorted({s for m in methods if m in grouped for s in grouped[m]['seeds']})

    ORACLE_METHODS = {'curiosity_critic_ours_ideal'}

    rows = []
    for m in methods:
        if m not in grouped:
            continue
        label = METHOD_STYLE[m][0]
        is_oracle = m in ORACLE_METHODS
        seed_to_error = {s: t[-1] for s, t in zip(grouped[m]['seeds'], grouped[m]['det_error_trace'])}
        vals = [seed_to_error.get(s, float('nan')) for s in all_seeds]
        arr = np.array(vals)
        mean = float(np.nanmean(arr))
        std  = float(np.nanstd(arr))
        rows.append((label, vals, mean, std, is_oracle))

    seed_headers = ' & '.join(f'Seed {s}' for s in all_seeds)
    col_spec = r'@{\extracolsep{\fill}}l' + 'c' * len(all_seeds) + 'c'
    lines = [
        r'\begin{table}[h]',
        r'    \centering',
        r'    \begin{tabular*}{\textwidth}{' + col_spec + '}',
        r'        \toprule',
        f'        Method & {seed_headers} & Mean $\\pm$ Std \\\\',
        r'        \midrule',
    ]
    for label, vals, mean, std, is_oracle in rows:
        val_cells = ' & '.join(f'{v:.3f}' if not np.isnan(v) else '--' for v in vals)
        row = f'        {label} & {val_cells} & ${mean:.3f} \\pm {std:.3f}$ \\\\'
        if is_oracle:
            row = f'        \\textit{{{label}}} & ' + ' & '.join(f'\\textit{{{v:.3f}}}' if not np.isnan(v) else '--' for v in vals) + f' & \\textit{{{mean:.3f}}} $\\pm$ \\textit{{{std:.3f}}} \\\\'
        lines.append(row)
    lines += [
        r'        \bottomrule',
        r'    \end{tabular*}',
        r'    \vspace{0.5em}',
        r'    \caption{Final mean L2 prediction error on deterministic cells (lower is better).'
        r' \textit{Italicised rows} denote oracle methods with privileged environment knowledge'
        r' and are included for reference only.}',
        r'    \label{tab:final_error}',
        r'\end{table}',
    ]

    out = os.path.join(output_dir, 'final_error_table.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Saved: {out}')


import math as _math
_ORACLE_DET_CRITIC   = 0.0
_ORACLE_STOCH_CRITIC = _math.sqrt(200) * 0.5   # ≈ 7.07


def save_critic_convergence(grouped, output_dir, log_interval):
    """
    Two-panel figure showing neural critic estimate convergence vs. oracle:
      Left  panel : mean critic estimate over deterministic cells across seeds.
      Right panel : mean critic estimate over stochastic cells across seeds.

    Only 'curiosity_critic_ours_nnet' data is plotted (line + ±1-std band).
    Oracle values are shown as horizontal dashed lines.
    """
    nnet_key = 'curiosity_critic_ours_nnet'
    if nnet_key not in grouped:
        print('save_critic_convergence: no nnet data found, skipping.')
        return

    traces_det   = [t for t in grouped[nnet_key]['nnet_critic_det_trace']   if t]
    traces_stoch = [t for t in grouped[nnet_key]['nnet_critic_stoch_trace'] if t]

    if not traces_det or not traces_stoch:
        print('save_critic_convergence: nnet critic traces are empty, skipping.')
        return

    min_len_det   = min(len(t) for t in traces_det)
    min_len_stoch = min(len(t) for t in traces_stoch)

    arr_det   = np.array([t[:min_len_det]   for t in traces_det])
    arr_stoch = np.array([t[:min_len_stoch] for t in traces_stoch])

    mean_det,   std_det   = arr_det.mean(axis=0),   arr_det.std(axis=0)
    mean_stoch, std_stoch = arr_stoch.mean(axis=0), arr_stoch.std(axis=0)

    x_det   = np.arange(1, min_len_det   + 1) * log_interval
    x_stoch = np.arange(1, min_len_stoch + 1) * log_interval

    nnet_color  = METHOD_STYLE[nnet_key][1]   # '#1f77b4'

    fig, (ax_det, ax_stoch) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: deterministic cells ─────────────────────────────────────────────
    ax_det.plot(x_det, mean_det, color=nnet_color, linewidth=2.4,
                label='Ours (Neural Critic Model)')
    ax_det.fill_between(x_det, mean_det - std_det, mean_det + std_det,
                        color=nnet_color, alpha=0.15)
    ax_det.axhline(_ORACLE_DET_CRITIC, color='#2ca02c', linestyle='--', linewidth=1.8,
                   label=f'Oracle baseline = {_ORACLE_DET_CRITIC:.2f}')
    ax_det.set_xlabel('Environment steps')
    ax_det.set_ylabel('Mean critic estimate')
    ax_det.set_title('Critic Estimate — Deterministic Cells', fontweight='bold')
    ax_det.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax_det.grid(True, axis='y')
    ax_det.legend(frameon=True, fancybox=False, edgecolor='#cccccc')

    # ── Right: stochastic cells ───────────────────────────────────────────────
    ax_stoch.plot(x_stoch, mean_stoch, color=nnet_color, linewidth=2.4,
                  label='Ours (Neural Critic Model)')
    ax_stoch.fill_between(x_stoch, mean_stoch - std_stoch, mean_stoch + std_stoch,
                          color=nnet_color, alpha=0.15)
    ax_stoch.axhline(_ORACLE_STOCH_CRITIC, color='#2ca02c', linestyle='--', linewidth=1.8,
                     label=f'Oracle baseline ≈ {_ORACLE_STOCH_CRITIC:.2f}')
    ax_stoch.set_xlabel('Environment steps')
    ax_stoch.set_ylabel('Mean critic estimate')
    ax_stoch.set_title('Critic Estimate — Stochastic Cells', fontweight='bold')
    ax_stoch.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax_stoch.grid(True, axis='y')
    ax_stoch.legend(frameon=True, fancybox=False, edgecolor='#cccccc')

    fig.suptitle(
        'Neural Critic Convergence to Asymptotic Error Baseline\n'
        r'(mean $\pm$ 1 std across seeds)',
        fontsize=14, fontweight='bold', y=1.02,
    )
    fig.tight_layout()

    out = os.path.join(output_dir, 'critic_convergence.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',  required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--zoom-steps', type=int, default=10_000,
                        help='Number of final steps to show in the zoomed plot (default: 10000).')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    grouped, log_interval = load_results(args.input_dir)
    methods = [m for m in METHODS_TO_PLOT if m in grouped]

    save_error(grouped, methods, args.output_dir, log_interval)
    save_zoomed_error(grouped, methods, args.output_dir, log_interval, args.zoom_steps)
    save_error_with_zoom(grouped, methods, args.output_dir, log_interval, args.zoom_steps)
    save_latex_table(grouped, methods, args.output_dir)
    save_critic_convergence(grouped, args.output_dir, log_interval)

if __name__ == '__main__':
    main()
