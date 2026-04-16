"""
Analyze per-cell visitation patterns from experiment pickle files.

Produces:
  1. Visitation heatmaps (end of training, seed-averaged) for each method.
  2. Deterministic-region visit fraction over time (all methods).
  3. Visitation heatmaps at early / mid / late training windows.
  4. Summary statistics printed to stdout.

Usage:
    python analyze_visits.py --input-dir results/ --output-dir analysis/
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
    'font.size':          12,
    'axes.titlesize':     13,
    'axes.labelsize':     12,
    'legend.fontsize':    11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'savefig.dpi':        200,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.15,
})

METHOD_LABEL = {
    'random':                      'Random',
    'curiosity_v1':                'Curiosity V1',
    'curiosity_v2':                'Curiosity V2',
    'visitation_count':            'Visitation Count',
    'curiosity_critic_ours_tabular_critic':       'Ours (Tabular Critic)',
    'curiosity_critic_ours_nnet':  'Ours (Neural Critic Model)',
    'curiosity_critic_ours_ideal': 'Ours Oracle (Ground-Truth Critic)',
}

METHOD_ORDER = [
    'random', 'curiosity_v1', 'curiosity_v2',
    'curiosity_critic_ours_nnet', 'curiosity_critic_ours_ideal',
]

METHOD_COLOR = {
    'random':                              '#888888',
    'curiosity_v1':                        '#d62728',
    'curiosity_v2':                        '#ff7f0e',
    'visitation_count':                    '#9467bd',
    'curiosity_critic_ours_tabular_critic': '#6baed6',
    'curiosity_critic_ours_nnet':          '#1f77b4',
    'curiosity_critic_ours_ideal':         '#2ca02c',
}


def load_all(input_dir):
    grouped = {}
    for path in sorted(Path(input_dir).glob('*.pkl')):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        m = d['method']
        if m == 'curiosity_critic_ours':
            m = 'curiosity_critic_ours_tabular_critic'
        if m not in grouped:
            grouped[m] = []
        grouped[m].append(d)
    return grouped


def visit_heatmap(trajectory, grid_size, steps_slice=None):
    """Count visits per cell in trajectory[steps_slice]."""
    traj = trajectory[steps_slice] if steps_slice else trajectory
    counts = np.zeros((grid_size, grid_size), dtype=np.float64)
    for (r, c) in traj:
        counts[r, c] += 1
    return counts


def det_boundary(d):
    g = d['grid']
    return g['det_cols']  # number of deterministic columns (cols 0..det_cols-1)


# ── 1. Heatmaps: end of training, seed-averaged ───────────────────────────────

def plot_heatmaps_end(grouped, output_dir, grid_size=30):
    methods = [m for m in METHOD_ORDER if m in grouped]
    n = len(methods)
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 4.5))
    axes = np.array(axes).flatten()

    # Derive total steps from the trajectory length of the first available run
    first_method = methods[0]
    total_steps = len(grouped[first_method][0]['trajectory'])
    total_steps_k = f'{total_steps // 1000}k' if total_steps >= 1000 else str(total_steps)
    n_seeds = max(len(grouped[m]) for m in methods)

    for ax_idx, m in enumerate(methods):
        ax = axes[ax_idx]
        maps = []
        for d in grouped[m]:
            maps.append(visit_heatmap(d['trajectory'], grid_size))
        avg = np.mean(maps, axis=0)
        avg_pct = avg / avg.sum() * 100  # percent of time

        det_cols = det_boundary(grouped[m][0])
        im = ax.imshow(avg_pct, origin='upper', cmap='hot', aspect='equal')
        ax.axvline(x=det_cols - 0.5, color='cyan', linewidth=1.5, linestyle='--', label='Det boundary')
        ax.set_title(METHOD_LABEL[m], fontweight='bold')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        plt.colorbar(im, ax=ax, label='% time', fraction=0.046, pad=0.04)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f'Visitation Heatmaps — Full Training ({total_steps_k} steps, {n_seeds} seeds, % time)',
        fontweight='bold', y=1.01)
    fig.tight_layout()
    out = os.path.join(output_dir, 'heatmap_end.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ── 2. Det-region visit fraction over time ────────────────────────────────────

def plot_visit_frac(grouped, output_dir):
    fig, ax = plt.subplots(figsize=(12, 5))

    for m in METHOD_ORDER:
        if m not in grouped:
            continue
        traces = [d['det_visit_frac_trace'] for d in grouped[m]]
        min_len = min(len(t) for t in traces)
        arr = np.array([t[:min_len] for t in traces])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        n    = arr.shape[0]
        log_interval = grouped[m][0]['config'].get('log_interval', 100)
        x    = np.arange(1, min_len + 1) * log_interval

        label = METHOD_LABEL[m]
        color = METHOD_COLOR[m]
        ax.plot(x, mean, label=label, color=color, linewidth=2)
        if n > 1:
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    ax.axhline(0.5, color='black', linewidth=0.8, linestyle=':', label='50% (uniform)')
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Fraction of steps in deterministic region')
    ax.set_title('Fraction of Time Spent in Deterministic Region', fontweight='bold')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))
    ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
              loc='upper center', bbox_to_anchor=(0.5, -0.18),
              ncol=3, borderpad=0.6, handlelength=2.5)
    fig.subplots_adjust(bottom=0.32)

    out = os.path.join(output_dir, 'visit_frac.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ── 3. Heatmaps at early / mid / late windows ─────────────────────────────────

def plot_heatmaps_windows(grouped, output_dir, grid_size=30, total_steps=None):
    methods = [m for m in METHOD_ORDER if m in grouped]

    # Derive total steps from data if not provided
    if total_steps is None:
        total_steps = len(grouped[methods[0]][0]['trajectory'])

    ts = total_steps
    early_end  = min(5_000, ts)
    mid_start  = ts // 2 - 2_500
    mid_end    = ts // 2 + 2_500
    late_start = max(0, ts - 5_000)

    def _k(x): return f'{x // 1000}k' if x >= 1000 else str(x)

    windows = [
        (f'Early  (steps 0–{_k(early_end)})',              slice(0,          early_end)),
        (f'Mid    (steps {_k(mid_start)}–{_k(mid_end)})',  slice(mid_start,  mid_end)),
        (f'Late   (steps {_k(late_start)}–{_k(ts)})',      slice(late_start, ts)),
    ]
    n_seeds = max(len(grouped[m]) for m in methods)

    # rows = windows (early/mid/late), cols = methods
    fig, axes = plt.subplots(len(windows), len(methods),
                             figsize=(3.5 * len(methods), 4 * len(windows)))

    for row, (win_label, slc) in enumerate(windows):
        for col, m in enumerate(methods):
            ax = axes[row, col]
            det_cols = det_boundary(grouped[m][0])
            maps = []
            for d in grouped[m]:
                maps.append(visit_heatmap(d['trajectory'], grid_size, slc))
            avg = np.mean(maps, axis=0)
            total = avg.sum()
            avg_pct = avg / total * 100 if total > 0 else avg

            im = ax.imshow(avg_pct, origin='upper', cmap='hot', aspect='equal')
            ax.axvline(x=det_cols - 0.5, color='cyan', linewidth=1.2, linestyle='--')
            if col == 0:
                ax.set_ylabel(win_label, fontsize=10, fontweight='bold')
            if row == 0:
                ax.set_title(METHOD_LABEL[m], fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Visitation Heatmaps Across Training ({_k(ts)} steps total, {n_seeds} seeds, % time per window)',
        fontweight='bold', y=1.01)
    fig.tight_layout()

    # Draw blue outline box around the nnet column spanning all rows,
    # wide enough to include the colorbar and tall enough to include the column title.
    if 'curiosity_critic_ours_nnet' in methods:
        nnet_col = methods.index('curiosity_critic_ours_nnet')
        ax_top    = axes[0, nnet_col]
        ax_bottom = axes[-1, nnet_col]

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()

        # Top edge: include the column title text
        title_win = ax_top.title.get_window_extent(renderer=renderer)
        top_fig   = inv.transform((0, title_win.y1))[1]

        # Bottom edge: bottom of the bottom axes
        bb_bottom = ax_bottom.get_position()
        bot_fig   = bb_bottom.y0

        # Left edge: left of the nnet axes
        bb_top = ax_top.get_position()
        left_fig = bb_top.x0

        # Right edge: rightmost extent of the colorbar for each nnet-column row,
        # including tick labels (use get_tightbbox which accounts for text).
        # Colorbars are appended to fig.axes after all main axes in row-major order.
        n_main = len(windows) * len(methods)
        n_cols = len(methods)
        right_disp = 0.0
        for row_i in range(len(windows)):
            cb_idx = n_main + row_i * n_cols + nnet_col
            if cb_idx < len(fig.axes):
                cb_bb = fig.axes[cb_idx].get_tightbbox(renderer)
                if cb_bb is not None:
                    right_disp = max(right_disp, cb_bb.x1)
        right_fig = inv.transform((right_disp, 0))[0] if right_disp else bb_top.x1

        pad = 0.010
        rect = matplotlib.patches.FancyBboxPatch(
            (left_fig - pad, bot_fig - pad),
            (right_fig - left_fig) + 2 * pad,
            (top_fig   - bot_fig)  + 2 * pad,
            boxstyle='round,pad=0',
            linewidth=2.5,
            edgecolor='#1f77b4',
            facecolor='none',
            transform=fig.transFigure,
            clip_on=False,
            zorder=10,
        )
        fig.add_artist(rect)

    out = os.path.join(output_dir, 'heatmap_windows.png')
    fig.savefig(out)
    plt.close(fig)
    print(f'Saved: {out}')


# ── 4. Summary statistics ──────────────────────────────────────────────────────

def print_summary(grouped, total_steps=35000):
    print('\n' + '='*70)
    print('VISITATION SUMMARY  (seed-averaged)')
    print('='*70)

    windows = [
        ('Early  (0–5k)',   slice(0,      5_000)),
        ('Mid  (15–20k)',   slice(15_000, 20_000)),
        ('Late (30–35k)',   slice(30_000, 35_000)),
        ('Full (0–35k)',    slice(0,      35_000)),
    ]

    for m in METHOD_ORDER:
        if m not in grouped:
            continue
        print(f'\n{METHOD_LABEL[m]}')
        det_cols = det_boundary(grouped[m][0])
        grid_size = grouped[m][0]['grid']['grid_size']

        for win_label, slc in windows:
            fracs = []
            for d in grouped[m]:
                traj = d['trajectory'][slc]
                n_det = sum(1 for (r, c) in traj if c < det_cols)
                fracs.append(n_det / len(traj) * 100)
            arr = np.array(fracs)
            print(f'  {win_label}: {arr.mean():.1f}% ± {arr.std():.1f}% in det region')

        # unique cells visited
        unique_det, unique_stoch = [], []
        for d in grouped[m]:
            visited = set(d['trajectory'])
            unique_det.append(sum(1 for (r, c) in visited if c < det_cols))
            unique_stoch.append(sum(1 for (r, c) in visited if c >= det_cols))
        n_det_cells  = grid_size * det_cols
        n_stoch_cells = grid_size * (grid_size - det_cols)
        print(f'  Unique det  cells visited: {np.mean(unique_det):.0f} / {n_det_cells}  '
              f'({np.mean(unique_det)/n_det_cells*100:.1f}%)')
        print(f'  Unique stoch cells visited: {np.mean(unique_stoch):.0f} / {n_stoch_cells}  '
              f'({np.mean(unique_stoch)/n_stoch_cells*100:.1f}%)')

    print('\n' + '='*70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',  required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    grouped = load_all(args.input_dir)

    print_summary(grouped)
    plot_visit_frac(grouped, args.output_dir)
    plot_heatmaps_end(grouped, args.output_dir)
    plot_heatmaps_windows(grouped, args.output_dir)


if __name__ == '__main__':
    main()
