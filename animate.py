"""
animate.py — Fast trajectory animation for the curiosity experiment.
=====================================================================

Modes:

  combined (default)
    One video total: all methods in a 3×4 grid, all seeds shown simultaneously
    as differently-coloured blobs within each method panel.
    Row 1: Random | Curiosity V1 | Curiosity V2 | Visitation Count
    Row 2: RND (State) | RND (Observation) | Ours (tabular) | Ours (Neural Critic)
    Row 3: Ours Oracle | empty | empty | Legend (cell types + seeds)

  single
    One video per (method, seed) pair.

Speed strategy
--------------
The static background is rendered once into a numpy uint8 array.  Each frame
is a plain array copy with agent blobs stamped via boolean mask indexing.
No matplotlib is used in the frame loop.  Frames stream directly to the video
file via imageio with no per-frame disk I/O.

Dependencies
------------
    pip install "imageio[ffmpeg]"

Usage
-----
    python animate.py --results-dir results/
    python animate.py --results-dir results/ --fps 200
    python animate.py --results-dir results/ --mode single --method curiosity_v1 --seed 1
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm


# ── Panel layout (combined mode) ──────────────────────────────────────────────

PANEL_LAYOUT = {
    'random':                       (0, 0),
    'curiosity_v1':                 (0, 1),
    'curiosity_v2':                 (0, 2),
    'visitation_count':             (0, 3),
    'rnd_state':                    (1, 0),
    'rnd_observation':              (1, 1),
    'curiosity_critic_ours_tabular_critic':        (1, 2),
    'curiosity_critic_ours_nnet':   (1, 3),   # primary method — green border
    'curiosity_critic_ours_ideal':  (2, 0),
    # (2, 3) is the shared legend (single panel width)
}

METHOD_DISPLAY = {
    'random':                      'Random',
    'curiosity_v1':                'Curiosity V1',
    'curiosity_v2':                'Curiosity V2',
    'visitation_count':            'Visitation Count',
    'rnd_state':                   'RND (State)',
    'rnd_observation':             'RND\n(Observation)',
    'curiosity_critic_ours_tabular_critic':       'Ours (Tabular Critic)',
    'curiosity_critic_ours_nnet':  'Ours (Neural\nCritic Model)',
    'curiosity_critic_ours_ideal': 'Ours Oracle\n(Ground-Truth Critic)',
}

# ── Seed colours (RGB) — chosen to avoid grid greens and greys ────────────────
# Distinct, vivid colours that stand out on both deterministic (green) and
# stochastic (grey) cell backgrounds.

SEED_COLOURS = [
    (220,  50,  47),   # red
    (255, 140,   0),   # orange
    (108,  52, 196),   # purple
    (  0, 168, 232),   # sky blue
    (220,  20, 140),   # deep pink
    ( 30, 180,  30),   # vivid green  (darker than cell green — still distinct)
    (255, 200,   0),   # yellow
    (  0, 180, 160),   # teal
]

# ── Plot style (matches plot.py) ─────────────────────────────────────────────
# (color, linestyle, zorder)
PLOT_METHOD_STYLE = {
    # (color, linestyle, zorder, linewidth)
    'random':                         ('#888888', ':',    5, 1.8),
    'curiosity_v1':                   ('#d62728', '-',   10, 1.8),
    'curiosity_v2':                   ('#ff7f0e', '-',   20, 1.8),
    'visitation_count':               ('#9467bd', '-',   15, 1.8),
    'rnd_state':                      ('#8c564b', '-',   18, 1.8),
    'rnd_observation':                ('#e377c2', '-',   19, 1.8),
    'curiosity_critic_ours_tabular_critic': ('#6baed6', '-',  30, 1.8),
    'curiosity_critic_ours_nnet':     ('#1f77b4', '-',   35, 1.8),
    'curiosity_critic_ours_ideal':    ('#2ca02c', ':',   25, 1.8),
}

# ── Layout / colour constants ─────────────────────────────────────────────────

_TITLE_H = 52    # pixels for the method-name bar at the top of each panel

_C_DET   = (168, 216, 168)
_C_STOCH = (208, 208, 208)
_C_START = ( 69, 123, 157)
_C_BG    = (248, 248, 248)
_C_LINE  = (160, 160, 160)
_C_TEXT  = ( 40,  40,  40)


# ── Font loader ───────────────────────────────────────────────────────────────

def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Try common system TrueType fonts; fall back to Pillow's built-in."""
    candidates = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ] if bold else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    try:
        return ImageFont.load_default(size=size)   # Pillow >= 10.0
    except TypeError:
        return ImageFont.load_default()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pad16(x: int) -> int:
    return x + (-x % 16)


def _is_det(r: int, c: int, grid: dict) -> bool:
    return (
        grid['det_row_start'] <= r < grid['det_row_start'] + grid['det_rows'] and
        grid['det_col_start'] <= c < grid['det_col_start'] + grid['det_cols']
    )


def _blob_offsets(cell_px: int) -> tuple:
    """
    Pre-compute (dy, dx) offset arrays for all pixels inside the agent disk.
    Used to stamp the blob at any arbitrary pixel centre — needed for
    sub-frame interpolation where the centre falls between cell boundaries.
    """
    radius = cell_px * 0.38
    r_int  = int(np.ceil(radius))
    ys, xs = np.ogrid[-r_int:r_int + 1, -r_int:r_int + 1]
    inside = ys ** 2 + xs ** 2 <= radius ** 2
    dys, dxs = np.where(inside)
    return dys - r_int, dxs - r_int   # offsets relative to centre


def _stamp(frame: np.ndarray, cy: int, cx: int,
           dys: np.ndarray, dxs: np.ndarray,
           colour: np.ndarray) -> None:
    """Stamp a pre-computed disk onto *frame* centred at (cy, cx)."""
    ys = cy + dys
    xs = cx + dxs
    valid = (ys >= 0) & (ys < frame.shape[0]) & \
            (xs >= 0) & (xs < frame.shape[1])
    frame[ys[valid], xs[valid]] = colour


# ── Panel builders ────────────────────────────────────────────────────────────

def _build_grid_panel(grid: dict, cell_px: int, method_label: str,
                      panel_h: int, panel_w: int) -> np.ndarray:
    gs = grid['grid_size']
    bg = np.full((panel_h, panel_w, 3), _C_BG, dtype=np.uint8)

    for r in range(gs):
        y0 = _TITLE_H + r * cell_px
        for c in range(gs):
            x0 = c * cell_px
            bg[y0:y0 + cell_px, x0:x0 + cell_px] = (
                _C_DET if _is_det(r, c, grid) else _C_STOCH
            )

    grid_h = gs * cell_px
    grid_w = gs * cell_px
    for i in range(gs + 1):
        y = _TITLE_H + i * cell_px
        if y < panel_h:
            bg[y, :grid_w] = _C_LINE
    for j in range(gs + 1):
        x = j * cell_px
        if x < panel_w:
            bg[_TITLE_H:_TITLE_H + grid_h, x] = _C_LINE

    # start-cell border (2 px, steel blue)
    sr, sc = 15, 15
    y0, x0 = _TITLE_H + sr * cell_px, sc * cell_px
    for t in range(2):
        bg[y0 + t,                x0:x0 + cell_px] = _C_START
        bg[y0 + cell_px - 1 - t, x0:x0 + cell_px] = _C_START
        bg[y0:y0 + cell_px, x0 + t               ] = _C_START
        bg[y0:y0 + cell_px, x0 + cell_px - 1 - t ] = _C_START

    img   = Image.fromarray(bg)
    draw  = ImageDraw.Draw(img)
    font  = _load_font(22, bold=True)
    lines = method_label.split('\n')
    line_h = 18
    y_text = (_TITLE_H - len(lines) * line_h) // 2
    for line in lines:
        draw.text((6, y_text), line, fill=_C_TEXT, font=font)
        y_text += line_h

    return np.array(img)


def _build_legend_panel(panel_h: int, panel_w: int,
                        seed_labels: list) -> np.ndarray:
    """
    Legend panel with cell-type and seed sections stacked vertically.

    Fits within a single grid-panel width.  Cell types are listed first,
    followed by a small gap, then seeds.

    seed_labels : list of (seed_number, rgb_colour) in seed order.
    """
    bg   = np.full((panel_h, panel_w, 3), _C_BG, dtype=np.uint8)
    img  = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)

    font_h = _load_font(22, bold=True)
    font_b = _load_font(19, bold=False)

    swatch   = 15
    gap      = 16          # gap between rows within a section
    sec_gap  = 22          # extra gap between the two sections
    lx       = max(10, (panel_w - 180) // 2)

    cell_items = [
        (_C_DET,   None,     "Deterministic"),
        (_C_STOCH, None,     "Stochastic"),
        (_C_BG,    _C_START, "Start cell"),
    ]

    # Total height of content for vertical centering
    n_cell_rows = 1 + len(cell_items)
    n_seed_rows = 1 + len(seed_labels)
    total_h = (n_cell_rows * (swatch + gap) + sec_gap +
               n_seed_rows * (swatch + gap))
    ly = max(8, (panel_h - total_h) // 2)

    # ── Cell types section ────────────────────────────────────────────────────
    draw.text((lx, ly), "Cell types", fill=_C_TEXT, font=font_h)
    ly += swatch + gap
    for fill, outline, text in cell_items:
        draw.rectangle([lx, ly, lx + swatch, ly + swatch],
                       fill=fill, outline=outline or fill)
        draw.text((lx + swatch + 8, ly + 1), text, fill=_C_TEXT, font=font_b)
        ly += swatch + gap

    ly += sec_gap

    # ── Seeds section ─────────────────────────────────────────────────────────
    draw.text((lx, ly), "Seeds", fill=_C_TEXT, font=font_h)
    ly += swatch + gap
    for seed_num, colour in seed_labels:
        draw.ellipse([lx, ly, lx + swatch, ly + swatch], fill=colour)
        draw.text((lx + swatch + 8, ly + 1), f"Seed {seed_num}",
                  fill=_C_TEXT, font=font_b)
        ly += swatch + gap

    return np.array(img)


# ── Background builders ───────────────────────────────────────────────────────

def _build_single_background(grid: dict, cell_px: int,
                              method_label: str) -> tuple:
    gs      = grid['grid_size']
    panel_h = _pad16(_TITLE_H + gs * cell_px)
    panel_w = _pad16(gs * cell_px)
    bg = _build_grid_panel(grid, cell_px, method_label, panel_h, panel_w)
    return bg, _TITLE_H


def _hex_to_rgb(hex_color: str) -> tuple:
    """Convert '#rrggbb' to (r, g, b) tuple."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _dash_pattern(linestyle: str) -> list:
    """
    Return a list of (on, off) pixel run lengths for each linestyle.
    Solid '-' returns [] meaning continuous fill.
    """
    return {
        '-':  [],
        '--': [(12, 6)],
        '-.': [(12, 5), (3, 5)],
        ':':  [(3, 6)],
    }.get(linestyle, [])


def _draw_border(canvas: np.ndarray, bx0: int, by0: int,
                 bx1: int, by1: int,
                 color: tuple, linestyle: str, thickness: int) -> None:
    """
    Draw a rectangular border on *canvas* in-place.

    Sides are drawn with the given color, line thickness (pixels), and
    dash pattern matching the matplotlib linestyle.
    """
    pattern = _dash_pattern(linestyle)
    rgb = np.array(color, dtype=np.uint8)

    def _fill_run(arr1d_indices, positions):
        """Apply dash pattern to a 1-D sequence of pixel positions."""
        if not pattern:
            # solid — fill all
            for pos in positions:
                arr1d_indices[pos] = True
            return
        # cycle through (on, off) segments
        cycle = []
        for on, off in pattern:
            cycle += [True] * on + [False] * off
        n = len(cycle)
        for i, pos in enumerate(positions):
            if cycle[i % n]:
                arr1d_indices[pos] = True

    # Top and bottom edges
    for edge_y_range in [range(by0, by0 + thickness), range(by1 - thickness + 1, by1 + 1)]:
        mask = np.zeros(bx1 - bx0 + 1, dtype=bool)
        _fill_run(mask, range(bx1 - bx0 + 1))
        xs = np.where(mask)[0] + bx0
        for y in edge_y_range:
            if 0 <= y < canvas.shape[0]:
                valid = xs[(xs >= 0) & (xs < canvas.shape[1])]
                canvas[y, valid] = rgb

    # Left and right edges
    for edge_x_range in [range(bx0, bx0 + thickness), range(bx1 - thickness + 1, bx1 + 1)]:
        mask = np.zeros(by1 - by0 + 1, dtype=bool)
        _fill_run(mask, range(by1 - by0 + 1))
        ys = np.where(mask)[0] + by0
        for x in edge_x_range:
            if 0 <= x < canvas.shape[1]:
                valid = ys[(ys >= 0) & (ys < canvas.shape[0])]
                canvas[valid, x] = rgb


def _build_combined_background(grid: dict, cell_px: int,
                                seed_labels: list) -> tuple:
    """
    Returns (canvas, title_h, panel_h, panel_w).

    Layout (3 rows × 4 columns):
      Row 0: Random | V1 | V2 | Visitation Count
      Row 1: RND (State) | RND (Observation) | Ours (tabular) | Ours (Neural Critic)
      Row 2: Ours Oracle | empty | empty | Legend

    Each method panel is outlined with a border whose color, linestyle, and
    thickness match its entry in PLOT_METHOD_STYLE.
    """
    gs      = grid['grid_size']
    panel_h = _pad16(_TITLE_H + gs * cell_px)
    panel_w = _pad16(gs * cell_px)

    panels = {}
    for method, (pr, pc) in PANEL_LAYOUT.items():
        label = METHOD_DISPLAY.get(method, method)
        panels[(pr, pc)] = _build_grid_panel(grid, cell_px, label,
                                              panel_h, panel_w)

    # Legend fits in a single panel width (col 3, row 2)
    legend = _build_legend_panel(panel_h, panel_w, seed_labels)
    blank = np.full((panel_h, panel_w, 3), _C_BG, dtype=np.uint8)

    row0 = np.concatenate([panels[(0, 0)], panels[(0, 1)],
                           panels[(0, 2)], panels[(0, 3)]], axis=1)
    row1 = np.concatenate([panels[(1, 0)], panels[(1, 1)],
                           panels[(1, 2)], panels[(1, 3)]], axis=1)
    row2 = np.concatenate([panels[(2, 0)], blank, blank, legend], axis=1)
    canvas = np.concatenate([row0, row1, row2], axis=0)

    # Draw per-method borders within the title bar only (rows 0.._TITLE_H per
    # panel). The title bar is padding above the grid, so borders never
    # overwrite grid cell pixels. Inset by 2px from panel edges so adjacent
    # panels' borders do not share pixels.
    inset = 2
    for method, (pr, pc) in PANEL_LAYOUT.items():
        if method not in PLOT_METHOD_STYLE:
            continue
        color_hex, ls, _, lw = PLOT_METHOD_STYLE[method]
        color_rgb = _hex_to_rgb(color_hex)
        thickness = max(2, round(lw))
        bx0 = pc * panel_w             + inset
        bx1 = (pc + 1) * panel_w - 1  - inset
        by0 = pr * panel_h             + inset
        by1 = pr * panel_h + _TITLE_H  - inset
        _draw_border(canvas, bx0, by0, bx1, by1, color_rgb, ls, thickness)

    return canvas, _TITLE_H, panel_h, panel_w


# ── Error-plot pre-renderer ───────────────────────────────────────────────────

def _prerender_plot_frames(all_error_traces: dict, log_interval: int,
                           plot_w: int, plot_h: int,
                           plot_window: int = 0) -> tuple:
    """
    Pre-render one matplotlib frame per error-trace checkpoint.

    all_error_traces : {method: {seed: [float, ...]}}
    Returns (frames, n_checkpoints) where *frames* is a list of uint8 arrays
    of shape (plot_h, plot_w, 3).
    """
    plt.rcParams.update({
        'font.family':        'serif',
        'font.size':          14,
        'axes.titlesize':     16,
        'axes.labelsize':     14,
        'legend.fontsize':    12,
        'xtick.labelsize':    13,
        'ytick.labelsize':    13,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'axes.linewidth':     0.8,
        'grid.alpha':         0.25,
        'grid.linewidth':     0.5,
        'lines.linewidth':    2.0,
        'savefig.dpi':        100,
    })

    # Aggregate per-method stats across seeds
    method_stats = {}
    for method, seed_traces in all_error_traces.items():
        traces = list(seed_traces.values())
        if not traces:
            continue
        min_len = min(len(t) for t in traces)
        raw  = np.array([t[:min_len] for t in traces], dtype=np.float64)
        method_stats[method] = {
            'mean': raw.mean(axis=0),
            'std':  raw.std(axis=0),
            'n':    len(traces),
            'x':    np.arange(1, min_len + 1) * log_interval,
        }

    if not method_stats:
        return [], 0

    n_checkpoints = max(len(s['mean']) for s in method_stats.values())
    total_steps   = max(st['x'][-1] for st in method_stats.values())
    # How many checkpoints fit in the requested window (0 = unlimited)
    window_ckpts = max(1, plot_window // log_interval) if plot_window > 0 else 0

    # Display order matches METHODS_TO_PLOT in plot.py
    _ORDER = ['random', 'curiosity_v1', 'curiosity_v2', 'visitation_count',
              'rnd_state', 'rnd_observation',
              'curiosity_critic_ours_tabular_critic', 'curiosity_critic_ours_nnet',
              'curiosity_critic_ours_ideal']
    methods_ord = [m for m in _ORDER if m in method_stats]

    dpi   = 100
    fig_w = plot_w / dpi
    fig_h = plot_h / dpi

    def _fmt(x, _):
        return f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'

    frames = []
    for ci in tqdm(range(n_checkpoints),
                   desc="  pre-rendering plot frames", unit="ckpt"):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

        for method in methods_ord:
            st    = method_stats[method]
            color, ls, zo, lw = PLOT_METHOD_STYLE[method]
            name  = METHOD_DISPLAY.get(method, method)
            start = max(0, ci + 1 - window_ckpts) if window_ckpts else 0
            x     = st['x'][start:ci + 1]
            mean  = st['mean'][start:ci + 1]
            std   = st['std'][start:ci + 1]
            n     = st['n']

            cur_mean = float(mean[-1]) if len(mean) else 0.0
            cur_std  = float(std[-1])  if len(std)  else 0.0
            tag = (f'{name}  [{cur_mean:.3f} \u00b1 {cur_std:.3f}]'
                   if n > 1 else f'{name}  [{cur_mean:.3f}]')

            ax.plot(x, mean, label=tag, color=color, linestyle=ls,
                    linewidth=lw, zorder=zo)
            if n > 1:
                ax.fill_between(x, mean - std, mean + std,
                                color=color, alpha=0.10, zorder=zo - 1)

        # Vertical marker for current training position
        ax.axvline(x=(ci + 1) * log_interval,
                   color='#888888', linewidth=0.8, linestyle='--', alpha=0.6)

        ax.set_xlabel(f'Environment steps (out of {_fmt(total_steps, None)})')
        ax.set_ylabel(
            f'Mean L2 prediction error (deterministic cells)\n'
            f'(logged every {log_interval} steps)')
        ax.set_title('World-Model Quality in the Learnable Region',
                     fontweight='bold')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
        ax.grid(True, axis='y')
        ax.legend(frameon=True, fancybox=False, edgecolor='#cccccc',
                  loc='lower left', borderpad=0.6, handlelength=2.0,
                  borderaxespad=0.5)

        fig.tight_layout(pad=0.8)
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        rgb = buf[:, :, :3]
        plt.close(fig)

        # Resize to exact panel dimensions (tight_layout may shift pixel size slightly)
        if rgb.shape[:2] != (plot_h, plot_w):
            rgb = np.array(
                Image.fromarray(rgb).resize((plot_w, plot_h), Image.LANCZOS))
        frames.append(rgb)

    return frames, n_checkpoints


# ── Animation writers ─────────────────────────────────────────────────────────

def _writer_kwargs(fps: int, fmt: str) -> dict:
    if fmt == 'mp4':
        return dict(fps=fps, codec='libx264', pixelformat='yuv420p',
                    output_params=['-r', str(fps)])
    return dict(duration=1.0 / fps)


def animate_single(pkl_path: Path, fps: int, cell_px: int, fmt: str,
                   smooth: int = -1) -> None:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    trajectory = data.get('trajectory')
    if not trajectory:
        print(f"  [skip] no trajectory in {pkl_path.name}")
        return

    grid   = data['grid']
    method = data['method']
    seed   = data['seed']
    label  = METHOD_DISPLAY.get(method, method)
    colour = np.array(SEED_COLOURS[seed % len(SEED_COLOURS)], dtype=np.uint8)

    bg, title_h = _build_single_background(grid, cell_px, label)
    dys, dxs    = _blob_offsets(cell_px)
    out_path    = pkl_path.with_name(pkl_path.stem + f"_animation.{fmt}")
    n_steps     = len(trajectory)
    n_sub       = max(1, smooth)
    internal_fps = fps * n_sub
    half        = cell_px // 2

    with imageio.get_writer(str(out_path), **_writer_kwargs(internal_fps, fmt)) as writer:
        for step_idx in tqdm(range(n_steps), desc="  rendering", unit="step"):
            r0, c0 = trajectory[step_idx]
            r1, c1 = trajectory[step_idx + 1] if step_idx + 1 < n_steps else (r0, c0)
            cy0 = title_h + r0 * cell_px + half
            cx0 = c0 * cell_px + half
            cy1 = title_h + r1 * cell_px + half
            cx1 = c1 * cell_px + half
            for t in range(n_sub):
                alpha = t / n_sub
                cy = round(cy0 + alpha * (cy1 - cy0))
                cx = round(cx0 + alpha * (cx1 - cx0))
                frame = bg.copy()
                _stamp(frame, cy, cx, dys, dxs, colour)
                writer.append_data(frame)

    total_frames = n_steps * n_sub
    print(f"  saved → {out_path.name}  "
          f"({n_steps:,} steps × {n_sub} sub-frames = {total_frames:,} frames, "
          f"{total_frames/internal_fps:.0f}s at {internal_fps} FPS)")


def animate_combined(all_trajectories: dict, grid: dict,
                     fps: int, cell_px: int, fmt: str,
                     output_dir: Path, smooth: int = -1,
                     all_error_traces: dict = None,
                     log_interval: int = 100,
                     plot_window: int = 0,
                     snapshot_every: int = 0) -> None:
    """
    all_trajectories  : {method: {seed: [(r,c), ...]}}
    all_error_traces  : {method: {seed: [float, ...]}}  (optional)

    Produces one video with all methods and all seeds visible simultaneously.
    When all_error_traces is provided a live error-plot panel is rendered on
    the right spanning the full canvas height.
    """
    all_seeds   = sorted({s for trajs in all_trajectories.values() for s in trajs})
    seed_labels = [(s, SEED_COLOURS[i % len(SEED_COLOURS)])
                   for i, s in enumerate(all_seeds)]
    seed_colour = {s: np.array(c, dtype=np.uint8) for s, c in seed_labels}

    combined_bg, title_h, panel_h, panel_w = _build_combined_background(
        grid, cell_px, seed_labels)
    dys, dxs     = _blob_offsets(cell_px)
    n_sub        = max(1, smooth)
    internal_fps = fps * n_sub
    half         = cell_px // 2

    n_steps = min(
        len(traj)
        for trajs in all_trajectories.values()
        for traj in trajs.values()
    )

    # ── Optional live error-plot panel ────────────────────────────────────────
    plot_frames    = None
    n_checkpoints  = 0
    plot_w         = 0
    canvas_h       = combined_bg.shape[0]   # 2 × panel_h

    if all_error_traces:
        plot_w = _pad16(panel_w * 2)
        print("  Pre-rendering error-plot frames …")
        plot_frames, n_checkpoints = _prerender_plot_frames(
            all_error_traces, log_interval, plot_w, canvas_h, plot_window)

    # ── Extend background canvas to include the plot column ───────────────────
    grid_w = combined_bg.shape[1]   # already pad16
    if plot_w > 0:
        plot_bg = np.full((canvas_h, plot_w, 3), _C_BG, dtype=np.uint8)
        combined_bg = np.concatenate([combined_bg, plot_bg], axis=1)

    # Steps at which to save a full-frame PNG snapshot
    if snapshot_every > 0:
        snap_dir = output_dir / "snapshots"
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_steps = (
            {0, 1} |
            set(range(0, n_steps, snapshot_every)) |
            {n_steps - 1}
        )
    else:
        snap_dir   = None
        snap_steps = set()

    out_path = output_dir / f"combined_all_seeds_animation.{fmt}"

    with imageio.get_writer(str(out_path), **_writer_kwargs(internal_fps, fmt)) as writer:
        for step_idx in tqdm(range(n_steps), desc="  rendering", unit="step"):
            # Current and next positions for every method/seed
            curr = {m: {s: traj[step_idx] for s, traj in trajs.items()}
                    for m, trajs in all_trajectories.items()}
            nxt  = {m: {s: traj[step_idx + 1] if step_idx + 1 < n_steps else traj[step_idx]
                        for s, traj in trajs.items()}
                    for m, trajs in all_trajectories.items()}

            # Determine which pre-rendered plot frame matches this step
            if plot_frames:
                plot_idx   = min(step_idx // log_interval, n_checkpoints - 1)
                plot_slice = plot_frames[plot_idx]

            for t in range(n_sub):
                alpha = t / n_sub
                frame = combined_bg.copy()
                for method, trajs in all_trajectories.items():
                    pr, pc = PANEL_LAYOUT[method]
                    for seed in trajs:
                        r0, c0 = curr[method][seed]
                        r1, c1 = nxt[method][seed]
                        cy0 = pr * panel_h + title_h + r0 * cell_px + half
                        cx0 = pc * panel_w + c0 * cell_px + half
                        cy1 = pr * panel_h + title_h + r1 * cell_px + half
                        cx1 = pc * panel_w + c1 * cell_px + half
                        cy  = round(cy0 + alpha * (cy1 - cy0))
                        cx  = round(cx0 + alpha * (cx1 - cx0))
                        _stamp(frame, cy, cx, dys, dxs, seed_colour[seed])
                if plot_frames:
                    frame[:, grid_w:grid_w + plot_w] = plot_slice
                # Save full-frame PNG snapshot at t=0 for designated steps
                if t == 0 and step_idx in snap_steps:
                    tag = ("init" if step_idx == 0
                           else f"step{step_idx:07d}")
                    Image.fromarray(frame).save(
                        str(snap_dir / f"frame_{tag}.png"))
                writer.append_data(frame)

    total_frames = n_steps * n_sub
    print(f"  saved → {out_path.name}  "
          f"({n_steps:,} steps × {n_sub} sub-frames = {total_frames:,} frames, "
          f"{total_frames/internal_fps:.0f}s at {internal_fps} FPS)")
    if snap_dir:
        print(f"  saved {len(snap_steps)} frame snapshots → {snap_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate curiosity-agent trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", required=True,
                        help="Directory containing result .pkl files.")
    parser.add_argument("--mode", choices=["combined", "single"],
                        default="combined",
                        help="combined: one video, all methods + all seeds. "
                             "single: one video per (method, seed).")
    parser.add_argument("--fps", type=int, default=200,
                        help="Frames per second. Every step is one frame.")
    parser.add_argument("--cell-px", type=int, default=14,
                        help="Pixel size of each grid cell.")
    parser.add_argument("--format", dest="fmt", choices=["mp4", "gif"],
                        default="mp4", help="Output format.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Only include this seed.")
    parser.add_argument("--method", default=None,
                        help="(single mode only) Only animate this method.")
    parser.add_argument("--snapshot-every", type=int, default=5000,
                        help="Save a PNG of the error plot every N environment steps "
                             "(0 = disabled). Always includes step 0 (init), step 1, "
                             "and the final step.")
    parser.add_argument("--plot-window", type=int, default=0,
                        help="Steps of error history to show in the live plot "
                             "(0 = show all history, default). "
                             "E.g. 5000 shows only the last 5000 steps.")
    parser.add_argument("--smooth-interpol", dest="smooth", type=int, default=-1,
                        help="Sub-frame interpolation: number of intermediate frames "
                             "between each environment step. FPS is scaled internally "
                             "by this factor so playback duration is unchanged. "
                             "-1 disables interpolation (default).")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    pkl_files   = sorted(results_dir.glob("result__*.pkl"))

    if not pkl_files:
        print(f"No result .pkl files found in: {results_dir}")
        return

    if args.seed is not None:
        pkl_files = [p for p in pkl_files if f"seed{args.seed:04d}" in p.name]
    if args.method is not None and args.mode == "single":
        pkl_files = [p for p in pkl_files if f"__{args.method}__" in p.name]

    if not pkl_files:
        print("No files matched the specified filters.")
        return

    if args.mode == "single":
        print(f"Single mode: {len(pkl_files)} file(s) at {args.fps} FPS → {args.fmt.upper()}\n")
        for pkl_path in pkl_files:
            print(f"  {pkl_path.name}")
            animate_single(pkl_path, args.fps, args.cell_px, args.fmt, args.smooth)

    else:  # combined
        # Load all pkl files, group by method → seed
        all_trajectories: dict  = defaultdict(dict)
        all_error_traces: dict  = defaultdict(dict)
        grid         = None
        log_interval = 100  # default; overridden from first pkl that has config

        for pkl_path in pkl_files:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            method = data.get('method')
            if method == 'curiosity_critic_ours':
                method = 'curiosity_critic_ours_tabular_critic'
            seed   = data.get('seed')
            traj   = data.get('trajectory', [])
            if method not in PANEL_LAYOUT or not traj:
                continue
            all_trajectories[method][seed] = traj
            err = data.get('det_error_trace')
            if err:
                all_error_traces[method][seed] = err
            if grid is None:
                grid = data['grid']
            if 'config' in data:
                log_interval = data['config'].get('log_interval', log_interval)

        if not all_trajectories or grid is None:
            print("No valid trajectory data found.")
            return

        methods      = sorted(all_trajectories, key=lambda m: PANEL_LAYOUT[m])
        all_seed_set = sorted({s for t in all_trajectories.values() for s in t})
        has_errors   = bool(all_error_traces)
        print(f"Combined mode: methods={methods}  seeds={all_seed_set}  "
              f"{args.fps} FPS → {args.fmt.upper()}  "
              f"error-plot={'yes' if has_errors else 'no'}\n")

        animate_combined(
            dict(all_trajectories), grid,
            args.fps, args.cell_px, args.fmt, results_dir, args.smooth,
            all_error_traces=dict(all_error_traces) if has_errors else None,
            log_interval=log_interval,
            plot_window=args.plot_window,
            snapshot_every=args.snapshot_every,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
