"""
Curiosity-Driven Exploration in a Stochastic Grid World
=======================================================

Overview
--------
A minimal, reproducible benchmark for comparing curiosity-based intrinsic
motivation methods in reinforcement learning.  The environment is a 30×30
grid world containing a 15-column deterministic left half and a 15-column 
stochastic right half.  The central question is whether a curiosity signal 
can distinguish between these two regimes and focus the agent's exploration 
on the learnable region.

Environment
-----------
The grid is 30×30.  The left half (cols 0–14, all rows) is deterministic —
450 cells, 50% of the grid.  The right half (cols 15–29) is stochastic.
The agent starts at (15, 15), at the center row on the stochastic side of
the boundary:

             0         1         2
             012345678901234567890123456789
    row  0   DDDDDDDDDDDDDDD···············
        ⋮    (rows 1–14: same left/right split)
    row 15   DDDDDDDDDDDDDDS···············   ← agent start
        ⋮    (rows 16–29: same left/right split)
    row 29   DDDDDDDDDDDDDDD···············

    D = deterministic cell (cols 0–14)
    · = stochastic cell    (cols 15–29)
    S = agent start (15, 15)

State transitions are fully deterministic: the agent always moves to the
intended neighbouring cell.  Boundaries are hard — actions that would leave
the grid are invalid and excluded from action selection.

Each cell emits a 200-dimensional binary observation ("TV pixels"):

  Deterministic region (all rows, cols 0–14):
    Every cell has a unique fixed binary pattern derived as a cyclic shift of
    a shared random base vector, so neighbouring cells have correlated
    observations.  Patterns are generated once from GRID_SEED and are
    identical across all experimental runs.  The world model can reduce
    prediction error on these cells to zero, and the low-frequency structure
    allows it to do so with fewer visits via the factored state encoding.

  Stochastic region (all rows, cols 15–29):
    Each visit independently re-samples an i.i.d. Bernoulli(0.5) vector.
    Prediction error here is irreducible regardless of the number of
    gradient updates the world model receives.

The agent starts at (15, 15), at the center row immediately on the stochastic
side of the deterministic boundary.

World model
-----------
A two-layer MLP trained to predict the TV-pixel observation at the agent's
current grid location:

    f_θ(state) → predicted observation at state

Input : factored 2×GRID_SIZE-dimensional encoding of the current cell —
        one-hot(row) concatenated with one-hot(col).
Output: NUM_PIXELS-dimensional vector of raw logit predictions (no output
        activation).

The factored encoding represents each cell as the sum of a row embedding
and a column embedding, so gradient updates for one cell propagate to all
cells sharing the same row or column.  This cross-cell generalisation can
accelerate learning when nearby cells carry correlated observations.

Training uses MSE loss on raw logits (no output activation).
Error for reward computation uses the same L2 norm of the difference between
the raw logit prediction and the true binary observation.

Methods
-------
random                  Uniform random action selection (unguided baseline).
curiosity_v1            Curiosity V1 [Schmidhuber, Feb 1991]: intrinsic reward equals the
                        L2 prediction error of the world model before the gradient update.
curiosity_v2            Curiosity V2 [Schmidhuber, Nov 1991]: intrinsic reward equals the reduction in prediction
                        error produced by one gradient update (learning progress signal).
rnd_state               Random Network Distillation on the factored grid state.  This is a
                        learned spatial-novelty / soft-count baseline over (row, col).
rnd_observation         Random Network Distillation on the observed TV-pixel vector.  This
                        is the citation-faithful RND baseline for the stochastic-observation
                        setting: stochastic cells keep producing new-looking observations.
curiosity_critic_ours_tabular_critic   Ours (Tabular Critic): intrinsic reward equals the world-model
                        prediction error minus a per-state EMA-mean tabular baseline
                        (decay=0.9) of residual errors after each gradient step.  As the
                        model learns a deterministic state, the residual shrinks and net
                        reward stays positive; for stochastic states, the residual
                        stabilises at the irreducible noise floor and net reward → 0,
                        suppressing further exploration of unlearnable cells.
curiosity_critic_ours_ideal  Ours Oracle (Ground-Truth Critic): uses the known analytical
                        irreducible error (sqrt(NUM_PIXELS) × 0.5 for stochastic cells,
                        0.0 for deterministic cells) instead of a learned estimate.
                        Requires oracle knowledge of cell type and noise level; serves as
                        an ideal upper bound for curiosity_critic_ours_tabular_critic.
curiosity_critic_ours_nnet  Ours (Neural Critic Model): identical to curiosity_critic_ours_tabular_critic
                        except the per-state EMA-mean tabular baseline is replaced by a
                        small neural network (CriticNNetModel) trained online to predict
                        error_after at each state.  After each world-model gradient step
                        the critic network receives one gradient update on
                        (state, error_after) and its updated prediction serves as the
                        baseline for the intrinsic reward.

Evaluation metrics
------------------
Recorded every LOG_INTERVAL environment steps:

1. Mean L2 prediction error over all deterministic cells (primary).
   Each cell is queried directly — the model predicts the observation at that
   cell and the L2 norm of the residual is averaged across all deterministic cells.
   Lower values indicate better world-model learning in the deterministic region.

2. Fraction of steps spent in the deterministic region within each window
   (secondary).  Shows how effectively each method directs the agent's
   attention toward learnable states.

Reproducibility
---------------
The observation patterns of deterministic cells are fixed by GRID_SEED and
are shared across all methods and seeds.  Each run uses an independent
numpy.random.Generator seeded by the user-supplied --seed argument.

Usage
-----
Run all methods across all seeds:
    python curiosity_experiment.py batch --output-dir results/
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Global constants
# ──────────────────────────────────────────────────────────────────────────────

GRID_SIZE: int   = 30          # grid is GRID_SIZE × GRID_SIZE
NUM_PIXELS: int  = 200         # dimensionality of each TV-pixel observation
NUM_ACTIONS: int = 4           # 0=Up, 1=Down, 2=Left, 3=Right
GRID_SEED: int   = 28081994    # fixed seed for observation patterns; shared by all runs
TOTAL_STEPS: int = 20_000      # environment steps per experiment
LOG_INTERVAL: int = 100        # steps between evaluation checkpoints
COLD_START_STEPS: int       = 1_000  # random steps used to estimate mean reward for V-table init
MODEL_COLD_START_SEED: int    = 67154131  # fixed seed for model warm-up; far from experiment seeds (0–~30)
VTABLE_COLD_START_SEED: int   = 41316715  # fixed seed for V-table cold-start; far from experiment seeds

# The deterministic region is the left half of the grid: all rows, cols 0–14.
# The right half (cols 15–29) is stochastic.
DET_ROWS: int      = GRID_SIZE      # all 30 rows
DET_COLS: int      = GRID_SIZE // 2 # left 15 columns
DET_ROW_START: int = 0
DET_COL_START: int = 0

# Hardcoded seeds and methods for the batch subcommand.
BATCH_SEEDS:   List[int] = list(range(1,6))
BATCH_METHODS: List[str] = ['random', 'curiosity_v1', 'curiosity_v2', 'visitation_count', 'rnd_state', 'rnd_observation', 'curiosity_critic_ours_tabular_critic', 'curiosity_critic_ours_nnet', 'curiosity_critic_ours_ideal']


# ──────────────────────────────────────────────────────────────────────────────
# Grid world
# ──────────────────────────────────────────────────────────────────────────────

def build_grid(
    grid_seed: int = GRID_SEED,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Assign a unique fixed NUM_PIXELS-dim binary observation pattern to every
    cell in the deterministic region (all rows, cols 0–DET_COLS-1 = 0–14).

    Patterns are constructed as cyclic shifts of a shared random base vector,
    so adjacent cells share correlated observations.  This low-frequency
    structure allows the world model to learn the deterministic region faster
    through its factored (row + column) embedding — gradient updates for one
    cell transfer partial information to neighbouring cells via shared weights.

    Parameters
    ----------
    grid_seed : RNG seed — must be held constant across all experimental runs.

    Returns
    -------
    patterns : dict mapping (row, col) → np.ndarray of shape (NUM_PIXELS,).
               Only deterministic cells appear as keys.
    """
    rng = np.random.default_rng(grid_seed)
    base = rng.random(size=NUM_PIXELS).astype(np.float32)
    patterns: Dict[Tuple[int, int], np.ndarray] = {}
    for r in range(DET_ROW_START, DET_ROW_START + DET_ROWS):
        for c in range(DET_COL_START, DET_COL_START + DET_COLS):
            offset = (r - DET_ROW_START) * DET_COLS + (c - DET_COL_START)
            pattern = np.roll(base, offset * 3)
            patterns[(r, c)] = (pattern > 0.5).astype(np.float32)
    return patterns


def is_deterministic(state: Tuple[int, int]) -> bool:
    """Return True if *state* lies in the deterministic region."""
    return (DET_ROW_START <= state[0] < DET_ROW_START + DET_ROWS and
            DET_COL_START <= state[1] < DET_COL_START + DET_COLS)


def step_environment(
    state: Tuple[int, int],
    action: int,
) -> Tuple[int, int]:
    """
    Execute a deterministic action from *state* with hard (reflecting) boundaries.

    Actions that would move the agent outside the grid are invalid — callers
    must only pass actions returned by valid_actions().

    Action indices: 0=Up, 1=Down, 2=Left, 3=Right.
    """
    r, c = state
    if action == 0:    return (r - 1, c)
    elif action == 1:  return (r + 1, c)
    elif action == 2:  return (r, c - 1)
    else:              return (r, c + 1)


def valid_actions(state: Tuple[int, int], size: int = GRID_SIZE) -> List[int]:
    """
    Return the list of actions available from *state* given hard grid boundaries.

    Only directional actions (Up/Down/Left/Right) that keep the agent within
    [0, size-1] × [0, size-1] are included.  Corner/edge cells have fewer
    valid actions.
    """
    r, c = state
    actions = []
    if r > 0:        actions.append(0)  # Up
    if r < size - 1: actions.append(1)  # Down
    if c > 0:        actions.append(2)  # Left
    if c < size - 1: actions.append(3)  # Right
    return actions


def get_observation(
    state: Tuple[int, int],
    patterns: Dict[Tuple[int, int], np.ndarray],
    rng: np.random.Generator,
    discrete: bool = True,
) -> np.ndarray:
    """
    Return the TV-pixel observation at *state*.

    Deterministic cell → fixed unique pattern from *patterns*.
    Stochastic cell    → freshly sampled i.i.d. binary (or uniform) vector.
    """
    if is_deterministic(state):
        return patterns[state].copy()
    if discrete:
        return rng.integers(0, 2, size=NUM_PIXELS).astype(np.float32)
    return rng.random(size=NUM_PIXELS).astype(np.float32)


def compute_deterministic_error(
    patterns: Dict[Tuple[int, int], np.ndarray],
    model: 'WorldModel',
) -> float:
    """
    Mean L2 prediction error over all deterministic cells.

    Each deterministic cell (r, c) is queried directly: the model predicts
    the observation at (r, c) and the L2 norm of the residual is averaged
    across all deterministic cells.  This is in the same units as the V1
    per-step reward (||pred - pixels||_2), making it easy to compare.

    Lower is better; the ideal agent drives this metric toward zero.
    """
    total = 0.0
    for (r, c), pattern in patterns.items():
        pred = model.predict((r, c))
        total += float(np.linalg.norm(pred - pattern))
    return total / len(patterns)


def encode_state_np(state: Tuple[int, int]) -> np.ndarray:
    """Factored one-hot(row) | one-hot(col) state encoding as a NumPy vector."""
    row_enc = np.zeros(GRID_SIZE, dtype=np.float32)
    col_enc = np.zeros(GRID_SIZE, dtype=np.float32)
    row_enc[state[0]] = 1.0
    col_enc[state[1]] = 1.0
    return np.concatenate([row_enc, col_enc], axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# World model  (PyTorch MLP)
# ──────────────────────────────────────────────────────────────────────────────

class _WorldModelNet(nn.Module):
    """
    Two-layer MLP:
        Linear(state_dim → hidden) → ReLU
        → Linear(hidden → output_dim)   [raw logits; no output activation]
    """

    def __init__(self, input_dim: int, hidden: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModel:
    """
    Predicts the TV-pixel observation at the agent's current grid location.

    Input  : [one-hot(row) | one-hot(col)]  →  (2×GRID_SIZE,)  =  (60,)
    Output : predicted observation at state →  (NUM_PIXELS,)   =  (200,)
    Training loss : MSELoss on raw logits (no output activation)
    Reward error  : L2 norm of (logits − true_pixels)
    Optimiser     : Adam

    State encoding
    --------------
    The state (r, c) is encoded as two concatenated one-hot vectors — one of
    length GRID_SIZE for the row index and one of length GRID_SIZE for the
    column index — giving a 2×GRID_SIZE = 60-dimensional input vector.

    The public interface accepts and returns plain NumPy arrays; all tensor
    conversions are handled internally.
    """

    _INPUT_DIM  = 2 * GRID_SIZE   # 60  (one-hot row + one-hot col)
    _OUTPUT_DIM = NUM_PIXELS      # 200

    def __init__(
        self,
        hidden: int = 4,
        lr: float   = 0.001,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._hidden     = hidden
        self._lr         = lr
        self._torch_seed = int(rng.integers(0, 2**31)) if rng is not None else 0
        self._build()

    # ── construction / reset ─────────────────────────────────────────────────

    def _build(self) -> None:
        torch.manual_seed(self._torch_seed)
        self._net  = _WorldModelNet(self._INPUT_DIM, self._hidden, self._OUTPUT_DIM)
        self._loss = nn.MSELoss()
        self._opt  = torch.optim.Adam(self._net.parameters(), lr=self._lr, betas=(0.9, 0.999), eps=1e-8)

    def reset(self) -> None:
        """Re-initialise all weights and optimiser state."""
        self._build()

    # ── encoding ─────────────────────────────────────────────────────────────

    def _encode(self, state: Tuple[int, int]) -> torch.Tensor:
        row_enc = torch.zeros(GRID_SIZE)
        col_enc = torch.zeros(GRID_SIZE)
        row_enc[state[0]] = 1.0
        col_enc[state[1]] = 1.0
        return torch.cat([row_enc, col_enc]).unsqueeze(0)  # (1, 2*GRID_SIZE)

    # ── public interface ──────────────────────────────────────────────────────

    def predict(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Return the predicted observation at *state*.
        Output shape: (NUM_PIXELS,).  Raw logits, no output activation.
        """
        self._net.eval()
        with torch.no_grad():
            logits = self._net(self._encode(state))
        return logits.squeeze(0).numpy()

    def train(
        self,
        state: Tuple[int, int],
        target: np.ndarray,
    ) -> None:
        """One Adam step minimising MSE loss on raw logits against binary target."""
        self._net.train()
        x = self._encode(state)
        y = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        self._opt.zero_grad()
        self._loss(self._net(x), y).backward()
        self._opt.step()


# ──────────────────────────────────────────────────────────────────────────────
# Reward normaliser
# ──────────────────────────────────────────────────────────────────────────────

class RewardNormalizer:
    """
    Normalises rewards to approximately unit scale before V-table updates.

    Divides each reward by a running estimate of its standard deviation
    (without subtracting the mean), so that non-negative rewards stay
    non-negative after normalisation:

        std  = sqrt(EMA(r²) - EMA(r)²)
        normalised_r = r / (std + ε)

    This decouples reward magnitude from lr_model without shifting the
    baseline, preserving the sign and positivity of the raw reward signal.
    """

    def __init__(self, decay: float = 0.95, eps: float = 1e-8) -> None:
        self._ema    = 0.0
        self._ema_sq = 0.0
        self._decay  = decay
        self._eps    = eps

    def normalise(self, reward: float) -> float:
        self._ema    = self._decay * self._ema    + (1.0 - self._decay) * reward
        self._ema_sq = self._decay * self._ema_sq + (1.0 - self._decay) * reward ** 2
        var = max(0.0, self._ema_sq - self._ema ** 2)
        std = math.sqrt(var)
        return reward / (std + self._eps)


# ──────────────────────────────────────────────────────────────────────────────
# V-table policy  (state-only value function, ε-greedy)
# ──────────────────────────────────────────────────────────────────────────────

class VTablePolicy:
    """
    Tabular value-based policy using a state-only V-table.

    Because transitions are deterministic and local (each action moves the
    agent to exactly one neighbouring cell), a state-only value function
    suffices.  At boundary cells only the subset of actions that stay within
    the grid are valid; the max and the ε-greedy sample are restricted to
    that subset.

    Update rule (Bellman optimality, off-policy TD):

        V(s) ← V(s) + α · [ r + γ · max_{a ∈ valid(s)} V(step(s, a)) − V(s) ]

    The max is over valid actions only, making this the V-table analogue of
    Q-learning (off-policy, greedy backup) with hard boundaries.
    """

    def __init__(
        self,
        init_value: float = 1.0,
        lr: float         = 0.1,
        gamma: float      = 0.99,
        epsilon: float    = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._lr      = lr
        self._gamma   = gamma
        self._epsilon = epsilon
        self._rng     = rng or np.random.default_rng()
        self.V        = np.full((GRID_SIZE, GRID_SIZE), init_value)

    def _valid_neighbor_values(self, state: Tuple[int, int]) -> Tuple[List[int], np.ndarray]:
        """Return (actions, V-values) restricted to valid actions from *state*."""
        actions = valid_actions(state)
        values  = np.array([self.V[step_environment(state, a)] for a in actions])
        return actions, values

    def select_action(self, state: Tuple[int, int]) -> int:
        actions, values = self._valid_neighbor_values(state)
        if self._rng.random() < self._epsilon:
            return int(self._rng.choice(actions))
        best = [a for a, v in zip(actions, values) if v == values.max()]
        return int(self._rng.choice(best))

    def update(self, state: Tuple[int, int], reward: float) -> None:
        _, values      = self._valid_neighbor_values(state)
        td_error       = reward + self._gamma * values.max() - self.V[state[0], state[1]]
        self.V[state[0], state[1]] += self._lr * td_error


# ──────────────────────────────────────────────────────────────────────────────
# Critic baseline estimation  (EMA-mean of residual error per state)
# ──────────────────────────────────────────────────────────────────────────────

class CriticBaselineEstimation:
    """
    Per-state EMA-mean baseline used by the Curiosity-Critic reward.

    Estimates the expected residual prediction error at each state by
    maintaining an exponential moving average of error_after with decay=0.9:

        ema(i,j) ← decay * ema(i,j) + (1 - decay) * error_after
        baseline(i,j) = ema(i,j)

    For stochastic states the residual error after each gradient step remains
    large regardless of training, so the baseline stabilises at the irreducible
    noise level and the net Curiosity-Critic reward → 0.  For deterministic
    states the residual shrinks as the world model learns, pulling the baseline
    down and keeping the net reward positive, sustaining focused exploration.
    """

    _DECAY = 0.9

    def __init__(self) -> None:
        self._ema = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)

    def update(self, state: Tuple[int, int], error_: float) -> None:
        i, j = state
        self._ema[i, j] = (
            self._DECAY * self._ema[i, j] + (1.0 - self._DECAY) * error_
        )

    def mean(self, state: Tuple[int, int]) -> float:
        return float(self._ema[state[0], state[1]])


# ──────────────────────────────────────────────────────────────────────────────
# Neural critic model  (predicts error_after per state)
# ──────────────────────────────────────────────────────────────────────────────

class CriticNNetModel:
    """
    Small MLP trained online to predict error_after at each state.

    Used by the Curiosity-Critic with Neural Critic method as a drop-in
    replacement for the tabular EMA-mean baseline in CriticBaselineEstimation.
    Rather than maintaining a per-state running average, this model learns a
    smooth function over the state space, allowing it to generalise across
    states that share row or column embeddings.

    Architecture
    ------------
    Input  : [one-hot(row) | one-hot(col)]  →  (2×GRID_SIZE,)  =  (60,)
    Hidden : Linear(60 → hidden) → ReLU
    Output : Linear(hidden → 1)  [scalar, predicted error_after]

    The hidden dimension is intentionally smaller than the world model's hidden
    dimension since predicting a scalar residual error is a considerably easier
    regression target than predicting the full 200-dimensional observation.

    Training
    --------
    One Adam step per environment step with MSE loss against the observed
    error_after.  The update is applied before predict() is called for the
    reward, so the prediction already reflects the current visit.

    Predict
    -------
    Returns max(0, output) — clamped to non-negative since error_after ≥ 0.
    """

    _INPUT_DIM = 2 * GRID_SIZE   # 60  (one-hot row + one-hot col)

    def __init__(
        self,
        hidden: int = 128,
        lr: float   = 0.001,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._hidden     = hidden
        self._lr         = lr
        self._torch_seed = int(rng.integers(0, 2**31)) if rng is not None else 0
        self._build()

    def _build(self) -> None:
        torch.manual_seed(self._torch_seed)
        self._net  = _WorldModelNet(self._INPUT_DIM, self._hidden, 1) # repurposing _WorldModelNet class for CriticNNetModel's MLP architecture
        self._loss = nn.MSELoss()
        self._opt  = torch.optim.Adam(self._net.parameters(), lr=self._lr, betas=(0.9, 0.999), eps=1e-8)

    def _encode(self, state: Tuple[int, int]) -> torch.Tensor:
        row_enc = torch.zeros(GRID_SIZE)
        col_enc = torch.zeros(GRID_SIZE)
        row_enc[state[0]] = 1.0
        col_enc[state[1]] = 1.0
        return torch.cat([row_enc, col_enc]).unsqueeze(0)  # (1, 2*GRID_SIZE)

    def train(self, state: Tuple[int, int], error_after: float) -> None:
        """One Adam step fitting the network to predict error_after at state."""
        self._net.train()
        x = self._encode(state)
        y = torch.tensor([[error_after]], dtype=torch.float32)
        self._opt.zero_grad()
        self._loss(self._net(x), y).backward()
        self._opt.step()

    def predict(self, state: Tuple[int, int]) -> float:
        """Return the predicted error_after at state (clamped to ≥ 0)."""
        self._net.eval()
        with torch.no_grad():
            out = self._net(self._encode(state))
        return float(max(0.0, out.item()))


# ──────────────────────────────────────────────────────────────────────────────
# Random Network Distillation model
# ──────────────────────────────────────────────────────────────────────────────

class RNDModel:
    """
    Random Network Distillation (Burda et al., 2018).

    A fixed random target network defines a deterministic feature map, and a
    trainable predictor network learns to match it on inputs the agent has
    visited.  The intrinsic reward is the predictor's MSE before the predictor
    trains on the current input, so a first visit receives novelty credit and
    repeated visits decay as the predictor catches up.

    Two input variants are used in this benchmark:
      - rnd_state: input is the factored one-hot row/column state (60 dims).
      - rnd_observation: input is the observed TV-pixel vector (200 dims).
    """

    def __init__(
        self,
        input_dim: int,
        hidden: int = 128,
        output_dim: int = 128,
        lr: float = 0.001,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self._input_dim = input_dim
        self._hidden = hidden
        self._output_dim = output_dim
        self._lr = lr
        rng = rng or np.random.default_rng()
        self._target_seed = int(rng.integers(0, 2**31))
        self._predictor_seed = int(rng.integers(0, 2**31))
        self._build()

    def _build(self) -> None:
        torch.manual_seed(self._target_seed)
        self._target = _WorldModelNet(self._input_dim, self._hidden, self._output_dim)
        self._target.eval()
        for param in self._target.parameters():
            param.requires_grad_(False)

        torch.manual_seed(self._predictor_seed)
        self._predictor = _WorldModelNet(self._input_dim, self._hidden, self._output_dim)
        self._loss = nn.MSELoss()
        self._opt = torch.optim.Adam(self._predictor.parameters(), lr=self._lr, betas=(0.9, 0.999), eps=1e-8)

    def _as_tensor(self, x: np.ndarray) -> torch.Tensor:
        arr = np.asarray(x, dtype=np.float32)
        if arr.shape != (self._input_dim,):
            raise ValueError(f"RND input has shape {arr.shape}; expected ({self._input_dim},)")
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

    def prediction_error(self, x: np.ndarray) -> float:
        """Return MSE between predictor and fixed target on input x."""
        self._predictor.eval()
        with torch.no_grad():
            xt = self._as_tensor(x)
            target = self._target(xt)
            pred = self._predictor(xt)
            return float(self._loss(pred, target).item())

    def train(self, x: np.ndarray) -> None:
        """One Adam step distilling the fixed random target on input x."""
        self._predictor.train()
        xt = self._as_tensor(x)
        with torch.no_grad():
            target = self._target(xt)
        self._opt.zero_grad()
        self._loss(self._predictor(xt), target).backward()
        self._opt.step()


# ──────────────────────────────────────────────────────────────────────────────
# Visitation count table  (Strehl & Littman, 2008)
# ──────────────────────────────────────────────────────────────────────────────

class VisitCountTable:
    """
    Per-state visitation counter for the count-based curiosity baseline.

    Implements the exploration bonus from Strehl & Littman (2008):

        r(s) = 1 / sqrt( N(s) )

    where N(s) is the number of times state s has been visited.  Counts are
    initialised to 1 (one phantom visit per state), encoding a uniform prior
    over the grid and ensuring the denominator never reaches zero.  The first
    real visit to a state therefore gives N=2 and bonus 1/sqrt(2) ≈ 0.707,
    decaying smoothly with subsequent visits.

    The count is incremented via increment() *before* calling bonus() so that
    the reward reflects the updated count at the moment of visit.

    Reference
    ---------
    A.L. Strehl and M.L. Littman. An analysis of model-based interval
    estimation for Markov Decision Processes. Journal of Computer and System
    Sciences, 74(8):1309–1331, 2008.
    """

    def __init__(self) -> None:
        # Initialise to 1 (phantom visit) so bonus is always finite.
        self._counts = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.int64)

    def increment(self, state: Tuple[int, int]) -> None:
        """Increment the visit count for *state*."""
        self._counts[state[0], state[1]] += 1

    def bonus(self, state: Tuple[int, int]) -> float:
        """Return the exploration bonus 1 / sqrt(N(s)) for *state*."""
        return 1.0 / math.sqrt(self._counts[state[0], state[1]])


# Analytical irreducible-error baseline used by the oracle variant.
# For stochastic cells (i.i.d. Bernoulli(0.5)), a fully converged model
# predicts 0.5 per pixel, so each error component is exactly |pixel - 0.5|
# ∈ {0.5} — deterministically 0.5 regardless of the Bernoulli draw.  The
# resulting L2 norm is sqrt(NUM_PIXELS * 0.25) = 0.5 * sqrt(NUM_PIXELS),
# which is both the pointwise value and the EMA-mean the critic converges to.
# Deterministic cells have zero irreducible error.
_STOCHASTIC_IRREDUCIBLE_ERROR: float = math.sqrt(NUM_PIXELS) * 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Curiosity reward functions
# ──────────────────────────────────────────────────────────────────────────────

def _curiosity_reward_v1(
    state, error_before, error_after, var_table
) -> float:
    """Curiosity V1: intrinsic reward = world-model prediction error at current state."""
    return error_before


def _curiosity_reward_v2(
    state, error_before, error_after, var_table
) -> float:
    """Curiosity V2: intrinsic reward = reduction in prediction error (learning progress)."""
    return error_before - error_after


def _curiosity_critic_reward_ours(
    state, error_before, error_after, var_table
) -> float:
    """
    Ours (Tabular Critic): reward = error_before - mean(error_after history).

    The EMA-mean baseline tracks error_after directly (residual L2 error after
    one gradient step), accumulated across all visits including the current one.
    Stochastic states keep high error_after regardless of training, so their
    EMA-mean quickly converges to the irreducible level and net reward → 0.
    Deterministic states see error_after shrink as the model learns, so their
    EMA-mean falls and net reward stays positive, sustaining focused exploration.
    """
    return error_before - var_table.mean(state)


def _curiosity_critic_nnet_reward_ours(
    state, error_before, error_after, critic_nnet
) -> float:
    """
    Ours (Neural Critic Model): reward = error_before - critic_nnet.predict(state).

    The critic network has already received one gradient update on (state, error_after)
    before this function is called.  Its updated prediction serves as the baseline,
    approximating the expected residual error at state via a learned function rather
    than a per-state EMA table.
    """
    return error_before - critic_nnet.predict(state)


def _curiosity_ideal_critic_reward_ours_oracle(
    state, error_before, error_after, var_table
) -> float:
    """
    Ours Oracle (Ground-Truth Critic): reward = error_before - irreducible_error(state).

    Uses the known analytical irreducible error rather than the EMA-mean
    estimate.  For stochastic states (i.i.d. Bernoulli(0.5)) the irreducible
    L2 error is sqrt(NUM_PIXELS) * 0.5; for deterministic states it is 0.0.
    Requires oracle knowledge of cell type and noise level.
    """
    irreducible = 0.0 if is_deterministic(state) else _STOCHASTIC_IRREDUCIBLE_ERROR
    return error_before - irreducible


def _curiosity_reward_visitation(
    state, error_before, error_after, visit_table
) -> float:
    """
    Count-based curiosity (Strehl & Littman, 2008):
    intrinsic reward = 1 / sqrt(N(s)).

    The visit count N(s) is incremented before this function is called
    (inside run_experiment), so the bonus already reflects the updated count
    at the current visit.  The visit_table argument is a VisitCountTable
    instance; the error arguments are unused but retained for API consistency
    with the other reward functions.
    """
    return visit_table.bonus(state)


_REWARD_FN = {
    'random':                        None,
    'curiosity_v1':                  _curiosity_reward_v1,
    'curiosity_v2':                  _curiosity_reward_v2,
    'rnd_state':                     None,
    'rnd_observation':               None,
    'curiosity_critic_ours_tabular_critic':         _curiosity_critic_reward_ours,
    'curiosity_critic_ours_nnet':    _curiosity_critic_nnet_reward_ours,
    'curiosity_critic_ours_ideal':   _curiosity_ideal_critic_reward_ours_oracle,
    'visitation_count':              _curiosity_reward_visitation,
}


# ──────────────────────────────────────────────────────────────────────────────
# Experiment configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    method:          str   = 'curiosity_v1'
    seed:            int   = 0
    grid_seed:       int   = GRID_SEED
    total_steps:     int   = TOTAL_STEPS
    log_interval:    int   = LOG_INTERVAL
    hidden:          int   = 4       # world-model hidden units
    hidden_critic:   int   = 128     # critic network hidden units (for curiosity_critic_ours_nnet)
    hidden_rnd:      int   = 128     # RND predictor/target hidden units
    rnd_output_dim:  int   = 128     # RND random feature dimension
    lr_model:        float = 0.001
    lr_critic:       float = 0.001   # critic network learning rate (for curiosity_critic_ours_nnet)
    lr_rnd:          float = 0.001   # RND predictor learning rate
    lr_policy:       float = 0.1
    gamma:           float = 0.99
    epsilon:         float = 0.1
    cold_start_steps:       int   = COLD_START_STEPS  # random steps for V-init estimation
    cold_start:             bool  = False   # if True, estimate v_init from cold-start rewards
    q_init:                 float = 3.0    # V-table init value when cold_start=False
    model_cold_start:       bool  = True   # if True, pre-train model on random policy before main loop
    model_cold_start_steps: int   = 100  # steps for model warm-up
    discrete_pixels:        bool  = True   # True = binary obs; False = uniform [0,1]


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ──────────────────────────────────────────────────────────────────────────────

def build_model_checkpoint(cfg: ExperimentConfig) -> dict:
    """
    Run the model warm-up once and return a deep-copied checkpoint of the
    world-model weights.

    The warm-up seed is MODEL_COLD_START_SEED + cfg.seed, so each experiment
    seed produces distinct initial weights while all methods sharing the same
    seed load from an identical starting point.  MODEL_COLD_START_SEED=67154131
    is chosen to be far from the per-experiment seeds (typically 0–~30).

    Returns
    -------
    dict with key 'net' (OrderedDict of parameter tensors), deep-copied so
    callers cannot accidentally mutate the stored checkpoint.  The Adam
    optimiser state is intentionally excluded — each run starts with a fresh
    optimiser so there is no momentum carry-over from the warm-up phase.
    """
    mcs_seed = MODEL_COLD_START_SEED + cfg.seed
    rng      = np.random.default_rng(mcs_seed)
    model    = WorldModel(
        hidden = cfg.hidden,
        lr     = cfg.lr_model,
        rng    = np.random.default_rng(mcs_seed + 1),
    )
    patterns = build_grid(grid_seed=cfg.grid_seed)
    state    = (15, 15)
    for _ in range(cfg.model_cold_start_steps):
        pixels = get_observation(state, patterns, rng, cfg.discrete_pixels)
        model.train(state, pixels)
        state  = step_environment(state, int(rng.choice(valid_actions(state))))
    return copy.deepcopy({'net': model._net.state_dict()})


def run_experiment(
    cfg: ExperimentConfig,
    model_checkpoint: Optional[dict] = None,
) -> Dict[str, List[float]]:
    """
    Execute one curiosity experiment and return evaluation traces.

    Returns
    -------
    dict with keys:
      'det_error_trace'       : mean L2 prediction error over deterministic
                                cells, recorded every *log_interval* steps.
      'det_visit_frac_trace'  : fraction of steps spent in the deterministic
                                region during each *log_interval* window.
    """
    # ── reproducible RNGs ────────────────────────────────────────────────────
    env_rng   = np.random.default_rng(cfg.seed)
    model_rng = np.random.default_rng(cfg.seed + 10_000)

    # ── environment ──────────────────────────────────────────────────────────
    patterns = build_grid(grid_seed=cfg.grid_seed)

    # ── components (initial construction) ────────────────────────────────────
    model = WorldModel(hidden=cfg.hidden, lr=cfg.lr_model, rng=model_rng)

    epsilon   = 1.0 if cfg.method == 'random' else cfg.epsilon
    reward_fn = _REWARD_FN[cfg.method]

    # ── reward normaliser ─────────────────────────────────────────────────────
    reward_normalizer = RewardNormalizer()

    # ── cold-start: estimate mean reward to set a fair V-table init ───────────
    if cfg.cold_start:
        cold_rng         = np.random.default_rng(VTABLE_COLD_START_SEED + cfg.seed)
        cold_var_table   = CriticBaselineEstimation()
        cold_visit_table = VisitCountTable()
        cold_critic_nnet = CriticNNetModel(
            hidden = cfg.hidden_critic,
            lr     = cfg.lr_critic,
            rng    = np.random.default_rng(VTABLE_COLD_START_SEED + cfg.seed + 1),
        )
        cold_rnd_state = RNDModel(
            input_dim  = 2 * GRID_SIZE,
            hidden     = cfg.hidden_rnd,
            output_dim = cfg.rnd_output_dim,
            lr         = cfg.lr_rnd,
            rng        = np.random.default_rng(VTABLE_COLD_START_SEED + cfg.seed + 2),
        )
        cold_rnd_observation = RNDModel(
            input_dim  = NUM_PIXELS,
            hidden     = cfg.hidden_rnd,
            output_dim = cfg.rnd_output_dim,
            lr         = cfg.lr_rnd,
            rng        = np.random.default_rng(VTABLE_COLD_START_SEED + cfg.seed + 3),
        )
        cold_state       = (15, 15)
        cold_rewards: List[float] = []

        _needs_error_after = ('curiosity_v2', 'curiosity_critic_ours_tabular_critic',
                               'curiosity_critic_ours_nnet')
        for _ in range(cfg.cold_start_steps):
            cold_pixels      = get_observation(cold_state, patterns, cold_rng, cfg.discrete_pixels)
            cold_pred_before = model.predict(cold_state)
            cold_error       = float(np.linalg.norm(cold_pred_before - cold_pixels))
            model.train(cold_state, cold_pixels)
            cold_error_after = (
                float(np.linalg.norm(model.predict(cold_state) - cold_pixels))
                if cfg.method in _needs_error_after else 0.0
            )
            cold_var_table.update(cold_state, cold_error_after)
            if cfg.method == 'curiosity_critic_ours_nnet':
                cold_critic_nnet.train(cold_state, cold_error_after)
            cold_visit_table.increment(cold_state)
            if cfg.method == 'rnd_state':
                rnd_input = encode_state_np(cold_state)
                raw_r = max(0.0, cold_rnd_state.prediction_error(rnd_input))
                cold_rnd_state.train(rnd_input)
                cold_rewards.append(reward_normalizer.normalise(raw_r))
            elif cfg.method == 'rnd_observation':
                raw_r = max(0.0, cold_rnd_observation.prediction_error(cold_pixels))
                cold_rnd_observation.train(cold_pixels)
                cold_rewards.append(reward_normalizer.normalise(raw_r))
            elif reward_fn is not None:
                if cfg.method == 'visitation_count':
                    cold_active_table = cold_visit_table
                elif cfg.method == 'curiosity_critic_ours_nnet':
                    cold_active_table = cold_critic_nnet
                else:
                    cold_active_table = cold_var_table
                raw_r = max(0.0, reward_fn(cold_state, cold_error, cold_error_after,
                                           cold_active_table))
                cold_rewards.append(reward_normalizer.normalise(raw_r))
            cold_valid  = valid_actions(cold_state)
            cold_action = int(cold_rng.choice(cold_valid))
            cold_next   = step_environment(cold_state, cold_action)
            cold_state = cold_next

        max_r  = float(np.max(cold_rewards)) if cold_rewards else 0.0
        v_init = max_r / (1.0 - cfg.gamma) if cfg.gamma < 1.0 else max_r
        model.reset()
        reward_normalizer = RewardNormalizer()   # reset after cold-start
    else:
        v_init = cfg.q_init

    # ── model warm-up: load per-seed checkpoint built by run_batch ───────────
    # The checkpoint is built once per seed in run_batch so every method for a
    # given seed loads from identical weights, while different seeds start from
    # distinct initializations (MODEL_COLD_START_SEED + seed).
    if model_checkpoint is not None:
        model._net.load_state_dict(copy.deepcopy(model_checkpoint['net']))
        # Adam is intentionally left fresh — only network weights are shared

    # Reset all per-method state so every method starts from the same model
    # checkpoint with fresh policy and statistics.
    env_rng         = np.random.default_rng(cfg.seed)   # reproducible main-loop RNG
    var_table         = CriticBaselineEstimation()
    visit_table       = VisitCountTable()
    critic_nnet       = CriticNNetModel(
        hidden = cfg.hidden_critic,
        lr     = cfg.lr_critic,
        rng    = np.random.default_rng(cfg.seed + 20_000),
    )
    rnd_state_model = RNDModel(
        input_dim  = 2 * GRID_SIZE,
        hidden     = cfg.hidden_rnd,
        output_dim = cfg.rnd_output_dim,
        lr         = cfg.lr_rnd,
        rng        = np.random.default_rng(cfg.seed + 30_000),
    )
    rnd_observation_model = RNDModel(
        input_dim  = NUM_PIXELS,
        hidden     = cfg.hidden_rnd,
        output_dim = cfg.rnd_output_dim,
        lr         = cfg.lr_rnd,
        rng        = np.random.default_rng(cfg.seed + 40_000),
    )
    reward_normalizer = RewardNormalizer()

    policy = VTablePolicy(
        init_value=v_init, lr=cfg.lr_policy,
        gamma=cfg.gamma, epsilon=epsilon, rng=env_rng,
    )

    state = (15, 15)

    det_error_trace:        List[float]          = []
    det_visit_frac_trace:   List[float]          = []
    trajectory:             List[Tuple[int,int]] = []
    window_det_visits:      int                  = 0   # det-region visits in current window
    nnet_critic_det_trace:  List[float]          = []  # mean critic estimate over det cells (nnet only)
    nnet_critic_stoch_trace: List[float]         = []  # mean critic estimate over stoch cells (nnet only)

    # ── main loop ────────────────────────────────────────────────────────────
    for step_idx in range(cfg.total_steps):

        # ── observe current state ─────────────────────────────────────────────
        trajectory.append(state)
        if is_deterministic(state):
            window_det_visits += 1

        pixels = get_observation(state, patterns, env_rng, cfg.discrete_pixels)

        # ── world model error before training ─────────────────────────────────
        # Error is L2 norm of pixel difference (used for reward computation).
        pred_before  = model.predict(state)
        error_before = float(np.linalg.norm(pred_before - pixels))

        # ── world model training (MSE loss on raw logits) ────────────────────
        model.train(state, pixels)

        # ── error after training ───────────────────────────────────────────────
        # Used by Curiosity V2 (learning progress = error_before - error_after)
        # and Curiosity-Critic (EMA-mean baseline tracks error_after directly
        # to estimate the residual error that persists after a gradient step).
        error_after = (
            float(np.linalg.norm(model.predict(state) - pixels))
        )

        # ── update running statistics (before reward) ─────────────────────────
        # EMA-mean critic baseline (decay=0.9): updated from error_after directly
        # so it tracks the expected residual error after a gradient step.  Kept
        # low for deterministic states (model learns → error_after → 0) and high
        # for stochastic states (irreducible noise → error_after stays large).
        var_table.update(state, error_after)

        # Neural critic: one gradient step fitting the network to predict
        # error_after at the current state.  The update is applied before
        # predict() is called below so the reward uses the freshest estimate.
        if cfg.method == 'curiosity_critic_ours_nnet':
            critic_nnet.train(state, error_after)

        # Visitation count: incremented here (before the reward call) so that
        # bonus() already reflects the count at the current visit.
        visit_table.increment(state)

        # ── intrinsic reward ──────────────────────────────────────────────────
        # Route each method to the state it needs for reward computation.
        # RND computes its predictor error before training on the current input.
        if cfg.method == 'rnd_state':
            rnd_input = encode_state_np(state)
            reward = rnd_state_model.prediction_error(rnd_input)
            rnd_state_model.train(rnd_input)
        elif cfg.method == 'rnd_observation':
            reward = rnd_observation_model.prediction_error(pixels)
            rnd_observation_model.train(pixels)
        elif cfg.method == 'visitation_count':
            active_table = visit_table
            reward = (
                0.0 if reward_fn is None
                else max(0.0, reward_fn(state, error_before, error_after,
                                        active_table))
            )
        else:
            if cfg.method == 'curiosity_critic_ours_nnet':
                active_table = critic_nnet
            else:
                active_table = var_table
            reward = (
                0.0 if reward_fn is None
                else max(0.0, reward_fn(state, error_before, error_after,
                                        active_table))
            )

        # ── action selection and transition ───────────────────────────────────
        # Select action before updating V(state), matching standard TD(0)
        # ordering: observe → compute reward → act → transition → update.
        # Valid actions are restricted to neighbours within the grid boundary
        # (hard walls; no wrap-around).  The greedy
        # action moves to the valid neighbour with the highest V-value.
        action     = policy.select_action(state)
        next_state = step_environment(state, action)

        # ── V-table update ────────────────────────────────────────────────────
        # Update V(s) after the transition to s'.  The Bellman backup uses the
        # current V-values of all neighbours of s (including s'), consistent
        # with standard off-policy TD(0) / Q-learning.
        #
        #   V(s) ← V(s) + α · [ r + γ · max_{a'} V(step(s,a')) − V(s) ]
        policy.update(state, reward_normalizer.normalise(reward))

        state = next_state

        # ── periodic evaluation ───────────────────────────────────────────────
        if (step_idx + 1) % cfg.log_interval == 0:
            det_error_trace.append(compute_deterministic_error(patterns, model))
            det_visit_frac_trace.append(window_det_visits / cfg.log_interval)
            window_det_visits = 0
            if cfg.method == 'curiosity_critic_ours_nnet':
                det_critic_vals = [
                    critic_nnet.predict((r, c))
                    for r in range(GRID_SIZE)
                    for c in range(DET_COLS)
                ]
                stoch_critic_vals = [
                    critic_nnet.predict((r, c))
                    for r in range(GRID_SIZE)
                    for c in range(DET_COLS, GRID_SIZE)
                ]
                nnet_critic_det_trace.append(float(np.mean(det_critic_vals)))
                nnet_critic_stoch_trace.append(float(np.mean(stoch_critic_vals)))

    return {
        'det_error_trace':        det_error_trace,
        'det_visit_frac_trace':   det_visit_frac_trace,
        'trajectory':             trajectory,
        'nnet_critic_det_trace':  nnet_critic_det_trace,
        'nnet_critic_stoch_trace': nnet_critic_stoch_trace,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_result(
    cfg: ExperimentConfig,
    traces: Dict[str, List[float]],
    output_dir: str,
) -> Path:
    """Persist experiment results as a binary pickle file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"result__{cfg.method}__seed{cfg.seed:04d}.pkl"
    path = Path(output_dir) / filename
    with open(path, 'wb') as f:
        pickle.dump({
            'method':    cfg.method,
            'seed':      cfg.seed,
            'grid_seed': cfg.grid_seed,
            'config':    cfg.__dict__,
            'grid': {
                'grid_size':     GRID_SIZE,
                'det_row_start': DET_ROW_START,
                'det_col_start': DET_COL_START,
                'det_rows':      DET_ROWS,
                'det_cols':      DET_COLS,
            },
            **traces,
        }, f)
    return path


def load_results(
    input_dir: str,
) -> Dict[str, Dict[str, List[List[float]]]]:
    """
    Load all .pkl result files from *input_dir* and group by method.

    Returns
    -------
    grouped : dict mapping method_name →
              {'det_error_trace': [[seed0_vals], [seed1_vals], ...],
               'det_visit_frac_trace': [[seed0_vals], [seed1_vals], ...]}
    """
    grouped: Dict[str, Dict[str, List]] = {}
    pkl_files = sorted(Path(input_dir).glob('*.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl result files found in: {input_dir}")
    for path in pkl_files:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        method = data['method']
        if method not in grouped:
            grouped[method] = {
                'det_error_trace':        [],
                'det_visit_frac_trace':   [],
                'nnet_critic_det_trace':  [],
                'nnet_critic_stoch_trace': [],
            }
        grouped[method]['det_error_trace'].append(data['det_error_trace'])
        grouped[method]['det_visit_frac_trace'].append(data['det_visit_frac_trace'])
        grouped[method]['nnet_critic_det_trace'].append(data.get('nnet_critic_det_trace', []))
        grouped[method]['nnet_critic_stoch_trace'].append(data.get('nnet_critic_stoch_trace', []))
    return grouped




# ──────────────────────────────────────────────────────────────────────────────
# Batch runner
# ──────────────────────────────────────────────────────────────────────────────

def run_batch(output_dir: str, **overrides) -> None:
    """
    Run every method in BATCH_METHODS across every seed in BATCH_SEEDS and
    save all result files to *output_dir*.

    Parameters
    ----------
    output_dir : directory where .pkl result files are written.
    **overrides : any ExperimentConfig field (e.g. total_steps, hidden) to
                  override the default value for all runs in the batch.
    """
    total = len(BATCH_METHODS) * len(BATCH_SEEDS)
    done  = 0

    for seed in BATCH_SEEDS:
        # Build one checkpoint per seed: MODEL_COLD_START_SEED + seed ensures
        # distinct initial weights across seeds while all methods for the same
        # seed load from an identical starting point.
        seed_cfg = ExperimentConfig(seed=seed, **{k: v for k, v in overrides.items()
                                                   if k in ExperimentConfig.__dataclass_fields__})
        if seed_cfg.model_cold_start:
            print(f"[batch] Building model checkpoint for seed={seed} "
                  f"({seed_cfg.model_cold_start_steps} warm-up steps, "
                  f"base_seed={MODEL_COLD_START_SEED}+{seed}) ...")
            model_checkpoint = build_model_checkpoint(seed_cfg)
            print("[batch] Checkpoint ready.\n")
        else:
            model_checkpoint = None

        for method in BATCH_METHODS:
            done += 1
            cfg = ExperimentConfig(method=method, seed=seed, **overrides)
            print(
                f"[batch {done}/{total}] method={method!r}  seed={seed}"
                f"  steps={cfg.total_steps:,}"
            )
            traces = run_experiment(cfg, model_checkpoint=model_checkpoint)
            path   = save_result(cfg, traces, output_dir)
            visit  = traces['det_visit_frac_trace']
            cumul_visit = sum(visit) / len(visit)
            print(
                f"[batch {done}/{total}] saved → {path}  "
                f"(final error={traces['det_error_trace'][-1]:.4f}  "
                f"det-visit={visit[-1]:.3f}  "
                f"cumul-det-visit={cumul_visit:.3f})"
            )
    print(f"\n[batch] All {total} runs complete.  Results in: {output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Command-line interface
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Curiosity-driven exploration experiment on a stochastic grid world.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── batch ─────────────────────────────────────────────────────────────────
    batch_p = sub.add_parser(
        'batch',
        help=(
            f'Run all methods ({", ".join(BATCH_METHODS)}) '
            f'over seeds {BATCH_SEEDS} and save results.'
        ),
    )
    batch_p.add_argument('--output-dir', default='results',
                         help='Directory for saving result files (default: results/).')
    batch_p.add_argument('--total-steps', type=int, default=35000,
                         help='Environment steps per run (default: 35,000).')
    batch_p.add_argument('--log-interval', type=int, default=LOG_INTERVAL,
                         help=f'Steps between evaluations (default: {LOG_INTERVAL:,}).')
    batch_p.add_argument('--hidden', type=int, default=1024,
                         help='World-model hidden units (default: 1024).')
    batch_p.add_argument('--lr-model',  type=float, default=0.001,
                         help='Adam learning rate for the world model (default: 0.001).')
    batch_p.add_argument('--hidden-rnd', type=int, default=128,
                         help='RND target/predictor hidden units (default: 128).')
    batch_p.add_argument('--rnd-output-dim', type=int, default=128,
                         help='RND random feature dimension (default: 128).')
    batch_p.add_argument('--lr-rnd', type=float, default=0.001,
                         help='Adam learning rate for the RND predictor (default: 0.001).')
    batch_p.add_argument('--lr-policy', type=float, default=0.05,
                         help='Q-learning step size α (default: 0.05).')
    batch_p.add_argument('--gamma',     type=float, default=0.0,
                         help='Discount factor γ (default: 0.0).')
    batch_p.add_argument('--epsilon',   type=float, default=0.3,
                         help='ε-greedy exploration rate (default: 0.3).')
    batch_p.add_argument('--cold-start-steps', type=int, default=COLD_START_STEPS,
                         help=f'Random steps used to estimate reward for V-table '
                              f'initialisation (default: {COLD_START_STEPS}).')
    batch_p.add_argument('--cold-start', action='store_true',
                         help='Enable cold-start V-table init from sampled rewards '
                              '(default: off; uses --q-init instead).')
    batch_p.add_argument('--q-init', type=float, default=3.0,
                         help='Fixed V-table init value used when cold-start is disabled '
                              '(default: 3.0).')
    batch_p.add_argument('--no-model-cold-start', action='store_true',
                         help='Disable model warm-up; world model starts from random '
                              'initialisation (default: warm-up enabled).')
    batch_p.add_argument('--model-cold-start-steps', type=int, default=100,
                         help='Random-policy steps used to warm up the world model '
                              'before the main loop (default: 100).')
    batch_p.add_argument('--continuous-pixels', action='store_true',
                         help='Use continuous Uniform[0,1] observations instead of binary.')

    return parser


def _print_config(cfg: ExperimentConfig, label: str = 'config') -> None:
    """Print all resolved ExperimentConfig fields before a run."""
    sep = '─' * 52
    print(sep)
    print(f"  [{label}] Resolved configuration")
    print(sep)
    print(f"  method           : {cfg.method}")
    print(f"  seed             : {cfg.seed}")
    print(f"  grid_seed        : {cfg.grid_seed}")
    print(f"  total_steps      : {cfg.total_steps:,}")
    print(f"  log_interval     : {cfg.log_interval}")
    print(f"  model_cold_start : {cfg.model_cold_start}  (steps={cfg.model_cold_start_steps})")
    print(f"  cold_start       : {cfg.cold_start}  (steps={cfg.cold_start_steps})")
    print(f"  q_init           : {cfg.q_init}  (V-table init when cold_start=False)")
    print(f"  det_region       : all rows, cols {DET_COL_START}–{DET_COL_START+DET_COLS-1}"
          f"  ({DET_ROWS}×{DET_COLS} = {DET_ROWS*DET_COLS} cells, left half)")
    print(f"  agent_start      : (15, 15)")
    print(f"  World model")
    print(f"    architecture   : Linear({2*GRID_SIZE}→{cfg.hidden})→ReLU→Linear({cfg.hidden}→{NUM_PIXELS})")
    print(f"    loss           : MSELoss (raw logits, no output activation)")
    print(f"    optimiser      : Adam  lr={cfg.lr_model}")
    print(f"  RND")
    print(f"    state input    : {2*GRID_SIZE} dims")
    print(f"    obs input      : {NUM_PIXELS} dims")
    print(f"    architecture   : Linear(input→{cfg.hidden_rnd})→ReLU→Linear({cfg.hidden_rnd}→{cfg.rnd_output_dim})")
    print(f"    predictor opt  : Adam  lr={cfg.lr_rnd}")
    print(f"  Policy")
    print(f"    type           : V-table  {GRID_SIZE}×{GRID_SIZE}")
    print(f"    lr (α)         : {cfg.lr_policy}")
    print(f"    gamma (γ)      : {cfg.gamma}")
    print(f"    epsilon (ε)    : {cfg.epsilon}")
    print(f"  discrete_pixels  : {cfg.discrete_pixels}")
    print(sep)


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.command == 'batch':
        _print_config(
            ExperimentConfig(
                total_steps      = args.total_steps,
                log_interval     = args.log_interval,
                hidden           = args.hidden,
                hidden_rnd       = args.hidden_rnd,
                rnd_output_dim   = args.rnd_output_dim,
                lr_model         = args.lr_model,
                lr_rnd           = args.lr_rnd,
                lr_policy        = args.lr_policy,
                gamma            = args.gamma,
                epsilon          = args.epsilon,
                cold_start_steps       = args.cold_start_steps,
                cold_start             = args.cold_start,
                q_init                 = args.q_init,
                model_cold_start       = not args.no_model_cold_start,
                model_cold_start_steps = args.model_cold_start_steps,
                discrete_pixels        = not args.continuous_pixels,
            ),
            label='batch — shared config (method/seed vary per run)',
        )
        print(f"  methods : {BATCH_METHODS}")
        print(f"  seeds   : {BATCH_SEEDS}")
        print(f"  output  : {args.output_dir!r}")
        print('─' * 52)
        run_batch(
            output_dir             = args.output_dir,
            total_steps            = args.total_steps,
            log_interval           = args.log_interval,
            hidden                 = args.hidden,
            hidden_rnd             = args.hidden_rnd,
            rnd_output_dim         = args.rnd_output_dim,
            lr_model               = args.lr_model,
            lr_rnd                 = args.lr_rnd,
            lr_policy              = args.lr_policy,
            gamma                  = args.gamma,
            epsilon                = args.epsilon,
            cold_start_steps       = args.cold_start_steps,
            cold_start             = args.cold_start,
            q_init                 = args.q_init,
            model_cold_start       = not args.no_model_cold_start,
            model_cold_start_steps = args.model_cold_start_steps,
            discrete_pixels        = not args.continuous_pixels,
        )


if __name__ == '__main__':
    main()
