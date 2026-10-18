# Connect4 AlphaZero

A minimal, educational implementation of AlphaZero for Connect Four from first principles using **PyTorch**, **OpenSpiel** (`pyspiel`), and **pygame-ce**.

## Architecture
- `mcts.py`: Monte Carlo Tree Search with canonical board transformation (`get_canonical_obs`), PUCT selection (`Node.puct_score`), masked neural leaf expansion, root Dirichlet exploration noise, and alternating zero-sum backpropagation (`value = -value`), plus a step-by-step generator (`search_step_by_step`) for live visualization.
- `model.py`: Dual-headed Residual Convolutional Network (`ConnectFourNet`, ~323k parameters: $3 \times 3$ stem + 3 residual blocks with `BatchNorm2d` + $1 \times 1$ policy and value heads).
- `train.py`: `SelfPlayTrainer` implementing two-stage trajectory collection, $2\times$ horizontal reflection symmetry augmentation, sliding-window replay buffer (`deque`), and 14-game duplicate opening-book Arena evaluation.
- `play.py`: Interactive 60 FPS `pygame-ce` visualizer displaying the playable Connect Four board alongside live animated MCTS visit distributions $\pi(a|s)$ vs. raw network priors $P(s,a)$, per-column $Q(s,a)$, and a vertical win-evaluation gauge (`Net V(s)` vs. `MCTS Q(s)`).
- `connect4_gen1.pt` / `connect4_best.pt`: Generation 1 (36 self-play games) and Generation 2 (336 cumulative self-play games, 86% win rate vs. Gen 1) model checkpoints.

## Quickstart

```bash
# Play against the trained AlphaZero agent (with live MCTS telemetry)
uv run python play.py

# Run self-play training and duplicate Arena evaluation
uv run python train.py
```

