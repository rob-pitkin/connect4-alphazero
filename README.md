# Connect4 AlphaZero

A minimal, educational implementation of AlphaZero for Connect Four from first principles in PyTorch.

## Architecture
- `game.py`: Canonical Connect Four board, step logic, legal action masking, terminal evaluation.
- `mcts.py`: Pure MCTS implementation with PUCT selection, leaf neural evaluation, and alternating zero-sum backpropagation.
- `model.py`: Dual-headed convolutional neural network predicting prior probabilities and board value.
- `train.py`: Self-play data collection, replay buffer, and policy improvement loop.
- `play.py`: Terminal interactive match against the trained model.
