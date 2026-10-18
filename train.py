import random
from collections import deque
from typing import Deque, List, Tuple
import numpy as np
import pyspiel
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mcts import MCTS, get_canonical_obs
from model import ConnectFourNet


class SelfPlayTrainer:
    def __init__(
        self,
        num_iterations: int = 15,
        episodes_per_iter: int = 20,
        num_simulations: int = 60,
        batch_size: int = 64,
        epochs_per_iter: int = 5,
        lr: float = 1e-3,
        buffer_maxlen: int = 10_000,
        device: str = "cpu",
        resume_from: str | None = None,
    ):
        self.num_iterations = num_iterations
        self.episodes_per_iter = episodes_per_iter
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.epochs_per_iter = epochs_per_iter
        self.device = device

        self.game = pyspiel.load_game("connect_four")
        self.model = ConnectFourNet().to(self.device)
        if resume_from is not None:
            state_dict = torch.load(resume_from, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"Warm-started model weights from {resume_from}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4
        )
        self.replay_buffer: Deque[Tuple[np.ndarray, np.ndarray, float]] = deque(
            maxlen=buffer_maxlen
        )

    def execute_episode(
        self, temp_threshold: int = 15
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Plays one full self-play game using MCTS.
        Stage A (in-game): records (canonical_obs_t, pi_t, current_player_t).
        Stage B (post-game): converts to (canonical_obs_t, pi_t, z_t) and adds horizontal reflection.
        """
        state = self.game.new_initial_state()
        mcts = MCTS(self.model, c_puct=1.5, device=self.device)

        # Stage A: temporary in-game trajectory holding current_player_t
        episode_history: List[Tuple[np.ndarray, np.ndarray, int]] = []
        move_count = 0

        while not state.is_terminal():
            canonical_obs = get_canonical_obs(state)
            curr_player = state.current_player()

            # Always record the soft (temperature=1.0) visit distribution as the policy target
            pi = mcts.search(
                state,
                num_simulations=self.num_simulations,
                temperature=1.0,
                add_noise=(move_count < temp_threshold),
            )
            episode_history.append((canonical_obs, pi, curr_player))

            # Early game: sample proportionally to visit counts (exploration)
            # Late game: play greedily to get a sharp terminal outcome z (exploitation)
            if move_count < temp_threshold:
                action = int(np.random.choice(7, p=pi))
            else:
                action = int(np.argmax(pi))

            state.apply_action(action)
            move_count += 1

        # Stage B: game is terminal; assign z_t = returns[player_t] and discard player_t
        returns = state.returns()
        training_samples: List[Tuple[np.ndarray, np.ndarray, float]] = []

        for canonical_obs, pi, player in episode_history:
            z = float(returns[player])

            # 1. Original canonical sample
            training_samples.append((canonical_obs, pi, z))

            # 2. Horizontal reflection symmetry (flip columns left-to-right)
            flipped_obs = np.flip(canonical_obs, axis=-1).copy()
            flipped_pi = pi[::-1].copy()
            training_samples.append((flipped_obs, flipped_pi, z))

        return training_samples

    def train_epoch(self) -> Tuple[float, float]:
        """
        Runs epochs_per_iter passes over random minibatches from replay_buffer.
        Returns (mean_policy_loss, mean_value_loss).
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        self.model.train()
        policy_losses, value_losses = [], []
        num_batches = max(1, len(self.replay_buffer) // self.batch_size)

        for _ in range(self.epochs_per_iter):
            for _ in range(num_batches):
                batch = random.sample(self.replay_buffer, self.batch_size)
                obs_b = torch.from_numpy(np.array([b[0] for b in batch])).to(self.device)
                pi_b = torch.from_numpy(np.array([b[1] for b in batch])).to(self.device)
                z_b = torch.tensor(
                    [b[2] for b in batch], dtype=torch.float32, device=self.device
                )

                logits, value = self.model(obs_b)

                # Policy Cross-Entropy: - sum_a pi_a * log p_a
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -torch.mean(torch.sum(pi_b * log_probs, dim=-1))

                # Value Mean Squared Error: (z - v)^2
                value_loss = F.mse_loss(value.squeeze(-1), z_b)

                total_loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        return float(np.mean(policy_losses)), float(np.mean(value_losses))

    def evaluate_vs_random(
        self, num_games: int = 20, eval_sims: int = 30
    ) -> Tuple[int, int, int]:
        """
        Plays num_games against a uniform random opponent (half as Player 0, half as Player 1).
        Returns (wins, draws, losses).
        """
        self.model.eval()
        mcts = MCTS(self.model, c_puct=1.5, device=self.device)
        wins, draws, losses = 0, 0, 0

        for g in range(num_games):
            ai_player = g % 2
            state = self.game.new_initial_state()

            while not state.is_terminal():
                if state.current_player() == ai_player:
                    pi = mcts.search(
                        state,
                        num_simulations=eval_sims,
                        temperature=0.0,
                        add_noise=False,
                    )
                    action = int(np.argmax(pi))
                else:
                    action = random.choice(state.legal_actions())
                state.apply_action(action)

            ret = state.returns()[ai_player]
            if ret > 0:
                wins += 1
            elif ret < 0:
                losses += 1
            else:
                draws += 1

        return wins, draws, losses

    def evaluate_vs_model(
        self, opponent_model: ConnectFourNet, eval_sims: int = 40
    ) -> Tuple[int, int, int]:
        """
        Plays a 14-game duplicate match (all 7 opening columns x both colors)
        against a frozen benchmark model using identical MCTS simulation budgets.
        Returns (wins, draws, losses) for self.model.
        """
        self.model.eval()
        opponent_model.eval()
        mcts_curr = MCTS(self.model, c_puct=1.5, device=self.device)
        mcts_opp = MCTS(opponent_model, c_puct=1.5, device=self.device)
        wins, draws, losses = 0, 0, 0

        for opening_col in range(7):
            for curr_player_id in (0, 1):
                state = self.game.new_initial_state()
                # Force the opening ply so all 7 columns are tested symmetrically
                state.apply_action(opening_col)

                while not state.is_terminal():
                    if state.current_player() == curr_player_id:
                        pi = mcts_curr.search(
                            state,
                            num_simulations=eval_sims,
                            temperature=0.0,
                            add_noise=False,
                        )
                    else:
                        pi = mcts_opp.search(
                            state,
                            num_simulations=eval_sims,
                            temperature=0.0,
                            add_noise=False,
                        )
                    action = int(np.argmax(pi))
                    state.apply_action(action)

                ret = state.returns()[curr_player_id]
                if ret > 0:
                    wins += 1
                elif ret < 0:
                    losses += 1
                else:
                    draws += 1

        return wins, draws, losses

    def train(
        self,
        checkpoint_path: str = "connect4_best.pt",
        benchmark_model: ConnectFourNet | None = None,
    ) -> None:
        print(
            f"Starting AlphaZero Connect Four Training ({self.num_iterations} iterations, "
            f"{self.episodes_per_iter} games/iter, {self.num_simulations} MCTS sims/move)"
        )

        best_score = 50.0

        for iteration in range(1, self.num_iterations + 1):
            for _ in tqdm(
                range(self.episodes_per_iter),
                desc=f"Iter {iteration:2d}/{self.num_iterations} [Self-Play]",
            ):
                samples = self.execute_episode()
                self.replay_buffer.extend(samples)

            p_loss, v_loss = self.train_epoch()

            if benchmark_model is not None:
                wins, draws, losses = self.evaluate_vs_model(
                    benchmark_model, eval_sims=40
                )
                total_g = wins + draws + losses
                score_pct = 100.0 * (wins + 0.5 * draws) / total_g
                eval_str = f"vs Gen1: {wins:2d}W-{draws:2d}D-{losses:2d}L ({score_pct:4.0f}%)"
            else:
                wins, draws, losses = self.evaluate_vs_random(num_games=20, eval_sims=30)
                score_pct = 100.0 * wins / 20.0
                eval_str = f"vs Random: {wins:2d}W-{draws:2d}D-{losses:2d}L ({score_pct:4.0f}%)"

            if score_pct >= best_score:
                best_score = score_pct
                torch.save(self.model.state_dict(), checkpoint_path)
                save_tag = f"★ Saved {checkpoint_path} ({score_pct:.0f}%)"
            else:
                save_tag = f"Kept best ({best_score:.0f}%)"

            print(
                f"  -> Buffer: {len(self.replay_buffer):5d} | "
                f"Policy Loss: {p_loss:.4f} | Value Loss: {v_loss:.4f} | "
                f"{eval_str} | {save_tag}"
            )


if __name__ == "__main__":
    import os
    import shutil

    gen1_path = "connect4_gen1.pt"
    best_path = "connect4_best.pt"

    if not os.path.exists(gen1_path) and os.path.exists(best_path):
        shutil.copyfile(best_path, gen1_path)
        print(f"Saved Generation 1 benchmark checkpoint to {gen1_path}")

    gen1_benchmark = None
    if os.path.exists(gen1_path):
        gen1_benchmark = ConnectFourNet()
        gen1_benchmark.load_state_dict(
            torch.load(gen1_path, map_location="cpu", weights_only=True)
        )
        gen1_benchmark.eval()

    trainer = SelfPlayTrainer(
        num_iterations=15,
        episodes_per_iter=20,
        num_simulations=60,
        batch_size=64,
        epochs_per_iter=4,
        resume_from=gen1_path if os.path.exists(gen1_path) else None,
    )
    trainer.train(checkpoint_path=best_path, benchmark_model=gen1_benchmark)

