import math
from typing import Dict, Optional, Tuple
import numpy as np
import torch


def get_canonical_obs(state) -> np.ndarray:
    """
    Converts OpenSpiel Connect Four observation into a (2, 6, 7) float32 array
    where channel 0 is always the current player's pieces and channel 1 is the opponent's.
    """
    obs = np.array(state.observation_tensor(0)).reshape(3, 6, 7)
    if state.current_player() == 1:
        return np.stack([obs[1], obs[0]], axis=0).astype(np.float32)
    return np.stack([obs[0], obs[1]], axis=0).astype(np.float32)


class Node:
    def __init__(
        self,
        parent: Optional["Node"] = None,
        action_taken: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children: Dict[int, "Node"] = {}
        self.visit_count: int = 0
        self.value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def puct_score(self, child: "Node", c_puct: float) -> float:
        """
        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * (sqrt(N(s)) / (1 + N(s, a)))
        Note: child.q_value is stored from the child's (opponent's) perspective,
        so from the parent's (current player's) perspective, exploitation is -child.q_value.
        """
        q = -child.q_value
        u = c_puct * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))
        return q + u


class MCTS:
    def __init__(self, model: torch.nn.Module, c_puct: float = 1.5, device: str = "cpu"):
        self.model = model
        self.c_puct = c_puct
        self.device = device

    def select(self, root_node: Node, root_state) -> Tuple[Node, object]:
        node = root_node
        state = root_state.clone()

        while node.is_expanded() and not state.is_terminal():
            best_action = max(
                node.children.keys(),
                key=lambda a: node.puct_score(node.children[a], self.c_puct),
            )
            state.apply_action(best_action)
            node = node.children[best_action]

        return node, state

    def expand(self, leaf_node: Node, state) -> float:
        if state.is_terminal():
            returns = state.returns()
            # If someone won, the previous player made the winning move.
            # From the current (terminal leaf) player's perspective, they lost (-1.0).
            if returns[0] != 0.0:
                return -1.0
            return 0.0

        canonical_obs = get_canonical_obs(state)
        obs_tensor = torch.from_numpy(canonical_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor)

        logits_np = logits.squeeze(0).cpu().numpy()
        value_scalar = float(value.item())

        # Slice strictly to legal actions for numerical stability
        legal_actions = state.legal_actions()
        legal_logits = logits_np[legal_actions]
        exp_logits = np.exp(legal_logits - np.max(legal_logits))
        legal_priors = exp_logits / np.sum(exp_logits)

        for action, prior in zip(legal_actions, legal_priors):
            leaf_node.children[action] = Node(
                parent=leaf_node,
                action_taken=action,
                prior=float(prior),
            )

        return value_scalar

    def backpropagate(self, leaf_node: Node, value: float) -> None:
        curr = leaf_node
        while curr is not None:
            curr.visit_count += 1
            curr.value_sum += value
            value = -value  # Flip perspective at each level up the tree
            curr = curr.parent

    def apply_dirichlet_noise(
        self, root: Node, alpha: float = 0.3, epsilon: float = 0.25
    ) -> None:
        actions = list(root.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = root.children[action]
            child.prior = (1.0 - epsilon) * child.prior + epsilon * float(n)

    def get_action_probs(self, root: Node, temperature: float = 1.0) -> np.ndarray:
        visits = np.zeros(7, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count

        if temperature < 1e-3:
            # Greedy argmax selection
            best_action = int(np.argmax(visits))
            probs = np.zeros(7, dtype=np.float32)
            probs[best_action] = 1.0
            return probs

        # Soft visit count distribution: N^(1/tau) / sum(N^(1/tau))
        scaled_visits = visits ** (1.0 / temperature)
        total = np.sum(scaled_visits)
        if total > 0:
            return scaled_visits / total
        return visits

    def search_step_by_step(
        self,
        root_state,
        num_simulations: int = 50,
        add_noise: bool = False,
    ):
        """
        Generator that yields (sim_idx, root_node, raw_priors, raw_value) after each
        MCTS simulation so an interactive UI can animate the search distribution in real time.
        """
        self.model.eval()
        root = Node()
        raw_value = self.expand(root, root_state)

        raw_priors = np.zeros(7, dtype=np.float32)
        for action, child in root.children.items():
            raw_priors[action] = child.prior

        if add_noise and len(root.children) > 0:
            self.apply_dirichlet_noise(root)

        for sim_idx in range(1, num_simulations + 1):
            leaf_node, sim_state = self.select(root, root_state)
            value = self.expand(leaf_node, sim_state)
            self.backpropagate(leaf_node, value)
            yield sim_idx, root, raw_priors, raw_value

    def search(
        self,
        root_state,
        num_simulations: int = 50,
        temperature: float = 1.0,
        add_noise: bool = False,
    ) -> np.ndarray:
        root = None
        for _, root, _, _ in self.search_step_by_step(
            root_state, num_simulations=num_simulations, add_noise=add_noise
        ):
            pass
        assert root is not None
        return self.get_action_probs(root, temperature=temperature)

