# q_learning.py
# Q-Learning agent for the traffic light environment
from typing import Dict, Tuple
import numpy as np
import random


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy policy.
    - Q-table: dict mapping state -> np.array for action values
    - Actions: 0 (keep), 1 (switch)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int = 42,
        discretize_cap: int = 20,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = random.Random(seed)
        self.Q: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
        self.discretize_cap = discretize_cap

    def _discretize_state(self, state: Tuple[int, int, int, int, int]) -> Tuple[int, int, int, int, int]:
        n, s, e, w, p = state
        return (
            min(n, self.discretize_cap),
            min(s, self.discretize_cap),
            min(e, self.discretize_cap),
            min(w, self.discretize_cap),
            p,
        )

    def _ensure_state(self, state: Tuple[int, int, int, int, int]):
        if state not in self.Q:
            self.Q[state] = np.zeros(2, dtype=float)

    def select_action(self, state: Tuple[int, int, int, int, int]) -> int:
        d_state = self._discretize_state(state)
        self._ensure_state(d_state)
        if self.rng.random() < self.epsilon:
            return self.rng.choice([0, 1])
        return int(np.argmax(self.Q[d_state]))

    def update(self, state, action: int, reward: float, next_state, done: bool):
        d_state = self._discretize_state(state)
        d_next = self._discretize_state(next_state)
        self._ensure_state(d_state)
        self._ensure_state(d_next)

        best_next = np.max(self.Q[d_next])
        td_target = reward + (0.0 if done else self.gamma * best_next)
        td_error = td_target - self.Q[d_state][action]
        self.Q[d_state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
