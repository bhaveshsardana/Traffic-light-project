# environment.py
# Custom 4-way intersection simulation for RL traffic light control
from typing import Dict, Tuple, List, Optional
import numpy as np
import random


class IntersectionEnv:
    """
    A simple 4-way intersection simulation with two phases:
    - Phase 0: North-South green, East-West red
    - Phase 1: East-West green, North-South red

    State:
    - Queue lengths: N, S, E, W (ints)
    - Current phase: 0 or 1

    Actions:
    - 0: Keep current phase
    - 1: Switch to next phase

    Dynamics:
    - Each second: cars arrive per direction with probability p_arrival[dir]
    - When phase green: cars depart per direction at rate depart_rate (per lane)

    Reward (per step):
    - r = -(alpha_wait * total_wait) - (alpha_queue * total_queue) + (beta_throughput * departed)

    Episode ends after max_steps.

    Metrics tracked per episode:
    - average_waiting_time (proxy): cumulative_wait / steps
    - average_queue_lengths: mean of queues per step
    - throughput: total departed vehicles
    - rewards: list of step rewards
    """

    def __init__(
        self,
        max_steps: int = 1000,
        initial_phase: int = 0,
        arrival_probs: Optional[Dict[str, float]] = None,
        depart_rate: int = 2,
        queue_cap: int = 50,
        seed: Optional[int] = None,
        alpha_wait: float = 1.0,
        alpha_queue: float = 0.2,
        beta_throughput: float = 0.5,
        switch_penalty: float = 0.0,
        animation: bool = False,
    ):
        self.directions = ["N", "S", "E", "W"]
        self.max_steps = max_steps
        self.initial_phase = initial_phase
        self.depart_rate = depart_rate
        self.queue_cap = queue_cap
        self.alpha_wait = alpha_wait
        self.alpha_queue = alpha_queue
        self.beta_throughput = beta_throughput
        self.switch_penalty = switch_penalty
        self.animation = animation

        if arrival_probs is None:
            self.arrival_probs = {"N": 0.3, "S": 0.3, "E": 0.3, "W": 0.3}
        else:
            self.arrival_probs = arrival_probs

        self.rng = random.Random(seed)
        np.random.seed(seed if seed is not None else np.random.randint(0, 10_000))

        self.step_count = 0
        self.phase = self.initial_phase
        self.queues: Dict[str, int] = {d: 0 for d in self.directions}

        self.cumulative_wait = 0.0
        self.cumulative_queue = 0.0
        self.total_departed = 0
        self.rewards: List[float] = []

    def reset(self) -> Tuple[Tuple[int, int, int, int, int], Dict]:
        self.step_count = 0
        self.phase = self.initial_phase
        self.queues = {d: 0 for d in self.directions}
        self.cumulative_wait = 0.0
        self.cumulative_queue = 0.0
        self.total_departed = 0
        self.rewards = []
        return self._get_state(), {}

    def _get_state(self) -> Tuple[int, int, int, int, int]:
        return (
            self.queues["N"],
            self.queues["S"],
            self.queues["E"],
            self.queues["W"],
            self.phase,
        )

    def _arrivals(self):
        for d in self.directions:
            if self.rng.random() < self.arrival_probs[d]:
                self.queues[d] = min(self.queue_cap, self.queues[d] + 1)

    def _departures(self) -> int:
        departed = 0
        if self.phase == 0:
            for d in ["N", "S"]:
                leave = min(self.depart_rate, self.queues[d])
                self.queues[d] -= leave
                departed += leave
        else:
            for d in ["E", "W"]:
                leave = min(self.depart_rate, self.queues[d])
                self.queues[d] -= leave
                departed += leave
        return departed

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int, int], float, bool, Dict]:
        assert action in [0, 1], "Invalid action: must be 0 (keep) or 1 (switch)"

        switch_cost = 0.0
        if action == 1:
            self.phase = 1 - self.phase
            switch_cost = self.switch_penalty

        self._arrivals()
        departed = self._departures()

        total_queue = sum(self.queues.values())
        total_wait_increment = total_queue

        self.cumulative_wait += total_wait_increment
        self.cumulative_queue += total_queue
        self.total_departed += departed

        reward = (
            -self.alpha_wait * total_wait_increment
            - self.alpha_queue * total_queue
            + self.beta_throughput * departed
            - switch_cost
        )
        self.rewards.append(reward)

        self.step_count += 1
        done = self.step_count >= self.max_steps

        if self.animation:
            self._render_ascii(departed, total_queue, reward)

        return self._get_state(), reward, done, {
            "departed": departed,
            "total_queue": total_queue,
            "step": self.step_count,
            "phase": self.phase,
        }

    def _render_ascii(self, departed: int, total_queue: int, reward: float):
        def bar(n): return "#" * min(n, 20)
        print(f"Step {self.step_count} | Phase: {'NS' if self.phase == 0 else 'EW'} | Departed: {departed} | Queue: {total_queue} | Reward: {reward:.2f}")
        print(f"N:{self.queues['N']:2d} {bar(self.queues['N'])}")
        print(f"S:{self.queues['S']:2d} {bar(self.queues['S'])}")
        print(f"E:{self.queues['E']:2d} {bar(self.queues['E'])}")
        print(f"W:{self.queues['W']:2d} {bar(self.queues['W'])}")
        print("-" * 50)

    def episode_metrics(self) -> Dict[str, float]:
        steps = max(1, self.step_count)
        return {
            "average_waiting_time": self.cumulative_wait / steps,
            "average_queue_length": self.cumulative_queue / steps,
            "throughput": self.total_departed,
            "total_reward": sum(self.rewards),
        }
