from abc import ABC
from itertools import cycle
from typing import Tuple, Optional, Mapping, List, Sequence, NamedTuple, Any

import gym
import numpy as np


GoalHashable = Tuple[float]


class ISettableGoalEnv(ABC):
    _possible_goals: cycle[Any]
    _successes_per_goal: Mapping[GoalHashable, List[bool]] = dict()
    max_episode_len: int
    starting_agent_pos: np.ndarray

    def set_possible_goals(self, goals: Optional[np.ndarray], entire_space=False) -> None:
        if goals is None and entire_space:
            self._possible_goals = None
            self._successes_per_goal = dict()
            return

        self._possible_goals = cycle(np.random.permutation(goals))
        self._successes_per_goal = {tuple(g): [] for g in goals}

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        return dict(self._successes_per_goal)
