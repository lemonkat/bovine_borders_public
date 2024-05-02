import random
from util import src_list, CORNERS, NB_INBOUNDS, NUM_ROUNDS

import numpy as np
import tf_agents as tfa


ACTION_SPEC = tfa.specs.array_spec.BoundedArraySpec(
    shape=(),
    dtype=np.int32,
    minimum=0,
    maximum=399,
    name="action",
)

# contains a mask of allowed moves
OBSERVATION_SPEC = tfa.specs.array_spec.BoundedArraySpec(
    shape=(
        20,
        20,
    ),
    dtype=np.int32,
    minimum=-1,
    maximum=3,
    name="observation",
), tfa.specs.array_spec.BoundedArraySpec(
    shape=(400,),
    dtype=np.int32,
    minimum=0,
    maximum=1,
    name="mask",
)

TIME_STEP_SPEC = tfa.trajectories.time_step_spec(OBSERVATION_SPEC)


class BBPyEnv(tfa.environments.py_environment.PyEnvironment):
    def __init__(self):
        self._board = np.empty((20, 20), dtype=np.int32)
        self._dst_result = np.empty((20, 20), dtype=np.int32)
        self._workspace = np.empty((2, 20, 20), dtype=bool)
        self._amts = np.empty((4,), dtype=np.int32)
        self._reset()

    def _reward(self):
        # return 1.0 - np.argsort(self._amts)[0] / 3.0
        return (self._amts[0] / 400.0) ** 2
        # return (self._amts[0] - np.mean(self._amts[1:])) / 400.0

    def _reset(self):
        self._episode_ended = False
        self._round_num = 0
        self._board[...] = -1
        for plr, corner in enumerate(CORNERS[:-1]):
            self._board[corner] = plr
        self._amts[:] = 1

        self._num_kills = 0
        self._time_step = tfa.trajectories.time_step.restart(self._get_observation())
        return self._time_step

    def _current_time_step(self):
        return self._time_step

    def __getitem__(self, pos):
        return self._board[pos]

    def __setitem__(self, pos, plr):
        tgt = self[pos]
        if plr == tgt:
            return
        self._board[pos] = plr
        if plr != -1:
            self._amts[plr] += 1
        if tgt != -1:
            self._amts[tgt] -= 1
            if pos == CORNERS[tgt]:
                self._board[self._board == tgt] = -1
                self._amts[tgt] = 0

                if plr == 0:
                    self._num_kills += 1

    def action_spec(self):
        return ACTION_SPEC

    def observation_spec(self):
        return OBSERVATION_SPEC

    def _get_observation(self):
        return self._board, self._allowed_moves(0).flatten()
        
    def _step(self, action):
        
        if self._episode_ended:
            return self.reset()
        
        self._num_kills = 0
        

        # get moves
        actions = [(0, (action // 20, action % 20))]
        for plr in range(1, 4):
            if self._amts[plr] > 0:
                moves = np.argwhere(self._allowed_moves(plr))
                actions.append((plr, tuple(random.choice(moves))))

        random.shuffle(actions)
        # handle moves in random order
        for plr, dst in actions:
            if self._amts[plr] > 0:
                if self[dst] == plr:
                    continue

                allowed_srcs = src_list(plr, dst, self._board)

                if not len(allowed_srcs):
                    continue

                src = random.choice(allowed_srcs)
                
                connected = self._is_connected(src)

                if self[dst] == -1:
                    self[dst] = plr
                    if not connected:
                        self[src] = -1
                else:
                    self[dst] = plr if connected else -1
                    self[src] = -1

        self._round_num += 1
        # game over
        if self._round_num == NUM_ROUNDS:
            self._episode_ended = True
            self._time_step = tfa.trajectories.time_step.termination(
                self._get_observation(),
                200.0 * self._num_kills,
            )

        # agent dead
        elif self._amts[0] == 0:
            self._episode_ended = True
            self._time_step = tfa.trajectories.time_step.termination(
                self._get_observation(),
                -600.0,
            )

        # all others dead
        elif np.all(self._amts[1:] == 0):
            self._episode_ended = True
            self._time_step = tfa.trajectories.time_step.termination(
                self._get_observation(),
                200 * self._num_kills + NUM_ROUNDS - self._round_num,
            )

        # otherwise
        else:
            self._time_step = tfa.trajectories.time_step.transition(
                self._get_observation(),
                reward=self._reward() + 200 * self._num_kills,
            )
        return self._time_step

    def render(self):
        result = []
        for row in self._board:
            result.append("".join("ABCD "[num] for num in row))
        return "\n".join(result)
    
    def _allowed_moves(self, plr):
        np.equal(self._board, plr, out=self._workspace[0])
        self._workspace[1, -1] = 0
        self._workspace[1, :-1, :] = self._workspace[0, 1:, :]
        self._workspace[1, 1:, :] |= self._workspace[0, :-1, :]
        self._workspace[1, :, :-1] |= self._workspace[0, :, 1:]
        self._workspace[1, :, 1:] |= self._workspace[0, :, :-1]
        np.logical_not(self._workspace[0], out=self._workspace[0])
        np.logical_and(self._workspace[0], self._workspace[1], out=self._dst_result)
        return self._dst_result
        
    def _is_connected(self, pos):
        plr = self[pos]
        tgt = CORNERS[plr]

        reached = self._workspace[0]

        np.not_equal(self._board, plr, out=reached)
        stack = [pos]

        while stack:
            pos = stack.pop()
            if reached[pos]:
                continue
            elif pos == tgt:
                return True
            reached[pos] = True
            stack.extend(NB_INBOUNDS[pos])
        return False
        

def BBTfEnv():
    return tfa.environments.tf_py_environment.TFPyEnvironment(
        BBPyEnv(),
    )

def BBProcEnv(batch_size):
    return tfa.environments.tf_py_environment.TFPyEnvironment(
        tfa.environments.parallel_py_environment.ParallelPyEnvironment(
            [lambda: BBPyEnv()] * batch_size,
        ),
    )

def BBThreadEnv(batch_size):
    return tfa.environments.tf_py_environment.TFPyEnvironment(
        tfa.environments.batched_py_environment.BatchedPyEnvironment(
            [BBPyEnv() for _ in range(batch_size)],
        )
    )

if __name__ == "__main__":
    from util import splitter
    tfa.environments.utils.validate_py_environment(BBPyEnv(), episodes=1, observation_and_action_constraint_splitter=splitter)
