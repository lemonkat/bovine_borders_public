import numpy as np
import tensorflow as tf
import tf_agents as tfa

# specification data for the environment
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

# does the same thing as in sample_ais, but way faster
def allowed_dst_grid(board: np.ndarray[np.int32]) -> np.ndarray[bool]:
    owned = board == 0
    result = np.zeros((20, 20), dtype=bool)
    result[1:, :] |= owned[:-1, :]
    result[:-1, :] |= owned[1:, :]
    result[:, 1:] |= owned[:, :-1]
    result[:, :-1] |= owned[:, 1:]
    return np.logical_and(result, np.logical_not(owned))

# "environment" that the policy can interact with
class BBIntPyEnv(tfa.environments.PyEnvironment):
    def __init__(self):
        self._board = None
        self._action = None
        self._time_step = None

    def _current_time_step(self):
        return self._time_step

    def _get_observation(self):
        return self._board, allowed_dst_grid(self._board).astype(np.int32).flatten()

    def _reset(self):
        self._time_step = tfa.trajectories.restart(self._get_observation())
        return self._time_step
    
    def _step(self, action):
        self._time_step = tfa.trajectories.transition(self._get_observation(), 0)
        return self._time_step
    
    def action_spec(self):
        return ACTION_SPEC

    def observation_spec(self):
        return OBSERVATION_SPEC

    def set_board(self, board):
        self._board = board

class Svarog:
    def __init__(self, filename: str = "svarog_v6_data"):
        self._py_env = BBIntPyEnv()
        self._tf_env = tfa.environments.TFPyEnvironment(self._py_env)
        self._policy = tf.saved_model.load(filename)

    def __call__(self, boardIn: list[list[int]]) -> tuple[int, int]:
        self._py_env.set_board(np.array(boardIn, dtype=np.int32))
        raw_action = self._policy.action(self._tf_env.step(0)).action.numpy()[0]
        return raw_action // 20, raw_action % 20
    
SVAROG_COLOR = [60, 60, 200]