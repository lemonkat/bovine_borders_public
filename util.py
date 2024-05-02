import numpy as np

NUM_ROUNDS = int(1e4)

CORNERS = [
    (0, 0),
    (0, 19),
    (19, 19),
    (19, 0),
    None,
]

BOARD_ROT_ARR = np.array(
    [
        [0, 1, 2, 3, -1],
        [3, 0, 1, 2, -1],
        [2, 3, 0, 1, -1],
        [1, 2, 3, 0, -1],
    ],
    dtype=np.int32,
)

NEW_BOARD = np.full((20, 20), -1, dtype=np.int32)
for plr, pos in enumerate(CORNERS[:-1]):
    NEW_BOARD[pos] = plr

def in_bounds(pos):
    return 0 <= pos[0] < 20 and 0 <= pos[1] < 20

def nearby(pos):
    return [
        (pos[0] + 1, pos[1]),
        (pos[0] - 1, pos[1]),
        (pos[0], pos[1] + 1),
        (pos[0], pos[1] - 1),
    ]

def splitter(observation):
    return observation[0], observation[1]

NB_INBOUNDS = np.empty((20, 20), dtype=object)
for i in range(20):
    for j in range(20):
        NB_INBOUNDS[i, j] = [pos for pos in nearby((i, j)) if in_bounds(pos)]

def src_list(plr, dst, board):
    return [pos for pos in NB_INBOUNDS[dst] if board[pos] == plr]

def evaluate(environment, policy):
    time_step = environment.reset()
    episode_return = 0.0
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward.numpy()[0]

    return episode_return, [np.sum(time_step.observation[0].numpy()[0] == plr, axis=(0, 1)) for plr in range(4)]

def display_game(environment, policy):
    returns = []
    areas = []
    episode_return = 0.0
    time_step = environment.reset()
    print(("\n" * 20) + "\x1b[20A", end="\x1b7")
    print("\x1b8", end=environment.pyenv.envs[0].render())
    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        episode_return += time_step.reward.numpy()[0]
        print("\x1b8", end=environment.pyenv.envs[0].render())
    returns.append(episode_return)

    return episode_return, areas


