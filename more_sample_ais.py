# a few sample bots

# AUTHOR @LemonKat
# AUTHOR @Vodiboi
# ignore the bot names, its fine - @LemonKat

import random
from collections import deque
from typing import Any

CAPITALS = [
    (0, 0),
    (0, 19),
    (19, 19),
    (19, 0),
]

# helper functions
# anyone is allowed to import and use these


def dist(a: tuple[int], b: tuple[int]) -> int:
    """
    returns manhattan distance between 2 points
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearby(pos: tuple[int]) -> set[tuple[int]]:
    return {
        (pos[0] + 1, pos[1]),
        (pos[0] - 1, pos[1]),
        (pos[0], pos[1] + 1),
        (pos[0], pos[1] - 1),
    }


def in_bounds(pos: tuple[int]) -> bool:
    """
    returns if pos is in bounds
    """
    return 0 <= pos[0] < 20 and 0 <= pos[1] < 20


def reachable_from(board: list[list[int]], pos: tuple[int]) -> set[tuple[int]]:
    """
    returns the set of all in-bounds coordinates P such that
    a soldier at P can reach pos
    """
    result = set()
    for pos2 in nearby(pos):
        if in_bounds(pos2):
            if board[pos2[0]][pos2[1]] == 0:
                result.add(pos2)
    return result


def allowed_moves(board):
    """
    returns the set of all legal moves on board
    """
    allowed = set()
    for i in range(20):
        for j in range(20):
            if board[i][j] != 0:
                opts = reachable_from(board, (i, j))
                for opt in opts:
                    allowed.add((opt, (i, j)))

    return allowed


def connected(board):
    """
    return 2d list of which cells are connected back to the capital
    uses depth-first search algorithm
    """

    result = [[False] * 20 for i in range(20)]

    def dfs(pos):
        nonlocal result
        # already logged
        if result[pos[0]][pos[1]]:
            return

        if board[pos[0]][pos[1]] != 0:
            return

        adj = [p for p in nearby(pos) if in_bounds(p)]
        result[pos[0]][pos[1]] = True
        for c in adj:
            dfs(c)

    dfs((0, 0))
    return result


snake_path = [(0, 0)]
timesWithCnt = [0] * 500


def barrys_snake_bot(board):
    connected_cells = connected(board)
    allowed = [dst for src, dst in allowed_moves(board)]
    while True:
        if not len(snake_path):
            src, dst = random.choice(
                list(allowed_moves(board)),
            )
            snake_path.append(src)
            snake_path.append(dst)
            return src, dst

        # if timesWithCnt[len(snake_path)] > 200:
        #     snake_path.pop()
        #     return None
        # stuck in loop, so break out by ret
        # move = random.choice(allowed_moves)
        # snake_path.append(move)
        # return move
        i1, j1 = snake_path[-1]

        if board[i1][j1] == 0 and connected_cells[i1][j1]:
            moves = [move for move in nearby((i1, j1)) if move in allowed]
            # moves.sort(key = lambda x: x[0]**2 + x[1]**2 - x[0]*x[1])
            if len(moves):
                next_point = random.choice(moves)  # moves[0]
                break

        snake_path.pop()

    snake_path.append(next_point)
    # timesWithCnt[len(snake_path)] += 1
    return (i1, j1), next_point


# # always passes
# def pass_bot(board):
#     return None


# returns a random legal move
def random_bot(board):
    return random.choice(list(allowed_moves(board)))


class HeatmapFuncBotBase:
    """
    this class behaves like a function.
    Given another function, it will randomly pick
    from the coordinates that maximize/minimize that
    function's output.
    """

    def __init__(self, func, lowest=False):
        """
        func must be a function from board, int, int -> float.
        If lowest is true, this bot will try to minimize the
        output of func.
        Otherwise, it will try to maximize the output of func.
        """
        self.func = func
        self.lowest = lowest

    def __call__(self, board):
        max_score = float("-inf")
        moves = []
        for (i1, j1), (i2, j2) in allowed_moves(board):
            score = self.func(board, i2, j2) * (-1 if self.lowest else 1)
            if score > max_score:
                max_score = score
                moves = []
            if score == max_score:
                moves.append(((i1, j1), (i2, j2)))

        return random.choice(moves) if len(moves) else None


curve_bot = HeatmapFuncBotBase(
    (
        lambda board, i, j: max(
            [i * j, 2 * i, 2 * j],
        )
        * (2 - (board[i][j] == -1))
    ),
    lowest=True,
)

curve_bot_2 = HeatmapFuncBotBase(
    lambda board, i, j: sum(
        [i * j, i * i / (j + 1), j * j / (i + 1)],
    ),
    lowest=True,
)

def_bot = HeatmapFuncBotBase(lambda board, i, j: i + j, True)


def pierce_func(board, i, j):
    choices = [i + j]
    if board[0][19] == 1:
        choices.append(i)
    if board[19][0] == 3:
        choices.append(j)
    if board[19][19] == 2:
        choices.append(abs(i - j))
        choices.append(abs(i - j))
    return random.choice(choices)


pierce_bot = HeatmapFuncBotBase(
    pierce_func,
    lowest=True,
)


def square_func(board, i, j):
    return random.choice([max(abs(10 - i), abs(10 - j)), i + j])


square_bot = HeatmapFuncBotBase(
    square_func,
    lowest=True,
)


class DeltaBot:
    def __init__(self):
        self.mem = [[-1] * 20 for i in range(20)]
        self.last_change = [[0] * 20 for i in range(20)]
        self.round = 0

    def __call__(self, board):
        for i in range(20):
            for j in range(20):
                if self.mem[i][j] != board[i][j]:
                    self.last_change[i][j] = self.round
        self.mem = board
        self.round += 1

        best_score = float("-inf")
        moves = []
        for (i1, j1), (i2, j2) in allowed_moves(board):
            score = self.last_change[i2][j2] - i2 - j2
            if score > best_score:
                best_score = score
                moves = []
            if score == best_score:
                moves.append(((i1, j1), (i2, j2)))

        return random.choice(moves) if len(moves) else None


def strict_static_map_bot(heatmap):
    def func(board):
        max_score = float("-inf")
        moves = []
        for (i1, j1), (i2, j2) in allowed_moves(board):
            score = heatmap[i2][j2]
            if score > max_score:
                max_score = score
                moves = []
            if score == max_score:
                moves.append(((i1, j1), (i2, j2)))
        if not len(moves):
            return None
        return random.choice(moves)

    return func


defensive_bot = strict_static_map_bot(
    [[-i - j for j in range(20)] for i in range(20)],
)
offensive_bot = strict_static_map_bot(
    [[i + j for j in range(20)] for i in range(20)],
)

defbot_2 = strict_static_map_bot(
    [[-min(i, j) for j in range(20)] for i in range(20)],
)

saboNeedle = strict_static_map_bot(
    [[-(20 - j - i) for j in range(20)] for i in range(20)]
)


curve0 = strict_static_map_bot(
    [[-(i + j + (i * j)) for j in range(20)] for i in range(20)]
)

curve1 = strict_static_map_bot(
    [[-(i * i + j * j - (i * j)) for j in range(20)] for i in range(20)]
)

curve2 = strict_static_map_bot([[(-(i * j)) for j in range(20)] for i in range(20)])

curve3 = strict_static_map_bot(
    [[-((20 - i) ^ 2 + (20 - j) ^ 2) for j in range(20)] for i in range(20)]
)

curve4 = strict_static_map_bot(
    [[(i * i + i * j - j * j) for j in range(20)] for i in range(20)]
)

curve5 = strict_static_map_bot(
    [[(400 - (i**2) - (j**2)) for j in range(20)] for i in range(20)]
)

curve6 = 0


# can do way better
# def pieceDist(board:list[list[int]], cell:tuple[int]) -> int:
#     if cell == -1: return False
#     a = False
#     for c in CAPITALS:
#         if c == cell:
#             a = True
#     if not a:
#         return False
#     q:deque = deque()
#     q.append((cell, 0))
#     seen:set[tuple[int]] = set()
#     valid = lambda x: (x >= 0 and x < len(board))
#     while (len(q) != 0):
#         spot, dist = q.popleft()
#         if any([
#             spot in seen,
#             not valid(spot[0]),
#             not valid(spot[1]),
#             board[spot[0]][spot[1]] != cell
#         ]):
#             continue
#         if spot in CAPITALS:
#             return dist
#     return False


EMPTY_CELL = -1
# ah yes infinity
INF = 1000


# this is lazy code. If speed becomes issue
# or want to exapnd, implement this with
# simple (or complex if needed) knapsack dp
#
# actually wait knapsack won't work well
# with all corners, maybe this is the fastest?
def distMap(board: list[list[int]]):
    # this is the fastest way to
    # initialize a matrix, in some
    # problems it actually matters
    ans = [[INF] * 20 for i in range(20)]
    invalid = lambda x: not (x >= 0 and x < len(board))
    for cx, cy in CAPITALS:
        q = deque()
        value = board[cy][cx]
        seen = set()
        if value == EMPTY_CELL:
            continue
        q.append(((cy, cx), 0))
        while len(q):
            pos, dist = q.popleft()
            y, x = pos
            if (
                invalid(y)
                or invalid(x)
                or pos in seen
                or ans[y][x] < dist
                or board[y][x] != value
            ):
                continue
            seen.add(pos)
            ans[y][x] = dist
            q.append(((y + 1, x), dist + 1))
            q.append(((y - 1, x), dist + 1))
            q.append(((y, x + 1), dist + 1))
            q.append(((y, x - 1), dist + 1))
    ans = [[-(ans[i][j] - i - j) for j in range(20)] for i in range(20)]
    return ans


debug = [0]
weird_1 = lambda board: (
    x := strict_static_map_bot(distMap(board))(board),
    debug.__setitem__(0, 1),
    x,
)[-1]

weird_2 = strict_static_map_bot(
    [[((i**2) / (j) - (j**2) / (i)) for j in range(1, 21)] for i in range(1, 21)]
)

weird_3 = strict_static_map_bot(
    [[(2**i) + (2**j) for j in range(20)] for i in range(20)]
)


# print(debug, end="--\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n")


def heatmapFromFile(file: str) -> list[list[int]]:
    with open(file, "r") as f:
        tabl = [list(map(int, i.split())) for i in f.read().split("\n")]
    return tabl


weird_4 = strict_static_map_bot(heatmapFromFile("mp2.txt"))

weird_5 = strict_static_map_bot([[-min(i, j) for j in range(20)] for i in range(20)])


moves = [(0, 0)]


def snaek(board):
    i = -1
    while (
        i >= -len(moves)
        and len([j for j in nearby(moves[i]) if board[j[0]][j[1]] > 0]) == 0
    ):
        i -= 1

    move = (
        random.choice([j for j in nearby(moves[i]) if j > 0])
        if i >= -len(moves)
        else random_bot(board)
    )
    moves.append(move)
    return move


# TODO: make
# def arulmozhivarmanWeightPos()


def vexing_vandiyadevan(board):
    # get all moves, pass if can't move
    moves = allowed_moves(board)
    if len(moves) == 0:
        return None
    moves = list(moves)

    # sort the moves by distance from capital,
    # plus a small random factor which sometimes
    # helps to break stalemates
    moves.sort(
        key=lambda move: (move[0][0] + move[0][1] + random.choice([0, 1, 2, 3, 5]))
    )
    # pick the first move x1,y1 -> x2,y2 that satisfies
    # |(x1)(x2) - (y1)(y2)| <= the number of possible moves.
    # this threshold is small initially, but increases as the
    # bot expands. The moves that best satisfy the inequality
    # tend to lay on the diagonal line from their capital.
    for e, s in moves:
        if abs(e[0] * s[0] - e[1] * s[1]) <= len(moves):
            return (e, s)
    # no valid moves were found, so see if any
    # of those moves goes to an empty spot. If
    # so, take it.
    for e, s in moves:
        if board[e[0]][e[1]] == 0:
            return (e, s)
    # ok the bot has now given up and will random it.
    return random.choice(moves)


class PersistentPunk:
    # im doing this LK, don't mess with it - @Vodiboi
    def __init__(self):
        self.cell = (0, 0)
        self._a()

    def sign(self, n: int):
        return 1 if n > 0 else -1 if n < 0 else 0

    def bfsNearest(self, board, cell):
        q = deque()
        q.append((cell, 0))
        seen = set()
        while len(q):
            c, d = q.popleft()
            if not in_bounds(c) or c in seen:
                continue
            seen.add(c)
            if board[c[0]][c[1]] == 0:
                return (c, d)
            for x in nearby(c):
                q.append((x, d + 1))
        return None

    def updateCellOne(self, board):
        # pick new random non-occupied cell
        nums = [(i, j) for i in range(20) for j in range(20)]
        random.shuffle(nums)
        # print(nums)
        for i in nums:
            if board[i[0]][i[1]] != 0:
                self.cell = i
                break

    def updateCellTwo(self, board, threshold=3):
        # pick a cell at most 3 away
        self.updateCellOne(board)  # randomly update
        nums = [(i, j) for i in range(20) for j in range(20)]
        random.shuffle(nums)
        for i in nums:
            m = self.allNotOccupiedInDist(board, i, threshold)
            if len(m):
                self.cell = random.choice(m)
                break

    def allNotOccupiedInDist(self, board, cell, dist=3):
        return [
            (i, j)
            for i in range(20)
            for j in range(20)
            if abs(i - cell[0]) + abs(j - cell[1]) == dist and board[i][j] != 0
        ]

    def updateCellThree(self, board, l=range(1, 40)):
        for r in l:
            m = self.allNotOccupiedInDist(board, (0, 0), r)
            if len(m):
                # weigh enemy over random
                for j in m:
                    if board[j[0]][j[1]] != -1:
                        self.cell = j
                        return
                self.cell = random.choice(m)
                return

    def _a(self):
        self.l1 = list(range(1, 40))
        self.l2 = list(range(1, 20)) + [23, 25, 29, 31, 35, 37] + list(range(20, 40))
        self.l3 = [i for i in range(1, 40) if i % 2] + [
            i for i in range(1, 40) if not i % 2
        ]
        self.l4 = [i for i in range(1, 40) if i % 5 in (0, 1)] + [
            i for i in range(1, 40) if i % 5 not in (0, 1)
        ]
        self.l5 = [i for i in range(1, 40) if i % 9 not in (3, 4)] + [
            i for i in range(1, 40) if i % 9 in (3, 4)
        ]

    def __call__(self, board) -> tuple[tuple[int]]:
        if board[self.cell[0]][self.cell[1]] == 0:
            self.updateCellThree(board, self.l1)
        # move towards cell
        # algorithm for doing so:
        # pick nearest cell that belongs to you
        # and move towards it. If somehow it is
        # blocked by itself, then that's a closer
        # cell.
        nearest, x = self.bfsNearest(board, self.cell)
        # figure out which way to move
        self.xDir = self.sign(-(nearest[0] - self.cell[0]))
        self.yDir = self.sign(-(nearest[1] - self.cell[1]))
        # print(self.xDir, self.yDir)
        if (random.randint(0, 1) and self.xDir != 0) or (
            self.xDir != 0 and self.yDir == 0
        ):
            # move in x
            return (nearest, (nearest[0] + self.xDir, nearest[1]))

        return (nearest, (nearest[0], nearest[1] + self.yDir))


__all__ = [
    "square_bot",
    "pierce_bot",
    "curve_bot",
    "curve_bot_2",
    "DeltaBot",
    "offensive_bot",
    "defensive_bot",
    "saboNeedle",
    "curve0",
    "curve1",
    "curve2",
    "curve3",
    "curve4",
    "curve5",
    "curve6",
    "weird_1",
    "weird_2",
    "weird_3",
    "weird_4",
    "weird_5",
    "snaek",
    "barrys_snake_bot",
    "vexing_vandiyadevan",
    "PersistentPunk",
]
