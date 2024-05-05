# a few sample bots

# AUTHOR @LemonKat
# AUTHOR @Vodiboi
# ignore the bot names, its fine - LemonKat

import random


######################
#  HELPER FUNCTIONS  #
######################

# anyone is allowed to use these


def dist(a: tuple[int], b: tuple[int]) -> int:
    """
    returns manhattan distance between 2 points
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def nearby(pos: tuple[int]) -> set[tuple[int]]:
    return [
        (pos[0] + 1, pos[1]),
        (pos[0] - 1, pos[1]),
        (pos[0], pos[1] + 1),
        (pos[0], pos[1] - 1),
    ]


def in_bounds(pos: tuple[int]) -> bool:
    """
    returns if pos is in bounds
    """
    return 0 <= pos[0] < 20 and 0 <= pos[1] < 20


def reachable_from(board: list[list[int]], pos: tuple[int]) -> set[tuple[int]]:
    """
    returns the set of all in-bounds coordinates X
    such that X -> pos is a legal move
    """
    result = []
    for pos2 in nearby(pos):
        if in_bounds(pos2):
            if board[pos2[0]][pos2[1]] == 0:
                result.append(pos2)
    return result


def allowed_moves(board):
    """
    returns a list of all legal moves on board
    """
    allowed = []
    for i in range(20):
        for j in range(20):
            if board[i][j] != 0:
                opts = reachable_from(board, (i, j))
                for opt in opts:
                    allowed.append((opt, (i, j)))

    return allowed


def allowed_dsts(board):
    """
    returns a list of all legal destinations on board
    """
    allowed = []
    for i in range(20):
        for j in range(20):
            if board[i][j] != 0:
                pos = i, j
                if any(
                    in_bounds(src) and board[src[0]][src[1]] == 0
                    for src in nearby(
                        pos,
                    )
                ):
                    allowed.append(pos)
    # print(allowed)
    return allowed


def connected(board):
    """
    return 2d list of which cells are connected back to (0, 0)
    uses depth-first search algorithm rather than reachable_from for speed
    """

    result = [[False] * 20 for i in range(20)]

    def dfs(pos):
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


def pick_random_source(board, dest):
    """
    given a random destination cell, picks a valid source cell for this
    """
    if len(reachable_from(board, dest)):
        return random.choice(reachable_from(board, dest))
    return None


##############
#  THE BOTS  #
##############


def pass_pompom(board):
    """
    always passes its turn
    """
    return None


def random_raven(board):
    """
    returns a random legal move
    name context: natasha is Raven in HI3.
    If you dont understand, thats fine
    """
    return random.choice(allowed_dsts(board))


def defensive_dan_heng(board):
    """
    Presents a hard to penetrate diagonal front-line
    """
    best_score = float("inf")
    moves = []
    for move in allowed_moves(board):
        score = move[1][0] + move[1][1]
        if score < best_score:
            best_score, moves = score, [move]
        elif score == best_score:
            moves.append(move)
    return random.choice(moves)


class AdjacentAsta:
    """
    Showcasing bot memory

    If an object has a method named "__call__",
    you can call that object like a function.
    By using this and instance variables, we can
    create bots that remember things.

    For example:

    bot = AdjacentAsta()
    asta_move = bot(board)

    is the same as

    bot = AdjacentAsta()
    asta_move = bot.__call__(board)


    How this bot works:
    Try to go next to where you previously went.
    If no such move is available, then pick a move at random.
    """

    def __init__(self):
        self._prev = 0, 0

    def __call__(self, board):
        a = connected(board)
        if board[self._prev[0]][self._prev[1]] == 0:
            allowed_dsts = []
            for pos in nearby(self._prev):
                if in_bounds(pos) and board[pos[0]][pos[1]] != 0:
                    allowed_dsts.append(pos)

            if len(allowed_dsts):
                move = self._prev, random.choice(allowed_dsts)
                self._prev = move[1]
                return move

        src, dst = random.choice(allowed_moves(board))
        self._prev = dst
        return src, dst


#################
#  YOUR BOT(S)  #
#################

# It is recommended to put all code for your bot here
