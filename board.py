# AUTHOR @LemonKat
# AUTHOR @Vodiboi

"""
main board code and helper functions
"""

import random
import time
import traceback
import shutil
from contextlib import redirect_stdout

import numpy as np
import tkinter as tk
from tkinter import font

pygame = None


__all__ = ["Board", "DEFAULT_COLORS"]

CORNERS = np.array(
    [[0, 0], [0, 19], [19, 19], [19, 0]],
    dtype=np.int16,
)

NEARBY = np.array(
    [[0, 1], [0, -1], [1, 0], [-1, 0]],
    dtype=np.int16,
)

_TK_MARGIN = 50


def _convert_ansi(color: tuple[int, int, int]) -> str:
    """
    converts RGB to an ANSI color code string.
    """

    r, g, b = color

    if r == g == b:
        # grayscale
        if r < 8:
            num = 16
        elif r > 248:
            num = 231

        else:
            num = 232 + int(r * 24 / 256)

    else:
        r = int(r * 6 / 256)
        g = int(g * 6 / 256)
        b = int(b * 6 / 256)

        num = 16 + 36 * r + 6 * g + b
    return f"\x1b[38;5;{num}m"


def _convert_hex(color: tuple[int, int, int]) -> str:
    """
    converts RGB to a hexadecimal string.
    """
    r, g, b = color
    hex_str = "0123456789ABCDEF"
    return "".join(
        [
            "#",
            hex_str[r >> 4],
            hex_str[r & 15],
            hex_str[g >> 4],
            hex_str[g & 15],
            hex_str[b >> 4],
            hex_str[b & 15],
        ]
    )


def _rotate_pos(steps: int, pos: tuple[int, int]) -> tuple[int, int]:
    """
    rotates pos int quarters ccw and returns that
    """
    i, j = pos
    for _ in range(steps % 4):
        i, j = 19 - j, i
    return i, j

def _rotate_move(steps: int, move: tuple[tuple[int]]) -> tuple[tuple[int]]:
    """
    rotates move int quarters ccw and returns that
    """
    return _rotate_pos(steps, move[0]), _rotate_pos(steps, move[1])


def _in_bounds(pos: tuple[int, int]) -> bool:
    """
    returns if pos is in-bounds on a 20x20 board
    """
    return 0 <= pos[0] < 20 and 0 <= pos[1] < 20


class _PrintRedirectorStream:
    """
    a system to redirect bot print statements to the log file
    """

    def __init__(self, out):
        self.out = out
        self.current_plr = None
        self.has_written = False

    def set_cur_plr(self, plr):
        self.current_plr = plr
        self.has_written = False

    def write(self, val):
        if not self.has_written:
            self.out._log(f"{self.current_plr} has printed: ")
            self.has_written = True
        self.out._log(val)


def nearby(pos: tuple[int]) -> set[tuple[int]]:
    """
    returns all nearby positions to `pos` even
    if they are invalid for some grid size

    Example usage:

    >>> nearby((4, 0))
    {(4, -1), (5, 0), (4, 1), (3, 0)}
    """
    return {
        (pos[0] + 1, pos[1]),
        (pos[0] - 1, pos[1]),
        (pos[0], pos[1] + 1),
        (pos[0], pos[1] - 1),
    }


DEFAULT_COLORS = [
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 0, 255),
]


class TerminalSizeOutOfBoundsException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class Board:
    """
    Class for a game board.
    """

    def __init__(
        self,
        bots,
        names=("0", "1", "2", "3"),
        colors=DEFAULT_COLORS,
        delay=-1,
        display_mode="terminal",
    ):
        """
        constructs a board

        bots is list of functions

        names and colors are names and colors of bots.
        colors in RGB, no matter the display mode

        delay is the time to wait between rounds,
        set to negative for no delay

        out is the name of the log file

        display_mode is the type of display.
        options:
        "terminal" -> display in terminal
        "tkinter" -> display in tkinter window
        None -> don't display anything
        """

        self.round_num = 0
        self._board = np.full((20, 20), -1, dtype=np.int16)
        self._bots = bots
        self.colors = list(colors)
        self.names = list(names)
        self._live = {0, 1, 2, 3}
        self._ter_amts = [1, 1, 1, 1]

        self.delay = delay
        self.display_mode = display_mode

        if self.display_mode is not None:
            self._ready_display()

        self._deaths = {}

        for plr, pos in enumerate(CORNERS):
            self[pos] = plr

        self.redirector = _PrintRedirectorStream(self)

    def _checkTerminalSize(self):
        cols, lines = shutil.get_terminal_size()
        if cols < 44 or lines < 28:
            raise TerminalSizeOutOfBoundsException(
                f"""Please make your terminal bigger.
It is currently {cols}x{lines} (wxh),
but must be at least 44x28."""
            )

    def _ready_display(self):
        """
        initializes board display
        """
        mode = self.display_mode

        if mode not in ["terminal", "tkinter", None]:
            raise ValueError(f'Invalid display mode "{mode}"')

        # truncate names

        self.display_names = []

        for name in self.names:
            if len(name) > 20:
                self.display_names.append(name[:17] + "...")
            else:
                self.display_names.append(
                    name + (" " * (20 - len(name))),
                )

        if mode == "tkinter":
            self._tk_root = tk.Tk()
            self._tk_root.configure(background="#000")

            self._tk_canvas = tk.Canvas(
                self._tk_root,
                width=528,
                height=567,
                background="#000",
                highlightthickness=0,
            )

            self._tk_font = font.Font(family="Menlo", size=20)

            # ready frame
            self._tk_canvas.create_text(
                0,
                0,
                text="#" * 44,
                anchor="nw",
                font=self._tk_font,
                fill="#FFF",
            )

            for y in range(21, 441, 21):
                self._tk_canvas.create_text(
                    0,
                    y,
                    text="##" + (" " * 40) + "##",
                    anchor="nw",
                    font=self._tk_font,
                    fill="#FFF",
                )

            self._tk_canvas.create_text(
                0,
                441,
                text="#" * 44,
                anchor="nw",
                font=self._tk_font,
                fill="#FFF",
            )

            # display metrics
            self._tk_canvas.create_text(
                204,
                462,
                text="Round",
                font=self._tk_font,
                fill="#FFF",
                anchor="nw",
            )

            for plr, y in enumerate(range(483, 567, 21)):
                self._tk_canvas.create_text(
                    0,
                    y,
                    text=self.display_names[plr],
                    font=self._tk_font,
                    fill=_convert_hex(self.colors[plr]),
                    anchor="nw",
                )
                self._tk_canvas.create_text(
                    280,
                    y,
                    text=": ",
                    font=self._tk_font,
                    fill="#FFF",
                    anchor="nw",
                )

            # cells and counters
            self._tk_rcount = self._tk_canvas.create_text(
                276,
                462,
                text="0",
                font=self._tk_font,
                fill="#FFF",
                anchor="nw",
            )

            self._tk_tcounts = [
                self._tk_canvas.create_text(
                    304,
                    y,
                    text="0",
                    font=self._tk_font,
                    fill="#FFF",
                    anchor="nw",
                )
                for plr, y in enumerate(range(483, 567, 21))
            ]

            self._tk_grid = [
                [
                    self._tk_canvas.create_rectangle(
                        x,
                        y,
                        x + 24,
                        y + 21,
                        fill="#000",
                        outline="#000",
                    )
                    for x in range(24, 504, 24)
                ]
                for y in range(21, 441, 21)
            ]

            self.squares = [_convert_hex(col) for col in self.colors]
            self.squares.append("#000")

        elif mode == "terminal":
            self._checkTerminalSize()

            self.squares = [_convert_ansi(col) + "██" for col in self.colors]
            self.squares.append("  ")

            print("#" * 44)
            for _ in range(20):
                print("##" + (" " * 40) + "##")
            print("#" * 44)
            print((" " * 15) + "round ")
            for plr in range(4):
                print(
                    "{}{}\x1b[0m: ".format(
                        _convert_ansi(self.colors[plr]),
                        self.display_names[plr],
                    )
                )
            print("\x1b[27A\r\x1b[0m\x1b7", end="")

    def _update_display(self):
        if self.display_mode == "terminal":
            self._checkTerminalSize()
            print(f"\x1b8\x1b[22B\x1b[22C\x1b[0m{self.round_num}")
            for plr in range(4):
                if plr in self._live:
                    print(f"\x1b[22C{self._ter_amts[plr]}    ")
                else:
                    print(f"\x1b[22CELIMINATED R{self._deaths[plr]}")

            print("\x1b[0m", end="")

        elif self.display_mode == "tkinter":
            self._tk_canvas.itemconfig(
                self._tk_rcount,
                text=str(self.round_num),
            )

            for plr, tk_id, amt in zip(
                range(4),
                self._tk_tcounts,
                self._ter_amts,
            ):
                if plr in self._live:
                    self._tk_canvas.itemconfig(
                        tk_id,
                        text=str(amt),
                    )

                else:
                    self._tk_canvas.itemconfig(
                        tk_id,
                        text=f"ELIMINATED R{self._deaths[plr]}",
                    )

            # update screen
            self._tk_canvas.pack(padx=_TK_MARGIN, pady=_TK_MARGIN)
            self._tk_root.update_idletasks()
            self._tk_root.update()

    def __getitem__(self, pos):
        return self._board[tuple(pos)] if _in_bounds(pos) else -1

    def __setitem__(self, pos, plr):
        """
        sets board
        updates display if necessary
        """

        pos = tuple(pos)

        tgt = self[pos]

        if plr == tgt:
            return

        # update territory counts
        if tgt != -1:
            self._ter_amts[tgt] -= 1
        if plr != -1:
            self._ter_amts[plr] += 1

        # update board
        self._board[pos] = plr

        # update display
        if self.display_mode is not None:
            if self.display_mode == "terminal":
                self._checkTerminalSize()
                print(
                    "\x1b8\x1b[{}B\x1b[{}C{}\x1b[0m".format(
                        pos[0] + 1,
                        pos[1] * 2 + 2,
                        self.squares[plr],
                    ),
                    end="",
                )

            elif self.display_mode == "tkinter":
                self._tk_canvas.itemconfig(
                    self._tk_grid[pos[0]][pos[1]],
                    fill=self.squares[plr],
                )

        # check for eliminations
        if tgt in self._live and pos == tuple(CORNERS[tgt]):
            self._live.remove(tgt)
            self._deaths[tgt] = self.round_num
            self._log(
                f"{self.names[tgt]} eliminated by {self.names[plr]}",
            )

            # clear out board
            for crd in np.argwhere(self._board == tgt):
                self[crd] = -1

            self._ter_amts[tgt] = 0

    def _log(self, *args):
        """
        log the given events to the output file
        """
        if not self._logged_round:
            self._logged_round = True
            self.log.write(f"[R{self.round_num}]\n")
        self.log.write("".join(f"{arg}\n" for arg in args))
        self.log.flush()

    def _get_valid_move(self, plr):
        """
        return a valid move
        if bot returns invalid move or throws error, log it
        """
        try:
            # call the bot
            cur_board = self._process_board(plr)
            with redirect_stdout(self.redirector) as f:
                f.set_cur_plr(self.names[plr])
                start = time.time_ns()
                move_raw = self._bots[plr](cur_board)
                stop = time.time_ns()
                if stop - start > 100000000:
                    self._log(
                        f"{self.names[plr]} is taking too long. ",
                        f"time: {(stop - start) // 1000000} ms",
                        f"move: {move_raw}",
                    )
                    return None

        except Exception:
            # bot threw an error
            self._log(
                f"{self.names[plr]} has thrown an error. ",
                "Your bot will forfeit its turn. ",
                traceback.format_exc(),
            )
            return None

        try:
            # bot passed its turn
            if move_raw is None:
                return None

            try:
                (i1, j1), (i2, j2) = move_raw

            # if user only provided dst
            except (AttributeError, TypeError, IndexError):
                i2, j2 = move_raw
                possible_srcs = []
                for src_raw in nearby((i2, j2)):
                    if _in_bounds(src_raw):
                        if self[_rotate_pos(-plr, src_raw)] == plr:
                            possible_srcs.append(src_raw)
                if len(possible_srcs):
                    i1, j1 = random.choice(possible_srcs)
                else:
                    # no valid source positions
                    return None

            i1, j1, i2, j2 = [int(x) for x in [i1, j1, i2, j2]]
            assert all(type(x) is int for x in [i1, j1, i2, j2])

            src, dst = _rotate_move(-plr, ((i1, j1), (i2, j2)))
            # move out of bounds
            if not _in_bounds(src) or not _in_bounds(dst):
                self._log(
                    f"{self.names[plr]} has made an invalid move. ",
                    f"move: {move_raw}",
                    "reason: out of bounds",
                )
                return None

            # move illegal
            if self[src] != plr or self[dst] == plr:
                self._log(
                    f"{self.names[plr]} has made an invalid move. ",
                    f"move: {move_raw}",
                    "reason: src is not yours, or dst is yours",
                )
                return None

            return src, dst

        except (AttributeError, TypeError, IndexError):
            # move processing went wrong
            self._log(
                f"{self.names[plr]} has made an invalid move. ",
                f"move: {move_raw}",
                "Error: ",
                traceback.format_exc(),
            )
            return None

    def _move(self, plr, src, dst):
        """
        executes plr moving from src to dst
        """

        # verify move is stil legal
        if self[src] != plr or self[dst] == plr:
            return

        # make move
        if self[dst] == -1:
            if not self._connected(src):
                self[src] = -1
            self[dst] = plr

        else:
            if self._connected(src):
                self[dst] = plr
            else:
                self[dst] = -1
            self[src] = -1

    def _process_board(self, plr):
        """
        returns a roatated copy of the board as an array
        """
        result = np.copy(np.rot90(self._board, k=plr))
        result[result != -1] = (result[result != -1] - plr) % 4
        return [[result[i, j] for j in range(20)] for i in range(20)]

    def _run_round(self):
        """
        runs 1 round of the game
        """
        # update screen
        if self.display_mode is not None:
            self._update_display()

        self._logged_round = False
        self.round_num += 1

        moves = [
            (
                plr,
                self._get_valid_move(plr),
            )
            for plr in self._live
        ]
        random.shuffle(moves)

        # execute moves
        for plr, move in moves:
            if move and plr in self._live:
                self._move(plr, *move)

    def run(self, rounds: int = 10000):
        """
        runs the game
        returns a list of the player indices, ordered by how well they did
        """
        try:
            self._u = 1
            self.log = open("log.txt", "w")
            self._logged_round = False
            self._log("---- GAME BEGINS ----")
            while all(
                [
                    len(self._live) > 1,
                    self.round_num < rounds,
                ]
            ):
                self._run_round()
                if self.delay > 0:
                    time.sleep(self.delay)

            self._log("---- GAME OVER ----")
            territories = sorted(
                (
                    self._ter_amts[plr],
                    plr,
                )
                for plr in self._live
            )
            self._log(
                *[
                    f"{self.names[plr]} has {amt} cells"
                    for (
                        amt,
                        plr,
                    ) in territories
                ],
            )
            territory_dict = {amt: [] for amt, plr in territories}
            for amt, plr in territories:
                territory_dict[amt].append(plr)
            return [plr for amt, plr in reversed(territories)] + sorted(
                self._deaths.keys(),
                key=lambda key: self._deaths[key],
                reverse=True,
            )
        except TerminalSizeOutOfBoundsException as e:
            self._u = 0
            raise e
        finally:
            self.log.close()
            if self._u:
                self._update_display()
                pass

            if self.display_mode == "tkinter":
                # keep the window running until someone closes it
                self._tk_root.mainloop()
            elif self.display_mode == "pygame":
                while self._pygameIsRunning:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            # sys.exit()
                            self._pygameIsRunning = 0

    def _connected(self, pos):
        """
        returns if pos is connected back to its capital
        """
        plr = self[pos]

        self.dfs_pos = pos
        self.dfs_set = set()

        def dfs():
            pos = self.dfs_pos
            if pos == tuple(CORNERS[plr]):
                return True
            adj = [p for p in nearby(pos) if _in_bounds(p)]
            self.dfs_set.add(pos)
            for c in adj:
                if self[c] == plr and c not in self.dfs_set:
                    self.dfs_pos = c
                    if dfs():
                        return True
            return False

        return dfs()


"""ANSI Control Codes used:
\x1b[0m -> reset color
\x1b[38;5;<color>m -> set text color
\x1b[<num> A -> move cursor up num
\x1b[<num> B -> move cursor down num
\x1b[<num> C -> move cursor left num
\x1b[<num> D -> move cursor right num
\x1b7 -> save cursor position
\x1b8 -> go to last saved cursor position

A comprehensive list of codes can be found here:
https://gist.github.com/fnky/458719343aabd01cfb17a3a4f7296797"""

if __name__ == "__main__":
    print("Run main.py, not this file!")
