from sample_ais import (
    pass_pompom,
    random_raven,
    defensive_dan_heng,
    AdjacentAsta,
)
from board import Board, DEFAULT_COLORS


print(
    Board(
        # list of bots
        [pass_pompom, random_raven, defensive_dan_heng, AdjacentAsta],
        # list of names
        ["Pass PomPom", "Random Raven", "Defensive Dan Heng", "Adjacent Asta"],
        # a list of tuples, where each tuple is the RGB of a color
        colors=DEFAULT_COLORS,
        # either "terminal" or "tkinter", depending on which display you prefer.
        display_mode="terminal",
        delay=-1,
    ).run()
)
