A software battlebot contest called Bovine Borders we designed and ran at school. Written by GitHub users @LemonKat and @Vodiboi.

The game is a little like Risk, and involves taking territory from other players. Contestants wrote their own bots using a variety of strategies and faced off against each other to take over the board.

I (@LemonKat) also spent some time training a reinforcement learning AI to play this game. It actually beat out most of the standard algorithmic bots! I used a Double DQN algorithm, with the reward for each turn of the game set to how much territory the AI had by the end of that turn.

Detailed instructions are in the rules PDF.

![game video](images/example_game_2.gif)

![game screenshot](images/screenshot1.png)

To run:
1. read the rules, in `rules.pdf`
2. code your bot up somewhere
3. change main.py to import the bot and add it to the game
4. run `python3 main.py`

There's a couple example bots in `sample_ais.py`, and more in `more_sample_ais.py`.

Have fun!
To set up and run the AI:
1. run `python3 train.py`
2. import the bot from `svarog.py`
3. run as normal