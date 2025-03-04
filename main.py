# This entrypoint file to be used in development. Start by reading README.md
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player, CountModel, DLModel
from unittest import main
from functools import partial

def partial_model(model):
    return partial(player, opponent_history=[], _model=[model])

def Chuck(): return partial_model(CountModel())
def Dale(): return partial_model(DLModel())

# play(Chuck(), quincy, 1000)
# play(Chuck(), abbey, 10000)
# play(Chuck(), kris, 1000)
# play(Chuck(), mrugesh, 1000)

# Uncomment line below to play interactively against a bot:
# play(human, abbey, 20, verbose=True)

# Uncomment line below to play against a bot that plays randomly:
# play(human, random_player, 1000)



# Uncomment line below to run unit tests automatically
main(module='test_module', exit=False)