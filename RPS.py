from random import randint
from tensorflow import keras
from dataclasses import dataclass, field
from itertools import product
import numpy as np
from random import randint
from abc import ABC, abstractmethod

MOVES = "RPS"
MOVE_TO_INDEX = dict(map(reversed, enumerate(MOVES)))

WINNING_MOVE = {
    'P': 'S',
    'S': 'R',
    'R': 'P'
}

class Model(ABC):
    @abstractmethod
    def update_and_predict(self, history: list[chr], new_move: chr) -> chr:
        ...

    @abstractmethod
    def __post_init__(self):
        ...

    def reset(self):
        self.__post_init__()

@dataclass
class CountModel(Model):
    window_length: int = 4
    matrix: np.array = field(init=False)
    indices: dict[str, int] = field(init=False)
    windows: list[str] = field(init=False)

    def __post_init__(self):
        windows = product(MOVES, repeat=self.window_length)
        windows = map(lambda t: "".join(t), windows)
        self.windows = list(windows)
        self.indices = dict(map(reversed, enumerate(self.windows)))
        self.matrix = np.zeros((len(self.windows), len(MOVES)))

    def update_and_predict(self, history: list[chr], new_move: chr) -> chr:
        if len(history) <= self.window_length:
            return MOVES[randint(0, len(MOVES) - 1)]

        window = "".join(history[-self.window_length:])
        old_index = self.indices[window]
        new_index = MOVE_TO_INDEX[new_move]
        self.matrix[old_index, new_index] += 1

        current_index = self.indices[window[1:] + new_move]
        return MOVES[np.argmax(self.matrix[current_index, :])]   


@dataclass
class DLModel(Model):
    window_length: int = 5
    model: keras.Sequential = field(init=False)

    def __post_init__(self):
        self.model = keras.Sequential([
            keras.layers.Input((self.window_length,)),
            keras.layers.Dense(15),
            keras.layers.Dense(30),
            keras.layers.Dense(1)
        ])

        self.model.compile( 
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

    def update_and_predict(self, history: list[chr], new_move: chr) -> chr:
        if new_move == '' or len(history) <= self.window_length:
            return MOVES[randint(0, len(MOVES) - 1)]

        self._update(history, new_move)

        next_window = "".join(history[-(self.window_length) + 1:]) + new_move
        data_array = np.zeros((len(MOVES), self.window_length))

        for i, w in enumerate(next_window[-self.window_length:]):
            data_array[MOVE_TO_INDEX[w], i] = 1

        result = self.model.predict_on_batch(data_array)
        index = np.argmax(result)
        return MOVES[index]


    def _update(self, history: list[str], prev_play: str):
        data_array = np.zeros((len(MOVES), self.window_length))
        for i, w in enumerate(history[-self.window_length:]):
            data_array[MOVE_TO_INDEX[w], i] = 1

        result_array = np.zeros((len(MOVES),))
        result_array[MOVE_TO_INDEX[prev_play]] = 0.99999

        self.model.train_on_batch(data_array, result_array)

def player(prev_play: chr, opponent_history: list[chr] = [], _model=[CountModel()]):
    [model] = _model
    if prev_play == '':
        model.reset()
        opponent_history.clear()

    move = model.update_and_predict(opponent_history, prev_play)
    opponent_history.append(prev_play)
    return WINNING_MOVE[move]