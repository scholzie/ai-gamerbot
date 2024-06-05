import keras
from keras import layers, models, optimizers
from collections import deque

# Hyperparameters
GAMMA = 0.99
LEARNING_RATE = 0.001
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
MAX_EPSILON = 1.0


class DQNAgent:
    def __init__(self, state_size, action_size) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.epsilon = MAX_EPSILON
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, activation='relu',
                         input_dim=self.state_size),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(
            learning_rate=LEARNING_RATE))
        return model

    # def remember(self):
    #     pass

    # def act(self):
    #     pass

    # def replay(self):
    #     pass
