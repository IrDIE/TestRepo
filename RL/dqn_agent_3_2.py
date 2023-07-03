import sys
from gym import Env
from gym.spaces import Discrete
import gym
from gym import spaces
import numpy as np
import time
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras import layers
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam


class CrossEnv2(Env):
    def __init__(self):
        self.action_space = Discrete(2)
        # nCars array
        self.observation_space = np.ndarray(shape=(4,))
        self.beautiful_balanсe = 1
        self.current_action = 0
        # Set start temp
        self.state = np.ndarray(shape=(4))
        # Set shower length
        self.duration = 70  # связано с продолжительностью эпохи в test

    def step(self, action):
        # получили action от модели
        with open("actions.txt", "w") as file:
            file.write(str(action))
            print("\nact\n*****\n", str(action), "\n")

        # отдаём action AL -- оно лежит в файте actions.txt
        # AnyLogic получил action - Отдаём состояние системы : число машин

        previous_state = self.state
        # ждём какое то время и получаем состояние окружения из Al
        acceleration = 20
        sleep = 10 / acceleration
        time.sleep(sleep)
        states_incorrect = True
        array = []
        while states_incorrect:
            try:
                with open('states.txt') as f:

                    for line in f:  # read rest of lines
                        array.append(*[int(x) for x in line.split()])
                        states_incorrect = False

            except:
                states_incorrect = True

        states_incorrect = True
        while states_incorrect:
            try:
                with open('mean_time_driving.txt') as f:
                    for line in f:  # read rest of lines
                        array.append(*[float(x) for x in line.split()])
                        states_incorrect = False

            except:
                states_incorrect = True

        self.state = np.array(array).reshape(4)
        # ---------- забрали число машин, считам награду

        # Calculate reward
        #  ------- reward for DQN_2- -------
        reward = 0
        current_dif = abs(self.state[2] - self.state[0])
        if (current_dif) < 10:
            reward += 0.9
        # --------------------------
        else:
            reward -= 1

        if self.duration <= 0:
            done = True
        else:
            done = False
        info = {}

        self.duration -= 1
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        # Reset start num cars
        self.state = np.ndarray(shape=(4,))
        self.duration = 70
        return self.state


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(1, states)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.add(layers.Reshape((2,)))
    return model


def build_agent(model, actions):
    # policy = BoltzmannQPolicy()
    policy = MaxBoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10)
    return dqn


if __name__ == "__main__":

    list_arg_cmd = sys.argv
    model_version = 'DQN_3_2'
    path_seved_model = os.path.join('Train', 'Saved Models', model_version) + '\dqn_weights_al.h5f'

    if list_arg_cmd[1] == 'train':
        print("----------------------\n\n TRAIN MODE---------------------- \n")
        with open("can_train.txt", "w") as file:
            file.write(str(0))

        env = CrossEnv2()
        states = len(env.observation_space)
        actions = env.action_space.n

        try:
            del model
        except:
            pass
        model = build_model(states, actions)
        dqn = build_agent(model, actions)
        dqn.compile(optimizer='adam', metrics=['mae'])
        # try to load saved model if exist
        # if exist then train  - and save new one
        try:
            dqn.load_weights(path_seved_model)
            print("\n****\n old weights are loaded \n\n")
        except:
            print("\n******\n\n start training new model")

        can_train = 0
        while can_train == 0:
            with open('can_train.txt') as f:
                for line in f:
                    can_train = int(line)
                    if can_train == 1:
                        print('\n\n ** TRAIN STARTED ** \n\n')
                        dqn.fit(env, nb_steps=5000, visualize=False, verbose=1, log_interval=1000)

        dqn.save_weights(path_seved_model, overwrite=True)

    if list_arg_cmd[1] == 'test':
        print("----------------------\n\n TEST MODE --------------\n")

        with open("can_train.txt", "w") as file:
            file.write(str(0))

        env = CrossEnv2()
        states = len(env.observation_space)
        actions = env.action_space.n
        model = build_model(states, actions)
        dqn = build_agent(model, actions)
        dqn.compile(optimizer='adam', metrics=['mae'])
        dqn.load_weights(path_seved_model)
        can_train = 0
        print("----------------------\n\n TEST started. Weights are loaded --------------\n")
        while can_train == 0:
            with open('can_train.txt') as f:
                for line in f:
                    can_train = int(line)
                    if (can_train == 1):
                        dqn.test(env, nb_episodes=40, visualize=False)
