from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tictactoe import TicTacToe
# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000


num_actions = 9

def unflatten(weights, w_layers):

    new_w = [ ]
    i = 0
    for layer in w_layers:
        size = layer.size
        new_w.append(weights[i:i+size].reshape(layer.shape))
        i += size

    return new_w

def flatten(model_weights):

    w = []
    for l in model_weights:
        w += list(l.flatten())

    return np.array(w)

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(3, 3,1))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 1, padding="same", activation="relu")(layer1)

    layer3 = layers.Flatten()(layer2)

    layer4 = layers.Dense(512, activation="relu")(layer3)
    action = layers.Dense(num_actions, activation="softmax")(layer4)

    return keras.Model(inputs=inputs, outputs=action)


class Agent:

    def __init__(self,weights,env):
        self._env = env
        self._model = create_q_model()
        self._model.set_weights(weights)

    def act(self,s):

        s = s.reshape((1,3,3))
        p = self._model.predict(s).flatten()

        legal_a = self._env.actions(s.flatten())

        illegal  = [a for a in range(9) if a not in legal_a]

        p[illegal] = 0

        p = p / p.sum()

        max_p = np.max(p)

        max_actions = [a for a in legal_a if p[a] == max_p]

        return np.random.choice(max_actions)


class ESP:
    """
    Evolutionary Strategies
    """
    def __init__(self,pop_size,alpha,sigma,eval_iters,eval_every,render):
        self._alpha = alpha
        self._env = TicTacToe()
        self._eval_iters = eval_iters
        self._eval_every = eval_every
        self._nn = create_q_model()
        self._current_weights = self._nn.get_weights()
        self._pop_size = pop_size
        self._render = render
        self._sigma = sigma

    def create_new_pop(self,weights):

        weights_f = flatten(weights)

        e = np.random.normal(0,1,(self._pop_size,len(weights_f)))

        w_e = np.ones(e.shape)*weights_f

        w_n = w_e + e*self._sigma

        return w_n, e

    def eval(self,p1,p2,render,eval_iters):

        result = 0
        for i in range(eval_iters):

            winner = self.play_game(p1, p2,render)

            if winner == 2:
                result -= 1
            elif winner == 1:
                result += 1

            winner2 = self.play_game(p2,p1,render)

            if winner2 == 2:
                result += 1
            elif winner == 1:
                result -= 1

        return result

    def get_agent_from_weights(self,weights):
        pass

    def play_game(self,agent1,agent2,render=False):

        self._env.reset()

        curr_player = agent1

        game_array = []

        while True:

            action = curr_player.act(self._env.board)

            curr_state, action, next_state, r = self._env.step(action)

            if render:
                game_array.append(self._env.board.copy().tolist())
                self._env.render()

            if r != 0:
                break

            curr_player = agent2 if curr_player == agent1 else agent1

        return self._env.winner

    def train(self,iters):
        """
        :param iters:
        :return:
        """

        current_agent = Agent(self._current_weights,self._env)
        for iter in range(iters):

            if (iter % self._eval_every) == 0:
                r = self.eval(current_agent,current_agent,True,1)
                print(f"Reward at iteration {iter} = {r}")

            new_pop, e_pop = self.create_new_pop(self._current_weights)

            F_total = np.zeros(e_pop[0].shape)
            for p_i in range(new_pop.shape[0]):
                p = unflatten(new_pop[p_i],self._current_weights)
                e = e_pop[p_i]
                p_agent = Agent(p,self._env)
                F = self.eval(p_agent,current_agent,self._render,self._eval_iters)
                F_total += F*e


            new_weights = flatten(self._current_weights) + self._alpha*(1.0 / (self._pop_size*self._sigma))*F_total

            self._current_weights = unflatten(new_weights,self._current_weights)

            current_agent = Agent(self._current_weights,self._env)


if __name__ == "__main__":
    pop_size = 25
    alpha = 0.001
    sigma = 0.1
    train_iters = 100
    eval_every = 5
    eval_iters = 1
    render = False

    esp = ESP(pop_size,alpha,sigma,eval_iters,eval_every,render)

    esp.train(train_iters)
    print("Done running ESP")




