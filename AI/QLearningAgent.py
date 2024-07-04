import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import QNet, QTrainer
from helper import plot
from agent import Agent

MAX_MEMORY = 100000
BATCH_SIZE = 1000

LR = 0.001

MODEL_CONFIG = {
    "input" : 11,
    "hiddenLayer" :  [
        {
            "size" : 64,
            "activation" : "relu"
        },
    ],
    "output" : 3
}
class QLearningAgent(Agent):

    def __init__(self):
        super().__init__()
        self.model = QNet(MODEL_CONFIG).to(self.device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=self.device)
        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = QLearningAgent()
    game = SnakeGameAI()
    while True:
        # get old state
        try:
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)
        except Exception as e:
            print(e)
            break

if __name__ == '__main__':
    train()