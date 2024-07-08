import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import  TargetNetworkQTrainer, QNet
from helper import plot
from agent import Agent
import cv2
import pygame

BATCH_SIZE = 1000

LR = 0.001
NUM_UPDATES = 20
SOFT_UPDATE = 0.001
EPSILON = 0


MODEL_CONFIG = {
    "input" : 32,
    "hiddenLayer" :  [
        {
            "size" : 256,
            "activation" : "relu"
        },
        {
            "size" : 256,
            "activation" : "relu"
        }
    ],
    "output" : 3
}

class TargetNetWorkAgent(Agent):

    def __init__(self):
        super().__init__()
        
        self.soft_update = SOFT_UPDATE
        self.model = QNet(MODEL_CONFIG).to(self.device)
        self.model_target = QNet(MODEL_CONFIG).to(self.device)
        self.trainer = TargetNetworkQTrainer(
            self.model, 
            self.model_target, 
            LR, 
            self.gamma, 
            self.soft_update,
            self.device
            )
            
    def update_descent(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = TargetNetWorkAgent()
    game = SnakeGameAI()
    pygame.display.set_caption('SnakeTargetNetwork')
    update = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    break
        try:
            state_old = agent.get_state(game)

            final_move = agent.get_action(state_old)

            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            agent.remember(state_old, final_move, reward, state_new, done)
            if update % NUM_UPDATES == 0 and len(agent.memory) > BATCH_SIZE:
                agent.update_descent()
                
            if done:
                game.reset()
                agent.n_games += 1

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)
                total_score += score
                mean_score = total_score / agent.n_games
                
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)
            update += 1
            
        except Exception as e:
            print(e)
            break
    

    np.save('plot_scores_TargetNetwork.npy', plot_scores)
    np.save('plot_mean_scores_TargetNetwork.npy', plot_mean_scores)

if __name__ == '__main__':
    train()
    plot(np.load('plot_scores_TargetNetwork.npy'), np.load('plot_mean_scores_TargetNetwork.npy'))
 