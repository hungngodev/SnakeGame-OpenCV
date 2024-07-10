from agent import QLearningAgent
from game import SnakeGameAI
import pygame
import numpy as np
from helper import plot
import wandb



MODEL_CONFIG = {
    "input" : 32,
    "hiddenLayer" :  [
        {
            "size" : 64,
            "activation" : "relu"
        },
    ],
    "output" : 3,
    "batch_size" : 1000,
    "learning_rate" : 0.001,   
    "gamma" : 0.9,
    "num_updates": 20,
}

wandb.init(
    project="Snake-QLearning",
    config=MODEL_CONFIG
)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = QLearningAgent(MODEL_CONFIG=MODEL_CONFIG)
    game = SnakeGameAI()
    pygame.display.set_caption('QLearning')
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    break
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
                    agent.trainer.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                total_score += score
                mean_score = total_score / agent.n_games
                
                plot_scores.append(score)
                plot_mean_scores.append(mean_score)
                wandb.log({"score": score, "mean_score": mean_score})
                
        except Exception as e:
            print(e)
            break
        
    np.save('./plotlib/plot_scores_QLearning.npy', plot_scores)
    np.save('./plotlib/plot_mean_scores_QLearning.npy', plot_mean_scores)
    
if __name__ == '__main__':
    train()
