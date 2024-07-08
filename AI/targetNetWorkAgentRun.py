import pygame
import numpy as np
from agent import TargetNetWorkAgent
from game import SnakeGameAI
from helper import plot
import wandb

MODEL_CONFIG2 = {
    "input" : 32,
    "hiddenLayer" :  [
        {
            "size" : 256,
            "activation" : "relu"
        },
        {
            "size" : 128,
            "activation" : "relu"
        },
    ],
    "output" : 3,
    "batch_size" : 1000,
    "learning_rate" : 0.001,
    "gamma" : 0.9,
    "soft_update": 0.001,
    "num_updates": 20,
}

wandb.init(
    project="Snake-TargetNetwork",
    config=MODEL_CONFIG2
)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = TargetNetWorkAgent(MODEL_CONFIG=MODEL_CONFIG2)
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
            if update % MODEL_CONFIG2['num_updates'] == 0 and len(agent.memory) > MODEL_CONFIG2['batch_size']:
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
                wandb.log({"score": score, "mean_score": mean_score})
            update += 1
            
        except Exception as e:
            print(e)
            break
    

    np.save('./plotlib/plot_scores_TargetNetwork.npy', plot_scores)
    np.save('./plotlib/plot_mean_scores_TargetNetwork.npy', plot_mean_scores)
    # wandb.finish()

if __name__ == '__main__':
    train()