import random
from collections import deque
from model import QNet, QTrainer, TargetNetworkQTrainer
import numpy as np
import torch

from game import Direction, Point

MAX_MEMORY = 1000000
GAMMA = 0.9
MIN_EPSILON = 0.05
INIT_EPSILON = 0.1
GAMES_EPSILON = 500    

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = INIT_EPSILON
        self.gamma =GAMMA
        print(torch.cuda.is_available())
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
            
            
    def get_state(self, game):
        head = game.snake[0]

        def findNewPoint(direction, i ):
            if direction == 'up left':
                return Point(i.x - 20, i.y - 20)
            elif direction =='up':
                return Point(i.x, i.y - 20)
            elif direction == 'up right':
                return Point(i.x + 20, i.y - 20)
            elif direction == 'right':
                return Point(i.x + 20, i.y)
            elif direction == 'down right':
                return Point(i.x + 20, i.y + 20)
            elif direction == 'down':
                return Point(i.x, i.y + 20)
            elif direction == 'down left':
                return Point(i.x - 20, i.y + 20)
            elif direction == 'left':
                return Point(i.x - 20, i.y)
        directions = ['up left', 'up', 'up right', 'right', 'down right', 'down', 'down left', 'left']
        dangerIn8Dir = [
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0],
            [0,0,0]
        ]
        j=0
        foundApple = False
        for dir in directions:
            i= head
            while i.x < game.w and i.x > 0 and i.y < game.h and i.y > 0:
                i = findNewPoint(dir, i)
                if (abs(i.x - head.x) > 20 or abs(i.y - head.y) > 20):
                        dangerIn8Dir[j][0] = 1
                if i in game.snake[1:] or (i.x == game.food.x and i.y == game.food.y):
                    if i in game.snake[1:]:
                        dangerIn8Dir[j][1] = 1
                    else:
                        dangerIn8Dir[j][2] = 1
                        foundApple= True
                    break
            j+=1

        if not foundApple:
            if game.food.x < game.head.x and game.food.y < game.head.y:
                dangerIn8Dir[0][2] = 1
            elif game.food.x == game.head.x and game.food.y < game.head.y:
                dangerIn8Dir[1][2] = 1
            elif game.food.x > game.head.x and game.food.y < game.head.y:
                dangerIn8Dir[2][2] = 1
            elif game.food.x > game.head.x and game.food.y == game.head.y:
                dangerIn8Dir[3][2] = 1
            elif game.food.x > game.head.x and game.food.y > game.head.y:
                dangerIn8Dir[4][2] = 1
            elif game.food.x == game.head.x and game.food.y > game.head.y:
                dangerIn8Dir[5][2] = 1
            elif game.food.x < game.head.x and game.food.y > game.head.y:
                dangerIn8Dir[6][2] = 1
            elif game.food.x < game.head.x and game.food.y == game.head.y:
                dangerIn8Dir[7][2] = 1
        
        dir_l = int(game.direction == Direction.LEFT)
        dir_r = int(game.direction == Direction.RIGHT)
        dir_u = int(game.direction == Direction.UP)
        dir_d = int(game.direction == Direction.DOWN)

        tail_l = int(game.snake[-1].x > game.snake[-2].x)
        tail_r = int(game.snake[-1].x < game.snake[-2].x)
        tail_up = int(game.snake[-1].y > game.snake[-2].y)
        tail_down = int(game.snake[-1].y < game.snake[-2].y)

        # dangerIn8Dir = [item for sublist in dangerIn8Dir for item in sublist]
        dangerIn8Dir = [item for sublist in dangerIn8Dir for item in sublist]
        state = dangerIn8Dir +  [
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            tail_l,
            tail_r,
            tail_up,
            tail_down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        coefficient = max((GAMES_EPSILON- self.n_games)/GAMES_EPSILON, 0)
        self.epsilon = (INIT_EPSILON - MIN_EPSILON) * coefficient + MIN_EPSILON
        choice = np.random.choice([0,1] , p=[1-self.epsilon, self.epsilon])
        
        final_move = [0,0,0]
        if choice:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


class Agent1:

    def __init__(self):
        self.n_games = 0
        self.epsilon = INIT_EPSILON
        self.gamma =GAMMA
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
            
            
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached



    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float, device=self.device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

class QLearningAgent(Agent):

    def __init__(self, MODEL_CONFIG):
        super().__init__()
        self.model = QNet(MODEL_CONFIG).to(self.device)
        self.trainer = QTrainer(self.model, lr=MODEL_CONFIG['learning_rate'], gamma = MODEL_CONFIG['gamma'], device=self.device)
        self.batch_size = MODEL_CONFIG['batch_size']
        self.max_memory = MODEL_CONFIG['max_memory']
        
    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
        
class TargetNetWorkAgent(Agent):

    def __init__(self, MODEL_CONFIG):
        super().__init__()
        
        self.model = QNet(MODEL_CONFIG).to(self.device)
        self.model_target = QNet(MODEL_CONFIG).to(self.device)
        self.trainer = TargetNetworkQTrainer(
            self.model, 
            self.model_target, 
            MODEL_CONFIG['learning_rate'],
            MODEL_CONFIG['gamma'],
            MODEL_CONFIG['soft_update'],
            self.device
            )
        self.batch_size = MODEL_CONFIG['batch_size']
    
    def update_target(self):
        self.trainer.update_target()
            
    def update_descent(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)