import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 1

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return game_over, self.score
    
    def _is_collision(self):
        # hits boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            print('boundary')
            return True
        # hits itself
        if self.head in self.snake[1:]:
            print('self')
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

if __name__ == '__main__':
    game = SnakeGame()
    def get_state( game):
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


    # game loop
    while True:
        game_over, score = game.play_step()
        print(get_state(game))
        if game_over == True:
            break
        
    print('Final Score', score)
        
        
    pygame.quit()