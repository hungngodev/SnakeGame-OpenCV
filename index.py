import random
import time
import cv2
import numpy as np
from AStarSearch import a_star_search

import csv

width = 20
length = 20
square =40
delay = 0.00005
eyeColor  = (255,0,255)
snakeHeadColor =(255,0,0)
snakeBodyColor =  (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
speed =square
initialLength = 1
keyDelay = 1
autoPlay =True

def randomPoint():
    return [np.random.randint(0, width-1) * square, np.random.randint(0, length-1) * square]

def eatApple(gameState):
    newPoint = randomPoint()
    availableSpace= set()
    for i in range(0, width):
        for j in range(0, length):
            availableSpace.add((i,j))
    availableSpace.remove((gameState['apple'][0]//square, gameState['apple'][1]//square))  
    for i in range(square//speed -1, len(gameState['snakeBody']), square//speed):
        availableSpace.remove((gameState['snakeBody'][i][0]//square, gameState['snakeBody'][i][1]//square))
    if len(availableSpace) == 0:
        print("You Won")
        return 1
    newPoint = random.choice(list(availableSpace))
    gameState['apple'] = [newPoint[0]*square, newPoint[1]*square]   
    gameState['score'] += 1
 

def wallCollide(snakeHead):
    if snakeHead[0]>=width*square or snakeHead[0]<0 or snakeHead[1]>=length*square or snakeHead[1]<0 :
        return 1
    else:
        return 0

def touchBody(snakeBody, snakeHead):
    if len(snakeBody) == 0:
        return 0
    snakeHead = snakeBody[0]
    return snakeHead in snakeBody[1:]

def drawing(snakeHead, img, snakeBody,apple, curDir, pathSolution):
    img = np.zeros((width *square, length*square,3), dtype='uint8')
    cv2.rectangle(img,(apple[0],apple[1]),(apple[0]+square,apple[1]+square),(0,0,255),-1)
    cv2.rectangle(img, (snakeHead[0], snakeHead[1]), (snakeHead[0]+square, snakeHead[1]+square), snakeHeadColor, 3)
    # cv2.circle(img, snakeHead, 10, eyeColor, -1)
    for i in range(0, width*square, square):
        cv2.line(img, (i,0), (i, length*square), (255,194,255), 1)
    for i in range(0, length*square, square):
        cv2.line(img, (0,i), (width*square, i), (255,114,255), 1)
    eyesCoordinates = []
    if curDir == 0: eyesCoordinates = [[snakeHead[0], snakeHead[1]], [snakeHead[0], snakeHead[1]+square]]
    elif curDir == 1: eyesCoordinates = [[snakeHead[0]+square, snakeHead[1]], [snakeHead[0]+square, snakeHead[1]+square]]
    elif curDir == 2: eyesCoordinates = [[snakeHead[0], snakeHead[1]+square], [snakeHead[0]+square, snakeHead[1]+square]]
    elif curDir == 3: eyesCoordinates = [[snakeHead[0], snakeHead[1]], [snakeHead[0]+square, snakeHead[1]]]
    cv2.circle(img,eyesCoordinates[0],10,eyeColor,-1)
    cv2.circle(img,eyesCoordinates[1],10,eyeColor,-1)
    
    for i in range(square//speed -1 , len(snakeBody), square//speed):
        cv2.circle(img, (snakeBody[i][0], snakeBody[i][1]), 5, eyeColor, -1)
        cv2.rectangle(img, (snakeBody[i][0], snakeBody[i][1]), (snakeBody[i][0]+square, snakeBody[i][1]+square), snakeBodyColor, 3)
    if autoPlay:
        for i in pathSolution['path']:
            cv2.rectangle(img, (i[0], i[1]), (i[0]+square, i[1]+square), (255,255,255), 1)
        for i in pathSolution['closedList']:
            cv2.rectangle(img, (i[0], i[1]), (i[0]+square, i[1]+square), (255,123,98), 1)
    cv2.imshow('a',img)
    return img

def createSnakeBody(snakeHead, curDir):
    snakeBody = []
  
    for i in range(1, initialLength * round(square // speed)):
            if curDir == 0:
                snakeBody.append([snakeHead[0] + i * speed, snakeHead[1]])
            elif curDir == 1:
                snakeBody.append([snakeHead[0] - i * speed, snakeHead[1]])
            elif curDir == 2:
                snakeBody.append([snakeHead[0], snakeHead[1] - i * speed])
            elif curDir == 3:
                snakeBody.append([snakeHead[0], snakeHead[1] + i * speed])
    return snakeBody

def takeKeyInput(gameState):
    k = -1
    if gameState['firstTime']:
        k = cv2.waitKey(0)
        gameState['firstTime'] = False
    else: 
        t_end = time.time() + delay
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(keyDelay)
            else:
                break
    if gameState['save']:
        k = gameState['key']
        gameState['save'] = False

    if k == ord('a') and gameState['prevDir'] not in [1, 0]: gameState['curDir'] = 0
    elif k == ord('d') and gameState['prevDir'] not in [0, 1]: gameState['curDir'] = 1
    elif k == ord('w') and gameState['prevDir'] not in [2, 3]: gameState['curDir'] = 3
    elif k == ord('s') and gameState['prevDir'] not in [3, 2]: gameState['curDir'] = 2
    elif k == ord('q'): return -2
    
    
def setUpGame():
    gameState= {
        "img": np.zeros((width *square, length*square,3), dtype='uint8'),
        "score": 0,
        "totalScore": 0,
        "prevDir": 1,
        "key": 1,
        "save": False,
        "firstTime":False,
        "snakeHead": randomPoint(),
        "apple": randomPoint(),
        "curDir": np.random.randint(0,4),
        "steps": 0
    }
    gameState['snakeBody'] = createSnakeBody(gameState['snakeHead'], gameState['curDir'])
    return gameState

gameState= setUpGame()

def setUpSolutionState(gameState):
    if autoPlay == False: return {}
    solution = a_star_search( np.ones((width *square, length*square), dtype='uint8'), gameState['snakeHead'], gameState['apple'], gameState['curDir'], gameState['snakeBody'], square, length*square, width*square, speed)
    solutionState = {
        "path": solution[0],
        "closedList": solution[1],
        "currentStep": 0,
        "changingDirection": False,
    }
    return solutionState
solutionState = setUpSolutionState(gameState)

def playFunc(gameState, solutionState):
    gameState['curDir'] = solutionState['path'][solutionState['currentStep']][2]
    solutionState['currentStep'] += 1
    
    
    
fields = ['Score', 'Total Score', 'Steps']
filename = "data.csv"
dicttionary = []

for i in range(10):
    while True:
        gameState['img'] = drawing(gameState['snakeHead'], gameState['img'], gameState['snakeBody'], gameState['apple'], gameState['curDir'], solutionState)
            
        k = takeKeyInput(gameState)
        if k == -2: break
        
        if autoPlay: 
       
            try:
                if solutionState['changingDirection']:
                        solutionState= setUpSolutionState(gameState)
                        solutionState['changingDirection'] = False
                playFunc(gameState, solutionState)
            except:
                cv2.putText(gameState['img'],'Your Score is {}'.format(gameState['totalScore']),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
                cv2.imshow('a',gameState['img'])
                print(gameState['score'])

                dicttionary.append({'Score': gameState['score'], 'Total Score': gameState['totalScore'], 'Steps': gameState['steps']})
                gameState= setUpGame()
                break
            
        if (gameState['prevDir'] != gameState['curDir'] and (gameState['snakeHead'][0] % square != 0 or gameState['snakeHead'][1] % square != 0)):
            gameState['key'] = k
            gameState['save'] = True
            gameState['curDir'] = gameState['prevDir']

        gameState['snakeBody'].insert(0,list(gameState['snakeHead']))  
        gameState['prevDir'] = gameState['curDir']

        if gameState['curDir'] == 1:
            gameState['snakeHead'][0] += speed
        elif gameState['curDir'] == 0:
            gameState['snakeHead'][0] -= speed
        elif gameState['curDir'] == 2:
            gameState['snakeHead'][1] += speed
        elif gameState['curDir'] == 3:
            gameState['snakeHead'][1] -= speed

        if gameState['snakeHead'] == gameState['apple']:
            gameState['snakeBody'].extend([list(gameState['snakeBody'][-1])]* (round(square//speed)-1))
            eatApple(gameState)
            if autoPlay:
                solutionState['changingDirection'] = True

        else:
            gameState['snakeBody'].pop()
            
        if gameState['snakeHead'][0] % square == 0 and gameState['snakeHead'][1] % square == 0:
            gameState['steps'] += 1
            total= (gameState['steps']* 0.1* -1 + gameState['score']*10)
            gameState['totalScore'] = total
            print(gameState['steps']* 0.1* -1 + gameState['score']*10)

        # keyDelay = 0 if gameState['totalScore'] > 400 else 1
        wallHit = wallCollide(gameState['snakeHead'])
        bodyHit = touchBody(gameState['snakeBody'], gameState['snakeHead'])
        
        if wallHit: print("Hit wall")
        
        if bodyHit: 
            print("Hit body")
        gameState['firstTime'] = False
        if wallHit or bodyHit :
            cv2.putText(gameState['img'],'Your Score is {}'.format(gameState['totalScore']),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',gameState['img'])
            print(gameState['score'])

            dicttionary.append({'Score': gameState['score'], 'Total Score': gameState['totalScore'], 'Steps': gameState['steps']})
            gameState= setUpGame()
            break


with open(filename, 'w') as csvfile:
    k =cv2.waitKey(0)
    print("Writing to csv file")    
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writeheader()
    writer.writerows(dicttionary)
    print("Writing complete")


    

