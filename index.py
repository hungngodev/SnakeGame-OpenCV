import random
import time
import cv2
import numpy as np
from HamiltonianCycle import pathWithDir
import csv

path = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (11, 1), (10, 1), (9, 1), (8, 1), (7, 1), (7, 2), (8, 2), (9, 2), (10, 2), (11, 2), (11, 3), (10, 3), (9, 3), (8, 3), (7, 3), (7, 4), (8, 4), (9, 4), (10, 4), (11, 4), (11, 5), (10, 5), (9, 5), (8, 5), (7, 5), (7, 6), (8, 6), (9, 6), (10, 6), (11, 6), (11, 7), (10, 7), (9, 7), (8, 7), (7, 7), (7, 8), (8, 8), (9, 8), (10, 8), (11, 8), (11, 9), (10, 9), (9, 9), (8, 9), (7, 9), (7, 10), (8, 10), (9, 10), (10, 10), (11, 10), (11, 11), (10, 11), (9, 11), (8, 11), (7, 11), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (11, 13), (10, 13), (9, 13), (8, 13), (7, 13), (7, 14), (8, 14), (9, 14), (10, 14), (11, 14), (11, 15), (10, 15), (9, 15), (8, 15), (7, 15), (7, 16), (8, 16), (9, 16), (10, 16), (11, 16), (12, 16), (13, 16), (14, 16), (14, 15), (13, 15), (12, 15), (12, 14), (13, 14), (14, 14), (14, 13), (13, 13), (12, 13), (12, 12), (13, 12), (14, 12), (14, 11), (13, 11), (12, 11), (12, 10), (13, 10), (14, 10), (14, 9), (13, 9), (12, 9), (12, 8), (13, 8), (14, 8), (14, 7), (13, 7), (12, 7), (12, 6), (13, 6), (14, 6), (14, 5), (13, 5), (12, 5), (12, 4), (13, 4), (14, 4), (14, 3), (13, 3), (12, 3), (12, 2), (13, 2), (14, 2), (14, 1), (13, 1), (12, 1), (12, 0), (13, 0), (14, 0), (15, 0), (15, 1), (15, 2), (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (15, 8), (15, 9), (15, 10), (15, 11), (15, 12), (15, 13), (15, 14), (15, 15), (15, 16), (16, 16), (17, 16), (18, 16), (18, 15), (17, 15), (16, 15), (16, 14), (17, 14), (18, 14), (18, 13), (17, 13), (16, 13), (16, 12), (17, 12), (18, 12), (18, 11), (17, 11), (16, 11), (16, 10), (17, 10), (18, 10), (18, 9), (17, 9), (16, 9), (16, 8), (17, 8), (18, 8), (18, 7), (17, 7), (16, 7), (16, 6), (17, 6), (18, 6), (18, 5), (17, 5), (16, 5), (16, 4), (17, 4), (18, 4), (18, 3), (17, 3), (16, 3), (16, 2), (17, 2), (18, 2), (18, 1), (17, 1), (16, 1), (16, 0), (17, 0), (18, 0), (19, 0), (19, 1), (19, 2), (19, 3), (19, 4), (19, 5), (19, 6), (19, 7), (19, 8), (19, 9), (19, 10), (19, 11), (19, 12), (19, 13), (19, 14), (19, 15), (19, 16), (19, 17), (18, 17), (17, 17), (17, 18), (18, 18), (19, 18), (19, 19), (18, 19), (17, 19), (16, 19), (15, 19), (14, 19), (14, 18), (15, 18), (16, 18), (16, 17), (15, 17), (14, 17), (13, 17), (12, 17), (11, 17), (10, 17), (9, 17), (8, 17), (7, 17), (7, 18), (8, 18), (9, 18), (10, 18), (11, 18), (12, 18), (13, 18), (13, 19), (12, 19), (11, 19), (10, 19), (9, 19), (8, 19), (7, 19), (6, 19), (5, 19), (4, 19), (3, 19), (2, 19), (1, 19), (0, 19), (0, 18), (1, 18), (2, 18), (3, 18), (4, 18), (5, 18), (6, 18), (6, 17), (5, 17), (4, 17), (3, 17), (2, 17), (1, 17), (0, 17), (0, 16), (0, 15), (0, 14), (1, 14), (1, 15), (1, 16), (2, 16), (3, 16), (4, 16), (5, 16), (6, 16), (6, 15), (5, 15), (4, 15), (3, 15), (2, 15), 
(2, 14), (3, 14), (4, 14), (5, 14), (6, 14), (6, 13), (5, 13), (4, 13), (3, 13), (2, 13), (1, 13), (0, 13), (0, 12), (1, 12), (2, 12), (3, 12), (4, 12), (5, 12), (6, 12), (6, 
11), (5, 11), (4, 11), (3, 11), (2, 11), (1, 11), (0, 11), (0, 10), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (6, 9), (5, 9), (4, 9), (3, 9), (2, 9), (1, 9), (0, 9), (0, 8), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8), (6, 7), (5, 7), (4, 7), (3, 7), (2, 7), (1, 7), (0, 7), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6), (6, 
5), (5, 5), (4, 5), (3, 5), (2, 5), (1, 5), (0, 5), (0, 4), (0, 3), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4), (6, 3), (5, 3), (4, 3), (3, 3), (2, 3), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (6, 1), (5, 1), (4, 1), (3, 1), (2, 1), (1, 1), (0, 1), (0, 0)]
gridDir = np.array( [[1, 3, 1, 3, 3, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 1, 3],
           [1, 0, 2, 2, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 2, 1, 0, 1, 0],
           [1, 0, 1, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 3, 1, 0, 1, 0],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
           [1, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0],
           [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 0],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
           [2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 0, 1, 0],
           [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 0, 1, 0],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 2, 0],
           [1, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 1, 3],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 0],
           [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 0, 3, 0],
           [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 0],
           [1, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 1, 0],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0]])

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
        path = pathSolution['path']
        for i in range(0, len(path)-1):
            cv2.line(img, (path[i][0]*square+ square//2, path[i][1]*square+ square//2), (path[i+1][0]*square + square//2, path[i+1][1]*square+ square//2), (255,255,255), 2)
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
        "firstTime": not autoPlay,
        "snakeHead": randomPoint(),
        "apple": randomPoint(),
        "curDir": np.random.randint(0,4),
        "steps": 0
    }
    gameState['snakeBody'] = createSnakeBody(gameState['snakeHead'], gameState['curDir'])
    return gameState

gameState= setUpGame()

def setUpSolutionState():
    if autoPlay == False: return {}
    solution= pathWithDir(length, width, square)
    return {
        "gridDir": solution[0],
        "path": solution[1],
        "wrapUp" : 0
    }
solutionState = setUpSolutionState()

def playFunc(gameState, solutionState):
    def detectInRange(snakeHead, point):
        return abs(snakeHead[0] - point[0]) <= 1 ^ abs(snakeHead[1] - point[1]) <= 1
    snakeHead = gameState['snakeHead']
    snakeBodyCoor = [ (gameState['snakeBody'][i][0]//square, gameState['snakeBody'][i][1]//square) for i in range(square//speed -1, len(gameState['snakeBody']), square//speed)]

    hamilDir = solutionState['gridDir'][snakeHead[0]//square][snakeHead[1]//square]
    prevDir = gameState['prevDir']
    gameState['curDir'] = hamilDir
    
    skipIfElse = gameState['score'] < 2
    if skipIfElse:
        applePerspective = (gameState['apple'][0]//square - snakeHead[0]//square, gameState['apple'][1]//square - snakeHead[1]//square)
        if applePerspective[0] > 0 and prevDir != 0:
            gameState['curDir'] = 1
        elif applePerspective[0] < 0 and prevDir != 1:
            gameState['curDir'] = 0
        elif applePerspective[1] > 0 and prevDir != 3:
            gameState['curDir'] = 2
        elif applePerspective[1] < 0 and prevDir != 2:
            gameState['curDir'] = 3
    else:    
        normalizedHead = (snakeHead[0]//square, snakeHead[1]//square)
        applePosition = solutionState['path'].index((gameState['apple'][0]//square, gameState['apple'][1]//square))
        headPosition = solutionState['path'].index(normalizedHead)
        
        reverse = solutionState['wrapUp'] == 1
        solutionState['wrapUp'] = 0 if reverse else 1
        if reverse:
            current = applePosition
        else:
            current = applePosition if applePosition > headPosition else 0
                    
        count =0
        skip = gameState['prevDir'] != hamilDir if gameState['totalScore'] > 440 else True
        skip2 = gameState['totalScore'] < 500
        if skip and skip2:
            if not( snakeHead[0] % square != 0 or snakeHead[1] % square != 0):
                found= False
                while count < len(solutionState['path']):
                    if detectInRange(normalizedHead, solutionState['path'][current]):
                        if not (solutionState['path'][current] in snakeBodyCoor):
                            found = True
                            print("Found point",solutionState['path'][current])
                            break
                        else:
                            print("snake body is here",solutionState['path'][current])
                    current -= 1
                    if reverse:
                        if current==0:
                            current = len(solutionState['path'])-2
                    else:
                        if current == 0 or current == headPosition:
                            break
                    count += 1
                if not found:
                    print("Not found")

                if normalizedHead[0] < solutionState['path'][current][0]:
                    gameState['curDir'] = 1
                elif normalizedHead[0] > solutionState['path'][current][0]:
                    gameState['curDir'] = 0
                elif normalizedHead[1] < solutionState['path'][current][1]:
                    gameState['curDir'] = 2
                else:
                    gameState['curDir'] = 3


fields = ['Score', 'Total Score', 'Steps']
filename = "data.csv"
dicttionary = []
for i in range(1):
    while True:
        gameState['img'] = drawing(gameState['snakeHead'], gameState['img'], gameState['snakeBody'], gameState['apple'], gameState['curDir'], solutionState)
            
        k = takeKeyInput(gameState)
        if k == -2: break
        
        if autoPlay: 
            playFunc(gameState, solutionState) 
            
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

        if wallHit or bodyHit :
            cv2.putText(gameState['img'],'Your Score is {}'.format(gameState['totalScore']),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('a',gameState['img'])
            print(gameState['score'])

            # k = cv2.waitKey(0)
            # if k == ord('q'): break
            # if k == ord('r'):            
            #     gameState= setUpGame()
            #     solutionState= setUpSolutionState()
            dicttionary.append({'Score': gameState['score'], 'Total Score': gameState['totalScore'], 'Steps': gameState['steps']})
            gameState= setUpGame()
            solutionState= setUpSolutionState()
            break


with open(filename, 'w') as csvfile:
    print("Writing to csv file")    
    writer = csv.DictWriter(csvfile, fieldnames = fields)
    writer.writeheader()
    writer.writerows(dicttionary)













    # if applePerspective[0] > 0 and prevDir != 0:
    #     gameState['curDir'] = 1
    # elif applePerspective[0] < 0 and prevDir != 1:
    #     gameState['curDir'] = 0
    # elif applePerspective[1] > 0 and prevDir != 3:
    #     gameState['curDir'] = 2
    # elif applePerspective[1] < 0 and prevDir != 2:
    #     gameState['curDir'] = 3
    # if applePerspective[0] > 0:
    #     if prevDir !=0:
    #         gameState['curDir'] = 1
    #     else: 
    #         if applePerspective[1] > 0:
    #             gameState['curDir'] = 2
    #         else:
    #             gameState['curDir'] = 3
    # elif applePerspective[0] < 0:
    #     if prevDir !=1:
    #         gameState['curDir'] = 0
    #     else: 
    #         if applePerspective[1] > 0:
    #             gameState['curDir'] = 2
    #         else:
    #             gameState['curDir'] = 3
    # elif applePerspective[1] > 0:
    #     if prevDir !=3:
    #         gameState['curDir'] = 2
    #     else: 
    #         if applePerspective[0] > 0:
    #             gameState['curDir'] = 1
    #         else:
    #             gameState['curDir'] = 0
    # elif applePerspective[1] < 0:
    #     if prevDir !=2:
    #         gameState['curDir'] = 3
    #     else: 
    #         if applePerspective[0] > 0:
    #             gameState['curDir'] = 1
    #         else:
    #             gameState['curDir'] = 0
    