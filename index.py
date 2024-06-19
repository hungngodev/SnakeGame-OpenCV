import random
import time
import cv2
import numpy as np


width = 30
length = 30
square =20
delay = 0.00005
eyeColor  = (255,0,255)
snakeHeadColor =(255,0,0)
snakeBodyColor =  (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
speed = 10
initialLength = 2
autoPlay = True

def randomPoint():
    return [np.random.randint(1, width-1) * square, np.random.randint(1, length-1) * square]

def eatApple(apple,score,snakeBody, snakeHead):
    newPoint = randomPoint()
    while newPoint in snakeBody or newPoint == apple or newPoint == snakeHead:
        newPoint = randomPoint()
    apple = newPoint
    score += 1
    return apple, score

def wallCollide(snakeHead):
    if snakeHead[0]>=width*square or snakeHead[0]<0 or snakeHead[1]>=length*square or snakeHead[1]<0 :
        return 1
    else:
        return 0

def touchBody(snakeBody, snakeHead):
    snakeHead = snakeBody[0]
    return snakeHead in snakeBody[1:]

def drawing(snakeHead, img, snakeBody,apple, curDir):
    img = np.zeros((width *square, length*square,3), dtype='uint8')
    cv2.rectangle(img,(apple[0],apple[1]),(apple[0]+square,apple[1]+square),(0,0,255),-1)
    cv2.rectangle(img, (snakeHead[0], snakeHead[1]), (snakeHead[0]+square, snakeHead[1]+square), snakeHeadColor, 3)
    # cv2.circle(img, snakeHead, 10, eyeColor, -1)
    eyesCoordinates = []
    if curDir == 0:
        eyesCoordinates = [[snakeHead[0], snakeHead[1]], [snakeHead[0], snakeHead[1]+square]]
    elif curDir == 1:
        eyesCoordinates = [[snakeHead[0]+square, snakeHead[1]], [snakeHead[0]+square, snakeHead[1]+square]]
    elif curDir == 2: 
        eyesCoordinates = [[snakeHead[0], snakeHead[1]+square], [snakeHead[0]+square, snakeHead[1]+square]]
    elif curDir == 3:
        eyesCoordinates = [[snakeHead[0], snakeHead[1]], [snakeHead[0]+square, snakeHead[1]]]
    cv2.circle(img,eyesCoordinates[0],3,eyeColor,-1)
    cv2.circle(img,eyesCoordinates[1],3,eyeColor,-1)
    for i in range(square//speed -1, len(snakeBody), square//speed):
        cv2.circle(img, (snakeBody[i][0], snakeBody[i][1]), 10, eyeColor, -1)
        cv2.rectangle(img, (snakeBody[i][0], snakeBody[i][1]), (snakeBody[i][0]+square, snakeBody[i][1]+square), snakeBodyColor, 3)
    cv2.imshow('a',img)
    return img

def createSnakeBody(snakeHead, curDir):
    snakeBody = []
    for i in range(1,initialLength* round(square//speed)):
        match curDir:
            case 0:
                snakeBody.append([snakeHead[0] + i*(speed), snakeHead[1]])
            case 1:
                snakeBody.append([snakeHead[0] - i*(speed), snakeHead[1]])
            case 2:
                snakeBody.append([snakeHead[0], snakeHead[1] - i*(speed)])
            case 3:
                snakeBody.append([snakeHead[0], snakeHead[1] + i*(speed)])
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
                k = cv2.waitKey(round(10))
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
    
    if (gameState['prevDir'] != gameState['curDir'] and (gameState['snakeHead'][0] % square != 0 or gameState['snakeHead'][1] % square != 0)):
        gameState['key'] = k
        gameState['save'] = True
        gameState['curDir'] = gameState['prevDir']

def autoPlay(gameState):
    appleLocation = gameState['apple']
    prevDir = gameState['prevDir']
    applePerspective = [appleLocation[0] - gameState['snakeHead'][0], appleLocation[1] - gameState['snakeHead'][1]]
    if applePerspective[0] > 0 and prevDir != 0:
        gameState['curDir'] = 1
    elif applePerspective[0] < 0 and prevDir != 1:
        gameState['curDir'] = 0
    elif applePerspective[1] > 0 and prevDir != 3:
        gameState['curDir'] = 2
    elif applePerspective[1] < 0 and prevDir != 2:
        gameState['curDir'] = 3
        
def setUpGame():
    gameState= {
        "img": np.zeros((width *square, length*square,3), dtype='uint8'),
        "score": 0,
        "prevDir": 1,
        "key": 1,
        "save": False,
        "firstTime": True,
        "snakeHead": randomPoint(),
        "apple": randomPoint(),
        "curDir": np.random.randint(0,4),
    }
    gameState['snakeBody'] = createSnakeBody(gameState['snakeHead'], gameState['curDir'])
    return gameState
gameState= setUpGame()

while True:
    gameState['img'] = drawing(gameState['snakeHead'], gameState['img'], gameState['snakeBody'], gameState['apple'], gameState['curDir'])
        
    k = takeKeyInput(gameState)
    if k == -2: break
    
    if autoPlay: autoPlay(gameState)

    gameState['prevDir'] = gameState['curDir']
    gameState['snakeBody'].insert(0,list(gameState['snakeHead']))
    
    if gameState['curDir'] == 1:
        gameState['snakeHead'][0] += speed
    elif gameState['curDir'] == 0:
        gameState['snakeHead'][0] -= speed
    elif gameState['curDir'] == 2:
        gameState['snakeHead'][1] += speed
    elif gameState['curDir'] == 3:
        gameState['snakeHead'][1] -= speed
        
    
    if gameState['snakeHead'] == gameState['apple']:
        gameState['apple'], gameState['score'] = eatApple(gameState['apple'], gameState['score'], gameState['snakeBody'], gameState['snakeHead'])
        gameState['snakeBody'].extend([list(gameState['snakeBody'][-1])]* (round(square//speed)-1))
    else:
        gameState['snakeBody'].pop()
         
    wallHit = wallCollide(gameState['snakeHead'])
    bodyHit = touchBody(gameState['snakeBody'], gameState['snakeHead'])
    
    if wallHit: print("Hit wall")
    
    if bodyHit: print("Hit Body")

    if wallHit or bodyHit :
        cv2.putText(gameState['img'],'Your Score is {}'.format(gameState['score']),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('a',gameState['img'])
        k = cv2.waitKey(0)
        if k == ord('q'): break
        if k == ord('r'):            
            gameState= setUpGame()
            continue
