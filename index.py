import numpy as np
import cv2
import random
import time

def randomPoint():
    return [np.random.randint(1, width-1) * square, np.random.randint(1, length-1) * square]

def eatApple(apple, score):
    apple = randomPoint()
    score += 1
    return apple, score

def wallCollide(snakeHead):
    if snakeHead[0]>=width*square or snakeHead[0]<0 or snakeHead[1]>=length*square or snakeHead[1]<0 :
        return 1
    else:
        return 0

def touchBody(snakeBody):
    snakeHead = snakeBody[0]
    return snakeHead in snakeBody[1:]
    
width = 30
length = 30
square =20
delay = 0.000002
eyeColor  = (255,0,255)
snakeColor = (255,0,0)
img = np.zeros((width *square, length*square,3),dtype='uint8')


snakeHead = randomPoint()
curDir = np.random.randint(0,4)
snakeBody =[]
match curDir:
    case 0:
        snakeBody = [
         [snakeHead[0] + i +1 , snakeHead[1]] for i in range(square)
            ]
    case 1:
        snakeBody = [
            [snakeHead[0] - i -1, snakeHead[1]] for i in range(square)
            ]
    case 2:
        snakeBody = [
            [snakeHead[0], snakeHead[1] - i -1] for i in range(square)
            ]
    case 3:
        snakeBody = [
            [snakeHead[0], snakeHead[1] + i +1] for i in range(square)
        ]

apple =randomPoint()
score = 0
prevDir = 1

def draw(img):
    cv2.imshow('a',img)
 
    img = np.zeros((width *square, length*square,3), dtype='uint8')
    cv2.rectangle(img,(apple[0],apple[1]),(apple[0]+square,apple[1]+square),(0,0,255),-1)
    
    cv2.rectangle(img, (snakeHead[0], snakeHead[1]), (snakeHead[0]+square, snakeHead[1]+square), snakeColor, -1)

    match curDir:
        case 0:
            cv2.circle(img, (snakeHead[0], snakeHead[1]), 5, eyeColor, -1)
            cv2.circle(img, (snakeHead[0], snakeHead[1]+square), 5,eyeColor, -1)
        case 1:
            cv2.circle(img, (snakeHead[0]+square, snakeHead[1]), 5, eyeColor, -1)
            cv2.circle(img, (snakeHead[0]+square, snakeHead[1]+square), 5,eyeColor, -1)
        case 2: 
            cv2.circle(img, (snakeHead[0], snakeHead[1]+square), 5, eyeColor, -1)
            cv2.circle(img, (snakeHead[0]+ square, snakeHead[1]+square), 5,eyeColor, -1)
        case 3:
            cv2.circle(img, (snakeHead[0], snakeHead[1]), 5, eyeColor, -1)
            cv2.circle(img, (snakeHead[0]+ square, snakeHead[1]), 5,eyeColor, -1)
    
    for i in range(len(snakeBody)):
        cv2.rectangle(img,(snakeBody[i][0],snakeBody[i][1]),(snakeBody[i][0]+square,snakeBody[i][1]+square),(0,255,0),-1)
    return img

while True:
    cv2.imshow('a',img)
 
    img = np.zeros((width *square, length*square,3), dtype='uint8')
    cv2.rectangle(img,(apple[0],apple[1]),(apple[0]+square,apple[1]+square),(0,0,255),-1)
    
    cv2.rectangle(img, (snakeHead[0], snakeHead[1]), (snakeHead[0]+square, snakeHead[1]+square), snakeColor, -1)

    match curDir:
        case 0:
            cv2.circle(img, (snakeHead[0], snakeHead[1]), 5, eyeColor, -1)
            cv2.circle(img, (snakeHead[0], snakeHead[1]+square), 5,eyeColor, -1)
        case 1:
            cv2.circle(img, (snakeHead[0]+square, snakeHead[1]), 5, eyeColor, -1)
            cv2.circle(img, (snakeHead[0]+square, snakeHead[1]+square), 5,eyeColor, -1)
        case 2: 
            cv2.circle(img, (snakeHead[0], snakeHead[1]+square), 5, eyeColor, -1)
            cv2.circle(img, (snakeHead[0]+ square, snakeHead[1]+square), 5,eyeColor, -1)
        case 3:
            cv2.circle(img, (snakeHead[0], snakeHead[1]), 5, eyeColor, -1)
            cv2.circle(img, (snakeHead[0]+ square, snakeHead[1]), 5,eyeColor, -1)
    
    for i in range(len(snakeBody)):
        cv2.rectangle(img,(snakeBody[i][0],snakeBody[i][1]),(snakeBody[i][0]+square,snakeBody[i][1]+square),(0,255,0),-1)
        
    t_end = time.time() + delay
    k = -1
    while time.time() < t_end:
        if k == -1:
            k = cv2.waitKey(round(10))
        else:
            continue
            
    havePressed = False
    if k == ord('a') and prevDir != 1 and prevDir != 0:
        curDir = 0
    elif k == ord('d') and prevDir != 0 and prevDir != 1:
        curDir = 1
    elif k == ord('w') and prevDir != 2 and prevDir != 3:
        curDir = 3
    elif k == ord('s') and prevDir != 3 and prevDir != 2:
        curDir = 2
    elif k == ord('q'):
        break
    else:
        curDir = curDir
        
    prevDir = curDir
    if curDir == 1:
        for i in range(square-1):
            snakeHead[0] += 1
            snakeBody.insert(0,list(snakeHead))
            img = draw(img)
            cv2.waitKey(10)
            snakeBody.pop()
    elif curDir == 0:
        for i in range(square-1):
            snakeHead[0] -= 1
            snakeBody.insert(0,list(snakeHead))
            img= draw(img)
            cv2.waitKey(10)
            snakeBody.pop()
    elif curDir == 2:
        for i in range(square-1):
            snakeHead[1] += 1
            snakeBody.insert(0,list(snakeHead))
            img= draw(img)
            cv2.waitKey(10)
            snakeBody.pop()
    elif curDir == 3:
        for i in range(square-1):
            snakeHead[1] -= 1
            snakeBody.insert(0,list(snakeHead))
            img =draw(img)
            cv2.waitKey(10)
            snakeBody.pop()
    print('done')
    
    if snakeHead == apple:
        apple, score = eatApple(apple, score)
        for i in range(square):
            snakeBody.append(list(snakeBody[len(snakeBody)-1]))
    else:
        nextToTail = snakeBody[len(snakeBody)-1]
        tail = snakeBody.pop()

        
    if (wallCollide(snakeHead)): print("Hit wall")
    if (touchBody(snakeBody)): print("Hit Body")
    if wallCollide(snakeHead)or touchBody(snakeBody) :
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'Your Score is {}'.format(score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA);
        cv2.imshow('a',img)
        cv2.waitKey(0)    
        break