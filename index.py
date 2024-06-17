import numpy as np
import cv2
import random
import time

def eatApple(apple, score):
    apple =[np.random.randint(1, size-1) * square, np.random.randint(1, size-1) * square]
    score += 1
    return apple, score

def wallCollide(snakeHead):
    if snakeHead[0]>size*square or snakeHead[0]<0 or snakeHead[1]>size*square or snakeHead[1]<0 :
        return 1
    else:
        return 0

def touchBody(snakeBody):
    snakeHead = snakeBody[0]
    if snakeHead in snakeBody[1:]:
        return 1
    else:
        return 0
    
size =40
square =20
img = np.zeros((size *square, size*square,3),dtype='uint8')


snakeHead = [
    np.random.randint(1, size-1) * square,
    np.random.randint(1, size-1) * square,
]
snakeBody = [
   [snakeHead[0] - square, snakeHead[1]],
   [snakeHead[0] - square*2, snakeHead[1]],
]
apple =[np.random.randint(1, size-1) * square, np.random.randint(1, size-1) * square]
score = 0
prevButton = 1
curDir = 1

while True:
    cv2.imshow('a',img)
 
    img = np.zeros((size *square, size*square,3),dtype='uint8')
    cv2.rectangle(img,(apple[0],apple[1]),(apple[0]+square,apple[1]+square),(0,0,255),3)
    for i in range(len(snakeBody)):
        if i != 0:
            cv2.rectangle(img,(snakeBody[i][0],snakeBody[i][1]),(snakeBody[i][0]+square,snakeBody[i][1]+square),(0,255,0),3)
        else:
            cv2.rectangle(img,(snakeBody[i][0],snakeBody[i][1]),(snakeBody[i][0]+square,snakeBody[i][1]+square),(255,0,0),3)

    t_end = time.time() + 0.05
    k = -1
    while time.time() < t_end:
        if k == -1:
            k = cv2.waitKey(10)
        else:
            continue
            
    havePressed = False
    if k == ord('a') and prevButton != 1:
        curDir = 0
    elif k == ord('d') and prevButton != 0:
        curDir = 1
    elif k == ord('w') and prevButton != 2:
        curDir = 3
    elif k == ord('s') and prevButton != 3:
        curDir = 2
    elif k == ord('q'):
        break
    else:
        curDir = curDir
    prevButton = curDir

    if curDir == 1:
        snakeHead[0] += square
    elif curDir == 0:
        snakeHead[0] -= square
    elif curDir == 2:
        snakeHead[1] += square
    elif curDir == 3:
        snakeHead[1] -= square
    snakeBody.insert(0,list(snakeHead))
    if snakeHead == apple:
        apple, score = eatApple(apple, score)
    else:
        snakeBody.pop()
    if (wallCollide(snakeHead)): print("Hit wall")
    if (touchBody(snakeBody)): print("Hit Body")
    if wallCollide(snakeHead)or touchBody(snakeBody) :
        font = cv2.FONT_HERSHEY_SIMPLEX
        img = np.zeros((size *square, size*square,3),dtype='uint8')
        cv2.putText(img,'Your Score is {}'.format(score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('a',img)
        cv2.waitKey(0)
        break
        
cv2.destroyAllWindows()