import cv2 as cv
import numpy as np

size = 10
matrix = np.zeros((size*size, size*size))
apple = np.random.randint(0, size-1), np.random.randint(0, size-1)
matrix[apple[0]][apple[1]] = 1
snakebody = [[
    np.random.randint(0, size-1), np.random.randint(0, size-1)
]]
cv.imshow("Display window", matrix)
