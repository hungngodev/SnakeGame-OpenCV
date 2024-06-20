import math
import heapq

class Cell:
	def __init__(self):
		self.parent_i = 0
		self.parent_j = 0
		self.f = float('inf')
		self.g = float('inf')
		self.h = 0
		self.dir = 0 

def is_valid(row, col, ROW, COL):
	return (row >= 0) and (row < ROW) and (col >= 0) and (col < COL)

def is_unblocked(grid, row, col):
	return grid[row][col]

def is_destination(row, col, dest):
	return row == dest[0] and col == dest[1]

def calculate_h_value(row, col, dest):
	return abs(row - dest[0]) + abs(col - dest[1])

def a_star_search(grid, src, dest, prevDir, snakeBody, square, COL, ROW, speed):
	snakeBodySquare = []
	for i in range(square//speed -1, len(snakeBody), square//speed):
		snakeBodySquare.append(snakeBody[i])
  
	for bodyCoordinate in snakeBodySquare:
		grid[bodyCoordinate[0]][bodyCoordinate[1]] = 0
  
	if not is_valid(src[0], src[1], ROW, COL):
		print("Source is invalid")
		return
	if not is_valid(dest[0], dest[1], ROW, COL):
		print("Destination is invalid")
		return
 
	if not is_unblocked(grid, src[0], src[1]):
		print("Source is blocked")
		return

	if not is_unblocked(grid, dest[0], dest[1]):
		print("Destination is blocked")
		return

	if is_destination(src[0], src[1], dest):
		print("We are already at the destination")
		return

	closed_list = [[False for _ in range(COL)] for _ in range(ROW)]
	cell_details = [[Cell() for _ in range(COL)] for _ in range(ROW)]

	i = src[0]
	j = src[1]
	cell_details[i][j].f = 0
	cell_details[i][j].g = 0
	cell_details[i][j].h = 0
	cell_details[i][j].parent_i = i
	cell_details[i][j].parent_j = j
	cell_details[i][j].dir = prevDir
	open_list = []
	heapq.heappush(open_list, (0.0, i, j))

	found_dest = False
	while len(open_list) > 0:
		p = heapq.heappop(open_list)

		i = p[1]
		j = p[2]
		closed_list[i][j] = True
		directions = []
		dirOfPrevMove = cell_details[i][j].dir
		directions = [0,1,2,3]
		if dirOfPrevMove == 0:
			directions.remove(1)
		elif dirOfPrevMove == 1:
			directions.remove(0)
		elif dirOfPrevMove == 2:
			directions.remove(3)
		elif dirOfPrevMove == 3:
			directions.remove(2)
		coordinates = [[-square,0], [square,0], [0,+square], [0,-square]]
		for dir in directions:
			new_i = i + coordinates[dir][0]
			new_j = j + coordinates[dir][1]

			if is_valid(new_i, new_j, ROW, COL) and is_unblocked(grid, new_i, new_j) and not closed_list[new_i][new_j]:
				if is_destination(new_i, new_j, dest):
					cell_details[new_i][new_j].parent_i = i
					cell_details[new_i][new_j].parent_j = j
					cell_details[new_i][new_j].dir = dir
					print("The destination cell is found")
					result = trace_path(cell_details, dest, src, speed, square)
					found_dest = True
					return result
				else:
					g_new = cell_details[i][j].g + 1.0
					h_new = calculate_h_value(new_i, new_j, dest)
					f_new = g_new + h_new

					if cell_details[new_i][new_j].f == float('inf') or cell_details[new_i][new_j].f > f_new:
						heapq.heappush(open_list, (f_new, new_i, new_j))
						cell_details[new_i][new_j].f = f_new
						cell_details[new_i][new_j].g = g_new
						cell_details[new_i][new_j].h = h_new
						cell_details[new_i][new_j].parent_i = i
						cell_details[new_i][new_j].parent_j = j
						cell_details[new_i][new_j].dir = dir
	if not found_dest:
		print("Failed to find the destination cell")
  
def trace_path(cell_details, dest, src, speed, square):
	path = []
	row = dest[0]
	col = dest[1]

	while not (cell_details[row][col].parent_i == row and cell_details[row][col].parent_j == col):
		path.append([row, col, cell_details[row][col].dir])
		temp_row = cell_details[row][col].parent_i
		temp_col = cell_details[row][col].parent_j
		row = temp_row
		col = temp_col
	path.reverse()
	print("Path", path)
	newPath = []
	for i in range(len(path)):
		for j in range(square//speed):
			newPath.append([path[i][0], path[i][1], path[i][2]])
	return newPath

