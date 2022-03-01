import argparse
from importlib.resources import path
import queue
from typing_extensions import Required
import numpy as np
import cv2
import math
from queue import PriorityQueue
import time
import imageio

# Creating a class Node with visited_node, parent_node and cost as members
class Node:

    def __init__(self, visited_node=False,parent_node=None, cost = math.inf ):
        self.visited_node = visited_node
        self.parent_node = parent_node
        self.cost = cost

# Get start node and check validation within for map range
def getInitialNode(dijkstra_map, coordinates):
    r ,c = dijkstra_map.shape[:2]
    row = r-coordinates[1]-1
    col = coordinates[0]
    if not nodeValidityCheck(dijkstra_map, row, col):
        print('Out of Range')
        return None
    return row, col

# Get goal node and check validation within for map range
def getGoalNode(dijkstra_map, coordinates):
    r, c = dijkstra_map.shape[:2]
    row = r-coordinates[1]-1
    col = coordinates[0]
    if not nodeValidityCheck(dijkstra_map, row, col):
        print('Error with Goal location, Out of Range')
        return None
    return row, col

# Step 2: Finding Mathematical constrain for free space to avoid obstacles
def nodeValidityCheck(my_map, r, c):
    row, col = my_map.shape[:2]

    if 0 <= r < row and 0 <= c < col and (my_map[r][c]==(0,0,0)).all():
        return True
    else :
        return False

#Step 1 Defining the Actions in a Mathematical format
# Defining action space: updating and validating each action set
def updateNodes(data, custom_map, c_node, o_list):

    m,n = c_node
    #Up node
    if nodeValidityCheck(custom_map,m-1,n):
        if (data[m][n].cost + 1 < data[m-1][n].cost):
            data[m-1][n].cost = data[m][n].cost + 1
            data[m-1][n].parent_node = c_node
            o_list.put((data[m-1][n].cost,(m-1,n)))

    #Up-left node
    if nodeValidityCheck(custom_map,m-1,n-1):
        if (data[m][n].cost + 1.41 < data[m-1][n-1].cost):
            data[m-1][n-1].cost = data[m][n].cost + 1.41
            data[m-1][n-1].parent_node = c_node
            o_list.put((data[m-1][n-1].cost,(m-1,n-1)))

    #left node
    if nodeValidityCheck(custom_map,m,n-1):
        if (data[m][n].cost + 1 < data[m][n-1].cost):
            data[m][n-1].cost = data[m][n].cost + 1
            data[m][n-1].parent_node = c_node
            o_list.put((data[m][n-1].cost,(m,n-1)))

    #Down-left node
    if nodeValidityCheck(custom_map,m+1,n-1):
        if (data[m][n].cost + 1.41 < data[m+1][n-1].cost):
            data[m+1][n-1].cost = data[m][n].cost + 1.41
            data[m+1][n-1].parent_node = c_node
            o_list.put((data[m+1][n-1].cost,(m+1,n-1)))

    #down_node
    if nodeValidityCheck(custom_map,m+1,n):
        if (data[m][n].cost + 1 < data[m+1][n].cost):
            data[m+1][n].cost = data[m][n].cost + 1
            data[m+1][n].parent_node = c_node
            o_list.put((data[m+1][n].cost,(m+1,n)))

    #down-right node
    if nodeValidityCheck(custom_map,m+1,n+1):
        if (data[m][n].cost + 1.41 < data[m+1][n+1].cost):
            data[m+1][n+1].cost = data[m][n].cost + 1.41
            data[m+1][n+1].parent_node = c_node
            o_list.put((data[m+1][n+1].cost,(m+1,n+1)))

    #right node
    if nodeValidityCheck(custom_map,m,n+1):
        if (data[m][n].cost + 1 < data[m][n+1].cost):
            data[m][n+1].cost = data[m][n].cost + 1
            data[m][n+1].parent_node = c_node
            o_list.put((data[m][n+1].cost,(m,n+1)))

    #Up-right node
    if nodeValidityCheck(custom_map,m-1,n+1):
        if (data[m][n].cost + 1.41 < data[m-1][n+1].cost):
            data[m-1][n+1].cost = data[m][n].cost + 1.41
            data[m-1][n+1].parent_node = c_node
            o_list.put((data[m-1][n+1].cost,(m-1,n+1)))

    return data

#Creating visualization for Dijkstra's Algorithm
def createMap():
    img= np.zeros((250,400,3))
    obstacle_circle(img)
    obstacle_polygon(img)
    obstacle_boomerang(img)
    cv2.imwrite('./dijkstra_map.jpg', img)
    cv2.imshow('Dijkstra Map', img)
    return img

# Step 4: Finding the optimal path: Backtracking the closed list
def backTrace(list, image , goal):
    n_list = []
    i_list = []
    result = './BackTrace.gif'
    current_node = goal
    while current_node is not None:
        n_list.append(current_node)
        current_node = list[current_node[0]][current_node[1]].parent_node
    n_list = n_list[::-1]
    for current_node in n_list:
        image[current_node[0]][current_node[1]] = (0, 0, 255)
        i_list.append(np.uint8(image.copy()))
        image[current_node[0]][current_node[1]] = (255, 0, 0)
    imageio.mimsave(result, i_list,fps=60)

# Step:3 Generating the Grap and checking for the goal node in each iteration
def dijkstra(inputs):
    # loading visualizer for exploring map
    global_map = createMap()
    # obtaining rows and columns of the designed map
    row, col = global_map.shape[:2]
    # Get initial node and goal node
    x_i = getInitialNode(global_map, inputs['source_node'])
    x_g = getGoalNode(global_map, inputs['goal_node'])
    # error handling if given nodes are empty
    if not (x_i and x_g):
        exit(1)
    list_ = np.array([[Node() for j in range(col)] for i in range(row)])
    list_[x_i[0]][x_i[1]].visited_node = True
    list_[x_i[0]][x_i[1]].cost = 0
    open_list = PriorityQueue()
    # Inserting initial node into queue
    open_list.put((list_[x_i[0]][x_i[1]].cost, x_i))
    # creating a list of all explored list
    closed_list = []
    timer_start = time.time()
    dijkstra_map = global_map.copy()
    dijkstra_map[x_g[0]][x_g[1]] = (0,0,255)
    while open_list:
        current_node = open_list.get()[1]
        #checking for the goal node in each iteration
        if current_node == x_g:
            timer_end = time.time()
            print('Goal found in:' + str( timer_end - timer_start) + "seconds")
            print('Printing Visualization: Please check the generated graphic Images')
            backTrace(list_, dijkstra_map, x_g)
            break
        list_[current_node[0]][current_node[1]].visited = True
        # Updating the nodes as it explores
        list_ = updateNodes(list_, global_map, current_node, open_list)
        dijkstra_map[current_node[0]][current_node[1]] = (137, 239, 216)  
        closed_list.append(np.uint8(dijkstra_map.copy()))
    # saving the explored map as GIF
    imageio.mimsave('explored_map.gif', closed_list, fps=60)
    cv2.destroyAllWindows()

# Creating circle as an obstacle on the map
def obstacle_circle(img):
    cv2.circle(img,(300,65), 40, (148,15,169), -1)

# Creating Polygon as an obstacle on the map
def obstacle_polygon(img):
 hexa = np.array([[[200,190],[235,170],[235,130],[200,110],[165,130],[165,170]]], np.int32)
 cv2.fillPoly(img, [hexa],(148,15,169))

# Creating Boomerang as an obstacle on the map
def obstacle_boomerang(img):
 hexa = np.array([[[36,65],[115,40],[80,70],[105,150]]], np.int32)
 cv2.fillPoly(img, [hexa],(148,15,169))

# Parsing source and destination location from the args/ commandline
input_parser = argparse.ArgumentParser()
input_parser.add_argument("-source", "--source_node", required= True, help="", nargs='+', type=int)
input_parser.add_argument("-destination", "--goal_node", required= False, help="", nargs='+', type=int)
algo_input = vars(input_parser.parse_args())
# feeding start and goal node to the dijkstras algorithm
dijkstra(algo_input)
cv2.waitKey(0);cv2.destroyAllWindows()

