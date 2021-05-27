import math
from numba import jit


class Node:
    
    def __init__(self, x_ind, y_ind, yaw_ind, direction,
                 x_list, y_list, yaw_list, directions,
                 steer=0.0, parent_index=None, cost=None):
        self.x_index = x_ind
        self.y_index = y_ind
        self.yaw_index = yaw_ind
        self.direction = direction
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        self.directions = directions
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost


class Path:
    
    def __init__(self, x_list, y_list, yaw_list, direction_list, cost):
        self.x_list = x_list
        self.y_list = y_list
        self.yaw_list = yaw_list
        # self.steer_list = steer_list
        self.direction_list = direction_list
        self.cost = cost


class Config:

    def __init__(self, ox, oy, xy_resolution, yaw_resolution):
        min_x_m = min(ox)
        min_y_m = min(oy)
        max_x_m = max(ox)
        max_y_m = max(oy)

        ox.append(min_x_m)
        oy.append(min_y_m)
        ox.append(max_x_m)
        oy.append(max_y_m)

        self.min_x = round(min_x_m / xy_resolution)
        self.min_y = round(min_y_m / xy_resolution)
        self.max_x = round(max_x_m / xy_resolution)
        self.max_y = round(max_y_m / xy_resolution)

        self.x_w = round(self.max_x - self.min_x)
        self.y_w = round(self.max_y - self.min_y)

        self.min_yaw = round(- math.pi / yaw_resolution) - 1
        self.max_yaw = round(math.pi / yaw_resolution)
        self.yaw_w = round(self.max_yaw - self.min_yaw)
