"""

A* grid based planning

author: Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import heapq
import math
import numpy as np
import matplotlib.pyplot as plt

show_animation = False


class Node:

    def __init__(self, x, y, cost, parent_index):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(
            self.cost) + "," + str(self.parent_index)


class DynamicProgrammingHeuristic:

    def calc_final_path(self, goal_node, closed_node_set, resolution):
        # generate final course
        rx, ry = [goal_node.x * resolution], [goal_node.y * resolution]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_node_set[parent_index]
            rx.append(n.x * resolution)
            ry.append(n.y * resolution)
            parent_index = n.parent_index

        return rx, ry

    def calc_distance_heuristic(self, gx, gy, ox, oy, resolution, rr):
        """
        gx: goal x position [m]
        gx: goal x position [m]
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        goal_node = Node(round(gx / resolution), round(gy / resolution), 0.0, -1)
        ox = [iox / resolution for iox in ox]
        oy = [ioy / resolution for ioy in oy]

        obstacle_map, min_x, min_y, max_x, max_y, x_w, y_w = self.calc_obstacle_map(
            ox, oy, resolution, rr)

        motion = self.get_motion_model()

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(goal_node, x_w, min_x, min_y)] = goal_node
        priority_queue = [(0, self.calc_index(goal_node, x_w, min_x, min_y))]

        while 1:
            if not priority_queue:
                break
            cost, c_id = heapq.heappop(priority_queue)
            if c_id in open_set:
                current = open_set[c_id]
                closed_set[c_id] = current
                open_set.pop(c_id)
            else:
                continue

            # show graph
            if show_animation:  # pragma: no cover
                plt.plot(current.x * resolution, current.y * resolution, "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 1 == 0:
                    plt.pause(0.001)

            # Remove the item from the open set

            # expand search grid based on motion model
            for i, _ in enumerate(motion):
                node = Node(current.x + motion[i][0],
                            current.y + motion[i][1],
                            current.cost + motion[i][2], c_id)
                n_id = self.calc_index(node, x_w, min_x, min_y)

                if n_id in closed_set:
                    continue

                if not self.verify_node(node, obstacle_map, min_x, min_y, max_x, max_y):
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # Discover a new node
                    heapq.heappush(
                        priority_queue,
                        (node.cost, self.calc_index(node, x_w, min_x, min_y)))
                else:
                    if open_set[n_id].cost >= node.cost:
                        # This path is the best until now. record it!
                        open_set[n_id] = node
                        heapq.heappush(
                            priority_queue,
                            (node.cost, self.calc_index(node, x_w, min_x, min_y)))

        return closed_set

    def verify_node(self, node, obstacle_map, min_x, min_y, max_x, max_y):
        if node.x < min_x:
            return False
        elif node.y < min_y:
            return False
        elif node.x >= max_x:
            return False
        elif node.y >= max_y:
            return False

        if obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy, resolution, vr):
        min_x = round(min(ox))
        min_y = round(min(oy))
        max_x = round(max(ox))
        max_y = round(max(oy))
        print("min_x:", min_x)
        print("min_y:", min_y)
        print("max_x:", max_x)
        print("max_y:", max_y)
        x_width = round(max_x - min_x)
        y_width = round(max_y - min_y)

        # obstacle map generation
        obstacle_map = [[False for _ in range(y_width)] for _ in range(x_width)]
        for ix in range(x_width):
            x = ix + min_x
            for iy in range(y_width):
                y = iy + min_y
                #  print(x, y)
                for iox, ioy in zip(ox, oy):
                    d = math.sqrt((iox - x) ** 2 + (ioy - y) ** 2)
                    if d <= vr / resolution:
                        obstacle_map[ix][iy] = True
                        break

        return obstacle_map, min_x, min_y, max_x, max_y, x_width, y_width

    def calc_index(self, node, x_width, x_min, y_min):
        return (node.y - y_min) * x_width + (node.x - x_min)

    def get_motion_model(self):
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion


def generate_obmap():
    ox, oy = [], []
    for i in range(10):
        ox.append(i)
        oy.append(25)
    for i in range(40):  # 60
        ox.append(i)
        oy.append(0.0)
    # for i in range(60):
    #     ox.append(60.0)
    #     oy.append(i)
    for i in range(41):  # 61
        ox.append(i)
        oy.append(60.0)
    for i in range(61):
        ox.append(0.0)
        oy.append(i)
    for i in range(40):
        ox.append(20.0)
        oy.append(i)
    for i in range(60):
        ox.append(40.0)
        oy.append(i)
    for i in range(10):
        ox.append(25.0)
        oy.append(60 - i)
    for i in range(10):
        ox.append(10 + i)
        oy.append(40)
    for i in range(10):
        ox.append(20 + i)
        oy.append(40)

    return np.array([ox, oy])


def init_startpoints():
    print("Start Hybrid A* planning")
    # Set Initial parameters
    start = [10.0, 10.0, np.deg2rad(90.0)]
    goal = [30.0, 15.0, np.deg2rad(0.0)]
    obst = generate_obmap()

    print("start : ", start)
    print("goal : ", goal)
    return start, goal, obst


if __name__ == '__main__':
    start, goal, obst = init_startpoints()
    dph = DynamicProgrammingHeuristic()
    dph.calc_distance_heuristic(30., 30., obst[0], obst[1], 8, 10)

