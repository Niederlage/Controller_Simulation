import numpy as np
import cv2
import matplotlib.pyplot as plt


class AStarSearcher:

    def __init__(self, img, obst, resolution, rr):

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obmap = None
        self.obst = obst
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion()
        self.calc_obmap(obst)  # generate obstacle map with bools
        self.show_animation = False

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.marks = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.marks) + "," + str(self.parent_index)

    @staticmethod
    def get_motion():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [1, 1, 1],
                  [0, 1, 1],
                  [-1, 1, 1],
                  [-1, 0, 1],
                  [-1, -1, 1],
                  [1, -1, 1]]

        return motion

    def planning(self, sx, sy):
        # Node : x, y, cost, parent_index
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(sx, self.min_x),
                              self.calc_xy_index(sy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node  # address node name to set

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set, key=lambda o: open_set[o].marks)
            current = open_set[c_id]  # x* = min(fx)

            # show graph and do not clear
            if self.show_animation:  # pragma: no cover
                # plt.cla()
                obstmap = np.argwhere(self.obst != 0)
                plt.plot(obstmap[:, 0], obstmap[:, 1], ".", color="gray")
                plt.plot(self.calc_grid_position(current.x, self.min_x), self.calc_grid_position(current.y, self.min_y)
                         , "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(
                                                 0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 100 == 0:
                    plt.pause(0.001)

            if self.goal_arrived(current, goal_node):
                print("reach start point")
                goal_node.parent_index = current.parent_index
                goal_node.marks = current.marks
                print("goal cost:", start_node.marks)
                break

            del open_set[c_id]

            closed_set[c_id] = current
            open_set = self.expand_nodes(current, c_id, open_set, closed_set)

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return [rx, ry]

    def goal_arrived(self, current, goal):
        dx = current.x - goal.x
        dy = current.y - goal.y
        if np.hypot(dx, dy) < 2 and goal.marks > 10:
            return True
        else:
            return False

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]

        # (reverse) searching from goal to start
        for n in closed_set.items():
            rx.append(self.calc_grid_position(n[1].x, self.min_x))
            ry.append(self.calc_grid_position(n[1].y, self.min_y))

        return rx, ry

    def calc_grid_position(self, index, min_position):
        return index * self.resolution + min_position

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # on the edges check
        if self.obmap[node.x, node.y]:
            return True

    def expand_nodes(self, current, current_id, open_set, closed_set):
        next_id = []
        for i, _ in enumerate(self.motion):
            node = self.Node(current.x + self.motion[i][0],
                             current.y + self.motion[i][1],
                             current.marks + self.motion[i][2], current_id)
            n_id = self.calc_grid_index(node)

            # If the node is valid, do nothing and continue
            if not self.verify_node(node):
                continue

            if n_id in closed_set:
                continue

            if n_id not in open_set:
                open_set[n_id] = node  # discovered a new node
                next_id.append(n_id)

        return open_set

    def calc_obmap(self, obst):
        obstmap = np.argwhere(obst != 0)
        # ox = obstmap[:, 0]
        # oy = obstmap[:, 1]
        # self.min_x = round(min(ox))
        # self.min_y = round(min(oy))
        # self.max_x = round(max(ox))
        # self.max_y = round(max(oy))
        self.min_x = 0
        self.min_y = 0
        self.max_x = len(obst[0])
        self.max_y = len(obst[1])

        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)
        self.obmap = np.array(obst, dtype=bool)


def get_rough_coordinates(filename):
    img = cv2.imread(filename, 0)

    img2 = np.copy(img)
    img[img != 0] = -1
    edges = img + 1
    hmax = len(edges[0])
    wmax = len(edges[1])
    asp = AStarSearcher(img2, edges, 1, 1)

    found_first_loc = False
    edges_list = []

    for i in range(hmax):
        for j in range(wmax):
            if found_first_loc:
                continue

            if edges[i, j] != 0 and not found_first_loc:
                shape = asp.planning(i, j)
                if shape is not None:
                    # print("found 1 shape")
                    edges_list.append(shape)
                    edges[shape[0], shape[1]] = 0.
                    break

    return edges_list


def remove_identicals(edges):
    new_list = []
    for edge in edges:
        origin = np.array(edge[0])
        down = np.array(edge[1])
        right = np.array(edge[2])
        difference_d_r = (down - right)

        if len(new_list) == 0:
            if np.any(difference_d_r) != 0:
                new_list.append(edge)

        else:
            last_origin = np.array(new_list[-1][0])
            difference_e = np.abs(origin - last_origin)

            if np.any(difference_e > 1):
                if np.any(difference_d_r != 0):
                    new_list.append(edge)

    return new_list


def remove_identicals1(edges):
    new_list = []
    for edge in edges:
        origin = np.array(edge[0])
        down = np.array(edge[1])
        right = np.array(edge[2])
        difference_d_r = (down - right)

        if np.any(difference_d_r == 0):
            new_list.append(edge)

    return new_list


def pixel2coordinates(edges, resolution, xmax, ymax):
    rot = np.array([[0, 1], [-1, 0]])
    trans = np.array([0, ymax])
    edges = np.array(edges)
    xlist = []
    ylist = []
    for i in range(len(edges.T)):
        res = (rot @ edges[:, i] + trans) * resolution
        xlist.append(res[0])
        ylist.append(-res[1])

    return [xlist, ylist]


if __name__ == '__main__':

    convert2coordinates = True
    file = "fig1.png"
    edges_list = get_rough_coordinates(file)
    print("size edge list:", len(edges_list))
    resolution = 15 / 300
    img = cv2.imread(file, 0)
    fig = plt.figure()
    plt.imshow(img, cmap='gray')
    if convert2coordinates:
        # plt.plot(edges_list[0][0], edges_list[0][1], "bo")
        for kant in edges_list:
            edge = pixel2coordinates(kant, 1, 300, 300)
            resolution = 2.05 / (max(edge[1]) - min(edge[1]))
            print("resolution:", resolution)
            biasx = min(edge[0]) * resolution
            print("biasx:", biasx)
            plt.plot(edge[0], edge[1], "ro")
            xmax = max(edge[0]) * resolution - biasx
            xmin = min(edge[0]) * resolution - biasx
            ymax = max(edge[1]) * resolution
            ymin = min(edge[1]) * resolution
            print("xmin:{}, xmax:{}, ymin:{}, ymax:{}".format(xmin, xmax, ymin, ymax))

    plt.show()

    print(1)
