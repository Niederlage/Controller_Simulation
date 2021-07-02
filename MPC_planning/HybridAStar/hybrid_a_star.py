"""

Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""
import heapq
import math
import yaml
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

try:
    from dynamic_programming_heuristic import DynamicProgrammingHeuristic
    from reeds_shepp_path_planning import ReedsSheppPathPlanning
    from car_modell.ackermann_car import AckermannCarModel
    from auxiliaries import Node, Path, Config
except Exception:
    raise
from obstacles.obstacles import Obstacles

# address = "../config_differ_smoother.yaml"
address = "../config_forklift.yaml"
# address = "../config_OBCA_large.yaml"
obmap_address = "../data/saved_obmap.npz"

class HybridAStar:
    def __init__(self):
        self.XY_GRID_RESOLUTION = 1  # [ m / grid]
        self.YAW_GRID_RESOLUTION = np.deg2rad(5.0)  # [rad/ grid]
        self.MOTION_RESOLUTION = 0.2  # [m/ grid] path interpolate resolution
        self.N_STEER = 10  # number of steer command
        self.ROBOT_RADIUS = 0.8  # robot radius

        self.SB_COST = 1e10  # switch back penalty cost
        self.BACK_COST = 1e15  # backward penalty cost
        self.STEER_CHANGE_COST = 1.0  # steer angle change penalty cost
        self.STEER_COST = 10.0  # steer angle change penalty cost
        self.H_COST = 15.0  # Heuristic cost
        self.rs = ReedsSheppPathPlanning()
        self.car = AckermannCarModel()
        self.dph = DynamicProgrammingHeuristic()
        self.show_animation = True
        self.obmap = Obstacles()

    def calc_motion_inputs(self):
        ############## get steer list interpolation with direction #####################
        for steer in np.concatenate((np.linspace(-self.car.MAX_STEER, self.car.MAX_STEER,
                                                 self.N_STEER), [0.0])):
            for d in [1, -1]:
                yield [steer, d]

    def get_neighbors(self, current, config, ox, oy, kd_tree):
        ############## get all neighbor nodes list  #####################
        for steer, d in self.calc_motion_inputs():
            node = self.calc_next_node(current, steer, d, config, ox, oy, kd_tree)
            if node and self.verify_index(node, config):
                yield node

    def calc_next_node(self, current, steer, direction, config, ox, oy, kd_tree):
        ############## get next node: car move + collision check #####################
        x, y, yaw = current.x_list[-1], current.y_list[-1], current.yaw_list[-1]

        arc_l = self.XY_GRID_RESOLUTION * 1.5
        x_list, y_list, yaw_list = [], [], []
        for _ in np.arange(0, arc_l, self.MOTION_RESOLUTION):  # get nonholonomic path
            x, y, yaw = self.car.move(x, y, yaw, self.MOTION_RESOLUTION * direction, steer)
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)
        # iteration til last state (x, y, yaw)
        if not self.car.check_car_collision(x_list, y_list, yaw_list, ox, oy, kd_tree):
            return None

        d = direction == 1  # bool check direction reverse?
        x_ind = round(x / self.XY_GRID_RESOLUTION)
        y_ind = round(y / self.XY_GRID_RESOLUTION)
        yaw_ind = round(yaw / self.YAW_GRID_RESOLUTION)

        added_cost = 0.0

        if d != current.direction:
            added_cost += self.SB_COST

        # steer penalty (linear)
        added_cost += self.STEER_COST * steer ** 2

        # steer change penalty
        added_cost += self.STEER_CHANGE_COST * (current.steer - steer) ** 2

        cost = current.cost + added_cost + arc_l ** 2

        node = Node(x_ind, y_ind, yaw_ind, d, x_list,
                    y_list, yaw_list, [d],
                    parent_index=self.calc_index(current, config),
                    cost=cost, steer=steer)

        return node

    # def is_same_grid(self, n1, n2):
    #     if n1.x_index == n2.x_index \
    #             and n1.y_index == n2.y_index \
    #             and n1.yaw_index == n2.yaw_index:
    #         return True
    #     return False

    def analytic_expansion(self, current, goal, ox, oy, kd_tree):
        ############## expand at the last point with RS algorithm #####################
        start_x = current.x_list[-1]
        start_y = current.y_list[-1]
        start_yaw = current.yaw_list[-1]

        goal_x = goal.x_list[-1]
        goal_y = goal.y_list[-1]
        goal_yaw = goal.yaw_list[-1]

        max_curvature = math.tan(self.car.MAX_STEER) / self.car.WB
        paths = self.rs.calc_paths(start_x, start_y, start_yaw,
                                   goal_x, goal_y, goal_yaw,
                                   max_curvature, step_size=self.MOTION_RESOLUTION)

        if not paths:
            return None

        best_path, best = None, None

        for path in paths:
            if self.car.check_car_collision(path.x, path.y, path.yaw, ox, oy, kd_tree):
                cost = self.calc_rs_path_cost(path)
                if not best or best > cost:
                    best = cost
                    best_path = path

        return best_path

    def update_node_with_analytic_expansion(self, current, goal,
                                            c, ox, oy, kd_tree):
        path = self.analytic_expansion(current, goal, ox, oy, kd_tree)

        if path:
            if self.show_animation:
                plt.plot(path.x, path.y, "g-")
            f_x = path.x[1:]
            f_y = path.y[1:]
            f_yaw = path.yaw[1:]

            f_cost = current.cost + self.calc_rs_path_cost(path)
            f_parent_index = self.calc_index(current, c)

            fd = []
            for d in path.directions[1:]:
                fd.append(d >= 0)

            f_steer = 0.0
            f_path = Node(current.x_index, current.y_index, current.yaw_index,
                          current.direction, f_x, f_y, f_yaw, fd,
                          cost=f_cost, parent_index=f_parent_index, steer=f_steer)
            return True, f_path

        return False, None

    def calc_rs_path_cost(self, reed_shepp_path):
        ############## cal RS path cost= length + switch bacjk + steer ##########
        cost = 0.0
        for length in reed_shepp_path.lengths:
            if length >= 0:  # forward
                cost += length
            else:  # back
                cost += abs(length) * self.BACK_COST

        # switch back penalty
        for i in range(len(reed_shepp_path.lengths) - 1):
            # switch back
            if reed_shepp_path.lengths[i] * reed_shepp_path.lengths[i + 1] < 0.0:
                cost += self.SB_COST

        # steer penalty
        for course_type in reed_shepp_path.ctypes:
            if course_type != "S":  # curve
                cost += self.STEER_COST * self.car.MAX_STEER ** 2

        # steer change penalty
        # calc steer profile
        n_ctypes = len(reed_shepp_path.ctypes)
        u_list = [0.0] * n_ctypes
        for i in range(n_ctypes):
            if reed_shepp_path.ctypes[i] == "R":
                u_list[i] = - self.car.MAX_STEER
            elif reed_shepp_path.ctypes[i] == "L":
                u_list[i] = self.car.MAX_STEER

        for i in range(len(reed_shepp_path.ctypes) - 1):
            cost += self.STEER_CHANGE_COST * (u_list[i + 1] - u_list[i]) ** 2

        return cost

    def hybrid_a_star_planning(self, start, goal, obst):
        """
        start: start node
        goal: goal node
        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        xy_resolution: grid resolution [m]
        yaw_resolution: yaw angle resolution [rad]
        """

        start[2], goal[2] = self.rs.pi_2_pi(start[2]), self.rs.pi_2_pi(goal[2])

        obstacle_kd_tree = cKDTree(obst.T)

        config = Config(list(obst[0]), list(obst[1]), self.XY_GRID_RESOLUTION, self.YAW_GRID_RESOLUTION)

        start_node = Node(round(start[0] / self.XY_GRID_RESOLUTION),
                          round(start[1] / self.XY_GRID_RESOLUTION),
                          round(start[2] / self.YAW_GRID_RESOLUTION), True,
                          [start[0]], [start[1]], [start[2]], [True], cost=0)
        goal_node = Node(round(goal[0] / self.XY_GRID_RESOLUTION),
                         round(goal[1] / self.XY_GRID_RESOLUTION),
                         round(goal[2] / self.YAW_GRID_RESOLUTION), True,
                         [goal[0]], [goal[1]], [goal[2]], [True])

        openList, closedList = {}, {}  # openlist: gird to search, closelist: gird searched

        h_dp = self.dph.calc_distance_heuristic(
            goal_node.x_list[-1], goal_node.y_list[-1],
            obst[0], obst[1], self.XY_GRID_RESOLUTION, self.ROBOT_RADIUS)

        pq = []
        openList[self.calc_index(start_node, config)] = start_node
        heapq.heappush(pq, (self.calc_cost(start_node, h_dp, config),
                            self.calc_index(start_node, config)))
        final_path = None

        while True:
            if not openList:
                print("Error: Cannot find path, No open set")
                return [], [], []

            cost, c_id = heapq.heappop(pq)
            if c_id in openList:
                current = openList.pop(c_id)
                closedList[c_id] = current
            else:
                continue

            if self.show_animation:  # pragma: no cover
                plt.plot(current.x_list[-1], current.y_list[-1], "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closedList.keys()) % 1000 == 0:
                    plt.pause(0.0001)

            is_updated, final_path = self.update_node_with_analytic_expansion(
                current, goal_node, config, obst[0], obst[1], obstacle_kd_tree)

            if is_updated:
                print("path found!!!!!!!!")
                break

            # cal next nodes in the neighborhood with motion constraints
            for neighbor in self.get_neighbors(current, config, obst[0], obst[1], obstacle_kd_tree):
                neighbor_index = self.calc_index(neighbor, config)
                if neighbor_index in closedList:
                    continue
                if neighbor not in openList or openList[neighbor_index].cost > neighbor.cost:
                    heapq.heappush(pq, (self.calc_cost(neighbor, h_dp, config), neighbor_index))
                    openList[neighbor_index] = neighbor

        path = self.get_final_path(closedList, final_path)
        print("the total length for search:", len(closedList.keys()))
        return path

    def calc_cost(self, n, h_dp, c):
        ind = (n.y_index - c.min_y) * c.x_w + (n.x_index - c.min_x)
        if ind not in h_dp:
            return n.cost + 999999999  # collision cost
        return n.cost + self.H_COST * h_dp[ind].cost

    def get_final_path(self, closed, goal_node):
        reversed_x, reversed_y, reversed_yaw = \
            list(reversed(goal_node.x_list)), list(reversed(goal_node.y_list)), \
            list(reversed(goal_node.yaw_list))  # , list(reversed(goal_node.steer_list))
        direction = list(reversed(goal_node.directions))
        nid = goal_node.parent_index
        final_cost = goal_node.cost

        while nid:
            n = closed[nid]
            reversed_x.extend(list(reversed(n.x_list)))
            reversed_y.extend(list(reversed(n.y_list)))
            reversed_yaw.extend(list(reversed(n.yaw_list)))
            # reversed_steer.extend(list(reversed(n.steer_list)))
            direction.extend(list(reversed(n.directions)))

            nid = n.parent_index

        reversed_x = list(reversed(reversed_x))
        reversed_y = list(reversed(reversed_y))
        reversed_yaw = list(reversed(reversed_yaw))
        # reversed_steer = list(reversed(reversed_steer))
        direction = list(reversed(direction))

        # adjust first direction
        direction[0] = direction[1]

        path = Path(reversed_x, reversed_y, reversed_yaw, direction, final_cost)

        return path

    def verify_index(self, node, c):
        x_ind, y_ind = node.x_index, node.y_index
        if c.min_x <= x_ind <= c.max_x and c.min_y <= y_ind <= c.max_y:
            return True

        return False

    def calc_index(self, node, c):
        ind = (node.yaw_index - c.min_yaw) * c.x_w * c.y_w + \
              (node.y_index - c.min_y) * c.x_w + (node.x_index - c.min_x)

        if ind <= 0:
            print("Error(calc_index):", ind)

        return ind

    def generate_obmap(self, loadmap=True):
        if loadmap:
            load = np.load(obmap_address, allow_pickle=True)
            samples = np.zeros((1, 2))
            for i, ob in enumerate(load["pointmap"]):
                samples = np.vstack((samples, ob))

            samples = np.delete(samples, 0, axis=0)
            bounds = self.obmap.get_bounds()
            self.obmap.obst_pointmap = load["pointmap"]
            return np.vstack([samples, bounds]).T
        else:
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

    def init_startpoints(self, address, loadmap=True):
        print("Start Hybrid A* planning")
        if loadmap:
            with open(address, 'r', encoding='utf-8') as f:
                param = yaml.load(f)
            sx = param["start"]
            ex = param["goal"]
            start = [sx[0], sx[1], np.deg2rad(sx[2])]
            goal = [ex[0], ex[1], np.deg2rad(ex[2])]
            # self.car.set_parameters(param)
        else:
            # Set Initial parameters
            start = [10.0, 10.0, np.deg2rad(90.0)]
            goal = [20.0, 45.0, np.deg2rad(0.0)]

        obst = self.generate_obmap(loadmap=loadmap)

        print("start : ", start)
        print("goal : ", goal)
        print("obst shape : ", obst.shape)
        return start, goal, obst

    def save_planned_path(self, path, planner, obst):
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list

        saved_hybrid_a_star_traj = np.array([x, y, yaw]) / planner.XY_GRID_RESOLUTION
        saved_hybrid_a_star_traj[2, :] *= planner.XY_GRID_RESOLUTION
        saved_hybrid_a_star_ob = obst / planner.XY_GRID_RESOLUTION
        np.savez("../data/saved_hybrid_a_star.npz", saved_traj=saved_hybrid_a_star_traj,
                 saved_ob=saved_hybrid_a_star_ob)


def main():
    planner = HybridAStar()
    start, goal, obst = planner.init_startpoints(address)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if planner.show_animation:
        # ax.plot(obst[0], obst[1], ".k")
        planner.obmap.plot_obst(ax)
        planner.rs.plot_arrow(start[0], start[1], start[2], fc='g')
        planner.rs.plot_arrow(goal[0], goal[1], goal[2])

        plt.grid(True)
        plt.axis("equal")

    path = planner.hybrid_a_star_planning(start, goal, obst)
    planner.save_planned_path(path, planner, obst)

    if planner.show_animation:
        f =plt.figure()

        plt.plot(np.rad2deg(path.yaw_list), "-g", label="Hybrid A* yaw")
        k = 0

        f2 = plt.figure()
        for i_x, i_y, i_yaw in zip(path.x_list, path.y_list, path.yaw_list):
            plt.cla()
            # plt.plot(obst[0], obst[1], ".k")
            planner.obmap.plot_obst(ax)
            plt.plot(path.x_list, path.y_list, "-r", label="Hybrid A* path")
            plt.grid(True)
            plt.axis("equal")
            planner.car.plot_robot(i_x, i_y, i_yaw)
            k += 1
            if k % 2 == 0:
                plt.pause(0.001)

    print(__file__ + " done!!")
    plt.show()


if __name__ == '__main__':

    main()
