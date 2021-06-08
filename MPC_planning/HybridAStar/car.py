"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

from math import sqrt, cos, sin, tan, pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class CarModel:
    def __init__(self):
        self.WB = 0.4  # rear to front wheel [m]
        self.W = 0.6  # width of car
        self.LF = 0.6  # distance from rear to vehicle front end
        self.LB = 0.2  # distance from rear to vehicle back end
        self.MAX_STEER = 35 / 180 * np.pi  # [rad] maximum steering angle
        self.SAFE_FRONT = self.LF + 0.1
        self.SAFE_BACK = self.LB + 0.1
        self.SAFE_WIDTH = self.W + 0.1
        self.W_BUBBLE_DIST = (self.LF - self.LB) / 2.0
        self.W_BUBBLE_R = sqrt(((self.LF + self.LB) / 2.0) ** 2 + 1)

        # vehicle rectangle vertices need at least 5 edges to draw all vertices
        self.VRX = [self.LF, -self.LB, -self.LB,
                    0., 0., 0., self.LF, self.LF]  #
        self.VRY = [self.W / 2, self.W / 2, -self.W / 2, -self.W / 2,
                    self.W / 2, -self.W / 2, -self.W / 2, self.W / 2]

    def set_parameters(self, param):
        self.WB = param["base"]
        self.LF = param["LF"]  # distance from rear to vehicle front end
        self.LB = param["LB"]  # distance from rear to vehicle back end
        self.W = param["W"]
        self.SAFE_FRONT = self.LF
        self.SAFE_BACK = self.LB
        self.SAFE_WIDTH = self.W
        self.W_BUBBLE_DIST = (self.LF - self.LB) / 2.0
        self.W_BUBBLE_R = sqrt(((self.LF + self.LB) / 2.0) ** 2 + 1)

        self.VRX = [self.LF, -self.LB, -self.LB,
                    0., 0., 0., self.LF, self.LF]  #
        self.VRY = [self.W / 2, self.W / 2, -self.W / 2, -self.W / 2,
                    self.W / 2, -self.W / 2, -self.W / 2, self.W / 2]
        print("")

    def check_car_collision(self, x_list, y_list, yaw_list, ox, oy, kd_tree):
        for i_x, i_y, i_yaw in zip(x_list, y_list, yaw_list):
            cx = i_x + self.W_BUBBLE_DIST * cos(i_yaw)
            cy = i_y + self.W_BUBBLE_DIST * sin(i_yaw)

            ids = kd_tree.query_ball_point([cx, cy], self.W_BUBBLE_R)  # circle out all the points inside Circle R

            if not ids:
                continue

            if not self.rectangle_check(i_x, i_y, i_yaw,
                                        [ox[i] for i in ids], [oy[i] for i in ids]):
                return False  # collision
        # print("car collision happens")
        return True  # no collision

    # @jit
    def rectangle_check(self, x, y, yaw, ox, oy):
        # transform obstacles to base link frame
        rotT = np.array([[np.cos(yaw), np.sin(yaw)],
                         [-np.sin(yaw), np.cos(yaw)]])

        for iox, ioy in zip(ox, oy):
            tx = iox - x
        ty = ioy - y
        # converted_xy = np.stack([tx, ty]).T @ rot
        converted_xy = rotT @ np.stack([tx, ty])
        rx, ry = converted_xy[0], converted_xy[1]

        if not (rx > self.SAFE_FRONT or
                rx < -self.SAFE_BACK or
                ry > self.SAFE_WIDTH / 2.0 or
                ry < -self.SAFE_WIDTH / 2.0):
            return False  # no collision
        # print("rectangular collision happens")
        return True  # collision

    def plot_arrow(self, x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
        """Plot arrow."""
        if not isinstance(x, float):
            for (i_x, i_y, i_yaw) in zip(x, y, yaw):
                self.plot_arrow(i_x, i_y, i_yaw)
        else:
            plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                      fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

    def plot_car(self, x, y, yaw):
        car_color = '-k'
        c, s = cos(yaw), sin(yaw)
        rot = Rot.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = rot @ np.stack([rx, ry])
            car_outline_x.append(converted_xy[0] + x)
            car_outline_y.append(converted_xy[1] + y)

        arrow_x, arrow_y, arrow_yaw = c * 1.5 + x, s * 1.5 + y, yaw
        self.plot_arrow(arrow_x, arrow_y, arrow_yaw)

        plt.plot(car_outline_x, car_outline_y, car_color)

    def pi_2_pi(self, angle):
        return (angle + pi) % (2 * pi) - pi

    def move(self, x, y, yaw, distance, steer):
        x += distance * cos(yaw)
        y += distance * sin(yaw)
        yaw += self.pi_2_pi(distance * tan(steer) / self.WB)  # distance/2

        return x, y, yaw


def main():
    x, y, yaw = 0., 0., 0.
    plt.axis('equal')
    car = CarModel()
    car.plot_car(x, y, yaw)
    plt.show()


if __name__ == '__main__':
    main()
