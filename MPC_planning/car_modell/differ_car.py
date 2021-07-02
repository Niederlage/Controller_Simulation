"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

from math import sqrt, cos, sin, tan, pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class DifferCarModel:
    def __init__(self):
        self.WB = 1.0  # rear to front wheel [m]
        self.W = 2.0  # width of car
        self.LF = 2.0  # distance from rear to vehicle front end
        self.LB = 1.0  # distance from rear to vehicle back end
        self.MAX_STEER = 50 / 180 * np.pi  # [rad] maximum steering angle
        self.SAFE_FRONT = self.LF + 0.05
        self.SAFE_BACK = self.LB + 0.05
        self.SAFE_WIDTH = self.W + 0.05
        self.W_BUBBLE_DIST = (self.LF - self.LB) / 2.0
        self.W_BUBBLE_R = sqrt(((self.LF + self.LB) / 2.0) ** 2 + 1)
        self.wheel_diameter = 0.5
        self.wheel_width = 0.2
        # # anticlockwise
        # # vehicle rectangle vertices need at least 5 edges to draw all vertices
        # self.VRX = [self.LF, -self.LB, -self.LB, self.LF, self.LF]  #
        # self.VRY = [self.W / 2, self.W / 2, -self.W / 2, -self.W / 2, self.W / 2]

        # clockwise
        # vehicle rectangle vertices need at least 5 edges to draw all vertices
        self.VRX = [self.LF, self.LF, -self.LB, -self.LB, self.LF]  #
        self.VRY = [self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2]

        self.wheelX = [self.wheel_diameter / 2, self.wheel_diameter / 2, -self.wheel_diameter / 2,
                       -self.wheel_diameter / 2, self.wheel_diameter / 2]  #
        self.wheelY = [self.wheel_width / 2, -self.wheel_width / 2, -self.wheel_width / 2, self.wheel_width / 2,
                       self.wheel_width / 2]

    def pi_2_pi(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

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

    def plot_wheel(self, x, y, yaw, offsetx, offsety, steer=0.):
        wheel_color = 'darkgoldenrod'
        rot = Rot.from_euler('z', yaw).as_matrix()[:2, :2]
        wheel_outline_x, wheel_outline_y = [], []
        # offset_rot = Rot.from_euler('z', steer).as_matrix()[0:2, 0:2] @ np.array([offsetx, offsety])
        for rx, ry in zip(self.wheelX, self.wheelY):
            offset_rot = Rot.from_euler('z', steer).as_matrix()[0:2, 0:2] @ np.array([rx, ry])
            converted_xy = rot @ np.stack([offset_rot[0] + offsetx, offset_rot[1] + offsety])
            wheel_outline_x.append(converted_xy[0] + x)
            wheel_outline_y.append(converted_xy[1] + y)

        plt.plot(wheel_outline_x, wheel_outline_y, wheel_color)

    def plot_wheels(self, x, y, yaw):
        self.plot_wheel(x, y, yaw, 0., 0.75 * self.W / 2)
        self.plot_wheel(x, y, yaw, 0., -0.75 * self.W / 2)
        # self.plot_wheel(x, y, yaw, self.WB, 0., steer=0.)

    def plot_car(self, x, y, yaw):
        car_color = '-k'
        c, s = cos(yaw), sin(yaw)
        rot = Rot.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = rot @ np.stack([rx, ry])
            car_outline_x.append(converted_xy[0] + x)
            car_outline_y.append(converted_xy[1] + y)
        offset = 0.
        arrow_x, arrow_y, arrow_yaw = c * offset + x, s * offset + y, yaw
        self.plot_arrow(arrow_x, arrow_y, arrow_yaw)
        self.plot_wheels(x, y, yaw)
        plt.plot(car_outline_x, car_outline_y, car_color)

    def move(self, x, y, yaw, distance, angle):
        x += distance * cos(yaw)
        y += distance * sin(yaw)
        yaw += self.pi_2_pi(angle)  # distance/2

        return x, y, self.pi_2_pi(yaw)

    def move_Runge_Kutta(self, x_, y_, yaw_, v_, omega_, dt, a_=0., omega_rate_=0.):

        k1_dx = v_ * np.cos(yaw_)
        k1_dy = v_ * np.sin(yaw_)
        k1_dyaw = omega_
        k1_dv = a_
        k1_domega = omega_rate_
        # k1_da = jerk_

        k2_dx = (v_ + 0.5 * dt * k1_dv) * np.cos(yaw_ + 0.5 * dt * k1_dyaw)
        k2_dy = (v_ + 0.5 * dt * k1_dv) * np.sin(yaw_ + 0.5 * dt * k1_dyaw)
        k2_dyaw = omega_ + 0.5 * dt * k1_domega
        k2_dv = a_
        k2_domega = omega_rate_
        # k2_da = jerk_

        k3_dx = (v_ + 0.5 * dt * k2_dv) * np.cos(yaw_ + 0.5 * dt * k2_dyaw)
        k3_dy = (v_ + 0.5 * dt * k2_dv) * np.sin(yaw_ + 0.5 * dt * k2_dyaw)
        k3_dyaw = omega_ + 0.5 * dt * k2_domega
        k3_dv = a_
        k3_domega = omega_rate_
        # k3_da = jerk_

        k4_dx = (v_ + dt * k3_dv) * np.cos(yaw_ + dt * k3_dyaw)
        k4_dy = (v_ + dt * k3_dv) * np.sin(yaw_ + dt * k3_dyaw)
        k4_dyaw = omega_ + 0.5 * dt * k3_domega
        # k4_dv = a_
        # k4_domega = omega_rate_
        # k4_da = jerk_

        x_ += dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
        y_ += dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
        yaw_ += dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
        # dv = dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6
        # domega = dt * (k1_domega + 2 * k2_domega + 2 * k3_domega + k4_domega) / 6
        # da = dt * (k1_da + 2 * k2_da + 2 * k3_da + k4_da) / 6

        return x_, y_, self.pi_2_pi(yaw_)


def main():
    x, y, yaw = 0., 0., 0.
    plt.axis('equal')
    car = DifferCarModel()
    car.plot_car(x, y, yaw)
    plt.show()


if __name__ == '__main__':
    main()
