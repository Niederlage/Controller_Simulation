"""

Car model for Hybrid A* path planning

author: Zheng Zh (@Zhengzh)

"""

from math import sqrt, cos, sin, tan, pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import yaml


class AckermannCarModel:
    def __init__(self):
        self.WB = 2.0  # rear to front wheel [m]
        self.W = 2.0  # width of car
        self.LF = 3.0  # distance from rear to vehicle front end
        self.LB = 1.0  # distance from rear to vehicle back end
        self.MAX_STEER = 40 / 180 * np.pi  # [rad] maximum steering angle
        self.SAFE_FRONT = self.LF + 0.05
        self.SAFE_BACK = self.LB + 0.05
        self.SAFE_WIDTH = self.W + 0.05
        self.W_BUBBLE_DIST = (self.LF - self.LB) / 2.0
        self.W_BUBBLE_R = sqrt(((self.LF + self.LB) / 2.0) ** 2 + 1)
        self.wheel_diameter = 0.4
        self.wheel_width = 0.2
        self.model_type = "forklift"

        # # anticlockwise
        # # vehicle rectangle vertices need at least 5 edges to draw all vertices
        # self.VRX = [self.LF, -self.LB, -self.LB, self.LF, self.LF]  #
        # self.VRY = [self.W / 2, self.W / 2, -self.W / 2, -self.W / 2, self.W / 2]

        # clockwise
        # vehicle rectangle vertices need at least 5 edges to draw all vertices
        self.VRX = np.array([self.LF, self.LF, -self.LB, -self.LB, self.LF])  #
        self.VRY = np.array([self.W / 2, -self.W / 2, -self.W / 2, self.W / 2, self.W / 2])
        # vehicle wheels rectangles
        self.wheelX = np.array([self.wheel_diameter / 2, self.wheel_diameter / 2, -self.wheel_diameter / 2,
                                -self.wheel_diameter / 2, self.wheel_diameter / 2])  #
        self.wheelY = np.array(
            [self.wheel_width / 2, -self.wheel_width / 2, -self.wheel_width / 2, self.wheel_width / 2,
             self.wheel_width / 2])

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
        #
        self.VRX = [self.LF, -self.LB, -self.LB, self.LF, self.LF]  #
        self.VRY = [self.W / 2, self.W / 2, -self.W / 2, -self.W / 2,
                    self.W / 2, -self.W / 2, -self.W / 2, self.W / 2]
        print("")

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

    def plot_arrow(self, x, y, yaw, length=0.8, width=0.4, fc="r", ec="k"):
        """Plot arrow."""
        if not isinstance(x, float):
            for (i_x, i_y, i_yaw) in zip(x, y, yaw):
                self.plot_arrow(i_x, i_y, i_yaw)
        else:
            plt.arrow(x, y, length * cos(yaw), length * sin(yaw),
                      fc=fc, ec=ec, head_width=width, head_length=width, alpha=0.4)

    def plot_wheel(self, x, y, yaw, offsetx, offsety, steer=0., scalex=1., scaley=1., wheel_color='darkgoldenrod'):
        rot = Rot.from_euler('z', yaw).as_matrix()[:2, :2]
        wheel_outline_x, wheel_outline_y = [], []
        # offset_rot = Rot.from_euler('z', steer).as_matrix()[0:2, 0:2] @ np.array([offsetx, offsety])
        for rx, ry in zip(self.wheelX * scalex, self.wheelY * scaley):
            offset_rot = Rot.from_euler('z', steer).as_matrix()[0:2, 0:2] @ np.array([rx, ry])
            converted_xy = rot @ np.stack([offset_rot[0] + offsetx, offset_rot[1] + offsety])
            wheel_outline_x.append(converted_xy[0] + x)
            wheel_outline_y.append(converted_xy[1] + y)

        plt.plot(wheel_outline_x, wheel_outline_y, wheel_color)

    def plot_wheels(self, x, y, yaw, steer):
        # self.plot_wheel(x, y, yaw, 0., 0.75 * self.W / 2)
        # self.plot_wheel(x, y, yaw, 0., -0.75 * self.W / 2)
        self.plot_wheel(x, y, yaw, self.WB, 0., steer=steer, wheel_color="darkorange")

    def plot_robot(self, x, y, yaw, steer=0.):
        if self.model_type == "car":
            # print(" use model: car ")
            self.plot_car(x, y, yaw, steer=steer)

        elif self.model_type == "forklift":
            # print(" use model: forklift ")
            self.plot_forklift(x, y, yaw, steer=steer)

        else:
            print("no model selected!")

    def plot_car(self, x, y, yaw, steer=0.):
        car_color = 'cadetblue'
        rot = Rot.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = rot @ np.stack([rx, ry])
            car_outline_x.append(converted_xy[0] + x)
            car_outline_y.append(converted_xy[1] + y)
        # offset = 0.
        # c, s = cos(yaw), sin(yaw)
        # arrow_x, arrow_y, arrow_yaw = c * offset + x, s * offset + y, yaw
        # self.plot_arrow(arrow_x, arrow_y, arrow_yaw)
        self.plot_wheels(x, y, yaw, steer)
        plt.plot(car_outline_x, car_outline_y, car_color)

    def plot_forklift(self, x, y, yaw, steer=0.):
        car_color = 'cadetblue'
        head_length = (self.LF - self.WB) * 2
        head_width = self.W * 4
        fork_length = 0.75 * self.LF / self.wheel_diameter
        fork_width = 0.5 / self.wheel_width

        rot = Rot.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        car_outline_x, car_outline_y = [], []
        for rx, ry in zip(self.VRX, self.VRY):
            converted_xy = rot @ np.stack([rx, ry])
            car_outline_x.append(converted_xy[0] + x)
            car_outline_y.append(converted_xy[1] + y)

        # plot front steer wheel & head
        self.plot_wheel(x, y, yaw, self.WB, 0., steer=steer, wheel_color="darkorange")
        self.plot_wheel(x, y, yaw, self.WB, 0.,
                        scalex=head_length, scaley=head_width, wheel_color='darkorchid')

        # plot rear wheel & forks
        self.plot_wheel(x, y, yaw, self.WB * 1.6 - self.LF, 0.5 * self.W / 2,
                        scalex=fork_length, scaley=fork_width, wheel_color='darkorchid')
        self.plot_wheel(x, y, yaw, self.WB * 1.6 - self.LF, -0.5 * self.W / 2,
                        scalex=fork_length, scaley=fork_width, wheel_color='darkorchid')
        self.plot_wheel(x, y, yaw, 0., 0.5 * self.W / 2)
        self.plot_wheel(x, y, yaw, 0., -0.5 * self.W / 2)

        # plt.plot(car_outline_x, car_outline_y, "-.", color=car_color)

    def move(self, x, y, yaw, distance, steer):
        x += distance * cos(yaw)
        y += distance * sin(yaw)
        yaw += distance * tan(steer) / self.WB  # distance/2

        return x, y, yaw

    def move_forklift(self, x, y, yaw, distance, steer):
        if abs(steer) >= self.MAX_STEER:
            steer = steer / abs(steer) * self.MAX_STEER

        x += distance * cos(yaw) * cos(steer)
        y += distance * sin(yaw) * cos(steer)
        yaw += distance * sin(steer) / self.WB  # distance/2

        return x, y, yaw

    def move_Runge_Kutta(self, x_, y_, yaw_, v_, steer_, dt, a_=0., steer_rate_=0.):

        k1_dx = v_ * np.cos(yaw_)
        k1_dy = v_ * np.sin(yaw_)
        k1_dyaw = v_ / self.WB * np.tan(steer_)
        k1_dv = a_
        k1_dsteer = steer_rate_
        # k1_da = jerk_

        k2_dx = (v_ + 0.5 * dt * k1_dv) * np.cos(yaw_ + 0.5 * dt * k1_dyaw)
        k2_dy = (v_ + 0.5 * dt * k1_dv) * np.sin(yaw_ + 0.5 * dt * k1_dyaw)
        k2_dyaw = (v_ + 0.5 * dt * k1_dv) / self.WB * np.tan(steer_ + 0.5 * dt * k1_dsteer)
        k2_dv = a_  # + 0.5 * dt * k1_da
        k2_dsteer = steer_rate_
        # k2_da = jerk_

        k3_dx = (v_ + 0.5 * dt * k2_dv) * np.cos(yaw_ + 0.5 * dt * k2_dyaw)
        k3_dy = (v_ + 0.5 * dt * k2_dv) * np.sin(yaw_ + 0.5 * dt * k2_dyaw)
        k3_dyaw = (v_ + 0.5 * dt * k2_dv) / self.WB * np.tan(steer_ + 0.5 * dt * k2_dsteer)
        k3_dv = a_  # + 0.5 * dt * k2_da
        k3_dsteer = steer_rate_

        k4_dx = (v_ + dt * k3_dv) * np.cos(yaw_ + dt * k3_dyaw)
        k4_dy = (v_ + dt * k3_dv) * np.sin(yaw_ + dt * k3_dyaw)
        k4_dyaw = (v_ + dt * k3_dv) / self.WB * np.tan(steer_ + dt * k3_dsteer)

        x_ += dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
        y_ += dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
        yaw_ += dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6

        return x_, y_, self.pi_2_pi(yaw_)

    def move_forklift_Runge_Kutta(self, x_, y_, yaw_, v_, steer_, dt, a_=0., steer_rate_=0.):

        k1_dx = v_ * np.cos(yaw_) * np.cos(steer_)
        k1_dy = v_ * np.sin(yaw_) * np.cos(steer_)
        k1_dyaw = v_ / self.WB * np.sin(steer_)
        k1_dv = a_
        k1_dsteer = steer_rate_
        # k1_da = jerk_

        k2_dx = (v_ + 0.5 * dt * k1_dv) * np.cos(yaw_ + 0.5 * dt * k1_dyaw) * np.cos(steer_ + dt * k1_dsteer)
        k2_dy = (v_ + 0.5 * dt * k1_dv) * np.sin(yaw_ + 0.5 * dt * k1_dyaw) * np.cos(steer_ + dt * k1_dsteer)
        k2_dyaw = (v_ + 0.5 * dt * k1_dv) / self.WB * np.sin(steer_ + 0.5 * dt * k1_dsteer)
        k2_dv = a_
        k2_dsteer = steer_rate_
        # k2_da = jerk_

        k3_dx = (v_ + 0.5 * dt * k2_dv) * np.cos(yaw_ + 0.5 * dt * k2_dyaw) * np.cos(steer_ + dt * k2_dsteer)
        k3_dy = (v_ + 0.5 * dt * k2_dv) * np.sin(yaw_ + 0.5 * dt * k2_dyaw) * np.cos(steer_ + dt * k2_dsteer)
        k3_dyaw = (v_ + 0.5 * dt * k2_dv) / self.WB * np.sin(steer_ + 0.5 * dt * k2_dsteer)
        k3_dv = a_  # + 0.5 * dt * k2_da
        k3_dsteer = steer_rate_

        k4_dx = (v_ + dt * k3_dv) * np.cos(yaw_ + dt * k3_dyaw) * np.cos(steer_ + dt * k3_dsteer)
        k4_dy = (v_ + dt * k3_dv) * np.sin(yaw_ + dt * k3_dyaw) * np.cos(steer_ + dt * k3_dsteer)
        k4_dyaw = (v_ + dt * k3_dv) / self.WB * np.sin(steer_ + dt * k3_dsteer)

        x_ += dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
        y_ += dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
        yaw_ += dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6

        return x_, y_, self.pi_2_pi(yaw_)


def main():
    x, y, yaw = 0., 0., 1.
    with open("../config_forklift.yaml", 'r', encoding='utf-8') as f:
        param = yaml.load(f)
    plt.axis('equal')
    car = AckermannCarModel()
    car.set_parameters(param)
    car.plot_car(x, y, yaw, steer=-0.4)
    # car.plot_forklift(x, y, yaw, steer=-0.4)
    plt.show()


if __name__ == '__main__':
    main()
