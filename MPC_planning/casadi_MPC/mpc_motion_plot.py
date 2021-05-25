#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de

import matplotlib.pyplot as plt
import numpy as np
from gears.polygon_generator import cal_coeff_mat


class UTurnMPC():
    def __init__(self):
        self.L = 2.0
        self.dt = 1e-1
        self.W = 2.  # width of car
        self.LF = 3.  # distance from rear to vehicle front end
        self.LB = 1.  # distance from rear to vehicle back end
        self.predicted_trajectory = None
        self.optimal_dt = None
        self.show_animation = True

    def motion_model(self, zst, u_in, dt):
        vr = u_in[0]
        ph = u_in[1]
        x = zst[0]
        y = zst[1]
        th = zst[2]

        x += vr * np.cos(th) * dt
        y += vr * np.sin(th) * dt
        th += vr * np.tan(ph) / self.L * dt

        return np.array([x, y, th])

    def plot_arrow(self, x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  head_length=width, head_width=width)
        plt.plot(x, y)

    def plot_robot(self, x, y, yaw):  # pragma: no cover
        # # 90 grad
        # outline = np.array([[self.W / 2, -self.W / 2, - self.W / 2, self.W / 2, self.W / 2],
        #                     [self.LF, self.LF, -self.LB, -self.LB, self.LF]])
        # old
        outline = np.array([
            [self.LF, self.LF, -self.LB, -self.LB, self.LF],
            [self.W / 2, -self.W / 2, - self.W / 2, self.W / 2, self.W / 2]])

        # outline = np.array([[1., -1, -1, 1, 1],
        #                     [1, 1, -1, -1, 1]])
        # outline[0, :] *= self.LF
        # outline[1, :] *= self.W / 2

        Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
        outline = Rot1 @ outline
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")

        arrow_x, arrow_y, arrow_yaw = np.cos(yaw) * 1.5 + x, np.sin(yaw) * 1.5 + y, yaw
        self.plot_arrow(arrow_x, arrow_y, arrow_yaw)

    def try_tracking(self, zst, u_op, trajectory, obst=None, ref_traj=None):
        k = 0
        f = plt.figure()
        while True:
            zst[:3] = self.motion_model(zst[:3], u_op[:, k], self.dt)  # simulate robot
            trajectory = np.vstack((trajectory, zst))  # store state history

            if self.show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                if ref_traj is not None:
                    plt.plot(ref_traj[0, 1:], ref_traj[1, 1:], "-", color="orange", label="warm start reference")

                plt.plot(zst[0], zst[1], "xr")
                self.plot_robot(zst[0], zst[1], zst[2])

                plt.plot(self.predicted_trajectory[0, :], self.predicted_trajectory[1, :], "-g", label="MPC prediciton")
                plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
                if obst is not None:
                    for obi in obst:
                        plt.plot(obi[:, 0], obi[:, 1], "o", color="black", label="obstacles")
                plt.legend()
                plt.plot(ref_traj[0, -1], ref_traj[1, -1], "xb")
                plt.plot(self.predicted_trajectory[0, -1], self.predicted_trajectory[1, -1], "x", color="purple")

                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.1)
                k += 1

            if k >= u_op.shape[1]:
                print("end point arrived...")
                break

        return trajectory

    def cal_distance(self, op_trajectories, iteration):
        diff_s = np.diff(np.array(op_trajectories), axis=1)
        sum_s = np.sum(np.hypot(diff_s[0, :], diff_s[1, :]))
        print("total number for iteration: ", iteration)
        print("total covered distance:{:.3f}m".format(sum_s))


def initialize_saved_data():
    loadtraj = np.load("../data/saved_hybrid_a_star.npz")
    ref_traj = loadtraj["saved_traj"]
    loadmap = np.load("../data/saved_obmap.npz", allow_pickle=True)
    ob1 = loadmap["pointmap"][0]
    ob2 = loadmap["pointmap"][1]
    ob3 = loadmap["pointmap"][2]
    ob = [ob1, ob2, ob3]

    ob_constraint_mat = loadmap["constraint_mat"]
    obst = []
    obst.append(ob_constraint_mat[:4, :])
    obst.append(ob_constraint_mat[4:8, :])
    obst.append(ob_constraint_mat[8:12, :])

    return ref_traj, ob, obst


def get_car_shape(ut):
    # # anticlockwise
    # car_outline = np.array([
    #     [ut.LF, -ut.LB, -ut.LB, ut.LF, ut.LF],
    #     [ut.W / 2, ut.W / 2, - ut.W / 2, -ut.W / 2, ut.W / 2]])
    # # 90 grad
    # car_outline = np.array([[1., -1, -1, 1, 1],
    #                         [1, 1, -1, -1, 1]])
    # car_outline[0, :] *= 2
    # car_outline[1, :] *= 1
    # clockwise
    car_outline = np.array(
        [[-ut.LB, ut.LF, ut.LF, ut.LB, -ut.LB],
         [ut.W / 2, ut.W / 2, -ut.W / 2, -ut.W / 2, ut.W / 2]])

    return cal_coeff_mat(car_outline.T)


if __name__ == '__main__':
    print(2)
