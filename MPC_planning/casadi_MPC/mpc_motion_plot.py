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
        self.reserve_footprint = False
        self.plot_arrows = False

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

    def plot_arrow(self, x, y, yaw, length=1.5, width=0.5):  # pragma: no cover
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  head_length=width, head_width=width)
        plt.plot(x, y)

    def plot_robot(self, x, y, yaw):  # pragma: no cover

        outline = np.array([
            [self.LF, self.LF, -self.LB, -self.LB, self.LF],
            [self.W / 2, -self.W / 2, - self.W / 2, self.W / 2, self.W / 2]])

        Rot1 = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
        outline = Rot1 @ outline
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-", color='cadetblue')
        if self.plot_arrows:
            arrow_x, arrow_y, arrow_yaw = x, y, yaw
            self.plot_arrow(arrow_x, arrow_y, arrow_yaw)

    def try_tracking(self, zst, u_op, trajectory, obst=None, ref_traj=None):
        k = 0
        f = plt.figure()
        ax = plt.subplot()
        while True:
            zst[:3] = self.motion_model(zst[:3], u_op[:, k], self.dt)  # simulate robot
            trajectory = np.vstack((trajectory, zst))  # store state history

            if self.show_animation:
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                if not self.reserve_footprint:
                    plt.cla()
                    self.plot_arrows = True

                if ref_traj is not None:
                    plt.plot(ref_traj[0, 1:], ref_traj[1, 1:], "-", color="orange", label="warm start reference")
                if obst is not None:
                    for obi in obst:
                        plt.plot(obi[:, 0], obi[:, 1], color="black", label="obstacles")

                ax.plot(self.predicted_trajectory[0, :], self.predicted_trajectory[1, :], "xg", label="MPC prediciton")
                ax.plot(trajectory[:, 0], trajectory[:, 1], "-r")

                ax.plot(ref_traj[0, -1], ref_traj[1, -1], "xb")
                ax.plot(self.predicted_trajectory[0, -1], self.predicted_trajectory[1, -1], "x", color="purple")
                self.plot_robot(zst[0], zst[1], zst[2])

                if not self.reserve_footprint:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, fontsize=10, loc="upper left")

                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.01)
                k += 1

            if k >= u_op.shape[1]:
                print("end point arrived...")
                break

        return trajectory

    def plot_results(self, op_dt, op_trajectories, op_controls, ref_traj, ob):

        self.predicted_trajectory = op_trajectories
        zst = ref_traj[:, 0]
        trajectory = np.copy(zst)

        self.cal_distance(op_trajectories[:2, :], len(ref_traj.T))
        self.dt = op_dt
        print("Time resolution:{:.3f}s, total time:{:.3f}s".format(self.dt, self.dt * len(ref_traj.T)))
        # np.savez("saved_traj", traj=predicted_trajectory)

        fig = plt.figure()
        ax = plt.subplot(211)
        ax.plot(op_controls[0, :], label="v")
        ax.plot(op_controls[1, :], label="steer")
        ax.plot(op_controls[2, :], "-.", label="acc")
        ax.plot(op_controls[3, :], "-.", label="steer rate")
        ax.plot(op_controls[4, :], "-.", label="jerk")
        ax.grid()
        ax.legend()

        ax = plt.subplot(212)
        ax.plot(op_trajectories[2, :] * 180 / np.pi, label="heading/grad")
        ax.grid()
        ax.legend()

        trajectory = self.try_tracking(zst, op_controls, trajectory, obst=ob, ref_traj=ref_traj)
        print("Done")
        plt.show()

    def plot_results_path_only(self, op_trajectories, ref_traj, obst):

        self.cal_distance(op_trajectories[:2, :], len(ref_traj.T))
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(op_trajectories[2, :] * 180 / np.pi, label="yaw/grad")
        ax.plot(op_trajectories[3, :] * 180 / np.pi, label="steer/grad")
        ax.grid()
        ax.legend()

        f = plt.figure()
        i = 0
        while True:
            if self.show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                if ref_traj is not None:
                    plt.plot(ref_traj[0, 1:], ref_traj[1, 1:], "-", color="orange", label="reference")

                plt.plot(op_trajectories[0, :], op_trajectories[1, :], "xg", label="MPC prediciton")

                if obst is not None:
                    for obi in obst:
                        plt.plot(obi[:, 0], obi[:, 1], "_", color="black", label="obstacles")

                plt.plot(op_trajectories[0, -1], op_trajectories[1, -1], "x", color="purple")
                self.plot_robot(op_trajectories[0, i], op_trajectories[1, i], op_trajectories[2, i])

                plt.legend()
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.01)
                i += 1

                if i >= op_trajectories.shape[1]:
                    print("end point arrived...")
                    break

        plt.show()

    def cal_distance(self, op_trajectories, iteration):
        diff_s = np.diff(np.array(op_trajectories), axis=1)
        sum_s = np.sum(np.hypot(diff_s[0, :], diff_s[1, :]))
        print("total number for iteration: ", iteration)
        print("total covered distance:{:.3f}m".format(sum_s))

    def initialize_saved_data(self):
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

    def get_car_shape(self):
        Lev2 = (self.LB + self.LF) / 2
        Wev2 = self.W / 2
        car_outline = np.array(
            [[-Lev2, Lev2, Lev2, -Lev2, -Lev2],
             [Wev2, Wev2, -Wev2, -Wev2, Wev2]])

        return cal_coeff_mat(car_outline.T)


if __name__ == '__main__':
    print(2)
