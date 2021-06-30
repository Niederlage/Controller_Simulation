#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
import numpy as np
from gears.polygon_generator import cal_coeff_mat
from CarModell.AckermannCar import AckermannCarModel


class UTurnMPC():
    def __init__(self):
        # self.L = 0.4
        # self.dt = 1e-1
        # self.W = 0.6  # width of car
        # self.LF = 0.6  # distance from rear to vehicle front end
        # self.LB = 0.2  # distance from rear to vehicle back end
        self.loc_start = np.array([-2.5, -1., np.deg2rad(70)])
        self.predicted_trajectory = None
        self.optimal_dt = None
        self.show_animation = True
        self.reserve_footprint = False
        self.plot_arrows = False
        self.car = AckermannCarModel()

    def ackermann_motion_model(self, zst, u_in, dt, Runge_Kutta=True):
        v_ = u_in[0]
        steer_ = u_in[1]
        x_ = zst[0]
        y_ = zst[1]
        yaw_ = zst[2]

        if not Runge_Kutta:
            x_, y_, yaw_ = self.car.move(x_, y_, yaw_, v_ * dt, steer_)
            return np.array([x_, y_, yaw_])
        else:
            x_, y_, yaw_ = self.car.move_Runge_Kutta(x_, y_, yaw_, v_, steer_, dt, a_=u_in[2], steer_rate_=u_in[3])

            return np.array([x_, y_, yaw_])

    def plot_op_controls(self, v_, acc_, jerk_, yaw_, yaw_rate_, steer_, steer_rate_):
        fig = plt.figure()
        ax = plt.subplot(211)
        ax.plot(v_, label="v", color="red")
        ax.plot(acc_, "-.", label="acc")
        ax.plot(jerk_, "-.", label="jerk")
        ax.grid()
        ax.legend()

        ax = plt.subplot(212)
        ax.plot(yaw_ * 180 / np.pi, label="heading/grad")
        ax.plot(yaw_rate_ * 180 / np.pi, label="yaw rate/grad")
        ax.plot(steer_ * 180 / np.pi, label="steer/grad", color="red")
        ax.plot(steer_rate_ * 180 / np.pi, "-.", label="steer rate/grad")
        ax.grid()
        ax.legend()

    def try_tracking(self, zst, u_op, trajectory, obst=None, ref_traj=None):
        k = 0
        f = plt.figure()
        ax = plt.subplot()

        while True:
            u_in = u_op[:, k]

            zst = self.ackermann_motion_model(zst, u_in, self.dt)  # simulate robot
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
                        plt.plot(obi[:, 0], obi[:, 1], ".", color="black", label="obstacles")

                ax.plot(self.predicted_trajectory[0, :], self.predicted_trajectory[1, :], "xg", label="MPC prediciton")
                ax.plot(trajectory[:, 0], trajectory[:, 1], "-r")

                ax.plot(ref_traj[0, -1], ref_traj[1, -1], "xb")
                ax.plot(self.predicted_trajectory[0, -1], self.predicted_trajectory[1, -1], "x", color="purple")
                self.car.plot_car(zst[0], zst[1], zst[2], u_in[1])

                if not self.reserve_footprint:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, fontsize=10, loc="upper left")

                plt.axis("equal")
                plt.grid(True)
                if u_op.shape[1] > 200:
                    if k % 5 == 0:
                        plt.pause(0.001)
                k += 1

            if k >= u_op.shape[1]:
                print("end point arrived...")
                break

        return trajectory

    def plot_results(self, op_dt, op_trajectories, op_controls, ref_traj, ob, four_states=False):

        self.predicted_trajectory = op_trajectories
        zst = ref_traj[:, 0]
        # zst = self.loc_start
        trajectory = np.copy(zst)

        self.cal_distance(op_trajectories[:2, :], len(ref_traj.T))
        self.dt = op_dt
        print("Time resolution:{:.3f}s, total time:{:.3f}s".format(self.dt, self.dt * len(ref_traj.T)))

        yaw_ = op_trajectories[2, :]
        yaw_rate = np.diff(yaw_) / op_dt
        yaw_rate_ = np.append(0., yaw_rate)

        v_ = op_controls[0, :]
        steer_ = op_controls[1, :]
        acc_ = op_controls[2, :]

        if four_states:
            steer_rate = np.diff(op_controls[1, :]) / op_dt
            steer_rate_ = np.append(0., steer_rate)

            jerk = np.diff(op_controls[2, :]) / op_dt
            jerk_ = np.append(0., jerk)
        else:
            steer_rate_ = op_controls[3, :]
            jerk_ = op_controls[4, :]

        self.plot_op_controls(v_, acc_, jerk_, yaw_, yaw_rate_, steer_, steer_rate_)

        trajectory = self.try_tracking(zst, op_controls, trajectory, obst=ob, ref_traj=ref_traj)

        print("Done")
        plt.show()

    def plot_results_path_only(self, op_trajectories, ref_traj, obst):

        self.cal_distance(op_trajectories[:2, :], len(ref_traj.T))
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(op_trajectories[2, :] * 180 / np.pi, label="yaw/grad")
        # ax.plot(op_trajectories[3, :] * 180 / np.pi, label="steer/grad")
        ax.grid()
        ax.legend()

        f = plt.figure()
        i = 0
        while True:
            if self.show_animation:
                if not self.reserve_footprint:
                    plt.cla()
                    self.plot_arrows = True

                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                if ref_traj is not None:
                    plt.plot(ref_traj[0, 1:], ref_traj[1, 1:], "-", color="orange", label="reference")

                plt.plot(op_trajectories[0, :], op_trajectories[1, :], "xg", label="MPC prediciton")

                if obst is not None:
                    for obi in obst:
                        plt.plot(obi[:, 0], obi[:, 1], color="black", label="obstacles")

                plt.plot(op_trajectories[0, -1], op_trajectories[1, -1], "x", color="purple")
                self.car.plot_car(op_trajectories[0, i], op_trajectories[1, i], op_trajectories[2, i])

                if not self.reserve_footprint:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, fontsize=10, loc="upper left")

                plt.axis("equal")
                plt.grid(True)
                if i % 5 == 0:
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
        loadtraj = np.load("../../data/saved_hybrid_a_star.npz")
        ref_traj = loadtraj["saved_traj"]
        loadmap = np.load("../../data/saved_obmap.npz", allow_pickle=True)
        ob1 = loadmap["pointmap"][0]
        ob2 = loadmap["pointmap"][1]
        ob3 = loadmap["pointmap"][2]
        ob4 = loadmap["pointmap"][3]
        ob = [ob1, ob2, ob3, ob4]

        ob_constraint_mat = loadmap["constraint_mat"]
        obst = []
        obst.append(ob_constraint_mat[:4, :])
        obst.append(ob_constraint_mat[4:8, :])
        obst.append(ob_constraint_mat[8:12, :])
        obst.append(ob_constraint_mat[12:16, :])
        # obst.append(ob_constraint_mat[:2, :])
        # obst.append(ob_constraint_mat[4:6, :])
        # obst.append(ob_constraint_mat[8:10, :])
        return ref_traj, ob, obst

    def get_car_shape(self):
        Lev2 = (self.car.LB + self.car.LF) / 2
        Wev2 = self.car.W / 2
        car_outline = np.array(
            [[-Lev2, Lev2, Lev2, -Lev2, -Lev2],
             [Wev2, Wev2, -Wev2, -Wev2, Wev2]])

        return cal_coeff_mat(car_outline.T)


if __name__ == '__main__':
    print(2)
