#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
import numpy as np
from obstacles.obstacles import Obstacles
from car_modell.ackermann_car import AckermannCarModel


class UTurnMPC():
    def __init__(self):

        self.loc_start = np.array([-2.5, -1., np.deg2rad(70)])
        self.predicted_trajectory = None
        self.optimal_dt = None
        self.show_animation = True
        self.reserve_footprint = False
        self.use_Runge_Kutta = False
        self.plot_arrows = False
        self.car = AckermannCarModel()
        self.obmap = Obstacles()
        self.show_obstacles = False

    def ackermann_motion_model(self, zst, u_in, dt, Runge_Kutta=False):
        v_ = u_in[0]
        steer_ = u_in[1]
        x_ = zst[0]
        y_ = zst[1]
        yaw_ = zst[2]

        if not Runge_Kutta:
            # x_, y_, yaw_ = self.car.move(x_, y_, yaw_, v_ * dt, steer_)
            x_, y_, yaw_ = self.car.move_forklift(x_, y_, yaw_, v_ * dt, steer_)
            return np.array([x_, y_, yaw_])
        else:
            # x_, y_, yaw_ = self.car.move_Runge_Kutta(x_, y_, yaw_, v_, steer_, dt, a_=u_in[2], steer_rate_=u_in[3])
            x_, y_, yaw_ = self.car.move_Runge_Kutta(x_, y_, yaw_, v_, steer_, dt, a_=u_in[2], steer_rate_=u_in[3])

            return np.array([x_, y_, yaw_])

    def plot_op_controls(self, v_, acc_, jerk_, yaw_, yaw_rate_, steer_, steer_rate_, four_states=True):
        fig = plt.figure()
        ax = plt.subplot(211)
        ax.plot(v_, label="v", color="red")
        ax.plot(acc_, "-.", label="acc")
        if not four_states:
            ax.plot(jerk_, "-.", label="jerk")
        ax.grid()
        ax.legend()

        ax = plt.subplot(212)
        ax.plot(yaw_ * 180 / np.pi, label="heading/grad")
        # ax.plot(yaw_rate_ * 180 / np.pi, label="yaw rate/grad")
        ax.plot(steer_ * 180 / np.pi, label="steer/grad", color="red")
        if not four_states:
            ax.plot(steer_rate_ * 180 / np.pi, "-.", label="steer rate/grad")
        ax.grid()
        ax.legend()

    def plot_animation(self, ax, k, zst, trajectory, ref_traj, steer):
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        if not self.reserve_footprint:
            plt.cla()
            self.plot_arrows = True
            self.obmap.plot_obst(ax)

        if ref_traj is not None:
            plt.plot(ref_traj[0, 1:], ref_traj[1, 1:], "-", color="orange", label="warm start reference")

        ax.plot(self.predicted_trajectory[0, :], self.predicted_trajectory[1, :], "xg", label="MPC prediciton")
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        ax.plot(ref_traj[0, -1], ref_traj[1, -1], "xb")
        ax.plot(self.predicted_trajectory[0, -1], self.predicted_trajectory[1, -1], "x", color="purple")
        self.car.plot_robot(zst[0], zst[1], zst[2], steer)

        if not self.reserve_footprint:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, fontsize=10, loc="upper left")

        plt.axis("equal")
        plt.grid(True)
        if self.predicted_trajectory.shape[1] > 200:
            if k % 5 == 0:
                plt.pause(0.001)
        else:
            plt.pause(0.001)

    def try_tracking(self, zst, u_op, trajectory, ref_traj=None, four_states=False):
        k = 0
        f = plt.figure()
        ax = plt.subplot()
        self.obmap.plot_obst(ax)

        while True:
            u_in = u_op[:, k]

            zst = self.ackermann_motion_model(zst, u_in, self.dt, Runge_Kutta=self.use_Runge_Kutta )  # simulate robot
            trajectory = np.vstack((trajectory, zst))  # store state history

            if self.show_animation:
                self.plot_animation(ax, k, zst, trajectory, ref_traj, u_in[1])
                k += 1

            if k >= u_op.shape[1]:
                print("end point arrived...")
                break

        return trajectory

    def plot_results(self, op_dt, op_trajectories, op_controls, ref_traj, four_states=True):

        self.predicted_trajectory = op_trajectories
        zst = ref_traj[:, 0]
        # zst = self.loc_start
        trajectory = np.copy(zst)

        self.cal_distance(op_trajectories[:2, :])
        self.dt = op_dt
        print("Time resolution:{:.3f}s, total time:{:.3f}s".format(self.dt, self.dt * len(op_trajectories.T)))

        yaw_ = op_trajectories[2, :]
        yaw_rate = np.diff(yaw_) / op_dt
        yaw_rate_ = np.append(0., yaw_rate)

        v_ = op_controls[0, :]
        steer_ = op_controls[1, :]

        if four_states:
            acc_ = np.diff(op_controls[0, :]) / op_dt
            acc_ = np.append(0., acc_)

            steer_rate = np.diff(op_controls[1, :]) / op_dt
            steer_rate_ = np.append(0., steer_rate)

            jerk = np.diff(acc_) / op_dt
            jerk_ = np.append(0., jerk)
        else:
            acc_ = op_controls[2, :]
            steer_rate_ = op_controls[3, :]
            jerk_ = op_controls[4, :]

        self.plot_op_controls(v_, acc_, jerk_, yaw_, yaw_rate_, steer_, steer_rate_, four_states)

        trajectory = self.try_tracking(zst, op_controls, trajectory, ref_traj=ref_traj, four_states=four_states)

        print("Done")
        plt.show()

    def cal_distance(self, op_trajectories):
        diff_s = np.diff(np.array(op_trajectories), axis=1)
        sum_s = np.sum(np.hypot(diff_s[0, :], diff_s[1, :]))
        print("total number for iteration: ", op_trajectories.shape[1])
        print("total covered distance:{:.3f}m".format(sum_s))

    def initialize_saved_data(self, traj_adress="../../data/saved_hybrid_a_star.npz",
                              map_adress="../../data/saved_obmap_obca.npz"):

        loadtraj = np.load(traj_adress)
        ref_traj = loadtraj["saved_traj"]
        loadmap = np.load(map_adress, allow_pickle=True)
        self.obmap.obst_pointmap = loadmap["pointmap"]
        return ref_traj, loadmap["pointmap"], loadmap["constraint_mat"]

    def get_car_shape(self):
        Lev2 = (self.car.LB + self.car.LF) / 2
        Wev2 = self.car.W / 2
        car_outline = np.array(
            [[-Lev2, Lev2, Lev2, -Lev2, -Lev2],
             [Wev2, Wev2, -Wev2, -Wev2, Wev2]])

        return self.obmap.cal_coeff_mat(car_outline.T)


if __name__ == '__main__':
    print(2)
