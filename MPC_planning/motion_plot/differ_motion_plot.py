#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de
import matplotlib.pyplot as plt
from obstacles.obstacles import Obstacles
from car_modell.differ_car import DifferCarModel

plt.switch_backend('TkAgg')
import numpy as np


class UTurnMPC():
    def __init__(self):

        self.loc_start = np.array([-2.5, -1., np.deg2rad(70)])
        self.predicted_trajectory = None
        self.optimal_dt = None
        self.show_animation = True
        self.reserve_footprint = False
        self.plot_arrows = False
        self.use_loc_start = False
        self.car = DifferCarModel()
        self.obmap = Obstacles()

    def normalize_angle(self, yaw):
        return (yaw + np.pi) % (2 * np.pi) - np.pi

    def differ_motion_model(self, zst, u_in, dt, Runge_Kutta=False):
        v_ = u_in[0]
        omega_ = u_in[1]
        x_ = zst[0]
        y_ = zst[1]
        yaw_ = zst[2]
        # k1_dx = v_ * np.cos(yaw_)
        # k1_dy = v_ * np.sin(yaw_)
        # k1_dyaw = omega_
        # # k1_dv = a_
        # # k1_domega = omega_rate_
        # # k1_da = jerk_
        #
        # k2_dx = v_ * np.cos(yaw_ + 0.5 * dt * k1_dyaw)
        # k2_dy = v_ * np.sin(yaw_ + 0.5 * dt * k1_dyaw)
        # k2_dyaw = omega_
        # # k2_dv = a_ + 0.5 * dt * k1_da
        # # k2_domega = omega_rate_
        # # k2_da = jerk_
        #
        # k3_dx = v_ * np.cos(yaw_ + 0.5 * dt * k2_dyaw)
        # k3_dy = v_ * np.sin(yaw_ + 0.5 * dt * k2_dyaw)
        # k3_dyaw = omega_
        # # k3_dv = a_ + 0.5 * dt * k2_da
        # # k3_domega = omega_rate_
        # # k3_da = jerk_
        #
        # k4_dx = v_ * np.cos(yaw_ + 0.5 * dt * k3_dyaw)
        # k4_dy = v_ * np.sin(yaw_ + 0.5 * dt * k3_dyaw)
        # k4_dyaw = omega_
        # # k4_dv = a_ + 0.5 * dt * k3_da
        # # k4_domega = omega_rate_
        # # k4_da = jerk_
        #
        # dx = dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
        # dy = dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
        # dyaw = dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
        # # dv = dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6
        # # domega = dt * (k1_domega + 2 * k2_domega + 2 * k3_domega + k4_domega) / 6
        # # da = dt * (k1_da + 2 * k2_da + 2 * k3_da + k4_da) / 6
        #
        # x_ += dx
        # y_ += dy
        # yaw_ += dyaw
        # yaw_ = self.normalize_angle(yaw_)

        if not Runge_Kutta:
            x_, y_, yaw_ = self.car.move(x_, y_, yaw_, v_ * dt, omega_ * dt)
            return np.array([x_, y_, yaw_])
        else:
            x_, y_, yaw_ = self.car.move_Runge_Kutta(x_, y_, yaw_, v_, omega_, dt, a_=u_in[2], omega_rate_=u_in[3])

            return np.array([x_, y_, yaw_])

    def plot_op_controls(self, v_, acc_, jerk_, yaw_, omega_, omega_rate_):

        fig = plt.figure()
        ax = plt.subplot(211)
        ax.plot(v_, label="v", color="red")
        ax.plot(acc_, "-.", label="acc")
        ax.plot(jerk_, "-.", label="jerk")
        ax.grid()
        ax.legend()

        ax = plt.subplot(212)
        ax.plot(yaw_ * 180 / np.pi, label="heading/grad")
        ax.plot(omega_ * 180 / np.pi, label="omega/grad")
        ax.plot(omega_rate_ * 180 / np.pi, "-.", label="omega rate/grad")
        ax.grid()
        ax.legend()

    def try_tracking(self, zst, u_op, trajectory, ref_traj=None):
        k = 0
        f = plt.figure()
        ax = plt.subplot()

        while True:
            u_in = u_op[:, k]
            zst = self.differ_motion_model(zst, u_in, self.dt)  # simulate robot

            trajectory = np.vstack((trajectory, zst))  # store state history

            if self.show_animation:
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                plt.cla()
                self.plot_arrows = True

                if ref_traj is not None:
                    plt.plot(ref_traj[0, 1:], ref_traj[1, 1:], "-", color="orange", label="warm start reference")

                ax.plot(self.predicted_trajectory[0, :], self.predicted_trajectory[1, :], "xg", label="MPC prediciton")
                ax.plot(trajectory[:, 0], trajectory[:, 1], "-r")

                ax.plot(ref_traj[0, -1], ref_traj[1, -1], "xb")
                ax.plot(self.predicted_trajectory[0, -1], self.predicted_trajectory[1, -1], "x", color="purple")
                self.car.plot_car(zst[0], zst[1], zst[2])

                plt.axis("equal")
                plt.grid(True)
                if u_op.shape[1] > 200:
                    if k % 20 == 0:
                        plt.pause(0.001)
                else:
                    plt.pause(0.001)

                k += 1

            if k >= u_op.shape[1]:
                print("end point arrived...")
                break

        return trajectory

    def plot_results_(self, op_dt, horizon, op_trajectories, op_controls, ref_traj, four_states=False):
        timeline = np.arange(0, horizon, op_dt)
        self.predicted_trajectory = op_trajectories
        zst = ref_traj[:, 0]
        if self.use_loc_start:
            zst = self.loc_start

        trajectory = np.copy(zst)

        self.cal_distance(op_trajectories[:2, :], len(ref_traj.T))
        self.dt = op_dt
        print("Time resolution:{:.3f}s, total time:{:.3f}s".format(self.dt, self.dt * len(op_trajectories.T)))
        yaw_ = op_trajectories[2, :]
        # omega_ = np.diff(yaw_) / op_dt
        # omega_rate_ = np.append(0., omega_)

        v_ = op_controls[0, :]
        omega_ = op_controls[1, :]
        acc_ = op_controls[2, :]

        if four_states:
            omega_rate = np.diff(op_controls[1, :]) / op_dt
            omega_rate_ = np.append(0., omega_rate)

            jerk = np.diff(op_controls[2, :]) / op_dt
            jerk_ = np.append(0., jerk)
        else:
            omega_rate_ = op_controls[3, :]
            jerk_ = op_controls[4, :]

        self.plot_op_controls(v_, acc_, jerk_, yaw_, omega_, omega_rate_)

        trajectory = self.try_tracking(zst, op_controls, trajectory, ref_traj=ref_traj)

    def plot_results_differ(self, op_dt, op_trajectories, op_controls, ref_traj, four_states=False):

        self.predicted_trajectory = op_trajectories
        zst = ref_traj[:, 0]
        trajectory = np.copy(zst)

        self.cal_distance(op_trajectories[:2, :], len(ref_traj.T))
        self.dt = op_dt
        print("Time resolution:{:.3f}s, total time:{:.3f}s".format(self.dt, self.dt * len(op_trajectories.T)))
        yaw_ = op_trajectories[2, :]
        # omega_ = np.diff(yaw_) / op_dt
        # omega_rate_ = np.append(0., omega_)

        v_ = op_controls[0, :]
        omega_ = op_controls[1, :]
        #

        if four_states:
            omega_rate = np.diff(op_controls[1, :]) / op_dt
            omega_rate_ = np.append(0., omega_rate)
            acc_ = np.diff(v_) / op_dt
            acc_ = np.append(0., acc_)

            jerk = np.diff(acc_) / op_dt
            jerk_ = np.append(0., jerk)
        else:
            acc_ = op_controls[2, :]
            omega_rate_ = op_controls[3, :]
            jerk_ = op_controls[4, :]

        self.plot_op_controls(v_, acc_, jerk_, yaw_, omega_, omega_rate_)

        trajectory = self.try_tracking(zst, op_controls, trajectory, ref_traj=ref_traj)
        self.show_plot()

    def show_plot(self):
        print("Done")
        plt.show()

    def cal_distance(self, op_trajectories, iteration):
        diff_s = np.diff(np.array(op_trajectories), axis=1)
        sum_s = np.sum(np.hypot(diff_s[0, :], diff_s[1, :]))
        print("total number for iteration: ", iteration)
        print("total covered distance:{:.3f}m".format(sum_s))

    def initialize_saved_data(self):
        # loadtraj = np.load("../../data/saved_hybrid_a_star.npz")
        # ref_traj = loadtraj["saved_traj"]
        # loadmap = np.load("../../data/saved_obmap_obca.npz", allow_pickle=True)
        # ob1 = loadmap["pointmap"][0]
        # ob2 = loadmap["pointmap"][1]
        # ob3 = loadmap["pointmap"][2]
        # ob = [ob1, ob2, ob3]
        #
        # ob_constraint_mat = loadmap["constraint_mat"]
        # obst = []
        # obst.append(ob_constraint_mat[:4, :])
        # obst.append(ob_constraint_mat[4:8, :])
        # obst.append(ob_constraint_mat[8:12, :])

        loadtraj = np.load("../../data/saved_hybrid_a_star.npz")
        ref_traj = loadtraj["saved_traj"]
        loadmap = np.load("../../data/saved_obmap_obca.npz", allow_pickle=True)
        self.obmap.obst_pointmap = loadmap["pointmap"]
        return ref_traj, loadmap["pointmap"], loadmap["constraint_mat"]


if __name__ == '__main__':
    print(2)
