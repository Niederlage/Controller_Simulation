#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')
import numpy as np
from gears.polygon_generator import cal_coeff_mat
from gears.Controller import Controller


class UTurnMPC():
    def __init__(self):
        self.L = 0.4
        self.dt = 1e-1
        self.W = 0.6  # width of car
        self.LF = 0.6  # distance from rear to vehicle front end
        self.LB = 0.2  # distance from rear to vehicle back end
        self.loc_start = np.array([-2.5, -1., np.deg2rad(70)])
        self.predicted_trajectory = None
        self.optimal_dt = None
        self.show_animation = True
        self.reserve_footprint = False
        self.plot_arrows = False
        self.use_differ_motion = False
        self.use_controller = True
        self.ctrl = Controller()
        self.u_regelung = None
        # self.Km = self.ctrl.cal_mat_K(self.ctrl.x_s)

    def set_parameters(self, param):
        self.L = param["base"]
        self.LF = param["LF"]  # distance from rear to vehicle front end
        self.LB = param["LB"]  # distance from rear to vehicle back end
        self.W = param["W"]

    def ackermann_motion_model(self, zst, u_in, dt, Runge_Kutta=True):
        v_ = u_in[0]
        steer_ = u_in[1]
        x_ = zst[0]
        y_ = zst[1]
        yaw_ = zst[2]

        if not Runge_Kutta:
            x_ += v_ * np.cos(yaw_) * dt
            y_ += v_ * np.sin(yaw_) * dt
            yaw_ += v_ * np.tan(steer_) / self.L * dt
            return np.array([x_, y_, yaw_])
        else:
            a_ = u_in[2]
            steer_rate_ = u_in[3]
            # jerk_ = u_in[4]

            k1_dx = v_ * np.cos(yaw_)
            k1_dy = v_ * np.sin(yaw_)
            k1_dyaw = v_ / self.L * np.tan(steer_)
            k1_dv = a_
            k1_dsteer = steer_rate_
            # k1_da = jerk_

            k2_dx = (v_ + 0.5 * dt * k1_dv) * np.cos(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dy = (v_ + 0.5 * dt * k1_dv) * np.sin(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dyaw = (v_ + 0.5 * dt * k1_dv) / self.L * np.tan(steer_ + 0.5 * dt * k1_dsteer)
            k2_dv = a_  # + 0.5 * dt * k1_da
            k2_dsteer = steer_rate_
            # k2_da = jerk_

            k3_dx = (v_ + 0.5 * dt * k2_dv) * np.cos(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dy = (v_ + 0.5 * dt * k2_dv) * np.sin(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dyaw = (v_ + 0.5 * dt * k2_dv) / self.L * np.tan(steer_ + 0.5 * dt * k2_dsteer)
            k3_dv = a_  # + 0.5 * dt * k2_da
            k3_dsteer = steer_rate_

            k4_dx = (v_ + 0.5 * dt * k3_dv) * np.cos(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dy = (v_ + 0.5 * dt * k3_dv) * np.sin(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dyaw = (v_ + 0.5 * dt * k3_dv) / self.L * np.tan(steer_ + 0.5 * dt * k3_dsteer)

            x_ += dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            y_ += dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            yaw_ += dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6

            return np.array([x_, y_, yaw_])

    def differ_motion_model(self, zst, u_in, dt, Runge_Kutta=True):
        v_ = u_in[0]
        omega_ = u_in[1]
        x_ = zst[0]
        y_ = zst[1]
        yaw_ = zst[2]

        x_ += v_ * np.cos(yaw_) * dt
        y_ += v_ * np.sin(yaw_) * dt
        yaw_ += omega_ * dt

        return np.array([x_, y_, yaw_])

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

    def plot_regelung(self, k):
        if k == 0:
            fig = plt.figure()
        plt.cla()
        ax = plt.subplot(211)
        ax.plot(self.u_regelung[0, :k], label="delta_a", color="red")
        ax.grid()
        ax.set_title("delta_a")
        # ax.legend()

        ax = plt.subplot(212)
        ax.plot(self.u_regelung[1, :k] * 180 / np.pi, label="delta_steer")
        ax.set_title("delta_steer")
        ax.grid()

        plt.pause(0.001)

    def lqr_regler(self, zst, u_in, k):
        if k == 0:
            state = np.block([zst, u_in[:2, k]])
        else:
            state = np.block([zst, u_in[:2, k]])

        ref_state = np.block([self.predicted_trajectory[:3, k], u_in[:2, k]])
        u_regelung = -self.ctrl.cal_mat_K(self.ctrl.x_s, self.dt) @ (state - ref_state)

        if abs(u_regelung[0]) > 4:
            u_regelung[0] = np.sign(u_regelung[0]) * 4
        if abs(u_regelung[1]) > 40 * np.pi / 180:
            u_regelung[1] = np.sign(u_regelung[1]) * 40 * np.pi / 180
        self.u_regelung[:, k] = u_regelung
        return u_regelung

    def try_tracking(self, zst, u_op, trajectory, obst=None, ref_traj=None):
        k = 0
        f = plt.figure()
        ax = plt.subplot()
        while True:
            u_in = u_op[:, k]
            if self.use_controller:
                u_regelung = self.lqr_regler(zst, u_op, k)
                u_in[2:4] += u_regelung

            if self.use_differ_motion:
                zst = self.differ_motion_model(zst, u_in, self.dt)  # simulate robot
            else:
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
                self.plot_robot(zst[0], zst[1], zst[2])

                if not self.reserve_footprint:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, fontsize=10, loc="upper left")

                plt.axis("equal")
                plt.grid(True)
                if k % 2 == 0:
                    plt.pause(0.001)
                self.plot_regelung(k)
                k += 1

            if k >= u_op.shape[1]:
                print("end point arrived...")
                break

        return trajectory

    def plot_results(self, op_dt, op_trajectories, op_controls, ref_traj, ob, four_states=False):

        self.predicted_trajectory = op_trajectories
        # zst = ref_traj[:, 0]
        zst = self.loc_start
        trajectory = np.copy(zst)

        self.cal_distance(op_trajectories[:2, :], len(ref_traj.T))
        self.dt = op_dt
        print("Time resolution:{:.3f}s, total time:{:.3f}s".format(self.dt, self.dt * len(ref_traj.T)))
        if self.use_controller:
            self.u_regelung = np.zeros((2, len(ref_traj.T)))
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
                self.plot_robot(op_trajectories[0, i], op_trajectories[1, i], op_trajectories[2, i])

                if not self.reserve_footprint:
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, fontsize=10, loc="upper left")

                plt.axis("equal")
                plt.grid(True)
                if i % 5 == 0:
                    plt.pause(0.0001)
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
