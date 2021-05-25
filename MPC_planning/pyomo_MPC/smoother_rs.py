#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
from MPC_Module_Family import MPC_Reference
from MPC_with_RS.casadi_MPC.reeds_shepp_path_planning import ReedsSheppPathPlanning
import yaml


class UTurnMPC():
    def __init__(self):
        self.L = 1.0
        self.dt = 1e-1
        self.robot_length = 0.5
        self.robot_width = 0.3
        self.optimal_dt = None

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

    def fitpolynorm(self, xvals, yvals, order):
        l_x = len(xvals)
        assert (l_x == len(yvals))
        A_mat = np.zeros((l_x, order + 1))
        A_mat[:, 0] = 1.0
        for i in range(l_x):
            for j in range(order):
                A_mat[i, j + 1] = A_mat[i, j] * xvals[i]

        return scipy.linalg.solve(A_mat, yvals)

    def plot_arrow(self, x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
        plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
                  head_length=width, head_width=width)
        plt.plot(x, y)

    def plot_robot(self, x, y, yaw, config):  # pragma: no cover

        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             (config.robot_length / 2), -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             - config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])
        Rot1 = np.array([[np.cos(yaw), np.sin(yaw)],
                         [-np.sin(yaw), np.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")

    def try_tracking(self, zst, u_op, trajectory, ref_traj=None, optimal_time=False):
        k = 0
        f = plt.figure()
        while True:

            zst[:3] = ut.motion_model(zst[:3], u_op[:, k], ut.dt)  # simulate robot

            e_lon = zst[0] - predicted_trajectory[0][1]
            e_lat = zst[1] - predicted_trajectory[1][1]
            zst[-2:] = np.array([e_lon, e_lat])
            trajectory = np.vstack((trajectory, zst))  # store state history

            if show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

                plt.plot(zst[0], zst[1], "xr")
                self.plot_robot(zst[0], zst[1], zst[2], ut)
                self.plot_arrow(zst[0], zst[1], zst[2])
                plt.plot(predicted_trajectory[0], predicted_trajectory[1], "-g", label="MPC prediciton")
                if ref_traj is not None:
                    plt.plot(ref_traj[0], ref_traj[1], "-", color="orange", label="RS reference")
                plt.plot(predicted_trajectory[0][-1], predicted_trajectory[1][-1], "x", color="purple")
                plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
                plt.legend()
                if plot_goal:
                    plt.plot(goal[0], goal[1], "xb")

                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.01)
                k += 1

            if k >= u_op.shape[1]:
                print("end point arrived...")
                break

        return trajectory


if __name__ == '__main__':
    show_animation = True
    tracking = True
    plot_goal = True
    optimal_time = True
    curvature = 2.
    step_size = 0.1

    ut = UTurnMPC()
    rs = ReedsSheppPathPlanning()

    with open("config.yaml", 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    goal = param["goal"]
    start = param["start"]
    px, py, pyaw, mode, clen = rs.reeds_shepp_path_planning(
        start[0], start[1], start[2] * np.pi / 180, goal[0], goal[1], goal[2] * np.pi / 180, 0.5, step_size=0.1)
    ref_traj = np.array([px, py, pyaw])
    np.savez("../casadi_MPC/saved_traj.npz", saved_traj=ref_traj)

    mpc = MPC_Reference(param, ref_traj=ref_traj)
    start[2] *= np.pi / 180
    zst = np.hstack((start, np.array([0., 0., 0., 0.])))
    coefficients = np.array([1, 1, 1, 1])
    trajectory = np.copy(zst)

    predicted_trajectory, u_op = mpc.Solve(zst, goal=goal, coeffs=coefficients)
    diff_s = np.diff(np.array(predicted_trajectory[:2]), axis=0)
    print("total number for iteration: ", len(predicted_trajectory[0]))
    print("total covered distance:{:.3f}m".format(np.linalg.norm(diff_s)))
    np.savez("saved_traj", traj=predicted_trajectory)
    if optimal_time:
        ut.dt = predicted_trajectory[-1]
    print("Time resolution:{:.3f}s, total time:{:.3f}s".format(ut.dt, ut.dt * len(u_op.T)))

    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.plot(u_op[0, :], label="v")
    ax1.plot(u_op[1, :], label="steering angle")
    ax1.plot(predicted_trajectory[3], "-.", label="acc")
    ax1.plot(predicted_trajectory[4], "-.", label="omega")
    ax1.grid()
    ax1.legend()

    if tracking:
        trajectory = ut.try_tracking(zst, u_op, trajectory, ref_traj=ref_traj, optimal_time=True)
        print("Done")

    else:
        f2 = plt.Figure()
        ax = f2.gca()
        ax.plot(predicted_trajectory[:, 3], label="v")

    plt.show()