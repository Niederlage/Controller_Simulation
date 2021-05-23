#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de

import matplotlib.pyplot as plt
import numpy as np


class UTurnMPC():
    def __init__(self):
        self.L = 1.0
        self.dt = 1e-1
        self.robot_length = 0.5
        self.robot_width = 0.3
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

        outline = np.array([[-self.robot_length / 2, self.robot_length / 2,
                             (self.robot_length / 2), -self.robot_length / 2,
                             -self.robot_length / 2],
                            [self.robot_width / 2, self.robot_width / 2,
                             - self.robot_width / 2, -self.robot_width / 2,
                             self.robot_width / 2]])
        Rot1 = np.array([[np.cos(yaw), np.sin(yaw)],
                         [-np.sin(yaw), np.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T
        outline[0, :] += x
        outline[1, :] += y
        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")

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
                    plt.plot(ref_traj[0, :], ref_traj[1, :], "-", color="orange", label="warm start reference")
                plt.plot(zst[0], zst[1], "xr")
                self.plot_robot(zst[0], zst[1], zst[2])
                self.plot_arrow(zst[0], zst[1], zst[2])
                plt.plot(self.predicted_trajectory[0, :], self.predicted_trajectory[1, :], "-g", label="MPC prediciton")
                plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
                if obst is not None:
                    plt.plot(obst[0], obst[1], "o", color="black", label="obstacles")
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


if __name__ == '__main__':
    print(2)
