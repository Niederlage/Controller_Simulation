#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de

import numpy as np
import matplotlib.pyplot as plt
from gears.cal_angle_from_track import CalAngleFromTracks


def continuum_yaw(yaw):
    dyaw = np.diff(yaw)
    stetig_yaw = [yaw[0]]
    sum_theta = yaw[0]
    for dtheta in dyaw:
        if abs(dtheta) < np.pi:
            sum_theta += dtheta
            stetig_yaw.append(sum_theta)
        else:
            sum_theta += (dtheta + np.pi) % (2 * np.pi) - np.pi
            stetig_yaw.append(sum_theta)

    return np.array(stetig_yaw)


def mod_theta(theta, t_last):
    if abs(theta - t_last) > np.pi / 2:
        if np.sign(theta) > 0:
            return theta - np.pi
        else:
            return theta + np.pi

    return (theta + np.pi) % (2 * np.pi) - np.pi


def interpolate_track(tracks, theta, res):
    path = []
    dtracks = tracks[:, -1] - tracks[:, 0]
    dtheta = theta[-1] - theta[0]

    l_dist = np.hypot(dtracks[0], dtracks[1])
    l_path_size = int(np.ceil(l_dist / res))
    th_list = []
    for i in range(l_path_size):
        if (i == 0):
            point = tracks[:, 0]
            th = theta[0]
        elif i == l_path_size:
            point = tracks[:, -1]
            th = theta[-1]
        else:
            point = tracks[:, 0] + i * dtracks / l_path_size
            th = theta[0] + i * dtheta / l_path_size

        path.append(point)
        th_list.append(th)

    # yaw = continuum_yaw(th_list)
    path = np.array(path).T

    return np.vstack((path, th_list))


def continuous_interpolate(tracks, theta, res):
    l_tracks = len(tracks.T)
    sum_path = np.zeros((3, 1))
    for i in range(l_tracks):
        if i + 2 <= l_tracks:
            temp_path = interpolate_track(tracks[:, i:i + 2], theta[i:i + 2], res)
        else:
            temp_path = np.hstack((tracks[:, -1], theta[-1]))[:, None]

        sum_path = np.hstack((sum_path, temp_path))

    return sum_path


if __name__ == '__main__':

    try_test = False
    resolution = 0.1
    test_track = np.array([[0, 1, 1.5, 1., 0.],
                           [0, 1, 1.1, 1.9, 1.5]])

    test_traj = np.load("../data/saved_hybrid_a_star.npz", allow_pickle=True)
    trackx = test_traj["saved_traj"][0]
    tracky = test_traj["saved_traj"][1]
    trackth = np.array(test_traj["saved_traj"][2])
    start = np.array([trackx[0], tracky[1], trackth[2]])
    tracks = np.vstack((trackx, tracky))

    ca = CalAngleFromTracks()

    if try_test:
        run_track = test_track
    else:
        run_track = tracks

    cal_vec, theta = ca.generate_orientation(run_track, np.zeros((3,)))
    cal_path = continuous_interpolate(run_track, theta, resolution)
    print("original track size:{}, size after interpolation:{}".format(len(run_track.T), len(cal_path.T)))
    cal_path[2, :] = continuum_yaw(cal_path[2, :])

    thetaline = np.linspace(0, 100, len(theta))
    cal_thetaline = np.linspace(0, 100, len(cal_path.T))
    fig = plt.figure()
    ax = plt.subplot(311)
    ax.plot(cal_path[0, :], cal_path[1, :], ".", label="interpolated tracks")
    ax.plot(run_track[0, :], run_track[1, :], ".", label="original tracks")
    ax.legend()

    ax = plt.subplot(312)
    ax.plot(theta, ".", label="original angle")
    ax.legend()
    ax = plt.subplot(313)
    ax.plot(cal_path[2, :], ".", label="interpolated angle")
    ax.legend()
    plt.show()
    print(1)
