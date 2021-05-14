#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author:tongjue.chen@fau.de

import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt


class Optimize_Dist(object):
    def __init__(self):
        self.N_s = 3
        self.wg = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        self.with_reference = False
        self.model = None

    def init_model(self, ref_path, wg, obst):
        N_p = np.shape(ref_path)[1]
        m = pyo.ConcreteModel()
        m.sNum = pyo.RangeSet(0, N_p - 1)  # predict traj N_p - 1 times
        m.sNum1 = pyo.RangeSet(0, N_p - 2)  # predict control N_p - 2 times

        # Parameters
        m.wg = pyo.Param(pyo.RangeSet(0, 5), initialize={0: wg[0], 1: wg[1], 2: wg[2], 3: wg[3], 4: wg[4], 5: wg[5]},
                         mutable=True)  # weights for adjusting

        m.states0 = pyo.Param(pyo.RangeSet(0, self.N_s - 1),
                              initialize={0: ref_path[0, 0], 1: ref_path[1, 0], 2: ref_path[2, 0]},
                              mutable=True)  # state initial
        m.goal = pyo.Param(pyo.RangeSet(0, 2), initialize={0: ref_path[0, -1], 1: ref_path[1, -1], 2: ref_path[2, -1]},
                           mutable=True)  # goal

        m.ref_trajx = pyo.Param(m.sNum, rule=lambda m, k: ref_path[0, k])
        m.ref_trajy = pyo.Param(m.sNum, rule=lambda m, k: ref_path[1, k])
        m.ref_heading = pyo.Param(m.sNum, rule=lambda m, k: ref_path[2, k])  # theta constraint
        m.N_ob = pyo.RangeSet(0, len(obst.T) - 1)
        m.obstx = pyo.Param(m.N_ob, rule=lambda m, k: obst[0, k])
        m.obsty = pyo.Param(m.N_ob, rule=lambda m, k: obst[1, k])

        # Variables
        m.states = pyo.Var(pyo.RangeSet(0, self.N_s - 1), m.sNum, bounds=(-1e5, 1e5))  # states as var NsxNs
        m.dist = pyo.Var(m.N_ob, m.sNum, bounds=(0.16, 1e4))

        # states&  0: x, 1: y, 2: theta
        # dist

        # Constraints
        # states_0 = states0
        m.states0_update = pyo.Constraint(pyo.RangeSet(0, self.N_s - 1),
                                          rule=lambda m, i: m.states[i, 0] == m.states0[i])
        # goal = states end
        m.goal_update = pyo.Constraint(pyo.RangeSet(0, 2), rule=lambda m, k: m.goal[k] == m.states[k, N_p - 1])

        # discretized Nonlinear Differential equations
        # (x' - x) * sin(theta) = (y' - y) * cos(theta)
        m.theta_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        (m.states[0, k + 1] - m.states[0, k]) * pyo.sin(m.states[2, k]) == (
                m.states[1, k + 1] - m.states[1, k]) * pyo.cos(m.states[2, k])
        if k < N_p - 1 else pyo.Constraint.Skip)

        def cal_dist(m):
            l_obst = len(obst.T)
            for i in m.sNum:
                for j in range(l_obst):
                    dx = m.states[0, i] - m.obstx[j]
                    dy = m.states[1, i] - m.obsty[j]
                    return (m.dist[j, i] == pyo.log10(dx ** 2 + dy ** 2))

        m.dist_update = pyo.Constraint(rule=cal_dist)

        # Objective function
        m.refxobj = m.wg[2] * sum((m.states[0, k] - m.ref_trajx[k]) ** 2 for k in m.sNum)
        m.refyobj = m.wg[2] * sum((m.states[1, k] - m.ref_trajy[k]) ** 2 for k in m.sNum)
        m.reftobj = m.wg[2] * sum((m.states[2, k] - m.ref_heading[k]) ** 2 for k in m.sNum)

        m.vxobj = m.wg[3] * sum((m.states[0, k + 1] - m.states[0, k]) ** 2 for k in m.sNum1)  # wg * ||v - v_ref||^2
        m.vyobj = m.wg[3] * sum((m.states[1, k + 1] - m.states[1, k]) ** 2 for k in m.sNum1)  # wg * ||v - v_ref||^2
        m.omegaobj = m.wg[3] * sum((m.states[2, k + 1] - m.states[2, k]) ** 2 for k in m.sNum1)  # wg * ||v - v_ref||^2

        m.distanceobj = m.wg[4] * sum(
            (m.states[0, k + 1] - m.states[0, k]) ** 2 + (m.states[1, k + 1] - m.states[1, k]) ** 2 for k in m.sNum1)

        m.obstobj = 0
        for i in m.N_ob:
            for j in m.sNum:
                m.obstobj += m.wg[5] * m.dist[i, j] ** 2

        m.obj = pyo.Objective(
            expr=m.refxobj + m.refyobj + m.reftobj + m.vxobj + m.vyobj + m.omegaobj + m.distanceobj - m.obstobj,
            sense=pyo.minimize)

        self.model = m

    def Solve(self, state, goal):

        self.model.states0.reconstruct({0: state[0], 1: state[1], 2: state[2]})
        self.model.goal.reconstruct({0: goal[0], 1: goal[1], 2: goal[2]})

        pyo.SolverFactory('ipopt').solve(self.model)

        x_pred_vals = [self.model.states[0, k]() for k in self.model.sNum]
        y_pred_vals = [self.model.states[1, k]() for k in self.model.sNum]
        th_pred_vals = [self.model.states[2, k]() for k in self.model.sNum]

        dist_pred_vals = np.zeros((len(self.model.N_ob), len(self.model.sNum)))
        for i in self.model.N_ob:
            for j in self.model.sNum:
                dist_pred_vals[i, j] = self.model.dist[i, j]()

        return np.array([x_pred_vals, y_pred_vals, th_pred_vals]), dist_pred_vals


if __name__ == '__main__':
    loadtraj = np.load("saved_hybrid_a_star.npz")
    ref_traj = loadtraj["saved_traj"]
    ob = loadtraj["saved_ob"]
    opt = Optimize_Dist()
    opt.init_model(ref_traj, opt.wg, obst=ob)
    pre_traj, dist = opt.Solve(ref_traj[:, 0], ref_traj[:, -1])
    print(dist[0, :])

    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax1.plot(pre_traj[0, :], pre_traj[1, :], label="pred_traj")
    ax1.plot(ref_traj[0, :], ref_traj[1, :], label="ref_traj")
    ax1.plot(ob[0, :], ob[1, :], "o", color="black", label="obstacles")
    plt.axis("equal")
    ax1.grid()
    ax1.legend()

    ax1 = plt.subplot(122)
    ax1.plot(pre_traj[2, :], label="pred_theta")
    ax1.plot(ref_traj[2, :], label="ref_theta")
    ax1.grid()
    ax1.legend()
    plt.show()
