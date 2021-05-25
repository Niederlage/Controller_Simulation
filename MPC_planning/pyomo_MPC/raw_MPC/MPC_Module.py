import numpy as np
import pyomo.environ as pyo
import yaml


# from pyomo.dae import *


class MPC(object):
    def __init__(self, param, ref_traj=None):
        coeffs = [1, 0.00734257, 0.0538795, 0.080728]
        N_p = param["PredictionNum"]
        self.N_s = param["StateNum"]
        self.dt = param["dt"]
        self.Lf = param["Lf"]
        self.ref_v = param["ref_v"]
        wg = param["wg"]
        self.uj_bound = param["uj_bound"]
        self.uo_bound = np.array(param["uo_bound"]) * np.pi / 180

        goal = param["goal"]
        goal[2] *= np.pi / 180
        self.with_reference = False

        if ref_traj is None:
            self.model = self.init_model_default(N_p, coeffs, wg, goal)
        else:
            self.model = self.init_model_reference(ref_traj, wg, goal)
            self.with_reference = True

    def init_model_default(self, N_p, coeffs, wg, goal):

        m = pyo.ConcreteModel()
        m.sNum = pyo.RangeSet(0, N_p - 1)  # predict traj N_p - 1 times
        m.uNum = pyo.RangeSet(0, N_p - 2)  # predict control N_p - 2 times
        m.uNum1 = pyo.RangeSet(0, N_p - 3)  # predict control

        # Parameters
        m.wg = pyo.Param(pyo.RangeSet(0, 5), initialize={0: wg[0], 1: wg[1], 2: wg[2], 3: wg[3], 4: wg[4], 5: wg[5]},
                         mutable=True)  # weights for adjusting
        m.dt = pyo.Param(initialize=self.dt, mutable=True)  # integration time
        m.Lf = pyo.Param(initialize=self.Lf, mutable=True)  # car length between wheels axes
        m.ref_v = pyo.Param(initialize=self.ref_v, mutable=True)  # reference speed
        m.ref_cte = pyo.Param(initialize=0.0, mutable=True)  # reference lat error
        m.ref_epsi = pyo.Param(initialize=0.0, mutable=True)  # reference heading error
        m.states0 = pyo.Param(pyo.RangeSet(0, self.N_s - 1),
                              initialize={0: 0., 1: 0., 2: 0., 3: 0., 4: 0., 5: 0., 6: 0.},
                              mutable=True)  # state initial
        m.coeffs = pyo.Param(pyo.RangeSet(0, 3),
                             initialize={0: coeffs[0], 1: coeffs[1], 2: coeffs[2], 3: coeffs[3]},
                             mutable=True)  # coeffs for spineline
        m.goal = pyo.Param(pyo.RangeSet(0, 4),
                           initialize={0: goal[0], 1: goal[1], 2: goal[2], 3: goal[3], 4: goal[4]})  # goal

        # Variables
        m.states = pyo.Var(pyo.RangeSet(0, self.N_s - 1), m.sNum)  # states as var NsxNs

        m.ref_traj = pyo.Var(m.sNum)  # reference traj

        m.ref_heading = pyo.Var(m.sNum)  # theta constraint
        m.ua = pyo.Var(m.uNum, bounds=(self.uj_bound[0], self.uj_bound[1]))  # control acc as var
        m.uo = pyo.Var(m.uNum,
                       bounds=(self.uo_bound[0], self.uo_bound[1]))  # control steering as var -0.436332, 0.436332

        # states&  0: x, 1: y, 2: theta, 3: v, 4: psi, 5: cte, 6: epsi
        # u&  0: a, 1: omega
        # Constraints
        # states_0 = states0
        m.states0_update = pyo.Constraint(pyo.RangeSet(0, self.N_s - 1),
                                          rule=lambda m, i: m.states[i, 0] == m.states0[i])

        # f(x) = c0 + c1 * x + c2 * x^2 + c3 * x^3
        m.ref_traj_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.ref_traj[k] == m.coeffs[0] + m.coeffs[1] * m.states[0, k] + m.coeffs[2] * m.states[0, k] ** 2 + m.coeffs[3] *
        m.states[0, k] * 3)

        # ref_heading = arctan(dy/dx) = arctan(f'(x))
        m.ref_heading_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.ref_heading[k] == pyo.atan(
            3 * m.coeffs[3] * m.states[0, k] ** 2 + 2 * m.coeffs[2] * m.states[0, k] + m.coeffs[1]))

        # goal = states end
        m.goal_update = pyo.Constraint(pyo.RangeSet(0, 4), rule=lambda m, k: m.goal[k] == m.states[k, N_p - 1])

        # discretized Nonlinear Differential equations
        # x' = x + v * cos(theta) * dt
        m.x_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[0, k + 1] == m.states[0, k] + m.states[3, k] * pyo.cos(m.states[2, k]) * m.dt
        if k < N_p - 1 else pyo.Constraint.Skip)

        # y' = y + v * sin(theta) * dt
        m.y_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[1, k + 1] == m.states[1, k] + m.states[3, k] * pyo.sin(
            m.states[2, k]) * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # th' = th + v * tan(psi) / Lf * dt
        m.theta_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[2, k + 1] == m.states[2, k] + m.states[3, k] * pyo.tan(
            m.states[4, k]) / m.Lf * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # v' = v + a * dt
        m.v_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[3, k + 1] == m.states[3, k] + m.ua[k] * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # psi' = psi + omega * dt
        m.omega_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[4, k + 1] == m.states[4, k] + m.uo[k] * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # cte = f(x) - y - v * sin(theta) * dt
        m.cte_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[5, k + 1] == (m.ref_traj[k] - m.states[1, k] - m.states[3, k] * pyo.sin(
            m.states[2, k]) * m.dt) if k < N_p - 1 else pyo.Constraint.Skip)

        # epsi = thdes - th - v * tan(psi) / Lf * dt
        m.epsi_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[6, k + 1] == m.ref_heading[k] - m.states[2, k] - m.states[3, k] * pyo.tan(
            m.states[4, k]) / m.Lf * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # Objective function
        m.vobj = m.wg[0] * sum((m.states[3, k] - m.ref_v) ** 2 for k in m.sNum)  # wg * ||v - v_ref||^2
        m.uoobj = m.wg[1] * sum(m.uo[k] ** 2 for k in m.uNum)  # wg * ||uo||^2
        m.uaobj = m.wg[1] * sum(m.ua[k] ** 2 for k in m.uNum)  # wg * ||ua||^2

        m.suoobj = m.wg[3] * sum((m.uo[k + 1] - m.uo[k]) ** 2 for k in m.uNum1)  # sum: wg * ||uo_next - uo||^2
        m.suaobj = m.wg[3] * sum((m.ua[k + 1] - m.ua[k]) ** 2 for k in m.uNum1)  # sum: wg * ||a_next - a||^2

        m.cteobj = m.wg[0] * sum((m.states[5, k] - m.ref_cte) ** 2 for k in m.sNum)  # sum: wg * ||cte - cte_ref||^2
        m.epsiobj = m.wg[0] * sum((m.states[6, k] - m.ref_epsi) ** 2 for k in m.sNum)  # sum: wg * ||epsi - epsi_ref||^2

        m.timeobj = m.wg[4] * m.dt * N_p

        # m.goalobj = m.wg[4] * sum(
        #     (m.goal[k] - m.states[k, N_p - 1]) ** 2 for k in range(5))  # wg * ||traj_end - goal||^2

        m.obj = pyo.Objective(
            expr=m.cteobj + m.epsiobj + m.vobj + m.uoobj + m.uaobj + m.suoobj + m.suaobj + m.timeobj,
            sense=pyo.minimize)

        return m  # .create_instance()

    def init_model_reference(self, ref_traj, wg, goal):
        N_p = np.shape(ref_traj)[1]
        self.N_s = 9
        # goal = [ref_traj[0, -1], ref_traj[1, -1], ref_traj[2, -1]]

        m = pyo.ConcreteModel()
        m.sNum = pyo.RangeSet(0, N_p - 1)  # predict traj N_p - 1 times
        m.uNum = pyo.RangeSet(0, N_p - 2)  # predict control N_p - 2 times
        m.uNum1 = pyo.RangeSet(0, N_p - 3)  # predict control

        # constant Parameters
        m.wg = pyo.Param(pyo.RangeSet(0, 5), initialize={0: wg[0], 1: wg[1], 2: wg[2], 3: wg[3], 4: wg[4], 5: wg[5]},
                         mutable=True)  # weights for adjusting
        m.Lf = pyo.Param(initialize=self.Lf, mutable=True)  # car length between wheels axes
        m.ref_v = pyo.Param(initialize=self.ref_v, mutable=True)  # reference speed
        m.ref_cte = pyo.Param(initialize=0.0, mutable=True)  # reference lat error
        m.ref_epsi = pyo.Param(initialize=0.0, mutable=True)  # reference heading error
        m.states0 = pyo.Param(pyo.RangeSet(0, self.N_s - 1),
                              initialize={0: ref_traj[0, 0], 1: ref_traj[1, 0], 2: ref_traj[2, 0], 3: 0., 4: 0., 5: 0.,
                                          6: 0., 7: 0., 8: 0.},
                              mutable=True)  # state initial
        m.goal = pyo.Param(pyo.RangeSet(0, 5),
                           initialize={0: goal[0], 1: goal[1], 2: goal[2], 3: 0., 4: 0., 5: 0.})  # goal

        m.ref_trajx = pyo.Param(m.sNum, rule=lambda m, k: ref_traj[0, k])
        m.ref_trajy = pyo.Param(m.sNum, rule=lambda m, k: ref_traj[1, k])
        m.ref_heading = pyo.Param(m.sNum, rule=lambda m, k: ref_traj[2, k])  # theta constraint

        # Variables for variation
        m.states = pyo.Var(pyo.RangeSet(0, self.N_s - 1), m.sNum)  # states as var NsxNs
        m.uj = pyo.Var(m.uNum, bounds=(self.uj_bound[0], self.uj_bound[1]))  # control jerk as var
        m.uo = pyo.Var(m.uNum, bounds=(self.uo_bound[0], self.uo_bound[1]))  # control steering as var
        m.dt = pyo.Var(initialize=0.1, bounds=(0.01, 1))
        # m.ref_traj = pyo.Var(pyo.RangeSet(0, 1), m.sNum)

        # states&  0: x, 1: y, 2: theta, 3: v, 4: psi, 5:acc,  6:ctex, 7: ctey, 8.epsi
        # u&  0: jerk, 1: omega

        # Constraints
        # states_0 = states0
        m.states0_update = pyo.Constraint(pyo.RangeSet(0, self.N_s - 1),
                                          rule=lambda m, i: m.states[i, 0] == m.states0[i])
        # goal = states end
        m.goal_update = pyo.Constraint(pyo.RangeSet(0, 5), rule=lambda m, i: m.goal[i] == m.states[i, N_p - 1])

        ############### discretized Nonlinear Differential equations as constraints ######################
        # x' = x + v * cos(theta) * dt
        m.x_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[0, k + 1] == m.states[0, k] + m.states[3, k] * pyo.cos(m.states[2, k]) * m.dt
        if k < N_p - 1 else pyo.Constraint.Skip)

        # y' = y + v * sin(theta) * dt
        m.y_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[1, k + 1] == m.states[1, k] + m.states[3, k] * pyo.sin(
            m.states[2, k]) * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # th' = th + v * tan(psi) / Lf * dt
        m.theta_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[2, k + 1] == m.states[2, k] + m.states[3, k] * pyo.tan(
            m.states[4, k]) / m.Lf * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # v' = v + a * dt
        m.v_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[3, k + 1] == m.states[3, k] + m.states[5, k] * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # psi' = psi + omega * dt
        m.omega_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[4, k + 1] == m.states[4, k] + m.uo[k] * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # a' = a + jerk * dt
        m.a_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[5, k + 1] == m.states[5, k] + m.uj[k] * m.dt if k < N_p - 1 else pyo.Constraint.Skip)

        # ctex = x_ref - x - v * cos(theta) * dt
        m.ctex_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[6, k + 1] == (m.ref_trajx[k] - m.states[0, k] - m.states[3, k] * pyo.cos(
            m.states[2, k]) * m.dt) if k < N_p - 1 else pyo.Constraint.Skip)

        # ctey = y_ref - y - v * sin(theta) * dt
        m.ctey_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[7, k + 1] == (m.ref_trajy[k] - m.states[1, k] - m.states[3, k] * pyo.sin(
            m.states[2, k]) * m.dt) if k < N_p - 1 else pyo.Constraint.Skip)

        # epsi = heading - th - v * tan(psi) / Lf * dt
        m.epsi_update = pyo.Constraint(m.sNum, rule=lambda m, k:
        m.states[8, k + 1] == (m.ref_heading[k] - m.states[2, k] - m.states[3, k] * pyo.tan(
            m.states[4, k]) / m.Lf * m.dt) if k < N_p - 1 else pyo.Constraint.Skip)

        # Objective function
        m.vobj = m.wg[2] * sum((m.states[3, k] - m.ref_v) ** 2 for k in m.sNum)  # wg * ||v - v_ref||^2
        m.uoobj = m.wg[1] * sum(m.uo[k] ** 2 for k in m.uNum)  # wg * ||uo||^2
        m.ujobj = m.wg[1] * sum(m.uj[k] ** 2 for k in m.uNum)  # wg * ||uj||^2

        m.suoobj = m.wg[3] * sum((m.uo[k + 1] - m.uo[k]) ** 2 for k in m.uNum1)  # sum: wg * ||uo_next - uo||^2
        m.sujobj = m.wg[3] * sum((m.uj[k + 1] - m.uj[k]) ** 2 for k in m.uNum1)  # sum: wg * ||j_next - j||^2

        m.ctexobj = m.wg[3] * sum((m.states[6, k] - m.ref_cte) ** 2 for k in m.sNum)  # sum: wg * ||ctex - cte_ref||^2
        m.cteyobj = m.wg[3] * sum((m.states[7, k] - m.ref_cte) ** 2 for k in m.sNum)  # sum: wg * ||ctey - cte_ref||^2
        m.epsiobj = m.wg[3] * sum((m.states[8, k] - m.ref_epsi) ** 2 for k in m.sNum)  # sum: wg * ||epsi - epsi_ref||^2

        m.timeobj = m.wg[3] * sum(m.dt ** 2 for k in m.sNum)
        m.distanceobj = m.wg[4] * sum(
            (m.states[0, k + 1] - m.states[0, k]) ** 2 + (m.states[1, k + 1] - m.states[1, k]) ** 2 for k in m.uNum1)

        m.obj = pyo.Objective(
            expr=m.ctexobj + m.cteyobj + m.epsiobj + m.vobj + m.uoobj + m.ujobj + m.suoobj + m.timeobj + m.distanceobj,
            sense=pyo.minimize)

        return m  # .create_instance()

    def Solve(self, state, coeffs=None):

        self.model.states0_update.reconstruct()
        self.model.goal_update.reconstruct()

        if not self.with_reference:
            self.model.states0.reconstruct(
                {0: state[0], 1: state[1], 2: state[2], 3: state[3], 4: state[4], 5: state[5]})
            self.model.coeffs.reconstruct({0: coeffs[0], 1: coeffs[1], 2: coeffs[2], 3: coeffs[3]})
            self.model.ref_traj_update.reconstruct()
            self.model.ref_heading_update.reconstruct()
        else:
            self.model.states0.reconstruct(
                {0: state[0], 1: state[1], 2: state[2], 3: state[3], 4: state[4], 5: state[5], 6: state[6],
                 7: state[7], 8: state[8]})

        pyo.SolverFactory('ipopt').solve(self.model)

        x_pred_vals = [self.model.states[0, k]() for k in self.model.sNum]
        y_pred_vals = [self.model.states[1, k]() for k in self.model.sNum]
        th_pred_vals = [self.model.states[2, k]() for k in self.model.sNum]
        v_forward = [self.model.states[3, k]() for k in self.model.sNum]
        steering_angle = [self.model.states[4, k]() for k in self.model.sNum]
        acc = [self.model.states[5, k]() for k in self.model.sNum]
        omega = [self.model.uo[k]() for k in self.model.uNum]
        dt = self.model.dt()

        return [x_pred_vals, y_pred_vals, th_pred_vals, acc, omega, dt], np.array([v_forward, steering_angle])


if __name__ == '__main__':
    with open("config.yaml", 'r', encoding='utf-8') as f:
        param = yaml.load(f)
    mpc = MPC(param)
    zst = np.array([0., 0., 0., 0., 0., 0.])
    coefficients = np.array([1, 1, 1, 1])
    pre_traj, u_op = mpc.Solve(zst, coefficients)
    print(1)
