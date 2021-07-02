import casadi as ca
import numpy as np
import time
from motion_plot.ackermann_motion_plot import UTurnMPC
from gears.cubic_spline_planner import calc_spline_course
import yaml


class CasADi_MPC_OBCA_reduced:
    def __init__(self):
        self.base = 2.0
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2
        self.x_max = 14
        self.y_max = 15
        self.nx = 6
        self.ng = 4
        self.v_max = 1.3
        self.steer_max = np.pi * 50 / 180
        self.a_max = 4.

        self.model = None
        self.x_opt = None
        self.obst_num = 0
        self.horizon = 0
        self.dt = 0.1
        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        self.dmin = 0.

    def set_parameters(self, param):
        self.base = param["base"]
        self.LF = param["LF"]  # distance from rear to vehicle front end
        self.LB = param["LB"]  # distance from rear to vehicle back end

    def Array2SX(self, array):
        rows, cols = np.shape(array)
        sx = ca.SX.zeros(rows * cols, 1)
        array_ = array.T.flatten()
        for i in range(rows * cols):
            sx[i] = array_[i]

        return ca.reshape(sx, rows, cols)

    def init_motion_constraints(self, x, dt):
        gx = ca.SX.sym("g1", self.ng, self.horizon - 1)
        for i in range(self.horizon - 1):
            x_ = x[0, i]
            y_ = x[1, i]
            yaw_ = x[2, i]
            v_ = x[3, i]
            steer_ = x[4, i]
            a_ = x[5, i]

            k1_dx = v_ * ca.cos(yaw_)
            k1_dy = v_ * ca.sin(yaw_)
            k1_dyaw = v_ / self.base * ca.tan(steer_)
            k1_dv = a_

            k2_dx = (v_ + 0.5 * dt * k1_dv) * ca.cos(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dy = (v_ + 0.5 * dt * k1_dv) * ca.sin(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dyaw = (v_ + 0.5 * dt * k1_dv) / self.base * ca.tan(steer_)
            k2_dv = a_

            k3_dx = (v_ + 0.5 * dt * k2_dv) * ca.cos(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dy = (v_ + 0.5 * dt * k2_dv) * ca.sin(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dyaw = (v_ + 0.5 * dt * k2_dv) / self.base * ca.tan(steer_)
            k3_dv = a_

            k4_dx = (v_ + 0.5 * dt * k3_dv) * ca.cos(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dy = (v_ + 0.5 * dt * k3_dv) * ca.sin(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dyaw = (v_ + 0.5 * dt * k3_dv) / self.base * ca.tan(steer_)
            k4_dv = a_

            dx = dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            dy = dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            dyaw = dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
            dv = dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6

            gx[0, i] = x_ + dx - x[0, i + 1]
            gx[1, i] = y_ + dy - x[1, i + 1]
            gx[2, i] = yaw_ + dyaw - x[2, i + 1]
            gx[3, i] = v_ + dv - x[3, i + 1]

        return gx

    def init_OBCA_constraints(self, x_, lambda_, mu_, shape, obst):
        gx = ca.SX.sym("g2", self.obst_num * 4, self.horizon - 1)
        gT = self.Array2SX(shape[:, 2][:, None].T)
        GT = self.Array2SX(shape[:, :2].T)

        # G = ca.SX.zeros(4, 2)
        # G[0, 0] = 1
        # G[1, 0] = -1
        # G[2, 1] = 1
        # G[3, 1] = -1
        #
        # g = ca.SX.zeros(4, 1)
        # g[0, 0] = 3
        # g[1, 0] = 1
        # g[2, 0] = 1
        # g[3, 0] = 1
        # GT = G.T
        # gT = g.T

        for i in range(self.horizon - 1):
            mu_i = mu_[:, i]
            yaw_i = x_[2, i]
            offset = ca.SX.zeros(2, 1)
            offset[0, 0] = 1 * ca.cos(yaw_i)
            offset[1, 0] = 1 * ca.sin(yaw_i)
            t_i = x_[:2, i] + offset
            rotT_i = ca.SX.zeros(2, 2)
            rotT_i[0, 0] = ca.cos(yaw_i)
            rotT_i[1, 1] = ca.cos(yaw_i)
            rotT_i[1, 0] = -ca.sin(yaw_i)
            rotT_i[0, 1] = ca.sin(yaw_i)

            for j in range(self.obst_num):
                Aj = self.Array2SX(obst[j][:, :2])
                bj = self.Array2SX(obst[j][:, 2][:, None])
                lambdaj = lambda_[(4 * j):(4 * (j + 1)), i]

                constraint1 = ca.mtimes(ca.mtimes(rotT_i, Aj.T), lambdaj) + ca.mtimes(GT, mu_i)
                constraint2 = ca.mtimes((ca.mtimes(Aj, t_i) - bj).T, lambdaj) - ca.mtimes(gT, mu_i)
                constraint3 = ca.sumsqr(ca.mtimes(Aj.T, lambdaj))
                constraint10 = constraint1[0, 0]
                constraint11 = constraint1[1, 0]

                gx[j, i] = ca.fabs(constraint10)
                gx[j + self.obst_num, i] = ca.fabs(constraint11)
                gx[j + 2 * self.obst_num, i] = constraint2
                gx[j + 3 * self.obst_num, i] = constraint3

        return gx

    def init_objects(self, x, lambda_, mu_, ref_path):
        sum_states_rate = 0
        sum_controls = 0
        sum_controls_rate = 0
        sum_dist_to_ref = 0
        # sum_g1 = 0.
        # sum_g2 = 0.
        # sum_g3 = 0.
        # gT = self.Array2SX(shape[:, 2][:, None].T)
        # GT = self.Array2SX(shape[:, :2].T)
        #
        # for i in range(self.horizon - 1):
        #     mu_i = mu_[:, i]
        #     yaw_i = x[2, i]
        #     offset = ca.SX.zeros(2, 1)
        #     offset[0, 0] = 1 * ca.cos(yaw_i)
        #     offset[1, 0] = 1 * ca.sin(yaw_i)
        #     t_i = x[:2, i] + offset
        #     rotT_i = ca.SX.zeros(2, 2)
        #     rotT_i[0, 0] = ca.cos(yaw_i)
        #     rotT_i[1, 1] = ca.cos(yaw_i)
        #     rotT_i[1, 0] = -ca.sin(yaw_i)
        #     rotT_i[0, 1] = ca.sin(yaw_i)
        #
        #     for j in range(self.obst_num):
        #         Aj = self.Array2SX(obst[j][:, :2])
        #         bj = self.Array2SX(obst[j][:, 2][:, None])
        #         lambdaj = lambda_[(4 * j):(4 * (j + 1)), i]
        #
        #         constraint1 = ca.mtimes(ca.mtimes(rotT_i, Aj.T), lambdaj) + ca.mtimes(GT, mu_i)
        #         constraint2 = ca.mtimes((ca.mtimes(Aj, t_i) - bj).T, lambdaj) - ca.mtimes(gT, mu_i)
        #         constraint3 = ca.sumsqr(ca.mtimes(Aj.T, lambdaj)) - 1
        #         sum_g1 += ca.sumsqr(constraint1)
        #         sum_g2 += ca.sumsqr(constraint2)
        #         sum_g3 += ca.sumsqr(constraint3)

        for i in range(self.horizon - 1):
            sum_controls += ca.sumsqr(x[3:, i])  # uk
            sum_dist_to_ref += ca.sumsqr(x[:3, i] - ref_path[:3, i])

            if i > 0:
                sum_states_rate += ca.sumsqr(x[:3, i] - x[:3, i - 1])  # xk - xk-1
                sum_controls_rate += ca.sumsqr(x[3:, i] - x[3:, i - 1])  # uk - uk-1

        obj = self.wg[3] * sum_states_rate \
              + self.wg[4] * sum_controls \
              + self.wg[9] * sum_dist_to_ref \
              + self.wg[3] * ca.sumsqr(x[:2, -1] - ref_path[:2, -1]) + \
              self.wg[9] * ca.sumsqr(x[2, -1] - ref_path[2, -1]) \
            # + 1e10 * self.wg[9] * sum_g1 + 1e10 * self.wg[6] * sum_g2 + 1e10 * self.wg[5] * sum_g3

        return obj

    def init_bounds_OBCA(self, start, goal):
        lbx = ca.DM(self.nx + (self.obst_num + 1) * 4, self.horizon)
        ubx = ca.DM(self.nx + (self.obst_num + 1) * 4, self.horizon)
        lbg = ca.DM(self.ng + 4 * self.obst_num, self.horizon - 1)
        ubg = ca.DM(self.ng + 4 * self.obst_num, self.horizon - 1)
        # lbg = ca.DM(self.ng, self.horizon - 1)
        # ubg = ca.DM(self.ng, self.horizon - 1)

        for i in range(self.horizon - 1):
            lbx[0, i] = -self.x_max  # x
            lbx[1, i] = -self.y_max  # y
            lbx[2, i] = -ca.pi  # th
            lbx[3, i] = -self.v_max  # v
            lbx[4, i] = -self.steer_max  # steer
            lbx[5, i] = -self.a_max  # a
            lbx[6:, i] = 0  # lambda, mu

            ubx[0, i] = self.x_max  # x
            ubx[1, i] = self.y_max  # y
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.v_max  # v
            ubx[4, i] = self.steer_max  # steer
            ubx[5, i] = self.a_max  # a
            ubx[6:, i] = 2.  # lambda, mu

            ubg[:self.ng, i] = 1e-5

            # constraint1 rotT_i, Aj.T * lambdaj + GT * mu_i
            lbg[self.ng: self.ng + 2 * self.obst_num, i] = 0.
            ubg[self.ng: self.ng + 2 * self.obst_num, i] = 1e-5

            # constraint2 (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i
            lbg[self.ng + 2 * self.obst_num:self.ng + 3 * self.obst_num, i] = 1e-5
            ubg[self.ng + 2 * self.obst_num:self.ng + 3 * self.obst_num, i] = 10.

            # constraint3  norm_2(Aj.T @ lambdaj) - 1
            lbg[self.ng + 3 * self.obst_num:self.ng + 4 * self.obst_num, i] = 0.
            ubg[self.ng + 3 * self.obst_num:self.ng + 4 * self.obst_num, i] = 1.

        lbx[0, 0] = start[0]
        lbx[1, 0] = start[1]
        lbx[2, 0] = start[2]
        # lbx[3:, 0] = 0.

        ubx[0, 0] = start[0]
        ubx[1, 0] = start[1]
        ubx[2, 0] = start[2]
        # ubx[3:, 0] = 0.

        # lbx[0, -1] = goal[0]
        # lbx[1, -1] = goal[1]
        # lbx[2, -1] = goal[2] - 0.1
        lbx[0, -1] = -self.x_max
        lbx[1, -1] = -self.y_max
        lbx[2, -1] = -ca.pi

        ubx[0, -1] = self.x_max
        ubx[1, -1] = self.y_max
        ubx[2, -1] = ca.pi
        # ubx[3:, -1] = 0.

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)

        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def states_initialization(self, reference_path):

        x0 = ca.DM(self.nx, self.horizon)
        for i in range(self.horizon):
            x0[0, i] = reference_path[0, i]
            x0[1, i] = reference_path[1, i]
            x0[2, i] = reference_path[2, i]
            if i > 1:
                ds = np.linalg.norm(reference_path[:2, i] - reference_path[:2, i - 1])
                x0[4, i] = ds / self.dt
                dyaw = reference_path[2, i] - reference_path[2, i - 1]
                steer = ca.atan2(dyaw * self.base / (ds + 1e-10), 1)
                x0[4, i] = steer

        return x0

    def dual_variables_initialization(self):
        v0 = ca.DM.ones(4 * (self.obst_num + 1), self.horizon) * 1e-8
        return v0

    def init_model_OBCA(self, reference_path, shape_m, obst_m):
        self.horizon = reference_path.shape[1]
        self.obst_num = len(obst_m)
        dt = self.dt

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x ,y ,theta ,steer)
        lambda_ = ca.SX.sym("lambda", 4 * self.obst_num, self.horizon)
        mu_ = ca.SX.sym("mu", 4, self.horizon)

        x0_ = self.states_initialization(reference_path)
        v0 = self.dual_variables_initialization()

        # initialize constraints
        g1 = self.init_motion_constraints(x, dt)
        g2 = self.init_OBCA_constraints(x, lambda_, mu_, shape_m, obst_m)
        gx = ca.vertcat(g1, g2)

        X, G = self.organize_variables(x, lambda_, mu_, gx)
        X0, XL, XU, GL, GU = self.organize_bounds(x0_, v0)

        # initialize objectives
        F = self.init_objects(x, lambda_, mu_, x0_)

        nlp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": True,
                        "ipopt.hessian_approximation": "exact",
                        'ipopt.max_iter': 100,
                        'ipopt.print_level': 3,
                        'print_time': 1,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', nlp, opts_setting)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def organize_variables(self, vx, vl, vm, cg):
        vs = ca.vertcat(vx, vl)
        vs = ca.vertcat(vs, vm)
        X = ca.reshape(vs, -1, 1)
        G = ca.reshape(cg, -1, 1)

        return X, G

    def organize_bounds(self, x0, y0):
        lbx, ubx, lbg, ubg = self.init_bounds_OBCA(x0[:, 0], x0[:, -1])
        x_all = ca.vertcat(x0, y0)
        X0 = ca.reshape(x_all, -1, 1)

        return X0, lbx, ubx, lbg, ubg

    def get_result_OBCA(self):
        cal_traj = ca.reshape(self.x_opt, self.nx + (self.obst_num + 1) * 4, self.horizon)
        op_trajectories = np.array(cal_traj[:3, :])
        op_controls = np.array(cal_traj[3:self.nx, :])
        vl = np.array(cal_traj[self.nx:self.nx + self.obst_num * 4, :])
        vm = np.array(cal_traj[self.nx + self.obst_num * 4:, :])
        return op_trajectories, op_controls, vl, vm


if __name__ == '__main__':
    start_time = time.time()
    with open("../../config_OBCA_large.yaml", 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    ut = UTurnMPC()
    ut.reserve_footprint = True

    cmpc = CasADi_MPC_OBCA_reduced()
    cmpc.set_parameters(param)
    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()

    rx, ry, ryaw, rk, s = calc_spline_course(ref_traj[0, :], ref_traj[1, :], ds=0.8 * cmpc.v_max * cmpc.dt)
    refpath = np.array([rx, ry, ryaw])

    # states: (x ,y ,theta ,v , steer, a)
    cmpc.init_model_OBCA(refpath, shape, obst)
    op_trajectories, op_control, vl, vm = cmpc.get_result_OBCA()
    print("OBCA total time:{:.3f}s".format(time.time() - start_time))
    # op_dt, op_trajectories, op_controls, ref_traj
    ut.plot_results(cmpc.dt, op_trajectories, op_control, ref_traj, four_states=True)
