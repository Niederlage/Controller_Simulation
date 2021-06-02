import casadi as ca
import numpy as np
import time
from mpc_motion_plot import UTurnMPC
import yaml


class CasADi_MPC_OBCA_PathOnly:
    def __init__(self):
        self.base = 2.0
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2

        self.nx = 4
        self.ng = 3
        self.min_distance = 0.2
        self.steer_max = np.pi * 40 / 180

        self.model = None
        self.x_opt = None
        self.obst_num = 0
        self.horizon = 0

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

    def init_motion_constraints(self, x, d_res, gx):
        for i in range(self.horizon - 1):
            x_ = x[0, i]
            y_ = x[1, i]
            yaw_ = x[2, i]
            steer_ = x[3, i]
            #
            # dx = d_res * ca.cos(yaw_)
            # dy = d_res * ca.sin(yaw_)
            # dyaw = d_res / self.base * ca.tan(steer_)

            k1_dx = d_res * ca.cos(yaw_)
            k1_dy = d_res * ca.sin(yaw_)
            k1_dyaw = d_res / self.base * ca.tan(steer_)

            k2_dx = d_res * ca.cos(yaw_ + 0.5 * k1_dyaw)
            k2_dy = d_res * ca.sin(yaw_ + 0.5 * k1_dyaw)
            k2_dyaw = d_res / self.base * ca.tan(steer_)

            k3_dx = d_res * ca.cos(yaw_ + 0.5 * k2_dyaw)
            k3_dy = d_res * ca.sin(yaw_ + 0.5 * k2_dyaw)
            k3_dyaw = d_res / self.base * ca.tan(steer_)

            k4_dx = d_res * ca.cos(yaw_ + 0.5 * k3_dyaw)
            k4_dy = d_res * ca.sin(yaw_ + 0.5 * k3_dyaw)
            k4_dyaw = d_res / self.base * ca.tan(steer_)

            dx = (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            dy = (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            dyaw = (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6

            gx[0, i] = x_ + dx - x[0, i + 1]
            gx[1, i] = y_ + dy - x[1, i + 1]
            gx[2, i] = yaw_ + dyaw - x[2, i + 1]

        return gx

    def init_OBCA_constraints(self, x_, lambda_o, mu_o, shape, obst, gx):
        gT = self.Array2SX(shape[:, 2][:, None].T)
        GT = self.Array2SX(shape[:, :2].T)

        lambda_v = ca.reshape(lambda_o, -1, 1)
        lambda_ = ca.reshape(lambda_v, self.horizon, 4 * self.obst_num).T
        mu_v = ca.reshape(mu_o, -1, 1)
        mu_ = ca.reshape(mu_v, self.horizon, 4).T

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

    def init_objects(self, x, ref_path):
        sum_states = 0
        sum_states_rate = 0
        sum_controls = 0
        sum_controls_rate = 0
        sum_dist_to_ref = 0

        for i in range(self.horizon - 1):
            sum_states += ca.sumsqr(x[:3, i])  # xk
            sum_controls += ca.sumsqr(x[3:, i])  # uk
            sum_dist_to_ref += ca.sumsqr(x[:3, i] - ref_path[:3, i])

            if i > 0:
                sum_states_rate += ca.sumsqr(x[:3, i] - x[:3, i - 1])  # xk - xk-1
                sum_controls_rate += ca.sumsqr(x[3:, i] - x[3:, i - 1])  # uk - uk-1

        obj = self.wg[2] * sum_states + self.wg[7] * sum_states_rate \
              + self.wg[3] * sum_controls + 1e1 * self.wg[9] * sum_controls_rate \
              + self.wg[2] * sum_dist_to_ref
              # + self.wg[2] * ca.sumsqr(x[:3, -1] - ref_path[:3, -1]) \


        return obj

    def init_bounds_OBCA(self, start, goal):
        lbx = ca.DM(self.nx + (self.obst_num + 1) * 4, self.horizon)
        ubx = ca.DM(self.nx + (self.obst_num + 1) * 4, self.horizon)
        lbg = ca.DM(self.ng + 4 * self.obst_num, self.horizon - 1)
        ubg = ca.DM(self.ng + 4 * self.obst_num, self.horizon - 1)

        for i in range(self.horizon - 1):
            lbx[0, i] = -10.  # x
            lbx[1, i] = -1.  # y
            lbx[2, i] = -ca.pi  # th
            lbx[3, i] = -self.steer_max  # steer
            lbx[4:, i] = 1e-10  # lambda, mu

            ubx[0, i] = 10.  # x
            ubx[1, i] = 10.  # y
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.steer_max  # steer
            ubx[4:, i] = 1.  # lambda, mu

            lbg[6:6 + 2 * self.obst_num, i] = 0.
            ubg[6:6 + 2 * self.obst_num, i] = 1e-5

            # constraint2 (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i
            lbg[self.ng + 2 * self.obst_num:self.ng + 3 * self.obst_num, i] = 0.
            ubg[self.ng + 2 * self.obst_num:self.ng + 3 * self.obst_num, i] = 1e-3

            # constraint3  norm_2(Aj.T @ lambdaj) - 1
            lbg[self.ng + 3 * self.obst_num:self.ng + 4 * self.obst_num, i] = 1.
            ubg[self.ng + 3 * self.obst_num:self.ng + 4 * self.obst_num, i] = 1.

        lbx[0, 0] = start[0]
        lbx[1, 0] = start[1]
        lbx[2, 0] = start[2]
        lbx[3:, 0] = 0.

        ubx[0, 0] = start[0]
        ubx[1, 0] = start[1]
        ubx[2, 0] = start[2]
        ubx[3:, 0] = 0.

        lbx[0, -1] = goal[0]
        lbx[1, -1] = goal[1]
        lbx[2, -1] = goal[2]
        lbx[3:, -1] = 0.

        ubx[0, -1] = goal[0]
        ubx[1, -1] = goal[1]
        ubx[2, -1] = goal[2]
        ubx[3:, -1] = 0.

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
                dyaw = reference_path[2, i] - reference_path[2, i - 1]
                steer = ca.atan2(dyaw * self.base / ds, 1)
                x0[3, i] = steer

        return x0

    def dual_variables_initialization(self):
        v0 = ca.DM.ones(4 * (self.obst_num + 1), self.horizon) * 1e-1
        return v0

    def init_model_OBCA(self, reference_path, shape_m, obst_m):
        self.horizon = reference_path.shape[1]
        self.obst_num = len(obst_m)

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x ,y ,theta ,steer)
        lambda_ = ca.SX.sym("lambda", 4 * self.obst_num, self.horizon)
        mu_ = ca.SX.sym("mu", 4, self.horizon)

        x0_ = self.states_initialization(reference_path)
        v0 = self.dual_variables_initialization()

        # initialize constraints
        g1 = ca.SX.sym("g1", self.ng, self.horizon - 1)
        g1 = self.init_motion_constraints(x, self.min_distance, g1)
        g2 = ca.SX.sym("g2", self.obst_num * 4, self.horizon - 1)
        g2 = self.init_OBCA_constraints(x, lambda_, mu_, shape_m, obst_m, g2)
        gx = ca.vertcat(g1, g2)

        X, G = self.organize_variables(x, lambda_, mu_, gx)
        X0, XL, XU, GL, GU = self.organize_bounds(x0_, v0)

        # initialize objectives
        F = self.init_objects(x, x0_)

        nlp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": True,
                        "ipopt.hessian_approximation": "limited-memory",
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
        op_trajectories = np.array(cal_traj[:4, :])
        vl = np.array(cal_traj[4:16, :])
        vm = np.array(cal_traj[16:, :])
        return op_trajectories


if __name__ == '__main__':
    start_time = time.time()
    large = True
    if large:
        address = "../config_OBCA_large.yaml"
    else:
        address = "../config_OBCA.yaml"
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    ut = UTurnMPC()
    ut.reserve_footprint = True
    ut.set_parameters(param)
    cmpc = CasADi_MPC_OBCA_PathOnly()
    cmpc.set_parameters(param)
    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()

    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc.init_model_OBCA(ref_traj, shape, obst)
    op_trajectories = cmpc.get_result_OBCA()
    print("OBCA total time:{:.3f}s".format(time.time() - start_time))
    ut.plot_results_path_only(op_trajectories, ref_traj, ob)
