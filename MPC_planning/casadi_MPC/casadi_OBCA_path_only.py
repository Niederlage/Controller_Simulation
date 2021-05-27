import casadi as ca
import numpy as np
import time
from mpc_motion_plot import UTurnMPC


class CasADi_MPC_OBCA_PathOnly:
    def __init__(self):
        self.base = 2.0
        self.nx = 4
        self.ng = 3
        self.min_distance = 0.2
        self.steer_max = np.pi * 40 / 180

        self.model = None
        self.x_opt = None
        self.obst_num = 0
        self.horizon = 0
        self.alpha = 1.2

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        self.dmin = 0.

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
            th_ = x[2, i]
            steer_ = x[3, i]

            dx = x_ + d_res * ca.cos(th_)
            dy = y_ + d_res * ca.sin(th_)
            dth = th_ + d_res / self.base * ca.tan(steer_)

            gx[0, i] = dx - x[0, i + 1]
            gx[1, i] = dy - x[1, i + 1]
            gx[2, i] = dth - x[2, i + 1]
            # gx[3, i] =  ca.cos(th_) - ca.sin(th_)

        return gx

    def init_OBCA_constraints(self, x_, lambda_, mu_, shape, obst, gx):
        # gT = self.Array2SX(shape[:, 2][:, None].T)
        # GT = self.Array2SX(shape[:, :2].T)

        G = ca.SX.zeros(4, 2)
        G[0, 0] = 1
        G[1, 0] = -1
        G[2, 1] = 1
        G[3, 1] = -1

        g = ca.SX.zeros(4, 1)
        g[0, 0] = 3
        g[1, 0] = 1
        g[2, 0] = 1
        g[3, 0] = 1
        GT = G.T
        gT = g.T

        for i in range(self.horizon - 1):
            mu_i = mu_[:, i]
            yaw_i = x_[2, i]
            # offset = ca.SX.zeros(2, 1)
            # offset[0, 0] = 1 * ca.cos(yaw_i)
            # offset[1, 0] = 1 * ca.sin(yaw_i)
            t_i = x_[:2, i]
            rotT_i = ca.SX.zeros(2, 2)
            rotT_i[0, 0] = ca.cos(yaw_i)
            rotT_i[1, 1] = ca.cos(yaw_i)
            rotT_i[1, 0] = -ca.sin(yaw_i)
            rotT_i[0, 1] = ca.sin(yaw_i)

            for j in range(self.obst_num):
                Aj = self.Array2SX(obst[j][:, :2])
                bj = self.Array2SX(obst[j][:, 2][:, None])
                lambdaj = lambda_[(4 * j):(4 * (j + 1)), i]

                constraint1 = rotT_i @ Aj.T @ lambdaj + GT @ mu_i
                constraint2 = (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i
                constraint3 = ca.sumsqr(Aj.T @ lambdaj)

                gx[j, i] = constraint1[0, 0]
                gx[j + self.obst_num, i] = constraint1[1, 0]
                gx[j + 2 * self.obst_num, i] = constraint2
                gx[j + 3 * self.obst_num, i] = constraint3

        return gx

    def init_objects(self, x, ref_path):
        sum_total_dist = 0
        sum_steer = 0
        sum_yaw_rate = 0
        sum_steer_rate = 0
        sum_dist_to_ref = 0

        for i in range(self.horizon - 1):
            sum_steer += ca.power(x[3, i], 2)
            sum_dist_to_ref += ca.sumsqr(x[:2, i] - ref_path[:2, i])

            if i > 1:
                sum_total_dist += ca.sumsqr(x[:2, i] - x[:2, i - 1])
                sum_yaw_rate += ca.power(x[2, i] - x[2, i - 1], 2)
                sum_steer_rate += ca.power(x[3, i] - x[3, i - 1], 2)

        obj = self.wg[3] * sum_total_dist + self.wg[4] * sum_yaw_rate \
              + self.wg[4] * sum_steer + self.wg[4] * sum_steer_rate \
              + self.wg[4] * sum_dist_to_ref

        return obj

    def init_bounds_OBCA(self, start, goal):
        lbx = ca.DM(self.nx + (self.obst_num + 1) * 4, self.horizon)
        ubx = ca.DM(self.nx + (self.obst_num + 1) * 4, self.horizon)
        lbg = ca.DM(self.ng + 4 * self.obst_num, self.horizon - 1)
        ubg = ca.DM(self.ng + 4 * self.obst_num, self.horizon - 1)

        for i in range(self.horizon - 1):
            lbx[0, i] = -10.  # x
            lbx[1, i] = 0.  # y
            lbx[2, i] = -ca.pi  # th
            lbx[3, i] = -self.steer_max  # steer
            lbx[4:, i] = 0.  # lambda, mu

            ubx[0, i] = 10.  # x
            ubx[1, i] = 10.  # y
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.steer_max  # steer
            ubx[4:, i] = ca.inf  # lambda, mu

            # lbg[6:6 + 2 * self.obst_num, i] = 0.
            # ubg[6:6 + 2 * self.obst_num, i] = 0.

            # constraint2 (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i
            lbg[self.ng + 2 * self.obst_num:self.ng + 3 * self.obst_num, i] = 0.
            ubg[self.ng + 2 * self.obst_num:self.ng + 3 * self.obst_num, i] = ca.inf

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
                x0[3, i] = ca.atan2(dyaw * self.base / ds, 1)

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

    ut = UTurnMPC()
    cmpc = CasADi_MPC_OBCA_PathOnly()
    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()

    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc.init_model_OBCA(ref_traj, shape, obst)
    op_trajectories = cmpc.get_result_OBCA()
    print("OBCA total time:{:.3f}s".format(time.time() - start_time))
    ut.plot_results_path_only(op_trajectories, ref_traj, ob)
