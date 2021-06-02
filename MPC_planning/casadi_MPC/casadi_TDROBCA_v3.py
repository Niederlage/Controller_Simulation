import casadi as ca
import numpy as np
import time
from mpc_motion_plot import UTurnMPC
import yaml


class CasADi_MPC_TDROBCA:
    def __init__(self):
        self.base = 2.0
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2

        self.nx = 8 - 2
        self.ng = 6 - 2
        self.obst_num = 0
        self.horizon = 0
        self.dt0 = 0.
        self.model = None
        self.x_opt = None
        self.op_lambda0 = None
        self.op_mu0 = None
        self.op_d0 = None
        self.op_control0 = None

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        self.v_max = 2.
        self.steer_max = ca.pi * 40 / 180
        self.omega_max = ca.pi
        self.a_max = 4.
        self.steer_rate_max = ca.pi * 40 / 180
        self.jerk_max = 5.
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

    def Array2DM(self, array):
        rows, cols = np.shape(array)
        sx = ca.DM.zeros(rows * cols, 1)
        array_ = array.T.flatten()
        for i in range(rows * cols):
            sx[i] = array_[i]

        return ca.reshape(sx, rows, cols)

    def init_dynamic_constraints(self, x, dt, gx):
        for i in range(self.horizon - 1):
            x_ = x[0, i]
            y_ = x[1, i]
            yaw_ = x[2, i]
            v_ = x[3, i]
            steer_ = x[4, i]
            a_ = x[5, i]
            # steer_rate_ = x[6, i]
            # jerk_ = x[7, i]

            k1_dx = v_ * ca.cos(yaw_)
            k1_dy = v_ * ca.sin(yaw_)
            k1_dyaw = v_ / self.base * ca.tan(steer_)
            k1_dv = a_
            # k1_dsteer = steer_rate_
            # k1_da = jerk_

            k2_dx = (v_ + 0.5 * dt * k1_dv) * ca.cos(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dy = (v_ + 0.5 * dt * k1_dv) * ca.sin(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dyaw = (v_ + 0.5 * dt * k1_dv) / self.base * ca.tan(steer_)
            k2_dv = a_
            # k2_dsteer = steer_rate_
            # k2_da = jerk_

            k3_dx = (v_ + 0.5 * dt * k2_dv) * ca.cos(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dy = (v_ + 0.5 * dt * k2_dv) * ca.sin(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dyaw = (v_ + 0.5 * dt * k2_dv) / self.base * ca.tan(steer_)
            k3_dv = a_
            # k3_dsteer = steer_rate_
            # k3_da = jerk_

            k4_dx = (v_ + 0.5 * dt * k3_dv) * ca.cos(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dy = (v_ + 0.5 * dt * k3_dv) * ca.sin(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dyaw = (v_ + 0.5 * dt * k3_dv) / self.base * ca.tan(steer_)
            k4_dv = a_
            # k4_dsteer = steer_rate_
            # k4_da = jerk_

            dx = dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            dy = dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            dyaw = dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
            dv = dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6
            # dsteer = dt * (k1_dsteer + 2 * k2_dsteer + 2 * k3_dsteer + k4_dsteer) / 6
            # da = dt * (k1_da + 2 * k2_da + 2 * k3_da + k4_da) / 6

            gx[0, i] = x_ + dx - x[0, i + 1]
            gx[1, i] = y_ + dy - x[1, i + 1]
            gx[2, i] = yaw_ + dyaw - x[2, i + 1]
            gx[3, i] = v_ + dv - x[3, i + 1]
            # gx[4, i] = steer_ + dsteer - x[4, i + 1]
            # gx[5, i] = a_ + da - x[5, i + 1]

        return gx

    def init_OBCA_constraints(self, x_, lambda_, mu_, d_, shape, obst, gx):
        gT = self.Array2SX(shape[:, 2][:, None].T)
        GT = self.Array2SX(shape[:, :2].T)

        # lambda_v = ca.reshape(lambda_, -1, 1)
        # lambda_ = ca.reshape(lambda_v, self.horizon, 4 * self.obst_num).T
        # mu_v = ca.reshape(mu_, -1, 1)
        # mu_ = ca.reshape(mu_v, self.horizon, 4).T

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

        l_ob = len(obst)
        for i in range(self.horizon - 1):
            mu_i = mu_[:, i]
            yaw_i = x_[2, i]
            offset = ca.SX.zeros(2, 1)
            offset[0, 0] = self.offset * ca.cos(yaw_i)
            offset[1, 0] = self.offset * ca.sin(yaw_i)
            t_i = x_[:2, i] + offset

            rotT_i = ca.SX.zeros(2, 2)
            rotT_i[0, 0] = ca.cos(yaw_i)
            rotT_i[1, 1] = ca.cos(yaw_i)
            rotT_i[1, 0] = -ca.sin(yaw_i)
            rotT_i[0, 1] = ca.sin(yaw_i)

            for j in range(l_ob):
                lambdaj = lambda_[(4 * j):(4 * (j + 1)), i]

                Aj = self.Array2SX(obst[j][:, :2])
                bj = self.Array2SX(obst[j][:, 2][:, None])
                constraint1 = ca.mtimes(ca.mtimes(rotT_i, Aj.T), lambdaj) + ca.mtimes(GT, mu_i)
                constraint2 = ca.mtimes((ca.mtimes(Aj, t_i) - bj).T, lambdaj) - ca.mtimes(gT, mu_i) + d_[j, i]
                constraint3 = ca.sumsqr(ca.mtimes(Aj.T, lambdaj))
                constraint10 = constraint1[0, 0]
                constraint11 = constraint1[1, 0]

                # Aj = obst[j][:, :2]
                # bj = obst[j][:, 2]
                # constraint10 = (mu_i[0] - mu_i[2]) \
                #                + ca.cos(x_[2, i]) * sum(Aj[k, 0] * lambdaj[k] for k in range(4)) \
                #                + ca.sin(x_[2, i]) * sum(Aj[k, 1] * lambdaj[k] for k in range(4))
                #
                # constraint11 = (mu_i[1] - mu_i[3]) \
                #                - ca.sin(x_[2, i]) * sum(Aj[k, 1] * lambdaj[k] for k in range(4)) \
                #                + ca.cos(x_[2, i]) * sum(Aj[k, 1] * lambdaj[k] for k in range(4))
                #
                # constraint2 = -(-sum(g[k] * mu_i[k] for k in range(4))) \
                #               + (x_[0, i] + self.offset * ca.cos(x_[2, i])) \
                #               * sum(Aj[k, 0] * lambdaj[k] for k in range(4)) \
                #               + (x_[1, i] + self.offset * ca.sin(x_[2, i])) \
                #               * sum(Aj[k, 1] * lambdaj[k] for k in range(4)) \
                #               - sum(bj[k] * lambdaj[k] for k in range(4)) + d_[j, i]
                #
                # constraint3 = sum(Aj[k, 0] * lambdaj[k] for k in range(4)) ** 2 \
                #               + sum(Aj[k, 1] * lambdaj[k] for k in range(4)) ** 2

                gx[j, i] = ca.fabs(constraint10)
                gx[j + l_ob, i] = ca.fabs(constraint11)
                gx[j + 2 * l_ob, i] = constraint2
                gx[j + 3 * l_ob, i] = constraint3

        return gx

    def init_objects(self, x_, d_, ref_path):

        sum_mindist = 0
        sum_states = 0
        sum_states_rate = 0
        sum_controls = 0
        sum_controls_rate = 0

        for i in range(self.horizon):
            sum_states += ca.sumsqr(x_[:4, i])  # xk
            sum_controls += ca.sumsqr(x_[4:, i])  # uk
            sum_mindist += ca.sumsqr(d_[:, i])
            if i > 0:
                sum_states_rate += ca.sumsqr(x_[:4, i] - x_[:4, i - 1])  # xk - xk-1
                sum_controls_rate += ca.sumsqr(x_[4:, i] - x_[4:, i - 1])  # uk - uk-1

        obj = self.wg[1] * sum_states + self.wg[6] * sum_states_rate + self.wg[3] * sum_controls \
              + 1e1 * self.wg[9] * sum_controls_rate + 1e8 * self.wg[9] * ca.sumsqr(x_[:3, -1] - ref_path[:3, -1]) \
              + 1e16 * self.wg[9] * sum_mindist

        return obj

    def init_bounds_OBCA(self, start, goal):
        lbx = ca.DM.zeros(self.nx + (self.obst_num + 1) * 4 + self.obst_num, self.horizon)
        ubx = ca.DM.zeros(self.nx + (self.obst_num + 1) * 4 + self.obst_num, self.horizon)
        lbg = ca.DM.zeros(self.ng + 4 * self.obst_num, self.horizon - 1)
        ubg = ca.DM.zeros(self.ng + 4 * self.obst_num, self.horizon - 1)

        for i in range(self.horizon):
            lbx[0, i] = -10.  # x
            ubx[0, i] = 10.  # x
            lbx[1, i] = -1.  # y
            ubx[1, i] = 10.  # 1.1y
            lbx[2, i] = -ca.pi  # th
            ubx[2, i] = ca.pi  # th
            lbx[3, i] = -self.v_max  # v
            ubx[3, i] = self.v_max  # v
            lbx[4, i] = -self.steer_max  # steer
            ubx[4, i] = self.steer_max  # steer
            lbx[5, i] = -self.a_max  # a
            ubx[5, i] = self.a_max  # a

            lbx[6:-self.obst_num, i] = 1e-5  # lambda, mu
            ubx[6:-self.obst_num, i] = 1.  # lambda, mu
            lbx[-self.obst_num:, i] = -1e-5  # dmin
            ubx[-self.obst_num:, i] = -1e-9  # -4e-5  # dmin

        # constraint1 rotT_i @ Aj.T @ lambdaj + GT @ mu_i == 0
        lbg[self.ng:self.ng + 2 * self.obst_num, :] = 1e-5
        ubg[self.ng:self.ng + 2 * self.obst_num, :] = 1e-5

        # constraint2 (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i + dmin == 0
        lbg[self.ng + 2 * self.obst_num:self.ng + 3 * self.obst_num, :] = 0.
        ubg[self.ng + 2 * self.obst_num:self.ng + 3 * self.obst_num, :] = 0.  # 1e-5

        # constraint3  norm_2(Aj.T @ lambdaj) <=1
        lbg[self.ng + 3 * self.obst_num:self.ng + 4 * self.obst_num, :] = 0.
        ubg[self.ng + 3 * self.obst_num:self.ng + 4 * self.obst_num, :] = 1.

        lbx[0, 0] = start[0]
        lbx[1, 0] = start[1]
        lbx[2, 0] = start[2]
        lbx[3:, 0] = 0.

        ubx[0, 0] = start[0]
        ubx[1, 0] = start[1]
        ubx[2, 0] = start[2]
        ubx[3:, 0] = 0.

        # lbx[0, -1] = goal[0]
        # lbx[1, -1] = goal[1]
        lbx[2, -1] = goal[2]
        lbx[3:, -1] = 0.

        # ubx[0, -1] = goal[0]
        # ubx[1, -1] = goal[1]
        ubx[2, -1] = goal[2]
        ubx[3:, -1] = 0.

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)

        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def states_initialization(self, reference_path):
        x0 = ca.DM(self.nx, self.horizon)
        diff_s = np.diff(reference_path[:2, :], axis=1)
        sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
        self.dt0 = 1.4 * (sum_s / self.v_max + self.v_max / self.a_max) / self.horizon

        last_v = 0.
        # last_a = 0.
        # last_steer = 0.
        for i in range(self.horizon):
            x0[0, i] = reference_path[0, i]
            x0[1, i] = reference_path[1, i]
            x0[2, i] = reference_path[2, i]

            if i > 1:
                ds = np.linalg.norm(reference_path[:2, i] - reference_path[:2, i - 1])
                dyaw = reference_path[2, i] - reference_path[2, i - 1]
                steer = ca.atan2(dyaw * self.base / ds, 1)
                x0[3, i] = ds / self.dt0
                x0[4, i] = steer
                x0[5, i] = (ds / self.dt0 - last_v) / self.dt0
                # x0[6, i] = (steer - last_steer) / self.dt0
                # x0[7, i] = ((ds / self.dt0 - last_v) / self.dt0 - last_a) / self.dt0

            last_v = x0[3, i]
            # last_steer = x0[4, i]
            # last_a = x0[5, i]

        if self.op_control0 is not None:
            x0[3:, :] = self.op_control0

        return x0

    def dual_variables_initialization(self):
        v0 = ca.DM.zeros(4 * (self.obst_num + 1) + self.obst_num, self.horizon)
        if (self.op_lambda0 is not None) and (self.op_mu0 is not None) and (self.op_d0 is not None):
            v0[:4 * self.obst_num, :] = self.Array2DM(self.op_lambda0)
            v0[4 * self.obst_num: -self.obst_num, :] = self.Array2DM(self.op_mu0)
            v0[-self.obst_num:, :] = self.Array2DM(self.op_d0)

        return v0

    def init_model_OBCA(self, reference_path, shape_m, obst_m):
        self.horizon = reference_path.shape[1]
        self.obst_num = len(obst_m)
        x0_ = self.states_initialization(reference_path)
        v0 = self.dual_variables_initialization()

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x, y, theta, v, steer, a, steer_rate, jerk)
        lambda_ = ca.SX.sym("lambda", 4 * self.obst_num, self.horizon)
        mu_ = ca.SX.sym("mu", 4, self.horizon)
        d_ = ca.SX.sym("d", self.obst_num, self.horizon)

        # initialize constraints
        g1 = ca.SX.sym("g1", self.ng, self.horizon - 1)
        g1 = self.init_dynamic_constraints(x, self.dt0, g1)
        g2 = ca.SX.sym("g2", self.obst_num * 4, self.horizon - 1)
        g2 = self.init_OBCA_constraints(x, lambda_, mu_, d_, shape_m, obst_m, g2)
        gx = ca.vertcat(g1, g2)

        X, G = self.organize_variables(x, lambda_, mu_, d_, gx)
        X0, XL, XU, GL, GU = self.organize_bounds(x0_, v0)

        # initialize objectives
        F = self.init_objects(x, d_, x0_)

        nlp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": False,
                        "ipopt.hessian_approximation": "exact",
                        'ipopt.max_iter': 100,
                        'ipopt.print_level': 3,
                        'print_time': 1,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', nlp, opts_setting)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def organize_variables(self, vx, vl, vm, vd, cg):
        vs = ca.vertcat(vx, vl)
        vs = ca.vertcat(vs, vm)
        vs = ca.vertcat(vs, vd)
        X = ca.reshape(vs, -1, 1)
        # X = ca.vertcat(dt, x_)
        G = ca.reshape(cg, -1, 1)

        return X, G

    def organize_bounds(self, x0, y0):
        lbx, ubx, lbg, ubg = self.init_bounds_OBCA(x0[:, 0], x0[:, -1])
        x_all = ca.vertcat(x0, y0)
        X0 = ca.reshape(x_all, -1, 1)
        # X0 = ca.vertcat(self.dt0, x0_)

        return X0, lbx, ubx, lbg, ubg

    def get_result_OBCA(self):
        op_dt = float(self.dt0)
        cal_traj = ca.reshape(self.x_opt, self.nx + (self.obst_num + 1) * 4 + self.obst_num, self.horizon)
        # cal_traj = ca.reshape(self.x_opt[1:], self.horizon, self.nx + (self.obst_num + 1) * 4).T
        op_controls = np.array(cal_traj[3:8, :])
        op_trajectories = np.array(cal_traj[:3, :])

        vl = np.array(cal_traj[8:20, :])
        vm = np.array(cal_traj[20:24, :])
        vd = np.array(cal_traj[24:, :])

        return op_dt, op_trajectories, op_controls, vl, vm


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
    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = CasADi_MPC_TDROBCA()
    cmpc.set_parameters(param)
    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()

    cmpc.init_model_OBCA(ref_traj, shape, obst)
    op_dt, op_trajectories, op_controls, vl, vm = cmpc.get_result_OBCA()
    print("TDROBCA total time:{:.3f}s".format(time.time() - start_time))
    ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob, four_states=True)
