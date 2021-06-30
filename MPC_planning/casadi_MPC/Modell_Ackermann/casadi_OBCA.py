import casadi as ca
import numpy as np
import time
from mpc_motion_plot import UTurnMPC
import yaml


class CasADi_MPC_OBCA:
    def __init__(self):
        self.base = 0.4
        self.LF = 0.6
        self.LB = 0.2
        self.offset = (self.LF - self.LB) / 2

        self.nx = 8
        self.ng = 6
        self.obst_num = 0
        self.horizon = 0
        self.dt0 = 0.
        self.model = None
        self.x_opt = None
        self.op_lambda0 = None
        self.op_mu0 = None
        self.op_control0 = None

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        self.v_max = 5.
        self.steer_max = ca.pi * 40 / 180
        self.omega_max = ca.pi
        self.a_max = 3.
        self.steer_rate_max = ca.pi * 40 / 180
        self.jerk_max = 3.
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
            steer_rate_ = x[6, i]
            jerk_ = x[7, i]

            k1_dx = v_ * ca.cos(yaw_)
            k1_dy = v_ * ca.sin(yaw_)
            k1_dyaw = v_ / self.base * ca.tan(steer_)
            k1_dv = a_
            k1_dsteer = steer_rate_
            k1_da = jerk_

            k2_dx = (v_ + 0.5 * dt * k1_dv) * ca.cos(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dy = (v_ + 0.5 * dt * k1_dv) * ca.sin(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dyaw = (v_ + 0.5 * dt * k1_dv) / self.base * ca.tan(steer_ + 0.5 * dt * k1_dsteer)
            k2_dv = a_ + 0.5 * dt * k1_da
            k2_dsteer = steer_rate_
            k2_da = jerk_

            k3_dx = (v_ + 0.5 * dt * k2_dv) * ca.cos(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dy = (v_ + 0.5 * dt * k2_dv) * ca.sin(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dyaw = (v_ + 0.5 * dt * k2_dv) / self.base * ca.tan(steer_ + 0.5 * dt * k2_dsteer)
            k3_dv = a_ + 0.5 * dt * k2_da
            k3_dsteer = steer_rate_
            k3_da = jerk_

            k4_dx = (v_ + 0.5 * dt * k3_dv) * ca.cos(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dy = (v_ + 0.5 * dt * k3_dv) * ca.sin(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dyaw = (v_ + 0.5 * dt * k3_dv) / self.base * ca.tan(steer_ + 0.5 * dt * k3_dsteer)
            k4_dv = a_ + 0.5 * dt * k3_da
            k4_dsteer = steer_rate_
            k4_da = jerk_

            dx = dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            dy = dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            dyaw = dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
            dv = dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6
            dsteer = dt * (k1_dsteer + 2 * k2_dsteer + 2 * k3_dsteer + k4_dsteer) / 6
            da = dt * (k1_da + 2 * k2_da + 2 * k3_da + k4_da) / 6

            gx[0, i] = x_ + dx - x[0, i + 1]
            gx[1, i] = y_ + dy - x[1, i + 1]
            gx[2, i] = yaw_ + dyaw - x[2, i + 1]
            gx[3, i] = v_ + dv - x[3, i + 1]
            gx[4, i] = steer_ + dsteer - x[4, i + 1]
            gx[5, i] = a_ + da - x[5, i + 1]

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

        l_ob = len(obst)
        for i in range(self.horizon - 1):
            mu_i = mu_[:, i]
            yaw_i = x_[2, i + 1]
            offset = ca.SX.zeros(2, 1)
            offset[0, 0] = self.offset * ca.cos(yaw_i)
            offset[1, 0] = self.offset * ca.sin(yaw_i)
            t_i = x_[:2, i + 1]

            rotT_i = ca.SX.zeros(2, 2)
            rotT_i[0, 0] = ca.cos(yaw_i)
            rotT_i[1, 1] = ca.cos(yaw_i)
            rotT_i[1, 0] = -ca.sin(yaw_i)
            rotT_i[0, 1] = ca.sin(yaw_i)

            for j in range(l_ob):
                Aj = self.Array2SX(obst[j][:, :2])
                bj = self.Array2SX(obst[j][:, 2][:, None])
                lambdaj = lambda_[(4 * j):(4 * (j + 1)), i]

                constraint1 = rotT_i @ Aj.T @ lambdaj + GT @ mu_i
                constraint2 = (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i + self.dmin
                constraint3 = ca.sumsqr(Aj.T @ lambdaj)

                gx[j, i] = constraint1[0, 0]
                gx[j + l_ob, i] = constraint1[1, 0]
                gx[j + 2 * l_ob, i] = constraint2
                gx[j + 3 * l_ob, i] = constraint3

        return gx

    def init_objects(self, x, dt, ref_path):
        sum_total_dist = 0
        sum_time = 0
        sum_vel = 0
        sum_steer = 0
        sum_a = 0
        sum_steer_rate = 0
        sum_dist_to_ref = 0
        sum_jerk = 0

        for i in range(self.horizon):
            sum_time += ca.power(dt, 2)
            sum_vel += ca.power(x[3, i], 2)
            sum_a += ca.power(x[5, i], 2)
            sum_jerk += ca.power(x[6, i], 2)
            sum_steer_rate += ca.power(x[6, i], 2)
            sum_dist_to_ref += ca.sumsqr(x[:3, i] - ref_path[:3, i])

            if i > 1:
                sum_total_dist += ca.sumsqr(x[:2, i] - x[:2, i - 1])
                sum_steer += ca.power(x[4, i] - x[4, i - 1], 2)

        obj = self.wg[6] * sum_total_dist + self.wg[9] * sum_time \
              + self.wg[3] * sum_vel + self.wg[6] * sum_steer \
              + self.wg[3] * sum_a + self.wg[5] * sum_steer_rate \
              + self.wg[3] * sum_dist_to_ref + self.wg[4] * sum_jerk

        return obj

    def init_bounds_OBCA(self, start, goal):
        lbx = ca.DM.zeros(self.nx + (self.obst_num + 1) * 4, self.horizon)
        ubx = ca.DM.zeros(self.nx + (self.obst_num + 1) * 4, self.horizon)
        lbg = ca.DM.zeros(self.ng + 4 * self.obst_num, self.horizon - 1)
        ubg = ca.DM.zeros(self.ng + 4 * self.obst_num, self.horizon - 1)

        for i in range(self.horizon):
            lbx[0, i] = -10.  # x
            ubx[0, i] = 20.  # x
            lbx[1, i] = 0.  # y
            ubx[1, i] = 15.  # 1.1y
            lbx[2, i] = -ca.pi  # th
            ubx[2, i] = ca.pi  # th
            lbx[3, i] = -self.v_max  # v
            ubx[3, i] = self.v_max  # v
            lbx[4, i] = -self.steer_max  # steer
            ubx[4, i] = self.steer_max  # steer
            lbx[5, i] = -self.a_max  # a
            ubx[5, i] = self.a_max  # a
            lbx[6, i] = -self.steer_rate_max  # steer_rate
            ubx[6, i] = self.steer_rate_max  # steer_rate
            lbx[7, i] = -self.jerk_max  # jerk
            ubx[7, i] = self.jerk_max  # jerk

            lbx[8:, i] = 0.  # lambda, mu
            ubx[8:, i] = 1.  # lambda, mu

        # constraint1  rotT_i @ Aj.T @ lambdaj + GT @ mu_i == 0
        lbg[6:6 + 2 * self.obst_num, :] = 0.
        ubg[6:6 + 2 * self.obst_num, :] = 0.  # 1e-5

        # constraint2 (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i
        lbg[6 + 2 * self.obst_num:6 + 3 * self.obst_num, :] = 0.
        ubg[6 + 2 * self.obst_num:6 + 3 * self.obst_num, :] = 1e-5  # 1e-5

        # constraint3  norm_2(Aj.T @ lambdaj) == 1
        lbg[6 + 3 * self.obst_num:6 + 4 * self.obst_num, :] = 0.
        ubg[6 + 3 * self.obst_num:6 + 4 * self.obst_num, :] = 1.

        lbx[0, 0] = start[0]
        lbx[1, 0] = start[1]
        lbx[2, 0] = start[2]
        lbx[3:, 0] = 0.

        ubx[0, 0] = start[0]
        ubx[1, 0] = start[1]
        ubx[2, 0] = start[2]
        ubx[3:, 0] = 0.

        lbx[0, -1] = -20.
        lbx[1, -1] = -20.
        lbx[2, -1] = -ca.pi
        lbx[3:, -1] = 0.

        # ubx[0, -1] = goal[0]
        # ubx[1, -1] = goal[1]
        # ubx[2, -1] = goal[2] + 0.1
        ubx[0, -1] = 20.
        ubx[1, -1] = 20.
        ubx[2, -1] = ca.pi
        ubx[3:, -1] = 0.

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)
        lbx_ = ca.vertcat(0.01, lbx_)
        ubx_ = ca.vertcat(0.2, ubx_)

        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def states_initialization(self, reference_path):
        x0 = ca.DM(self.nx, self.horizon)
        diff_s = np.diff(reference_path[:2, :], axis=1)
        sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
        self.dt0 = 1.3 * (sum_s / self.v_max + self.v_max / self.a_max) / self.horizon
        last_v = 0.
        last_a = 0.
        last_steer = 0.
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
                x0[6, i] = (steer - last_steer) / self.dt0
                x0[7, i] = ((ds / self.dt0 - last_v) / self.dt0 - last_a) / self.dt0

            last_v = x0[3, i]
            last_steer = x0[4, i]
            last_a = x0[5, i]

        if self.op_control0 is not None:
            x0[3:, :] = self.op_control0

        return x0

    def dual_variables_initialization(self):
        t0 = ca.DM.zeros(4 * (self.obst_num + 1), self.horizon)
        if (self.op_lambda0 is not None) and (self.op_mu0 is not None):
            t0[:4 * self.obst_num, :] = self.Array2DM(self.op_lambda0)
            t0[4 * self.obst_num:, :] = self.Array2DM(self.op_mu0)

        return t0

    def init_model_OBCA(self, reference_path, shape_m, obst_m):
        self.horizon = reference_path.shape[1]
        self.obst_num = len(obst_m)
        x0_ = self.states_initialization(reference_path)
        v0 = self.dual_variables_initialization()

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x, y, theta, v, steer, a, steer_rate, jerk)
        dt = ca.SX.sym("dt")
        lambda_ = ca.SX.sym("lambda", 4 * self.obst_num, self.horizon)
        mu_ = ca.SX.sym("mu", 4, self.horizon)

        # initialize constraints
        g1 = ca.SX.sym("g1", self.ng, self.horizon - 1)
        g1 = self.init_dynamic_constraints(x, dt, g1)
        g2 = ca.SX.sym("g2", self.obst_num * 4, self.horizon - 1)
        g2 = self.init_OBCA_constraints(x, lambda_, mu_, shape_m, obst_m, g2)
        gx = ca.vertcat(g1, g2)

        X, G = self.organize_variables(dt, x, lambda_, mu_, gx)
        X0, XL, XU, GL, GU = self.organize_bounds(x0_, v0)

        # initialize objectives
        F = self.init_objects(x, dt, x0_)

        nlp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": True,
                        # "ipopt.hessian_approximation": "limited-memory",
                        'ipopt.max_iter': 500,
                        'ipopt.print_level': 3,
                        'print_time': 1,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', nlp, opts_setting)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def organize_variables(self, dt, vx, vl, vm, cg):
        vs = ca.vertcat(vx, vl)
        vs = ca.vertcat(vs, vm)
        x_ = ca.reshape(vs, -1, 1)
        X = ca.vertcat(dt, x_)
        G = ca.reshape(cg, -1, 1)

        return X, G

    def organize_bounds(self, x0, y0):
        lbx, ubx, lbg, ubg = self.init_bounds_OBCA(x0[:, 0], x0[:, -1])
        x_all = ca.vertcat(x0, y0)
        x0_ = ca.reshape(x_all, -1, 1)
        X0 = ca.vertcat(self.dt0, x0_)

        return X0, lbx, ubx, lbg, ubg

    def get_result_OBCA(self):
        op_dt = float(self.x_opt[0])
        cal_traj = ca.reshape(self.x_opt[1:], self.nx + (self.obst_num + 1) * 4, self.horizon)
        # cal_traj = ca.reshape(self.x_opt[1:], self.horizon, self.nx + (self.obst_num + 1) * 4).T
        op_controls = np.array(cal_traj[3:8, :])
        op_trajectories = np.array(cal_traj[:3, :])

        vl = np.array(cal_traj[8:8 + self.obst_num * 4, :])
        vm = np.array(cal_traj[8 + self.obst_num * 4:, :])
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
    ut.set_parameters(param)

    ut.reserve_footprint = True
    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = CasADi_MPC_OBCA()
    cmpc.set_parameters(param)

    # def initialize_saved_data():
    #     loadtraj = np.load("../data/saved_hybrid_a_star.npz")
    #     ref_traj = loadtraj["saved_traj"]
    #     loadmap = np.load("../data/saved_obmap.npz", allow_pickle=True)
    #     # ob1 = loadmap["pointmap"][0]
    #     ob2 = loadmap["pointmap"][1]
    #     # ob3 = loadmap["pointmap"][2]
    #     # ob = [ob1, ob2, ob3]
    #     ob = [ob2]
    #     ob_constraint_mat = loadmap["constraint_mat"]
    #     obst = []
    #     # obst.append(ob_constraint_mat[:4, :])
    #     obst.append(ob_constraint_mat[4:8, :])
    #     # obst.append(ob_constraint_mat[8:12, :])

    # return ref_traj, ob, obst

    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()

    cmpc.init_model_OBCA(ref_traj, shape, obst)
    op_dt, op_trajectories, op_controls, vl, vm = cmpc.get_result_OBCA()
    print("OBCA total time:{:.3f}s".format(time.time() - start_time))
    ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob)
