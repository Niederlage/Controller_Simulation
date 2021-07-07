import casadi as ca
import numpy as np
import time
from motion_plot.ackermann_motion_plot import UTurnMPC
from gears.cubic_spline_planner import calc_spline_course
import yaml


class CasADi_MPC_differ_TDROBCA:
    def __init__(self):
        self.base = 2.0
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2

        self.nx = 8
        self.ng = 6 + 1
        self.obst_num = 0
        self.horizon = 0
        self.dt0 = 0.1
        self.model = None
        self.x_opt = None
        self.op_lambda0 = None
        self.op_mu0 = None
        self.op_d0 = None
        self.op_control0 = None
        self.sides = 4
        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        self.x_max = 20.
        self.y_max = 20.
        self.v_max = 5.
        self.omega_max = ca.pi * 40 / 180
        self.steer_max = ca.pi * 60 / 180
        self.a_max = 10.
        self.omega_rate_max = ca.pi * 120 / 180
        self.jerk_max = 10.
        self.dmin = 0.
        self.optimize_dt = True
        self.reduced_states = False

    def set_parameters(self, param):
        self.base = param["base"]
        self.LF = param["LF"]  # distance from rear to vehicle front end
        self.LB = param["LB"]  # distance from rear to vehicle back end
        self.x_max = param["x_max"]
        self.y_max = param["y_max"]

    def normalize_angle(self, yaw):
        return (yaw + np.pi) % (2 * np.pi) - np.pi

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

    def init_dynamic_constraints(self, x, dt):
        g1 = ca.SX.sym("g1", self.ng, self.horizon - 1)
        # states: x, y, yaw, v, omega, a, omega_rate, jerk, lateral
        for i in range(self.horizon - 1):
            x_ = x[0, i]
            y_ = x[1, i]
            yaw_ = x[2, i]
            v_ = x[3, i]
            omega_ = x[4, i]
            a_ = x[5, i]
            omega_rate_ = x[6, i]
            jerk_ = x[7, i]

            dx = v_ * ca.cos(yaw_) * dt
            dy = v_ * ca.sin(yaw_) * dt
            dyaw = omega_ * dt
            dv = a_ * dt
            domega = omega_rate_ * dt
            da = jerk_ * dt

            # k1_dx = v_ * ca.cos(yaw_)
            # k1_dy = v_ * ca.sin(yaw_)
            # k1_dyaw = omega_
            # k1_dv = a_
            # k1_domega = omega_rate_
            # k1_da = jerk_
            # # k1_de = v_ * (1 - e_ * kappa_r) * ca.tan(yaw_ - yaw_r)
            # # k1_dpsi = k1_dyaw - v_ * ca.cos(psi_) * ca.tan(steer_) / (self.base - ca.tan(steer_) * e_)
            #
            # k2_dx = (v_ + 0.5 * dt * k1_dv) * ca.cos(yaw_ + 0.5 * dt * k1_dyaw)
            # k2_dy = (v_ + 0.5 * dt * k1_dv) * ca.sin(yaw_ + 0.5 * dt * k1_dyaw)
            # k2_dyaw = (omega_ + 0.5 * dt * k1_domega)
            # k2_dv = a_ + 0.5 * dt * k1_da
            # k2_domega = omega_rate_
            # k2_da = jerk_
            # # k2_de = (v_ + 0.5 * dt * k1_dv) * (1 - (e_ + 0.5 * dt * k1_de) * kappa_r) * ca.tan(
            # #     yaw_ + 0.5 * dt * k1_dyaw - yaw_r)
            #
            # k3_dx = (v_ + 0.5 * dt * k2_dv) * ca.cos(yaw_ + 0.5 * dt * k2_dyaw)
            # k3_dy = (v_ + 0.5 * dt * k2_dv) * ca.sin(yaw_ + 0.5 * dt * k2_dyaw)
            # k3_dyaw = (omega_ + 0.5 * dt * k2_domega)
            # k3_dv = a_ + 0.5 * dt * k2_da
            # k3_domega = omega_rate_
            # k3_da = jerk_
            # # k3_de = (v_ + 0.5 * dt * k2_dv) * (1 - (e_ + 0.5 * dt * k2_de) * kappa_r) * ca.tan(
            # #     yaw_ + 0.5 * dt * k2_dyaw - yaw_r)
            #
            # k4_dx = (v_ + 0.5 * dt * k3_dv) * ca.cos(yaw_ + 0.5 * dt * k3_dyaw)
            # k4_dy = (v_ + 0.5 * dt * k3_dv) * ca.sin(yaw_ + 0.5 * dt * k3_dyaw)
            # k4_dyaw = (omega_ + 0.5 * dt * k3_domega)
            # k4_dv = a_ + 0.5 * dt * k3_da
            # k4_domega = omega_rate_
            # k4_da = jerk_
            # # k4_de = (v_ + 0.5 * dt * k3_dv) * (1 - (e_ + 0.5 * dt * k3_de) * kappa_r) * ca.tan(
            # #     yaw_ + 0.5 * dt * k3_dyaw - yaw_r)
            #
            # dx = dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            # dy = dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            # dyaw = dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
            # dv = dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6
            # domega = dt * (k1_domega + 2 * k2_domega + 2 * k3_domega + k4_domega) / 6
            # da = dt * (k1_da + 2 * k2_da + 2 * k3_da + k4_da) / 6
            # # de = dt * (k1_de + 2 * k2_de + 2 * k3_de + k4_de) / 6
            # # dpsi = dt * (k1_dpsi + 2 * k2_dpsi + 2 * k3_dpsi + k4_dpsi) / 6

            g1[0, i] = x_ + dx - x[0, i + 1]
            g1[1, i] = y_ + dy - x[1, i + 1]
            g1[2, i] = yaw_ + dyaw - x[2, i + 1]
            g1[3, i] = v_ + dv - x[3, i + 1]
            g1[4, i] = omega_ + domega - x[4, i + 1]
            g1[5, i] = a_ + da - x[5, i + 1]
            g1[6, i] = ca.atan2(self.base * omega_, v_)
            # g1[6, i] = e_ + de - x[8, i + 1]
            # g1[7, i] = dy * ca.cos(yaw_) - dx * ca.sin(yaw_)

        return g1

    def init_OBCA_constraints(self, x_, lambda_o, mu_o, d_, shape, obst):
        g2 = ca.SX.sym("g2", self.obst_num * self.sides, self.horizon - 1)
        gT = self.Array2SX(shape[:, 2][:, None].T)
        GT = self.Array2SX(shape[:, :2].T)

        lambda_v = ca.reshape(lambda_o, -1, 1)
        lambda_ = ca.reshape(lambda_v, self.horizon, self.sides * self.obst_num).T
        mu_v = ca.reshape(mu_o, -1, 1)
        mu_ = ca.reshape(mu_v, self.horizon, self.sides).T
        # G = ca.SX.zeros(self.sides, 2)
        # G[0, 0] = 1
        # G[1, 0] = -1
        # G[2, 1] = 1
        # G[3, 1] = -1
        #
        # g = ca.SX.zeros(self.sides, 1)
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
            t_i = x_[:2, i] + offset * 0

            rotT_i = ca.SX.zeros(2, 2)
            rotT_i[0, 0] = ca.cos(yaw_i)
            rotT_i[1, 1] = ca.cos(yaw_i)
            rotT_i[1, 0] = -ca.sin(yaw_i)
            rotT_i[0, 1] = ca.sin(yaw_i)

            for j in range(l_ob):
                Aj = self.Array2SX(obst[j][:, :2])
                bj = self.Array2SX(obst[j][:, 2][:, None])
                lambdaj = lambda_[(self.sides * j):(self.sides * (j + 1)), i]

                constraint1 = rotT_i @ Aj.T @ lambdaj + GT @ mu_i
                constraint2 = (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i + d_[j, i]
                constraint3 = ca.sumsqr(Aj.T @ lambdaj)

                g2[j, i] = constraint1[0, 0]
                g2[j + l_ob, i] = constraint1[1, 0]
                g2[j + 2 * l_ob, i] = constraint2
                g2[j + 3 * l_ob, i] = constraint3

        return g2

    def init_objects(self, dt, x_, d_, ref_path):
        sum_time = 0
        sum_mindist = 0
        sum_reference = 0
        sum_v = 0
        sum_omega = 0
        sum_v_rate = 0.
        sum_omega_rate = 0.

        for i in range(self.horizon):
            if self.optimize_dt:
                sum_time += ca.power(dt, 2)
            sum_reference = ca.sumsqr(x_[:3, i] - ref_path[:3, i])
            sum_mindist += ca.sumsqr(d_[:, i])
            sum_v += ca.sumsqr(x_[3, i])
            sum_omega += ca.sumsqr(x_[4, i])
            if i > 0:
                sum_v_rate += ca.sumsqr((x_[3, i] - x_[3, i - 1]) / dt)
                sum_omega_rate += ca.sumsqr((x_[4, i] - x_[4, i - 1]) / dt)

        obj = - 9 * self.wg[8] * sum_mindist \
              + self.wg[1] * sum_reference \
              + 5 * self.wg[6] * sum_time \
              + self.wg[1] * sum_v \
              + self.wg[1] * sum_omega \
              + self.wg[5] * sum_v_rate + \
              + self.wg[7] * sum_omega_rate
        return obj

    def init_bounds_OBCA2(self, refpath):

        lbx = ca.DM.zeros(self.nx, self.horizon)
        lblambda = ca.DM.zeros(self.obst_num * self.sides, self.horizon)
        lbmu = ca.DM.zeros(self.sides, self.horizon)
        lbd = ca.DM.zeros(self.obst_num, self.horizon)

        ubx = ca.DM.zeros(self.nx, self.horizon)
        ublambda = ca.DM.zeros(self.obst_num * self.sides, self.horizon)
        ubmu = ca.DM.zeros(self.sides, self.horizon)
        ubd = ca.DM.zeros(self.obst_num, self.horizon)

        lbg = ca.DM.zeros(self.ng, self.horizon - 1)
        lbobca = ca.DM.zeros(self.sides * self.obst_num, self.horizon - 1)
        ubg = ca.DM.zeros(self.ng, self.horizon - 1)
        ubobca = ca.DM.zeros(self.sides * self.obst_num, self.horizon - 1)

        for i in range(self.horizon):
            lbx[0, i] = -self.x_max  # x
            ubx[0, i] = self.x_max  # x
            lbx[1, i] = -self.y_max  # y
            ubx[1, i] = self.y_max  # 1.1y
            lbx[2, i] = -ca.pi  # th
            ubx[2, i] = ca.pi  # th
            lbx[3, i] = -self.v_max  # v
            ubx[3, i] = self.v_max  # v
            lbx[4, i] = -self.omega_max  # steer
            ubx[4, i] = self.omega_max  # steer

            if not self.reduced_states:
                lbx[5, i] = -self.a_max  # a
                ubx[5, i] = self.a_max  # a
                lbx[6, i] = -self.omega_rate_max  # steer_rate
                ubx[6, i] = self.omega_rate_max  # steer_rate
                lbx[7, i] = -self.jerk_max  # jerk
                ubx[7, i] = self.jerk_max  # jerk

            lblambda[:, i] = 1e-8  # lambda, mu
            lbmu[:, i] = 1e-8
            lbd[:, i] = -ca.inf
            ublambda[:, i] = ca.inf  # lambda, mu
            ubmu[:, i] = ca.inf
            ubd[:, i] = -1e-8

        lbg[:, :] = -1e-5
        ubg[:, :] = 1e-5
        lbg[6, :] = -self.steer_max
        ubg[6, :] = self.steer_max

        lbobca[:2 * self.obst_num, :] = 0.
        ubobca[:2 * self.obst_num, :] = 1e-5
        lbobca[2 * self.obst_num:3 * self.obst_num, :] = 0.
        ubobca[2 * self.obst_num:3 * self.obst_num, :] = 0.
        lbobca[3 * self.obst_num:4 * self.obst_num, :] = 0.
        ubobca[3 * self.obst_num:4 * self.obst_num, :] = 1.

        lbx[0, 0] = refpath[0, 0]
        lbx[1, 0] = refpath[1, 0]
        lbx[2, 0] = refpath[2, 0]
        lbx[3:, 0] = 0.

        ubx[0, 0] = refpath[0, 0]
        ubx[1, 0] = refpath[1, 0]
        ubx[2, 0] = refpath[2, 0]
        ubx[3:, 0] = 0.

        lbx[0, -1] = refpath[0, -1] - 0.2
        lbx[1, -1] = refpath[1, -1] - 0.2
        lbx[2, -1] = refpath[2, -1]
        lbx[3:, -1] = 0.

        ubx[0, -1] = refpath[0, -1] + 0.2
        ubx[1, -1] = refpath[1, -1] + 0.2
        ubx[2, -1] = refpath[2, -1]
        ubx[3:, -1] = 0.

        lbx_ = ca.vertcat(lbx, lblambda, lbmu, lbd)
        ubx_ = ca.vertcat(ubx, ublambda, ubmu, ubd)
        lbg_ = ca.vertcat(lbg, lbobca)
        ubg_ = ca.vertcat(ubg, ubobca)

        lbx_ = ca.reshape(lbx_, -1, 1)
        ubx_ = ca.reshape(ubx_, -1, 1)

        if self.optimize_dt:
            lbx_ = ca.vertcat(0.01, lbx_)
            ubx_ = ca.vertcat(0.2, ubx_)

        lbg_ = ca.reshape(lbg_, -1, 1)
        ubg_ = ca.reshape(ubg_, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def get_dt(self, ref_path):
        diff_s = np.diff(ref_path[:2, :], axis=1)
        sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
        self.dt0 = 1.5 * (sum_s / self.v_max + self.v_max / self.a_max) / len(ref_path.T)
        # self.dt0 = 0.1

    def get_total_sides(self, obst):
        num = 0
        for ob in obst:
            num += len(ob)
        return num

    def states_initialization(self, reference_path):
        x0 = ca.DM(self.nx, self.horizon)

        last_v = 0.
        last_a = 0.
        last_omega = 0.
        for i in range(self.horizon):
            x0[0, i] = reference_path[0, i]
            x0[1, i] = reference_path[1, i]
            x0[2, i] = reference_path[2, i]

            if i > 1:
                ds = np.linalg.norm(reference_path[:2, i] - reference_path[:2, i - 1])
                dyaw = reference_path[2, i] - reference_path[2, i - 1]
                x0[3, i] = ds / self.dt0
                x0[4, i] = dyaw / self.dt0
                if not self.reduced_states:
                    x0[5, i] = (ds / self.dt0 - last_v) / self.dt0
                    x0[6, i] = (dyaw / self.dt0 - last_omega) / self.dt0
                    x0[7, i] = ((ds / self.dt0 - last_v) / self.dt0 - last_a) / self.dt0
            if not self.reduced_states:
                last_v = x0[3, i]
                last_omega = x0[4, i]
                last_a = x0[5, i]
        # x0[2, 0] = reference_path[2, 0]
        # x0[2, -1] = reference_path[2, -1]
        if self.op_control0 is not None:
            x0[3:, :] = self.op_control0

        return x0

    def dual_variables_initialization(self):
        v0 = ca.DM.zeros(self.sides * (self.obst_num + 1) + self.obst_num, self.horizon)
        if (self.op_lambda0 is not None) and (self.op_mu0 is not None) and (self.op_d0 is not None):
            v0[:self.sides * self.obst_num, :] = self.Array2DM(self.op_lambda0)
            v0[self.sides * self.obst_num: -self.obst_num, :] = self.Array2DM(self.op_mu0)
            v0[-self.obst_num:, :] = self.Array2DM(self.op_d0)

        return v0

    def init_model_OBCA(self, reference_path, shape_m, obst_m):
        if self.reduced_states:
            self.nx = 5
            self.ng = 3

        self.horizon = reference_path.shape[1]
        self.obst_num = len(obst_m)
        # sides = self.get_total_sides(obst_m)
        x0_ = self.states_initialization(reference_path)
        v0 = self.dual_variables_initialization()

        # initialize variables
        if self.optimize_dt:
            dt = ca.SX.sym("dt")
        else:
            dt = self.dt0

        x = ca.SX.sym("x", self.nx, self.horizon)  # (x, y, theta, v, steer, a, steer_rate, jerk)
        lambda_ = ca.SX.sym("lambda", self.sides * self.obst_num, self.horizon)
        mu_ = ca.SX.sym("mu", self.sides, self.horizon)
        d_ = ca.SX.sym("d", self.obst_num, self.horizon)

        # initialize constraints
        g1 = self.init_dynamic_constraints(x, dt)
        g2 = self.init_OBCA_constraints(x, lambda_, mu_, d_, shape_m, obst_m)
        gx = ca.vertcat(g1, g2)

        X, G = self.organize_variables(x, lambda_, mu_, d_, gx)
        XL, XU, GL, GU = self.init_bounds_OBCA2(reference_path)
        x_all = ca.vertcat(x0_, v0)
        X0 = ca.reshape(x_all, -1, 1)

        if self.optimize_dt:
            X = ca.vertcat(dt, X)
            X0 = ca.vertcat(self.dt0, X0)

        # initialize objectives
        F = self.init_objects(dt, x, d_, x0_)

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

    def front_wheel_control(self, inputs):
        v_ = inputs[0, :]
        omega_ = inputs[1, :]
        steer_ = np.arctan2(self.base * omega_, v_)
        v_h = v_ / np.cos(steer_)

        return np.vstack((v_h, steer_))

    def get_result_OBCA(self):
        op_dt = float(self.dt0)
        startp = 0
        if self.optimize_dt:
            op_dt = float(self.x_opt[0])
            startp = 1

        cal_traj = ca.reshape(self.x_opt[startp:], self.nx + (self.obst_num + 1) * self.sides + self.obst_num,
                              self.horizon)
        op_trajectories = np.array(cal_traj[:3, :])
        op_controls = self.front_wheel_control(np.array(cal_traj[3:8, :]))

        vl = np.array(cal_traj[self.nx:self.nx + self.sides * self.obst_num, :])
        vm = np.array(
            cal_traj[self.nx + self.sides * self.obst_num:self.nx + self.sides * self.obst_num + self.sides, :])
        vd = np.array(cal_traj[self.nx + self.sides * self.obst_num + self.sides:, :])

        return op_dt, op_trajectories, op_controls, vl, vm, vd


def expand_path(refpath, ds):
    x = refpath[0, :]
    y = refpath[1, :]
    theta0 = refpath[2, 0]
    rx, ry, ryaw, rk, s = calc_spline_course(x, y, ds)

    return np.array([rx, ry, ryaw])


if __name__ == '__main__':
    start_time = time.time()

    large = True
    if large:
        address = "../../config_OBCA_large.yaml"
    else:
        address = "../config_OBCA.yaml"
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    ut = UTurnMPC()
    ut.reserve_footprint = True
    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = CasADi_MPC_differ_TDROBCA()
    cmpc.set_parameters(param)
    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()
    # cmpc.get_dt(ref_traj)

    ref_traj = expand_path(ref_traj, cmpc.dt0 * cmpc.v_max * 0.5)
    cmpc.init_model_OBCA(ref_traj, shape, obst)
    op_dt, op_trajectories, op_controls, vl, vm, vd = cmpc.get_result_OBCA()
    print("TDROBCA total time:{:.3f}s".format(time.time() - start_time))

    ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, four_states=True)
