import casadi as ca
import numpy as np
import time
from mpc_motion_plot import UTurnMPC
import yaml


class OptimalPathSmoother:
    def __init__(self):
        self.base = 2.
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2

        self.nx = 5
        self.ng = 4
        self.obst_num = 0
        self.horizon = 0
        self.dt0 = 0.01
        self.model = None
        self.x_opt = None
        self.op_lambda0 = None
        self.op_mu0 = None
        self.op_control0 = None

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        self.v_max = 2.
        self.steer_max = ca.pi * 40 / 180
        self.omega_max = ca.pi
        self.a_max = 3.
        self.steer_rate_max = ca.pi * 60 / 180
        self.jerk_max = 3.
        self.lateral_error_max = 1.
        self.heading_error_max = ca.pi * 40 / 180
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

    def init_dynamic_constraints(self, x, dt):
        gx = ca.SX.sym("g1", self.ng, self.horizon - 1)
        for i in range(self.horizon - 1):
            s_ = x[0, i]
            s_1 = x[1, i]
            s_2 = x[2, i]
            s_3 = x[3, i]
            s_4 = x[4, i]

            k1_ds = s_1
            k1_d1s = s_2
            k1_d2s = s_3
            k1_d3s = s_4

            k2_ds = s_1 + 0.5 * dt * k1_d1s
            k2_d1s = s_2 + 0.5 * dt * k1_d2s
            k2_d2s = s_3 + 0.5 * dt * k1_d3s
            k2_d3s = s_4

            k3_ds = s_1 + 0.5 * dt * k2_d1s
            k3_d1s = s_2 + 0.5 * dt * k2_d2s
            k3_d2s = s_3 + 0.5 * dt * k2_d3s
            k3_d3s = s_4

            k4_ds = s_1 + 0.5 * dt * k3_d1s
            k4_d1s = s_2 + 0.5 * dt * k3_d2s
            k4_d2s = s_3 + 0.5 * dt * k3_d3s
            k4_d3s = s_4

            ds = dt * (k1_ds + 2 * k2_ds + 2 * k3_ds + k4_ds) / 6
            d1s = dt * (k1_d1s + 2 * k2_d1s + 2 * k3_d1s + k4_d1s) / 6
            d2s = dt * (k1_d2s + 2 * k2_d2s + 2 * k3_d2s + k4_d2s) / 6
            d3s = dt * (k1_d3s + 2 * k2_d3s + 2 * k3_d3s + k4_d3s) / 6

            gx[0, i] = s_ + ds - x[0, i + 1]
            gx[1, i] = s_1 + d1s - x[1, i + 1]
            gx[2, i] = s_2 + d2s - x[2, i + 1]
            gx[3, i] = s_3 + d3s - x[3, i + 1]

        return gx

    def init_path_constraints(self, x, refpath, ct, ck):
        gx = ca.SX.sym("g2", self.ng, self.horizon - 1)
        for i in range(self.horizon - 1):
            s_ = x[0, i]

            gx[0, i] = ct[0] * s_ + ct[1] * ca.power(s_, 2) + ct[2] * ca.power(s_, 3) + ct[3] * ca.power(s_, 4) + ct[
                4] * ca.power(s_, 5) - refpath[2, i]

        return gx

    def init_bounds_reference_line(self, refpath):
        lbx = ca.DM.zeros(self.nx, self.horizon)
        ubx = ca.DM.zeros(self.nx, self.horizon)
        lbg = ca.DM.zeros(self.ng, self.horizon - 1)
        ubg = ca.DM.zeros(self.ng, self.horizon - 1)
        # bounds = self.get_fading_bounds(refpath)

        for i in range(self.horizon):
            lbx[0, i] = -ca.inf
            lbx[1, i] = -ca.inf
            lbx[2, i] = -ca.pi / 2  # th
            lbx[3, i] = -self.v_max  # v
            lbx[4, i] = -self.steer_max  # steer
            lbx[5, i] = -self.a_max  # a
            lbx[6, i] = -self.steer_rate_max  # steer_rate
            lbx[7, i] = -self.jerk_max  # jerk
            lbx[8, i] = -self.lateral_error_max

            ubx[0, i] = ca.inf
            ubx[1, i] = ca.inf
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.v_max  # v
            ubx[4, i] = self.steer_max  # steer
            ubx[5, i] = self.a_max  # a
            ubx[6, i] = self.steer_rate_max  # steer_rate
            ubx[7, i] = self.jerk_max  # jerk
            ubx[8, i] = self.lateral_error_max

        lbx[0, 0] = refpath[0, 0]
        lbx[1, 0] = refpath[1, 0]
        lbx[2, 0] = refpath[2, 0]
        lbx[3:, 0] = 0.

        ubx[0, 0] = refpath[0, 0]
        ubx[1, 0] = refpath[1, 0]
        ubx[2, 0] = refpath[2, 0]
        ubx[3:, 0] = 0.

        lbx[0, -1] = refpath[0, -1]
        lbx[1, -1] = refpath[1, -1]
        lbx[2, -1] = refpath[2, -1]
        lbx[3:, -1] = 0.

        ubx[0, -1] = refpath[0, -1]
        ubx[1, -1] = refpath[1, -1]
        ubx[2, -1] = refpath[2, -1]
        ubx[3:, -1] = 0.

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)
        lbx_ = ca.vertcat(0.01, lbx_)
        ubx_ = ca.vertcat(0.2, ubx_)

        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def init_objects(self, x, ct, ref_path):

        # sum_time = 0
        sum_error_theta = 0.
        sum_states_rate = 0.
        # sum_controls = 0.
        # sum_controls_rate = 0.
        # sum_e = 0

        for i in range(self.horizon):
            s_ = x[0, i]
            th_s = ct[0] + ct[1] * s_ + ct[2] * ca.power(s_, 2) + ct[3] * ca.power(s_, 3) + ct[4] * ca.power(s_, 4) + \
                   ct[5] * ca.power(s_, 5)
            sum_error_theta += ca.sumsqr(th_s - ref_path[2, i])
            if i > 0:
                sum_states_rate += ca.sumsqr(x[1:, i] - x[1:, i - 1])

        obj = self.wg[9] * sum_error_theta + self.wg[2] * sum_states_rate
        # + self.wg[3] * sum_dist_to_ref
        # + 1e2 * self.wg[9] * ca.sumsqr(x[:3, -1] - ref_path[:3, -1])

        return obj

    def init_horizon(self, refpath):
        diff_s = np.diff(refpath[:2, :], axis=1)
        sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
        return int(1.2 * (sum_s / self.v_max + self.v_max / self.a_max) / self.dt0)

    def states_initialization(self, reference_path):
        x0 = ca.DM(self.nx, self.horizon)

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

    def init_model_path_smoother(self, reference_path):
        self.horizon = self.init_horizon(reference_path)
        # x0_ = self.states_initialization(reference_path)

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x, y, theta, v, steer, a, steer_rate, jerk, e, psi)
        dt = self.dt0
        ctheta = ca.SX.sym("ct", 6)

        # initialize constraints
        g1 = self.init_dynamic_constraints(x, dt)
        x_ = ca.reshape(x, -1, 1)
        X = ca.vertcat(dt, x_)
        G = ca.reshape(g1, -1, 1)

        # initialize objectives
        F = self.init_objects(x, ctheta, reference_path)

        nlp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": True,
                        "ipopt.hessian_approximation": "exact",
                        'ipopt.max_iter': 100,
                        'ipopt.print_level': 3,
                        'print_time': 1,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', nlp, opts_setting)

        XL, XU, GL, GU = self.init_bounds_reference_line(x0_)
        x0_ = ca.reshape(x0_, -1, 1)
        X0 = ca.vertcat(self.dt0, x0_)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def get_result_reference_line(self):
        op_dt = float(self.x_opt[0])
        cal_traj = ca.reshape(self.x_opt[1:], self.nx, self.horizon)
        op_controls = np.array(cal_traj[3:, :])
        op_trajectories = np.array(cal_traj[:3, :])

        return op_dt, op_trajectories, op_controls


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
    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = OptimalPathSmoother()
    cmpc.set_parameters(param)
    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()

    cmpc.init_model_path_smoother(ref_traj)
    op_dt, op_trajectories, op_controls = cmpc.get_result_reference_line()
    print("OBCA total time:{:.3f}s".format(time.time() - start_time))
    ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob)
