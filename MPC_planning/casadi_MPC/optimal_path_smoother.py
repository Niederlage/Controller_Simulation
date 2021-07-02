import casadi as ca
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt


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
        self.dt0 = 0.6
        self.model = None
        self.x_opt = None
        self.op_lambda0 = None
        self.op_mu0 = None
        self.op_control0 = None

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        self.v_max = 2.
        self.a_max = 3.
        self.jerk_max = 4.
        self.snap_max = 5.

        self.steer_max = ca.pi * 40 / 180
        self.omega_max = ca.pi
        self.steer_rate_max = ca.pi * 60 / 180

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
        sum_s = self.get_sum_s(refpath)
        lbx = ca.DM.zeros(self.nx, self.horizon)
        ubx = ca.DM.zeros(self.nx, self.horizon)
        lbg = ca.DM.zeros(self.ng, self.horizon - 1)
        ubg = ca.DM.zeros(self.ng, self.horizon - 1)
        # bounds = self.get_fading_bounds(refpath)

        for i in range(self.horizon):
            lbx[0, i] = 0.
            lbx[1, i] = -self.v_max
            lbx[2, i] = -self.a_max  # th
            lbx[3, i] = -self.jerk_max
            lbx[4, i] = -self.snap_max

            ubx[0, i] = ca.inf
            ubx[1, i] = self.v_max
            ubx[2, i] = self.a_max  # th
            ubx[3, i] = self.jerk_max  # v
            ubx[4, i] = self.snap_max  # steer

        lbx[:, 0] = 0.

        ubx[:, 0] = 0.

        lbx[0, -1] = sum_s
        lbx[1:, -1] = 0

        ubx[0, -1] = sum_s
        ubx[1:, -1] = 0.

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)
        lct = -ca.DM_inf(6)
        uct = ca.DM_inf(6)
        lbx_ = ca.vertcat(lct, lbx_)
        ubx_ = ca.vertcat(uct, ubx_)
        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def init_objects(self, x, ct, ref_path):

        # sum_time = 0
        sum_error_theta = 0.
        sum_states_rate = 0.

        for i in range(self.horizon):
            s_ = x[0, i]
            th_s = ct[0] + ct[1] * s_ + ct[2] * ca.power(s_, 2) + ct[3] * ca.power(s_, 3) + ct[4] * ca.power(s_, 4) + \
                   ct[5] * ca.power(s_, 5)
            sum_error_theta += ca.sumsqr(th_s - ref_path[2, i])
            if i > 0:
                sum_states_rate += ca.sumsqr(x[1:, i] - x[1:, i - 1])

        obj = 1e3 * self.wg[9] * sum_error_theta + self.wg[2] * sum_states_rate
        # + self.wg[3] * sum_dist_to_ref
        # + 1e2 * self.wg[9] * ca.sumsqr(x[:3, -1] - ref_path[:3, -1])

        return obj

    def init_dt(self, refpath):
        sum_s = self.get_sum_s(refpath)
        return int(1.2 * (sum_s / self.v_max + self.v_max / self.a_max) / self.horizon)

    def get_sum_s(self, refpath):
        diff_s = np.diff(refpath[:2, :], axis=1)
        return np.sum(np.hypot(diff_s[0], diff_s[1]))

    def states_initialization(self, reference_path):
        x0 = ca.DM(self.nx, self.horizon)
        ki = 3
        ds = np.linalg.norm(reference_path[:2, ki] - reference_path[:2, ki - 1])

        for i in range(self.horizon):
            x0[0, i] = ds * i

        return x0

    def init_model_path_smoother(self, reference_path):
        self.horizon = reference_path.shape[1]
        x0_ = self.states_initialization(reference_path)

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x, y, theta, v, steer, a, steer_rate, jerk, e, psi)
        # self.dt0 = self.init_dt(reference_path)
        dt = self.dt0
        ctheta = ca.SX.sym("ct", 6)

        # initialize constraints
        g1 = self.init_dynamic_constraints(x, dt)
        G = ca.reshape(g1, -1, 1)

        # initialize objectives
        F = self.init_objects(x, ctheta, reference_path)

        x_ = ca.reshape(x, -1, 1)
        X = ca.vertcat(ctheta, x_)

        nlp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": True,
                        "ipopt.hessian_approximation": "exact",
                        'ipopt.max_iter': 100,
                        'ipopt.print_level': 4,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', nlp, opts_setting)

        XL, XU, GL, GU = self.init_bounds_reference_line(x0_)
        x0_ = ca.reshape(x0_, -1, 1)
        X0 = ca.vertcat(ca.DM.ones(6), x0_)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def get_result_reference_line(self):
        cal_coeff = self.x_opt[:6]
        cal_traj = ca.reshape(self.x_opt[6:], self.nx, self.horizon)

        return np.array(cal_coeff).flatten(), np.array(cal_traj)


def get_calc_theta(ct, ds, total_s):
    yawlist = []
    Ns = int(total_s / ds)
    for i in range(Ns):
        s_ = ds * i
        yaw_i = ct[0] + ct[1] * s_ + ct[2] * np.power(s_, 2) + ct[3] * np.power(s_, 3) + ct[4] * np.power(s_, 4) + \
                ct[5] * np.power(s_, 5)
        yawlist.append(yaw_i)
    return np.array(yawlist)


if __name__ == '__main__':
    start_time = time.time()
    large = True
    if large:
        address = "../config_OBCA_large.yaml"
    else:
        address = "../config_OBCA.yaml"
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = OptimalPathSmoother()
    loads = np.load("../data/smoothed_traj.npz")
    ref_traj = loads["refpath"]

    cmpc.init_model_path_smoother(ref_traj)
    coeff, trajs = cmpc.get_result_reference_line()

    print("optimize total time:{:.3f}s\n\n".format(time.time() - start_time))
    print("coeff:", coeff)
    f = plt.figure()
    plt.plot(trajs[0, :], label="s")
    plt.plot(trajs[1, :], label="v")
    plt.plot(trajs[2, :], label="a")
    plt.plot(trajs[3, :], label="jerk")
    plt.plot(trajs[4, :], label="snap")
    plt.legend()
    plt.grid()
    cal_yaw = get_calc_theta(coeff, 0.2, trajs[0, -1])

    f2 = plt.figure()
    ax = plt.subplot(211)
    ax.plot(cal_yaw, label="fitting", color="red")
    ax = plt.subplot(212)
    ax.plot(ref_traj[2, :], label="reference", color="blue")
    ax.grid()
    ax.legend()
    plt.show()
