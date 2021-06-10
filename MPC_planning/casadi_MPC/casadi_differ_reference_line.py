import casadi as ca
import numpy as np
import time
from mpc_motion_plot import UTurnMPC
import yaml


class CasADi_MPC_differ:
    def __init__(self):
        self.base = 2.
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2

        self.nx = 8 + 1
        self.ng = 6 + 1
        self.obst_num = 0
        self.horizon = 0
        self.dt0 = 0.1
        self.model = None
        self.x_opt = None
        self.op_lambda0 = None
        self.op_mu0 = None
        self.op_control0 = None

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        self.v_max = 2.
        self.omega_max = ca.pi * 40 / 180
        self.a_max = 2.
        self.omega_rate_max = ca.pi * 60 / 180
        self.jerk_max = 3.
        self.lateral_error_max = 0.5
        self.heading_error_max = ca.pi * 40 / 180
        self.dmin = 0.

        self.v0 = 1.
        self.omega0 = 0.1
        self.v_end = 1.5
        self.omega_end = 0.5

    def init_dynamic_constraints(self, x, dt, x0):
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
            e_ = x[8, i]

            kappa_r = x0[4, i] / x0[3, i]
            yaw_r = x0[2, i]

            k1_dx = v_ * ca.cos(yaw_)
            k1_dy = v_ * ca.sin(yaw_)
            k1_dyaw = omega_
            k1_dv = a_
            k1_domega = omega_rate_
            k1_da = jerk_
            k1_de = v_ * (1 - e_ * kappa_r) * ca.tan(yaw_ - yaw_r)
            # k1_dpsi = k1_dyaw - v_ * ca.cos(psi_) * ca.tan(steer_) / (self.base - ca.tan(steer_) * e_)

            k2_dx = (v_ + 0.5 * dt * k1_dv) * ca.cos(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dy = (v_ + 0.5 * dt * k1_dv) * ca.sin(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dyaw = (omega_ + 0.5 * dt * k1_domega)
            k2_dv = a_ + 0.5 * dt * k1_da
            k2_domega = omega_rate_
            k2_da = jerk_
            k2_de = (v_ + 0.5 * dt * k1_dv) * (1 - (e_ + 0.5 * dt * k1_de) * kappa_r) * ca.tan(
                yaw_ + 0.5 * dt * k1_dyaw - yaw_r)

            k3_dx = (v_ + 0.5 * dt * k2_dv) * ca.cos(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dy = (v_ + 0.5 * dt * k2_dv) * ca.sin(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dyaw = (omega_ + 0.5 * dt * k2_domega)
            k3_dv = a_ + 0.5 * dt * k2_da
            k3_domega = omega_rate_
            k3_da = jerk_
            k3_de = (v_ + 0.5 * dt * k2_dv) * (1 - (e_ + 0.5 * dt * k2_de) * kappa_r) * ca.tan(
                yaw_ + 0.5 * dt * k2_dyaw - yaw_r)

            k4_dx = (v_ + 0.5 * dt * k3_dv) * ca.cos(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dy = (v_ + 0.5 * dt * k3_dv) * ca.sin(yaw_ + 0.5 * dt * k3_dyaw)
            k4_dyaw = (omega_ + 0.5 * dt * k3_domega)
            k4_dv = a_ + 0.5 * dt * k3_da
            k4_domega = omega_rate_
            k4_da = jerk_
            k4_de = (v_ + 0.5 * dt * k3_dv) * (1 - (e_ + 0.5 * dt * k3_de) * kappa_r) * ca.tan(
                yaw_ + 0.5 * dt * k3_dyaw - yaw_r)

            dx = dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            dy = dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            dyaw = dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
            dv = dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6
            domega = dt * (k1_domega + 2 * k2_domega + 2 * k3_domega + k4_domega) / 6
            da = dt * (k1_da + 2 * k2_da + 2 * k3_da + k4_da) / 6
            de = dt * (k1_de + 2 * k2_de + 2 * k3_de + k4_de) / 6
            # dpsi = dt * (k1_dpsi + 2 * k2_dpsi + 2 * k3_dpsi + k4_dpsi) / 6

            g1[0, i] = x_ + dx - x[0, i + 1]
            g1[1, i] = y_ + dy - x[1, i + 1]
            g1[2, i] = yaw_ + dyaw - x[2, i + 1]
            g1[3, i] = v_ + dv - x[3, i + 1]
            g1[4, i] = omega_ + domega - x[4, i + 1]
            g1[5, i] = a_ + da - x[5, i + 1]
            g1[6, i] = e_ + de - x[8, i + 1]
            # g1[7, i] = dy * ca.cos(yaw_) - dx * ca.sin(yaw_)

        return g1

    def init_bounds_reference_line(self, refpath):
        lbx = ca.DM.zeros(self.nx, self.horizon)
        ubx = ca.DM.zeros(self.nx, self.horizon)
        lbg = ca.DM.zeros(self.ng, self.horizon - 1)
        ubg = ca.DM.zeros(self.ng, self.horizon - 1)

        for i in range(self.horizon):
            lbx[0, i] = -ca.inf
            lbx[1, i] = -ca.inf
            lbx[2, i] = -ca.pi / 2  # th
            lbx[3, i] = 0.  # v
            lbx[4, i] = -self.omega_max  # omega
            lbx[5, i] = -self.a_max  # a
            lbx[6, i] = -self.omega_rate_max  # omega_rate
            lbx[7, i] = -self.jerk_max  # jerk
            lbx[8, i] = -self.lateral_error_max

            ubx[0, i] = ca.inf
            ubx[1, i] = ca.inf
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.v_max  # v
            ubx[4, i] = self.omega_max  # omega
            ubx[5, i] = self.a_max  # a
            ubx[6, i] = self.omega_rate_max  # omega_rate
            ubx[7, i] = self.jerk_max  # jerk
            ubx[8, i] = self.lateral_error_max

        lbx[0, 0] = refpath[0, 0]
        lbx[1, 0] = refpath[1, 0]
        lbx[2, 0] = refpath[2, 0]
        lbx[3, 0] = self.v0
        lbx[4, 0] = self.omega0

        ubx[0, 0] = refpath[0, 0]
        ubx[1, 0] = refpath[1, 0]
        ubx[2, 0] = refpath[2, 0]
        ubx[3, 0] = self.v0
        ubx[4, 0] = self.omega0

        lbx[0, -1] = refpath[0, -1]
        lbx[1, -1] = refpath[1, -1]
        lbx[2, -1] = refpath[2, -1]
        lbx[3, -1] = 0.
        lbx[4, -1] = -self.omega_end

        ubx[0, -1] = refpath[0, -1]
        ubx[1, -1] = refpath[1, -1]
        ubx[2, -1] = refpath[2, -1]
        ubx[3, -1] = self.v_end
        ubx[4, -1] = self.omega_end

        # lbg[, :] = 0.
        # ubg[-1, :] = 1e-5

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)
        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def init_objects(self, x, ref_path):
        sum_total_dist = 0
        sum_dist_to_ref = 0
        sum_states = 0.
        sum_states_rate = 0.
        sum_controls = 0.
        sum_controls_rate = 0.
        sum_e = 0

        for i in range(self.horizon):

            sum_states += ca.sumsqr(x[:3, i])
            sum_controls += ca.sumsqr(x[3:8, i])
            sum_dist_to_ref += ca.sumsqr(x[:3, i] - ref_path[:3, i])
            sum_e += ca.power(x[8, i], 2)

            if i > 0:
                sum_total_dist += ca.sumsqr(x[:2, i] - x[:2, i - 1])
                sum_states_rate += ca.sumsqr(x[:3, i] - x[:3, i - 1])
                sum_controls_rate += ca.sumsqr(x[3:8, i] - x[3:8, i - 1])

        obj = self.wg[3] * sum_states + self.wg[9] * sum_states_rate \
              + self.wg[5] * sum_controls + self.wg[7] * sum_controls_rate \
              + self.wg[9] * sum_e  # + self.wg[9] * sum_total_dist
        # + self.wg[3] * sum_dist_to_ref
        # + 1e2 * self.wg[9] * ca.sumsqr(x[:3, -1] - ref_path[:3, -1])

        return obj

    def states_initialization(self, reference_path):
        x0 = ca.DM(self.nx, self.horizon)
        diff_s = np.diff(reference_path[:2, :], axis=1)
        sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
        self.dt0 = 1.4 * (sum_s / self.v_max + self.v_max / self.a_max) / self.horizon
        print("predicted_dt:", self.dt0)
        last_v = 0.
        last_a = 0.
        last_omega = 0.
        # states: x, y, yaw, v, omega, a, omega_rate, jerk, lateral
        for i in range(self.horizon):
            x0[0, i] = reference_path[0, i]
            x0[1, i] = reference_path[1, i]
            x0[2, i] = reference_path[2, i]

            if i > 0:
                ds = np.linalg.norm(reference_path[:2, i] - reference_path[:2, i - 1])
                dyaw = reference_path[2, i] - reference_path[2, i - 1]

                x0[3, i] = ds / self.dt0
                x0[4, i] = dyaw / self.dt0
                x0[5, i] = (ds / self.dt0 - last_v) / self.dt0
                x0[6, i] = (dyaw / self.dt0 - last_omega) / self.dt0
                x0[7, i] = ((ds / self.dt0 - last_v) / self.dt0 - last_a) / self.dt0

            last_v = x0[3, i]
            last_omega = x0[4, i]
            last_a = x0[5, i]

        if self.op_control0 is not None:
            x0[3:, :] = self.op_control0

        return x0

    def init_model_reference_line(self, reference_path):
        self.horizon = reference_path.shape[1]
        x0_ = self.states_initialization(reference_path)

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x, y, theta, v, steer, a, steer_rate, jerk, e, psi)
        dt = self.dt0

        # initialize constraints

        g1 = self.init_dynamic_constraints(x, dt, x0_)
        X = ca.reshape(x, -1, 1)
        G = ca.reshape(g1, -1, 1)

        # initialize objectives
        F = self.init_objects(x, x0_)

        nlp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": True,
                        "ipopt.hessian_approximation": "exact",
                        'ipopt.max_iter': 200,
                        'ipopt.print_level': 3,
                        'print_time': 1,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', nlp, opts_setting)

        XL, XU, GL, GU = self.init_bounds_reference_line(x0_)
        X0 = ca.reshape(x0_, -1, 1)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def get_result_reference_line(self):
        op_dt = float(self.dt0)
        cal_traj = ca.reshape(self.x_opt, self.nx, self.horizon)
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
    cmpc = CasADi_MPC_differ()

    ref_traj, ob, obst = ut.initialize_saved_data()
    ut.use_differ_motion = True
    cmpc.init_model_reference_line(ref_traj)
    op_dt, op_trajectories, op_controls = cmpc.get_result_reference_line()
    print("OBCA total time:{:.3f}s".format(time.time() - start_time))
    ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob)
