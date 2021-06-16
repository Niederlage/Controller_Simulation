import casadi as ca
import numpy as np
import time
from gears.curve_fitting import Curve_Fitting
from mpc_motion_plot import UTurnMPC
import yaml


class CasADi_MPC_differ:
    def __init__(self):
        self.base = 2.
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2

        self.nx = 10
        self.ng = 8
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
        self.centripetal = 0.8

        self.v0 = 1.
        self.omega0 = 0.1
        self.v_end = 2.
        self.omega_end = 0.5

    def init_dynamic_constraints_cxcy(self, cx, cy, x, dt):
        g1 = ca.SX.sym("g1", self.ng, self.horizon - 1)
        s = 0.
        # states: x, y, yaw, v, omega, a, omega_rate, jerk, lateral_error, heading_error
        for i in range(self.horizon - 1):
            x_ = x[0, i]
            y_ = x[1, i]
            yaw_ = x[2, i]
            v_ = x[3, i]
            omega_ = x[4, i]
            a_ = x[5, i]
            omega_rate_ = x[6, i]
            jerk_ = x[7, i]
            e_y = x[8, i]
            e_th = x[9, i]

            k1_dx = v_ * ca.cos(yaw_)
            k1_dy = v_ * ca.sin(yaw_)
            k1_dyaw = omega_
            k1_dv = a_
            k1_domega = omega_rate_
            k1_da = jerk_
            # k1_de = v_ * (1 - e_ * kappa_r) * ca.tan(yaw_ - yaw_r)
            # k1_dpsi = k1_dyaw - v_ * ca.cos(psi_) * ca.tan(steer_) / (self.base - ca.tan(steer_) * e_)

            k2_dx = (v_ + 0.5 * dt * k1_dv) * ca.cos(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dy = (v_ + 0.5 * dt * k1_dv) * ca.sin(yaw_ + 0.5 * dt * k1_dyaw)
            k2_dyaw = (omega_ + 0.5 * dt * k1_domega)
            k2_dv = a_ + 0.5 * dt * k1_da
            k2_domega = omega_rate_
            k2_da = jerk_
            # k2_de = (v_ + 0.5 * dt * k1_dv) * (1 - (e_ + 0.5 * dt * k1_de) * kappa_r) * ca.tan(
            #     yaw_ + 0.5 * dt * k1_dyaw - yaw_r)

            k3_dx = (v_ + 0.5 * dt * k2_dv) * ca.cos(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dy = (v_ + 0.5 * dt * k2_dv) * ca.sin(yaw_ + 0.5 * dt * k2_dyaw)
            k3_dyaw = (omega_ + 0.5 * dt * k2_domega)
            k3_dv = a_ + 0.5 * dt * k2_da
            k3_domega = omega_rate_
            k3_da = jerk_
            # k3_de = (v_ + 0.5 * dt * k2_dv) * (1 - (e_ + 0.5 * dt * k2_de) * kappa_r) * ca.tan(
            #     yaw_ + 0.5 * dt * k2_dyaw - yaw_r)

            k4_dx = (v_ + dt * k3_dv) * ca.cos(yaw_ + dt * k3_dyaw)
            k4_dy = (v_ + dt * k3_dv) * ca.sin(yaw_ + dt * k3_dyaw)
            k4_dyaw = (omega_ + dt * k3_domega)
            k4_dv = a_ + dt * k3_da
            k4_domega = omega_rate_
            k4_da = jerk_
            # k4_de = (v_ + dt * k3_dv) * (1 - (e_ + dt * k3_de) * kappa_r) * ca.tan(
            #     yaw_ + dt * k3_dyaw - yaw_r)

            dx = dt * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx) / 6
            dy = dt * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy) / 6
            dyaw = dt * (k1_dyaw + 2 * k2_dyaw + 2 * k3_dyaw + k4_dyaw) / 6
            dv = dt * (k1_dv + 2 * k2_dv + 2 * k3_dv + k4_dv) / 6
            domega = dt * (k1_domega + 2 * k2_domega + 2 * k3_domega + k4_domega) / 6
            da = dt * (k1_da + 2 * k2_da + 2 * k3_da + k4_da) / 6
            # de = dt * (k1_de + 2 * k2_de + 2 * k3_de + k4_de) / 6

            s += ca.sqrt(ca.power(dx, 2) + ca.power(dy, 2))
            dx_predict = 3 * cx[3] * ca.power(s, 2) + 2 * cx[2] * s + cx[1]
            dy_predict = 3 * cy[3] * ca.power(s, 2) + 2 * cy[2] * s + cy[1]
            py_ref = cy[3] * ca.power(s, 3) + cy[2] * ca.power(s, 2) + cy[1] * s + cy[0]
            pyaw_ref = ca.atan2(dy_predict, dx_predict)

            de_y = py_ref - (y_ + dy)
            de_th = pyaw_ref - (dyaw + yaw_)

            g1[0, i] = x_ + dx - x[0, i + 1]
            g1[1, i] = y_ + dy - x[1, i + 1]
            g1[2, i] = yaw_ + dyaw - x[2, i + 1]
            g1[3, i] = v_ + dv - x[3, i + 1]
            g1[4, i] = omega_ + domega - x[4, i + 1]
            g1[5, i] = a_ + da - x[5, i + 1]
            g1[6, i] = e_y + de_y - x[8, i + 1]
            g1[7, i] = e_th + de_th - x[9, i + 1]

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
            lbx[9, i] = -self.heading_error_max

            ubx[0, i] = ca.inf
            ubx[1, i] = ca.inf
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.v_max  # v
            ubx[4, i] = self.omega_max  # omega
            ubx[5, i] = self.a_max  # a
            ubx[6, i] = self.omega_rate_max  # omega_rate
            ubx[7, i] = self.jerk_max  # jerk
            ubx[8, i] = self.lateral_error_max
            ubx[9, i] = self.heading_error_max

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

        lbx[0, -1] = -ca.inf
        lbx[1, -1] = -ca.inf
        lbx[2, -1] = -ca.pi
        lbx[3, -1] = 0.
        lbx[4, -1] = -self.omega_end

        ubx[0, -1] = ca.inf
        ubx[1, -1] = ca.inf
        ubx[2, -1] = ca.pi
        ubx[3, -1] = self.v_end
        ubx[4, -1] = self.omega_end

        lbg[:, :] = 0.
        ubg[:, :] = 1e-5

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)
        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def init_objects(self, x):
        sum_states_rate = 0.
        sum_controls = 0.
        sum_controls_rate = 0.
        sum_error = 0
        sum_turning_rate = 0.

        # states: x, y, yaw, v, omega, a, omega_rate, jerk, lateral_error, heading_error
        for i in range(self.horizon):
            sum_controls += ca.sumsqr(x[3:8, i])
            sum_error += ca.sumsqr(x[8:, i])
            sum_turning_rate += ca.power(x[3, i] * x[4, i] - self.centripetal, 2)
            if i > 0:
                sum_states_rate += ca.sumsqr(x[:3, i] - x[:3, i - 1])
                sum_controls_rate += ca.sumsqr(x[3:8, i] - x[3:8, i - 1])

        obj = self.wg[5] * sum_states_rate \
              + self.wg[5] * sum_controls \
              + self.wg[7] * sum_controls_rate \
              + self.wg[9] * sum_error \
              + self.wg[9] * sum_turning_rate

        return obj

    def states_initialization(self, reference_path):
        x0 = ca.DM(self.nx, self.horizon)
        x0[0, 0] = reference_path[0, 0]
        x0[1, 0] = reference_path[1, 0]
        x0[2, 0] = reference_path[2, 0]

        return x0

    def init_model_reference_line(self, cx, cy, reference_path):
        x0_ = self.states_initialization(reference_path)

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (x, y, theta, v, steer, a, steer_rate, jerk, e, psi)
        dt = self.dt0

        # initialize constraints
        gx = self.init_dynamic_constraints_cxcy(cx, cy, x, dt)
        X = ca.reshape(x, -1, 1)
        G = ca.reshape(gx, -1, 1)

        # initialize objectives
        F = self.init_objects(x)

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
        X0 = ca.reshape(x0_, -1, 1)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def get_result_reference_line(self):
        cal_traj = ca.reshape(self.x_opt, self.nx, self.horizon)
        op_controls = np.array(cal_traj[3:8, :])
        op_trajectories = np.array(cal_traj[:3, :])
        op_error = np.array(cal_traj[8:, :])

        return op_trajectories, op_controls, op_error


if __name__ == '__main__':
    start_time = time.time()

    address = "../../config_differ_smoother.yaml"

    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    calco = Curve_Fitting()
    ut = UTurnMPC()
    ut.set_parameters(param)
    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = CasADi_MPC_differ()

    ref_traj, ob, obst = ut.initialize_saved_data()
    coeffx, coeffy, coefft, s0 = calco.cal_coefficient(ref_traj)
    ut.use_differ_motion = True
    cmpc.dt0 = 0.1
    cmpc.horizon = int(10 / cmpc.dt0)

    cmpc.init_model_reference_line(coeffx, coeffy, ref_traj)
    op_trajectories, op_controls, op_error = cmpc.get_result_reference_line()
    print("MPC total time:{:.3f}s".format(time.time() - start_time))
    ut.plot_results(cmpc.dt0, op_trajectories, op_controls, ref_traj, None)
