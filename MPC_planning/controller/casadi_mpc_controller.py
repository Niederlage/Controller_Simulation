import casadi as ca
import numpy as np
import time
from gears.cubic_spline_planner import Spline2D
import yaml


class CasADi_MPC_differ_Control:
    def __init__(self):
        self.base = 2.
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2

        self.nx = 7
        self.ng = 5
        self.obst_num = 0
        self.horizon = 5
        self.dt0 = 0.1
        self.model = None
        self.x_opt = None
        self.op_lambda0 = None
        self.op_mu0 = None
        self.op_control0 = None

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
        self.v_max = 2.
        self.omega_max = ca.pi * 40 / 180
        self.a_max = 1.
        self.omega_rate_max = ca.pi * 120 / 180
        self.jerk_max = 3.
        self.lateral_error_max = 0.1
        self.heading_error_max = ca.pi * 40 / 180
        self.u_range = 1e-5

        self.v0 = 1.
        self.omega0 = 0.1
        self.v_end = 1.5
        self.omega_end = 0.5
        self.centripetal = 0.8

    def get_linear_model_matrix(self, k, state, refpath, dt):
        A_ = self.get_A_bk(k, refpath, dt)
        B_ = self.get_B_bk(k, refpath, dt)

        Q_ = ca.DM.eye(5 * self.horizon)
        R_ = ca.DM.eye(2 * self.horizon)
        beta = 0.8

        u_max = [self.a_max - refpath[5, k], self.omega_rate_max - refpath[6, k]]
        u_min = [-self.a_max - refpath[5, k], -self.omega_rate_max - refpath[6, k]]
        # for i in range(1, self.horizon):
        #     Q_[5 * i:5 * (i + 1), 5 * i:5 * (i + 1)] = ca.power(beta, i)

        D_ = ca.DM.eye(2 * self.horizon)
        lba = ca.DM.zeros(2 * self.horizon, 1)
        uba = ca.DM.zeros(2 * self.horizon, 1)

        for i in range(1,self.horizon):
            uba[2 * i] = u_max[0]  # acc
            uba[2 * i - 1] = u_max[1]  # o_rate
            lba[2 * i] = u_min[0]
            lba[2 * i - 1] = u_min[1]

        H_ = 2 * (B_.T @ Q_ @ B_ + R_)
        f_ = 2 * B_.T @ Q_ @ A_ @ state
        H_ = (H_ + H_.T) / 2
        return H_, f_, D_, lba, uba

    def get_A(self, vr, yawr, dt):
        A = ca.DM.eye(5)
        A[0, 2] = - vr * ca.sin(yawr) * dt
        A[0, 3] = vr * ca.cos(yawr) * dt
        A[1, 2] = ca.cos(yawr) * dt
        A[1, 3] = ca.sin(yawr) * dt
        A[2, 4] = dt
        return A

    def get_A_bk(self, k, refpath, dt):
        vr_ = refpath[3, :]
        yawr_ = refpath[4, :]
        A_bk = ca.DM.zeros(5 * self.horizon, 5)
        alpha = ca.DM.eye(5)
        for i in range(self.horizon - 1):
            if k + i >= len(refpath):
                k = -i + len(refpath) - 1
            alpha = alpha @ self.get_A(vr_[k + i], yawr_[k + i], dt)
            A_bk[5 * i:5 * (i + 1), :] = alpha
        return A_bk

    def get_B_bk(self, k, refpath, dt):
        vr_ = refpath[3, :]
        yawr_ = refpath[4, :]
        B_bk = ca.DM.zeros(5 * self.horizon, 2 * self.horizon)
        Bk = ca.DM.zeros(5, 2)
        Bk[3, 0] = dt
        Bk[4, 1] = dt

        for j in range(self.horizon - 1):
            alpha = ca.DM.eye(5)
            for i in range(j, self.horizon - 1):
                B_bk[5 * i:5 * (i + 1), 2 * i:2 * (i + 1)] = alpha @ Bk
                if k + i >= len(refpath):
                    k = -i + len(refpath) - 1
                alpha = alpha @ self.get_A(vr_[k + i], yawr_[k + i], dt)
        return B_bk

    def init_model_controller_iterative(self, index, start, reference_path):

        dt = self.dt0
        H_, f_, A_, lba, uba = self.get_linear_model_matrix(index, start, reference_path, dt)
        qp = {"h": H_.sparsity(), "a": A_.sparsity()}
        opts = {'sparse': True,
                'epsDen': 1e-5,
                'CPUtime': 1e-3}
        # "printLevel": }
        Sol = ca.conic('S', 'qpoases', qp, opts)
        result = Sol(h=H_, g=f_, a=A_, lba=lba, uba=uba)
        self.x_opt = result["x"]

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
            omega_rate = x[6, i]

            dx = dt * (ca.cos(x0[2, i]) * v_ - x0[3, i] * ca.sin(x0[2, i]) * yaw_)
            dy = dt * (ca.sin(x0[2, i]) * v_ + x0[3, i] * ca.cos(x0[2, i]) * yaw_)
            dyaw = dt * omega_
            dv = dt * a_
            domega = dt * omega_rate

            g1[0, i] = x_ + dx - x[0, i + 1]
            g1[1, i] = y_ + dy - x[1, i + 1]
            g1[2, i] = yaw_ + dyaw - x[2, i + 1]
            g1[3, i] = v_ + dv - x[3, i + 1]
            g1[4, i] = omega_ + domega - x[4, i + 1]
            # g1[7, i] = dy * ca.cos(yaw_) - dx * ca.sin(yaw_)

        return g1

    def init_bounds_reference_line(self, start, refpath):
        lbx = ca.DM.zeros(self.nx, self.horizon)
        ubx = ca.DM.zeros(self.nx, self.horizon)
        lbg = ca.DM.zeros(self.ng, self.horizon - 1)
        ubg = ca.DM.zeros(self.ng, self.horizon - 1)

        for i in range(self.horizon):
            lbx[0, i] = -ca.inf
            lbx[1, i] = -ca.inf
            lbx[2, i] = -ca.pi  # th
            lbx[3, i] = -self.v_max  # v
            lbx[4, i] = -self.omega_max  # omega
            lbx[5, i] = -self.a_max  # a
            lbx[6, i] = -self.omega_rate_max  # omega_rate
            # lbx[7, i] = -self.jerk_max  # jerk
            # lbx[8, i] = -self.lateral_error_max

            ubx[0, i] = ca.inf
            ubx[1, i] = ca.inf
            ubx[2, i] = ca.pi  # th
            ubx[3, i] = self.v_max  # v
            ubx[4, i] = self.omega_max  # omega
            ubx[5, i] = self.a_max  # a
            ubx[6, i] = self.omega_rate_max  # omega_rate
            # ubx[7, i] = self.jerk_max  # jerk
            # ubx[8, i] = self.lateral_error_max

        lbx[0, 0] = start[0] - refpath[0, 0]
        lbx[1, 0] = start[1] - refpath[1, 0]
        lbx[2, 0] = start[2] - refpath[2, 0]
        lbx[3, 0] = start[3] - refpath[3, 0]
        lbx[4, 0] = start[4] - refpath[4, 0]

        ubx[0, 0] = start[0] - refpath[0, 0]
        ubx[1, 0] = start[1] - refpath[1, 0]
        ubx[2, 0] = start[2] - refpath[2, 0]
        ubx[3, 0] = start[3] - refpath[3, 0]
        ubx[4, 0] = start[4] - refpath[4, 0]

        # lbx[0, -1] = refpath[0, -1]
        # lbx[1, -1] = refpath[1, -1]
        # lbx[2, -1] = refpath[2, -1]
        # lbx[3, -1] = 0.
        # lbx[4, -1] = -self.omega_end
        #
        # ubx[0, -1] = refpath[0, -1]
        # ubx[1, -1] = refpath[1, -1]
        # ubx[2, -1] = refpath[2, -1]
        # ubx[3, -1] = self.v_end
        # ubx[4, -1] = self.omega_end

        # lbx[0, -1] = -ca.inf
        # lbx[1, -1] = -ca.inf
        # lbx[2, -1] = -ca.pi
        # lbx[3, -1] = 0.
        # lbx[4, -1] = -self.omega_end
        #
        # ubx[0, -1] = ca.inf
        # ubx[1, -1] = ca.inf
        # ubx[2, -1] = ca.pi
        # ubx[3, -1] = self.v_end
        # ubx[4, -1] = self.omega_end

        # lbg[, :] = 0.
        # ubg[-1, :] = 1e-5

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)
        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def init_objects(self, x):
        sum_x_error = 0.
        sum_u_error = 0.

        for i in range(self.horizon):
            sum_x_error += ca.sumsqr(x[:3, i])
            sum_u_error += ca.sumsqr(x[3:, i])

        obj = self.wg[5] * sum_x_error + self.wg[6] * sum_u_error
        return obj

    def init_model_controller(self, start, reference_path):

        # initialize variables
        x = ca.SX.sym("x", self.nx, self.horizon)  # (ex, ey, etheta, ev, eomega a, a)
        dt = self.dt0

        # initialize constraints
        g1 = self.init_dynamic_constraints(x, dt, reference_path)
        X = ca.reshape(x, -1, 1)
        G = ca.reshape(g1, -1, 1)

        # initialize objectives
        F = self.init_objects(x)

        qp = {"x": X, "f": F, "g": G}
        opts = {'sparse': True,
                'epsDen': 1e-5,
                'CPUtime': 1e-3}
        # "printLevel": }
        Sol = ca.qpsol('S', 'qpoases', qp, opts)

        XL, XU, GL, GU = self.init_bounds_reference_line(start, reference_path)

        result = Sol(lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.x_opt = result["x"]

    def get_mpc_result(self):
        bb = self.x_opt
        # cal_traj = ca.reshape(self.x_opt, self.nx, self.horizon)
        cal_traj = ca.reshape(self.x_opt, 2, self.horizon)
        # op_controls = np.array(cal_traj[3:, :])
        # op_trajectories = np.array(cal_traj[:3, :])
        op_trajectories = np.array(cal_traj)
        return op_trajectories


def main():
    start_time = time.time()

    address = "../../config_OBCA_large.yaml"
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    cmpc = CasADi_MPC_differ_Control()

    start = [ref_path[0, 0], ref_path[1, 0], ref_path[2, 0], 0.5, 0.5]
    cmpc.dt0 = dt
    ref_traj = expand_path(ref_path, 0.8 * dt * cmpc.v_max)

    cmpc.horizon = int(local_horizon / dt)
    cmpc.init_model_reference_line(start, ref_traj)
    op_path, op_input = cmpc.get_result_reference_line()
    print("ds:", dt * cmpc.v_max, " horizon after expasion:", len(ref_traj.T))
    print("MPC total time:{:.3f}s".format(time.time() - start_time))
    np.savez("../../data/smoothed_traj_differ", dt=dt, traj=op_path, control=op_input,
             refpath=ref_path)
    ut.plot_results(dt, local_horizon, op_path, op_input, ref_traj)
    ut.show_plot()


if __name__ == '__main__':
    dt = 0.1
    local_horizon = 15
    large = True
    main()
