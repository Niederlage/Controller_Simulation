import casadi as ca
import numpy as np
import time
from gears.cubic_spline_planner import Spline2D
import yaml


class CasADi_MPC_differ_Control:
    def __init__(self):

        self.horizon = 10
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
        self.omega_rate_max = ca.pi * 720 / 180
        self.jerk_max = 3.
        self.lateral_error_max = 0.1
        self.heading_error_max = ca.pi * 40 / 180
        self.u_range = 1e-5

        self.v0 = 1.
        self.omega0 = 0.1
        self.v_end = 1.5
        self.omega_end = 0.5
        self.centripetal = 0.8

    def get_A(self, vr, yawr, dt):
        A = ca.DM.eye(5)
        A[0, 2] = - vr * ca.sin(yawr) * dt
        A[1, 3] = vr * ca.cos(yawr) * dt
        A[1, 2] = ca.cos(yawr) * dt
        A[1, 3] = ca.sin(yawr) * dt
        A[2, 4] = dt
        return A

    def get_A_bk(self, k, refpath, dt):
        vr_ = refpath[3, :]
        yawr_ = refpath[4, :]
        A_bk = ca.DM.zeros(5 * self.horizon, 5)
        alpha = ca.DM.eye(5)
        lpath = len(vr_)
        for i in range(self.horizon):
            if (k + i >= lpath):
                p = lpath
            else:
                p = k + i

            alpha = alpha @ self.get_A(vr_[p], yawr_[p], dt)
            A_bk[5 * i:5 * (i + 1), :] = alpha
        return A_bk

    def get_B_bk(self, k, refpath, dt):
        vr_ = refpath[3, :]
        yawr_ = refpath[4, :]
        B_bk = ca.DM.zeros(5 * self.horizon, 2 * self.horizon)
        Bk = ca.DM.zeros(5, 2)
        Bk[3, 0] = dt
        Bk[4, 1] = dt
        lpath = len(vr_)
        for j in range(self.horizon):
            alpha = ca.DM.eye(5)
            ll = list(range(j + 1))[::-1]
            for i in ll:
                B_bk[5 * j:5 * (j + 1), 2 * i:2 * (i + 1)] = alpha @ Bk
                print(j)
                if (k + i >= lpath):
                    p = lpath
                else:
                    p = k + i
                alpha = alpha @ self.get_A(vr_[p], yawr_[p], dt)

        return B_bk

    def get_linear_model_matrix(self, k, state, refpath, dt):
        A_ = self.get_A_bk(k, refpath, dt)
        B_ = self.get_B_bk(k, refpath, dt)
        print(B_)
        Q_ = ca.DM.eye(5 * self.horizon)
        R_ = ca.DM.eye(2 * self.horizon)
        beta = 0.8
        Qk = ca.DM.eye(5)
        Qk[0, 0] = 1e3
        Qk[1, 1] = 1e3
        Qk[2, 2] = 1e1
        Qk[3, 3] = 1e4
        Qk[4, 4] = 1e2

        D_ = ca.DM.eye(2 * self.horizon)
        lba = ca.DM.zeros(2 * self.horizon, 1)
        uba = ca.DM.zeros(2 * self.horizon, 1)
        lpath = len(refpath.T)
        for i in range(self.horizon - 1):
            Q_[5 * i:5 * (i + 1), 5 * i:5 * (i + 1)] = Qk
            if (k + i) >= lpath:
                p = lpath
            else:
                p = k + i
            uba[2 * i] = self.a_max - refpath[5, p]  # acc
            uba[2 * i + 1] = self.omega_rate_max - refpath[6, p]  # o_rate
            lba[2 * i] = -self.a_max - refpath[5, p]
            lba[2 * i + 1] = -self.omega_rate_max - refpath[6, p]

        H_ = 2 * (B_.T @ Q_ @ B_ + R_)
        f_ = 2 * B_.T @ Q_ @ A_ @ state
        H_ = (H_ + H_.T) / 2
        return H_, f_, D_, lba, uba

    def init_model_controller_conic(self, index, start, reference_path):

        dt = self.dt0
        H_, f_, A_, lba, uba = self.get_linear_model_matrix(index, start, reference_path, dt)
        qp = {"h": H_.sparsity(), "a": A_.sparsity()}
        # opts = {'sparse': True,
        #         'epsDen': 1e-5,
        #         'CPUtime': 1e-3}
        # "printLevel": }

        opts = {'print_iter': False,
                'max_iter': 10,
                'print_header': False}
        Sol = ca.conic('S', 'qrqp', qp, opts)
        result = Sol(h=H_, g=f_, lbx=lba, ubx=uba)
        self.x_opt = result["x"]

    def init_model_controller_qpsol(self, index, start, reference_path):

        dt = self.dt0
        H_, f_, A_, lba, uba = self.get_linear_model_matrix(index, start, reference_path, dt)
        x = ca.SX.sym("x", 2 * self.horizon, 1)
        F = 0.5 * x.T @ H_ @ x + f_.T @ x

        qp = {"f": F}
        # opts = {'sparse': True,
        #         'epsDen': 1e-5,
        #         'CPUtime': 1e-3}
        # "printLevel": }

        opts = {'print_iter': True,
                'max_iter': 10,
                'print_header': True}
        Sol = ca.qpsol('S', 'qrqp', qp)
        result = Sol(lbx=lba, ubx=uba)
        self.x_opt = result["x"]

    def get_mpc_result(self):
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


if __name__ == '__main__':
    dt = 0.1
    local_horizon = 15
    large = True
    main()
