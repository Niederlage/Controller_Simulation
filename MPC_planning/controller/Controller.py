import numpy as np
import control as ctrl


class Controller:
    def __init__(self):
        self.wheelbase = 2.
        self.Bm = np.zeros((5, 2))
        self.Bm[-2:, :] = np.eye(2)
        self.Cm = np.zeros((3, 5))
        self.Cm[:, :3] = np.eye(3)
        self.v_f = 2.
        self.yaw_f = 40 * np.pi / 180
        self.steer_f = 20 * np.pi / 180
        self.x_s = np.array([0., 0., self.yaw_f, self.v_f, self.steer_f])
        self.Q_diag = np.array([1e1, 1e1, 1e2, 1e8, 1e8])
        self.R_diag = np.array([1e7, 1e7])

    def get_mat_A(self, state):
        Am = np.zeros((5, 5))
        yaw_ = state[2]
        v_ = state[3]
        steer_ = state[4]
        Am[0, 2] = - v_ * np.sin(yaw_)
        Am[0, 3] = np.cos(yaw_)
        Am[1, 2] = v_ * np.cos(yaw_)
        Am[1, 3] = np.sin(yaw_)
        Am[2, 3] = np.tan(steer_) / self.wheelbase
        Am[2, 4] = v_ / (np.cos(steer_) ** 2 * self.wheelbase)

        return Am

    def get_mat_A3(self, state):
        Am = np.zeros((3, 3))
        yaw_ = state[2]
        v_ = state[3]
        steer_ = state[4]
        Am[0, 2] = - v_ * np.sin(yaw_)
        Am[1, 2] = v_ * np.cos(yaw_)
        return Am

    def get_mat_B3(self, state):
        Am = np.zeros((3, 2))
        yaw_ = state[2]
        v_ = state[3]
        steer_ = state[4]
        Am[0, 0] = np.cos(yaw_)
        Am[1, 0] = np.sin(yaw_)
        Am[2, 0] = np.tan(steer_) / self.wheelbase
        Am[2, 1] = v_ / (np.cos(steer_) ** 2 * self.wheelbase)
        return Am

    def cal_mat_K(self, current_state, dt):
        Am = self.get_mat_A(current_state) * dt + np.eye(5)
        Qm = np.diag(self.Q_diag)
        Rm = np.diag(self.R_diag)
        Km, S, pol = ctrl.lqr(Am, self.Bm * dt, Qm, Rm)
        return Km

    def cal_mat_K3(self, current_state):
        Am = self.get_mat_A3(current_state)
        Bm = self.get_mat_B3(current_state)
        Qm = np.diag(self.Q_diag[:3])
        Rm = np.diag(self.R_diag)
        Km, S, pol = ctrl.lqr(Am, Bm, Qm, Rm)
        return Km
