import casadi as ca
import numpy as np
from mpc_motion_plot import UTurnMPC


class CasADi_MPC_WarmUp:
    def __init__(self):
        self.base = 2.0
        self.LF = 3.
        self.LB = 1.
        self.offset = (self.LF - self.LB) / 2

        self.ng = 6
        self.model = None
        self.d_opt = None
        self.op_dist0 = None
        self.op_lambda0 = None
        self.op_mu0 = None

        self.obst_num = 0
        self.horizon = 0
        self.alpha = 1.2

        self.wg = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        self.dmin = -1e-6

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

    def init_warmup_constraints(self, x0, d_, lambda_, mu_, shape, obst, gx):
        gT = self.Array2SX(shape[:, 2][:, None].T)
        GT = self.Array2SX(shape[:, :2].T)

        l_ob = len(obst)
        for i in range(self.horizon - 1):
            mu_i = mu_[:, i]
            yaw_i = x0[2, i]
            offset = ca.SX.zeros(2, 1)
            offset[0, 0] = self.offset * ca.cos(yaw_i)
            offset[1, 0] = self.offset * ca.sin(yaw_i)
            t_i = x0[:2, i] - offset

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
                constraint2 = (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i + d_[j, i]
                temp = Aj.T @ lambdaj
                constraint3 = ca.sumsqr(temp)

                gx[j, i] = constraint1[0, 0]
                gx[j + l_ob, i] = constraint1[1, 0]
                gx[j + 2 * l_ob, i] = constraint2
                gx[j + 3 * l_ob, i] = constraint3

        return gx

    def init_objects_warmup(self, d_):
        sum_total_dist = 0
        for i in range(self.horizon - 1):
            for j in range(self.obst_num):
                sum_total_dist += ca.power(d_[j, i], 2)

        return sum_total_dist

    def init_bounds_warmup(self):
        lbx = ca.DM(self.obst_num + (self.obst_num + 1) * 4, self.horizon)
        ubx = ca.DM(self.obst_num + (self.obst_num + 1) * 4, self.horizon)
        lbg = ca.DM(self.ng, self.horizon - 1)
        ubg = ca.DM(self.ng, self.horizon - 1)

        for i in range(self.horizon - 1):
            lbx[:self.obst_num, i] = -1e-5   # dist
            lbx[self.obst_num:, i] = 0.  # lambda, mu

            ubx[:self.obst_num, i] = 0. # dist
            ubx[self.obst_num:, i] = 1e-2  # lambda, mu

            # # constraint2 (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i + d == 0
            lbg[2 * self.obst_num:3 * self.obst_num, i] = 1e-10
            ubg[2 * self.obst_num:3 * self.obst_num, i] = 1e-6

            # # constraint2 (Aj @ t_i - bj).T @ lambdaj - gT @ mu_i + d == 0
            lbg[2 * self.obst_num:3 * self.obst_num, i] = 1e-10
            ubg[2 * self.obst_num:3 * self.obst_num, i] = 1e-6

            # constraint3  norm_2(Aj.T @ lambdaj) <= 1
            lbg[3 * self.obst_num:, i] = 1.
            ubg[3 * self.obst_num:, i] = 1.

        lbx_ = ca.reshape(lbx, -1, 1)
        ubx_ = ca.reshape(ubx, -1, 1)

        lbg_ = ca.reshape(lbg, -1, 1)
        ubg_ = ca.reshape(ubg, -1, 1)

        return lbx_, ubx_, lbg_, ubg_

    def x0_initialization(self, reference_path):

        x0 = ca.DM(3, self.horizon)
        for i in range(self.horizon):
            x0[0, i] = reference_path[0, i]
            x0[1, i] = reference_path[1, i]
            x0[2, i] = reference_path[2, i]

        return x0

    def dual_variables_initialization(self):
        t0 = ca.DM.ones(self.obst_num + (self.obst_num + 1) * 4, self.horizon) * 1e-8
        if (self.op_dist0 is not None) and (self.op_lambda0 is not None) and (self.op_mu0 is not None):
            t0[:self.obst_num, :] = self.op_dist0
            t0[self.obst_num:5 * self.obst_num, :] = self.op_lambda0
            t0[5 * self.obst_num:, :] = self.op_mu0
        return t0

    def init_model_warmup(self, reference_path, shape_m, obst_m):
        self.horizon = reference_path.shape[1]
        self.obst_num = len(obst_m)
        self.ng = 4 * len(obst_m)
        # initialize variables
        d_ = ca.SX.sym("d", self.obst_num, self.horizon)  # dist
        lambda_ = ca.SX.sym("l", 4 * self.obst_num, self.horizon)
        mu_ = ca.SX.sym("m", 4, self.horizon)

        x0 = self.x0_initialization(reference_path)
        v0 = self.dual_variables_initialization()

        # initialize constraints
        gx = ca.SX.sym("g", self.ng, self.horizon - 1)
        gx = self.init_warmup_constraints(x0, d_, lambda_, mu_, shape_m, obst_m, gx)

        X, G = self.organize_variables(d_, lambda_, mu_, gx)
        X0, XL, XU, GL, GU = self.organize_bounds(v0)

        # initialize objectives
        F = self.init_objects_warmup(d_)

        qp = {"x": X, "f": F, "g": G}
        opts_setting = {"expand": True,
                        "ipopt.hessian_approximation": "limited-memory",
                        'ipopt.max_iter': 200,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        Sol = ca.nlpsol('S', 'ipopt', qp, opts_setting)

        result = Sol(x0=X0, lbx=XL, ubx=XU, lbg=GL, ubg=GU)
        self.d_opt = result["x"]

    def organize_variables(self, vx, vl, vm, cg):
        vs = ca.vertcat(vx, vl)
        vs = ca.vertcat(vs, vm)
        X = ca.reshape(vs, -1, 1)
        G = ca.reshape(cg, -1, 1)

        return X, G

    def organize_bounds(self, v0):
        lbx, ubx, lbg, ubg = self.init_bounds_warmup()
        X0 = ca.reshape(v0, -1, 1)

        return X0, lbx, ubx, lbg, ubg

    def get_result_warmup(self):
        cal_traj = ca.reshape(self.d_opt, self.obst_num + (self.obst_num + 1) * 4, self.horizon)
        op_dist = np.array(cal_traj[:self.obst_num, :])
        op_lambda = np.array(cal_traj[self.obst_num:self.obst_num + self.obst_num * 4, :])
        op_mu = np.array(cal_traj[self.obst_num + self.obst_num * 4:, :])
        return op_dist, op_lambda, op_mu


if __name__ == '__main__':
    ut = UTurnMPC()
    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()

    cmpc = CasADi_MPC_WarmUp()
    cmpc.init_model_warmup(ref_traj, shape, obst)
    op_dist, op_lambda, op_mu = cmpc.get_result_warmup()
    np.savez("../data/saved_OBCA_warmup.npz", op_d=op_dist, op_lambda=op_lambda, op_mu=op_mu)
    print("optimal warm-up successful!")
