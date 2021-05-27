import time

import numpy as np

from mpc_motion_plot import UTurnMPC
from casadi_MPC.casadi_OBCA_warmup import CasADi_MPC_WarmUp
from casadi_MPC.casadi_OBCA import CasADi_MPC_OBCA

if __name__ == '__main__':
    tracking = True
    start_time = time.time()
    ut = UTurnMPC()
    ref_traj, ob, obst = ut.initialize_saved_data()
    shape = ut.get_car_shape()

    op_trajectories = ref_traj
    op_controls = np.zeros((5, len(ref_traj.T)))
    op_dist = np.zeros((len(obst), len(ref_traj.T)))
    op_lambda = np.zeros((4 * len(obst), len(ref_traj.T)))
    op_mu = np.zeros((4, len(ref_traj.T)))
    op_dt = 0.1

    for i in range(5):
        warmup_time = time.time()
        # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
        warmup_qp = CasADi_MPC_WarmUp()
        warmup_qp.op_dist0 = op_dist
        warmup_qp.op_lambda0 = op_lambda
        warmup_qp.op_mu0 = op_mu
        warmup_qp.init_model_warmup(op_trajectories, shape, obst)
        op_dist, op_lambda, op_mu = warmup_qp.get_result_warmup()
        print("warm up time:{:.3f}s".format(time.time() - warmup_time))

        obca = CasADi_MPC_OBCA()
        obca.op_lambda0 = op_lambda
        obca.op_mu0 = op_mu
        obca.init_model_OBCA(op_trajectories, shape, obst)
        op_dt, op_trajectories, op_controls, op_lambda, op_mu = obca.get_result_OBCA()
        obca.op_control0 = op_controls

    print("warm up OBCA total time:{:.3f}s".format(time.time() - start_time))
    ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, ob)
