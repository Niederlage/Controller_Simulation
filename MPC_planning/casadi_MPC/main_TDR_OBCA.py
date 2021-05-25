import numpy as np
import matplotlib.pyplot as plt
from mpc_motion_plot import UTurnMPC, get_car_shape, initialize_saved_data
from casadi_MPC.casadi_OBCA_warmup import CasADi_MPC_WarmUp
from casadi_MPC.casadi_OBCA import CasADi_MPC_OBCA


if __name__ == '__main__':

    tracking = True
    ut = UTurnMPC()
    ref_traj, ob, obst = initialize_saved_data()
    shape = get_car_shape(ut)

    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    warmup_qp = CasADi_MPC_WarmUp()
    warmup_qp.init_model_warmup(ref_traj, shape, obst)
    op_dist, op_lambda, op_mu = warmup_qp.get_result_warmup()

    obca = CasADi_MPC_OBCA()
    obca.op_lambda0 = op_lambda
    obca.op_mu0 = op_mu
    obca.init_model_OBCA(ref_traj, shape, obst)
    op_dt, op_trajectories, op_controls = obca.get_result_OBCA()

    ut.predicted_trajectory = op_trajectories
    zst = ref_traj[:, 0]
    trajectory = np.copy(zst)

    ut.cal_distance(op_trajectories[:2, :], obca.horizon)
    ut.dt = op_dt
    print("Time resolution:{:.3f}s, total time:{:.3f}s".format(ut.dt, ut.dt * len(ref_traj.T)))
    # np.savez("saved_traj", traj=predicted_trajectory)

    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.plot(op_controls[0, :], label="v")
    ax1.plot(op_controls[1, :], label="steer")
    ax1.plot(op_controls[2, :], "-.", label="acc")
    ax1.plot(op_controls[3, :], "-.", label="steer rate")
    ax1.grid()
    ax1.legend()

    if tracking:
        trajectory = ut.try_tracking(zst, op_controls, trajectory, obst=ob, ref_traj=ref_traj)
        print("Done")

    plt.show()
