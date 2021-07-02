import time
import numpy as np
import yaml
from gears.cubic_spline_planner import Spline2D
from motion_plot.ackermann_motion_plot import UTurnMPC
from Modell_Ackermann.casadi_TDROBCA import CasADi_MPC_TDROBCA
from Modell_Ackermann.casadi_OBCA_warmup import CasADi_MPC_WarmUp
from MPC_planning.HybridAStar.hybrid_a_star import HybridAStar


def hybrid_a_star_initialization(ut):
    with open("../config_OBCA_large.yaml", 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    coarse_planner = HybridAStar()
    coarse_planner.show_animation = False
    sx = param["start"]
    ex = param["goal"]
    start = [sx[0], sx[1], np.deg2rad(sx[2])]
    goal = [ex[0], ex[1], np.deg2rad(ex[2])]
    coarse_planner.car.set_parameters(param)
    traj_adress = "../data/saved_hybrid_a_star.npz"
    map_adress = "../data/saved_obmap_obca.npz"

    samples = np.zeros((1, 2))
    ref_traj, obpoints, obst = ut.initialize_saved_data(traj_adress=traj_adress,
                                                        map_adress=map_adress)
    for i, ob in enumerate(obpoints):
        samples = np.vstack((samples, ob))

    samples = np.delete(samples, 0, axis=0)
    bounds = ut.obmap.bounds
    obpoints = np.vstack([samples, bounds]).T

    path = coarse_planner.hybrid_a_star_planning(start, goal, obpoints)

    if path is not None:
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list
        saved_path = np.array([x, y, yaw])
    else:
        saved_path = None

    return saved_path, param, obst, obpoints


def normalize_angle(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def expand_path(refpath, ds):
    x = refpath[0, :]
    y = refpath[1, :]
    theta0 = refpath[2, 0]
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []
    yaw_last = theta0
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        yaw_ = sp.calc_yaw(i_s)
        ryaw.append(normalize_angle(yaw_))
        # rk.append(sp.calc_curvature(i_s))
    return np.array([rx, ry, ryaw])


def run_segment_OBCA_mpc(param, ref_traj, shape, obst):
    s_half = int(len(ref_traj.T) / 2)
    s_half2 = len(ref_traj.T) - s_half

    op_trajectories = ref_traj[:, :s_half]
    op_dist = np.zeros((len(obst), s_half))
    op_lambda = np.zeros((4 * len(obst), s_half))
    op_mu = np.zeros((4, s_half))

    all_traj = []
    for i in range(2):
        warmup_time = time.time()
        # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
        if i > 0:
            op_trajectories = ref_traj[:, s_half - 1:]
            op_dist = np.zeros((len(obst), s_half2 + 1))
            op_lambda = np.zeros((4 * len(obst), s_half2 + 1))
            op_mu = np.zeros((4, s_half2 + 1))

        obca = CasADi_MPC_TDROBCA()
        obca.get_dt(ref_traj)
        obca.set_parameters(param)
        obca.op_lambda0 = op_lambda
        obca.op_mu0 = op_mu
        obca.op_d0 = op_dist
        obca.init_model_OBCA(op_trajectories, shape, obst)
        op_dt, op_trajectories, op_controls, op_lambda, op_mu = obca.get_result_OBCA()
        all_traj.append([op_dt, op_trajectories, op_controls, op_lambda, op_mu])

    dt_op = (all_traj[0][0] + all_traj[1][0]) / 2
    traj_op = np.block([all_traj[0][1], all_traj[1][1]])
    controls_op = np.block([all_traj[0][2], all_traj[1][2]])

    return dt_op, traj_op, controls_op


def run_TDROBCA_mpc(param, ref_traj, shape, obst):
    warmup_time = time.time()

    # pathonly = CasADi_MPC_OBCA_PathOnly()
    # pathonly.set_parameters(param)
    # pathonly.init_model_OBCA(ref_path, shape, obst)
    # ref_traj, kappa, vl, vm = pathonly.get_result_OBCA()

    # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
    # warmup_qp = CasADi_MPC_WarmUp()
    # warmup_qp.set_parameters(param)
    # warmup_qp.init_model_warmup(ref_traj, shape, obst)
    # op_dist, op_lambda, op_mu = warmup_qp.get_result_warmup()
    # print("warm up time:{:.3f}s".format(time.time() - warmup_time))

    obca = CasADi_MPC_TDROBCA()
    obca.get_dt(ref_traj)
    print("cal dt:", obca.dt0)
    obca.dt0 = 0.1
    # ref_traj = expand_path(ref_traj, 0.5 * obca.dt0 * obca.v_max)
    # ref_traj = expand_path(ref_traj, obca.dt0 * obca.v_max * 0.8)
    obca.set_parameters(param)
    # obca.op_lambda0 = op_lambda
    # obca.op_mu0 = op_mu
    # obca.op_d0 = op_dist
    obca.init_model_OBCA(ref_traj, shape, obst)
    op_dt, op_trajectories, op_controls, op_lambda, op_mu = obca.get_result_OBCA()

    return op_dt, op_trajectories, op_controls


def main():
    address = "../../config_OBCA_large.yaml"
    HOBCA_mpc = False
    try_segment = False
    load_file = False
    large = True
    ds = 0.1
    ut = UTurnMPC()

    if not load_file:
        ref_traj, param, obst, ob_points = hybrid_a_star_initialization(ut)
        # ref_traj = ref_path
        if len(ref_traj.T) > 500:
            ref_traj = None

        if ref_traj is not None:
            ut.car.model_type = param["use_model_type"]
            ut.reserve_footprint = True
            shape = ut.get_car_shape()
            start_time = time.time()

            if try_segment:
                op_dt, op_trajectories, op_controls = run_segment_OBCA_mpc(param, ref_traj, shape, obst)
            else:
                op_dt, op_trajectories, op_controls = run_TDROBCA_mpc(param, ref_traj, shape, obst)

            print("warm up OBCA total time:{:.3f}s".format(time.time() - start_time))
            # np.savez("../data/smoothed_traj", dt=op_dt, traj=op_trajectories, control=op_controls, refpath=ref_traj)

            ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj)

        else:
            print("Hybrid A Star initialization failed ....")

    else:
        loads = np.load("../data/smoothed_traj.npz")
        op_dt = loads["dt"]
        op_trajectories = loads["traj"]
        op_controls = loads["control"]
        ref_traj = loads["refpath"]

        print("load file to check!")
        ut = UTurnMPC()
        ut.car.model_type = loads["use_model_type"]
        ut.reserve_footprint = True
        ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, four_states=True)


if __name__ == '__main__':
    main()
