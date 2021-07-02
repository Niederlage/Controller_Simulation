import time
import numpy as np
import yaml
from gears.cubic_spline_planner import Spline2D
from motion_plot.ackermann_motion_plot import UTurnMPC
from Modell_Ackermann.casadi_reference_line import CasADi_MPC_reference_line
from MPC_planning.HybridAStar.hybrid_a_star import HybridAStar


def hybrid_a_star_initialization(address, res):
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    coarse_planner = HybridAStar()
    coarse_planner.MOTION_RESOLUTION = res
    coarse_planner.show_animation = False
    sx = param["start"]
    ex = param["goal"]
    start = [sx[0], sx[1], np.deg2rad(sx[2])]
    goal = [ex[0], ex[1], np.deg2rad(ex[2])]
    print("start:", start, "\ngoal:", goal)
    coarse_planner.car.set_parameters(param)
    coarse_planner.obmap.generate_polygon_map()
    obst = coarse_planner.generate_obmap()

    path = coarse_planner.hybrid_a_star_planning(start, goal, obst)

    if path is not None:
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list
        saved_path = np.array([x, y, yaw])
    else:
        saved_path = None

    return saved_path, param, obst


def normalize_angle(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def expand_path(refpath, ds):
    x = refpath[0, :]
    y = refpath[1, :]
    theta0 = refpath[2, 0]
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []

    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        yaw_ = sp.calc_yaw(i_s)
        ryaw.append(normalize_angle(yaw_))
        rk.append(sp.calc_curvature(i_s))

    return np.array([rx, ry, ryaw, rk])


def expand_path2(refpath, ds):
    pass


def run_reference_line_mpc(param, ref_traj):
    warmup_time = time.time()

    forkplan = CasADi_MPC_reference_line()
    # forkplan.dt0 = 0.1
    # diff_s = np.diff(ref_traj[:2, :], axis=1)
    # sum_s = np.sum(np.hypot(diff_s[0], diff_s[1]))
    # forkplan.dt0 = 1.2 * (sum_s / forkplan.v_max + forkplan.v_max / forkplan.a_max) / ref_traj.shape[1]

    ref_traj = expand_path(ref_traj, forkplan.dt0 * forkplan.v_max * 1)
    forkplan.set_parameters(param)

    forkplan.init_model_reference_line(ref_traj)
    op_dt, op_trajectories, op_controls = forkplan.get_result_reference_line()

    return op_dt, op_trajectories, op_controls


def main():
    address = "../config_forklift.yaml"
    HOBCA_mpc = False
    try_segment = False
    load_file = False

    ds = 2 * 0.1

    if not load_file:
        ref_traj, param, obst = hybrid_a_star_initialization(address, ds)

        if len(ref_traj.T) > 500:
            ref_traj = None

        if ref_traj is not None:

            ut = UTurnMPC()
            ut.obmap.generate_polygon_map()
            ut.car.model_type = param["use_model_type"]
            ut.reserve_footprint = True
            start_time = time.time()

            op_dt, op_trajectories, op_controls = run_reference_line_mpc(param, ref_traj)

            print(" total time:{:.3f}s".format(time.time() - start_time))
            # np.savez("../data/smoothed_traj", dt=op_dt, traj=op_trajectories, control=op_controls, refpath=ref_traj)

            ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj)

        else:
            print("Hybrid A Star initialization failed ....")


if __name__ == '__main__':
    main()
