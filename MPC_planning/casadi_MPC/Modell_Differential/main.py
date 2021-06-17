import time
import numpy as np
import yaml
from mpc_motion_plot import UTurnMPC
from casadi_differ_reference_line import CasADi_MPC_differ
from gears.cubic_spline_planner import Spline2D
from MPC_planning.HybridAStar.hybrid_a_star import HybridAStar


def hybrid_a_star_initialization(address):
    # address = "../../config_OBCA_large.yaml"
    with open(address, 'r', encoding='utf-8') as f:
        param = yaml.load(f)

    coarse_planner = HybridAStar()
    coarse_planner.show_animation = False
    start, goal, obst = coarse_planner.init_startpoints(address)
    sx = param["start"]
    ex = param["goal"]
    start = [sx[0], sx[1], np.deg2rad(sx[2])]
    goal = [ex[0], ex[1], np.deg2rad(ex[2])]
    coarse_planner.car.set_parameters(param)
    path = coarse_planner.hybrid_a_star_planning(start, goal, obst)

    if path is not None:
        x = path.x_list
        y = path.y_list
        yaw = path.yaw_list
        saved_path = np.array([x, y, yaw])
    else:
        saved_path = None

    return saved_path, param, obst


def expand_path(refpath, ds):
    x = refpath[0, :]
    y = refpath[1, :]
    sp = Spline2D(x, y)
    s = np.arange(0, sp.s[-1], ds)

    rx, ry, ryaw, rk = [], [], [], []

    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        yaw_ = sp.calc_yaw(i_s)
        ryaw.append(yaw_)
        # yaw_last = yaw_
        # rk.append(sp.calc_curvature(i_s))
    return np.array([rx, ry, ryaw])


def normalize_angle(yaw):
    return (yaw + np.pi) % (2 * np.pi) - np.pi


def coordinate_transform(yaw, t, path, mode):
    rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]])
    trans = np.repeat(t[:, None], len(path.T), axis=1)

    if mode == "body to world":
        newpath = rot @ path[:2, :] + trans
        newyaw = path[2, :] + yaw
    else:
        newpath = rot.T @ (path[:2, :] - trans)
        newyaw = path[2, :] - yaw

    newyaw = [normalize_angle(i) for i in newyaw]

    return np.vstack((newpath, newyaw))


def main():
    address = "../../config_differ_smoother.yaml"
    load_file = False
    dt = 0.05  # [s]
    local_horizon = 30  # [s]

    if not load_file:
        original_path, param, obst = hybrid_a_star_initialization(address)
        ref_path = coordinate_transform(original_path[2, 0], original_path[:2, 0], original_path, mode="world to body")

        if len(original_path.T) > 500:
            original_path = None

        if original_path is not None:

            start_time = time.time()
            address = "../../config_differ_smoother.yaml"
            with open(address, 'r', encoding='utf-8') as f:
                param = yaml.load(f)
            ut = UTurnMPC()
            ut.set_parameters(param)

            # states: (x ,y ,theta ,v , steer, a, steer_rate, jerk)
            cmpc = CasADi_MPC_differ()
            cmpc.dt0 = dt
            ref_traj = expand_path(ref_path, 0.8 * dt * cmpc.v_max)
            cmpc.horizon = int(local_horizon / dt)
            cmpc.v_end = cmpc.v_max
            cmpc.omega_end = cmpc.omega_max

            cmpc.init_model_reference_line(ref_traj)
            op_dt, op_trajectories, op_controls = cmpc.get_result_reference_line()
            print("ds:", dt * cmpc.v_max, " horizon after expasion:", len(ref_traj.T))
            print("MPC total time:{:.3f}s".format(time.time() - start_time))

            op_path = coordinate_transform(original_path[2, 0], original_path[:2, 0], op_trajectories,
                                           mode="body to world")

            np.savez("../../data/smoothed_traj_differ", dt=op_dt, traj=op_path, control=op_controls,
                     refpath=original_path)
            ut.plot_results(op_dt, op_path, op_controls, original_path)
            ut.show_plot()

        else:
            print("Hybrid A Star initialization failed ....")

    else:
        loads = np.load("../../data/smoothed_traj_differ.npz")
        op_dt = loads["dt"]
        op_trajectories = loads["traj"]
        op_controls = loads["control"]
        ref_traj = loads["refpath"]

        print("load file to check!")
        ut = UTurnMPC()
        with open(address, 'r', encoding='utf-8') as f:
            param = yaml.load(f)

        ut.set_parameters(param)
        ut.reserve_footprint = True
        ut.plot_results(op_dt, op_trajectories, op_controls, ref_traj, four_states=True)
        ut.show_plot()


if __name__ == '__main__':
    main()
