import numpy as np
import scipy.linalg as sl
import matplotlib.pyplot as plt
from gears.interpolate_traj import continuous_interpolate


class Curve_Fitting:

    def cal_coefficient(self, refpath):
        # Ns = len(refpath.T)
        # num = int(Ns / 4.5)
        # spath = np.append(s[::num], s[-1])
        # S = get_S_mat(spath)
        # ct = sl.solve(Q_, refpath[2, ::int(len(refpath.T)/5)])
        # xpath = np.append(refpath[0, ::num], refpath[0, -1])
        # ypath = np.append(refpath[1, ::num], refpath[1, -1])
        # tpath = np.append(refpath[2, ::num], refpath[2, -1])

        s = self.get_s_(refpath[:2, :])
        S = self.get_S_mat(s, 6)
        xpath = refpath[0, :]
        ypath = refpath[1, :]
        tpath = refpath[2, :]

        cx = self.solve_coeff(S, xpath)
        cy = self.solve_coeff(S, ypath)
        ct = self.solve_coeff(S, tpath)

        return cx, cy, ct, s

    def solve_coeff(self, S, vals):
        Q_ = S.T @ S
        ct = sl.solve(Q_, S.T @ vals)
        return ct

    def get_s_(self, path0):
        dpath = np.diff(path0, axis=1)
        s2 = np.square(dpath)
        s_ = np.cumsum(np.sqrt(np.sum(s2, axis=0)))
        s_ = np.append(0., s_)
        return s_

    def get_S_mat(self, s_, order):
        S = np.zeros((len(s_), order))
        S[:, 0] = 1.
        for i in range(1, order):
            S[:, i] = np.power(s_, i)

        return S

    def get_curve(self, ct, res, ends):
        s_ = np.arange(0, ends, res)
        yaw = np.zeros((len(s_),))
        yaw += ct[0]
        for i in range(1, len(ct)):
            # yaw += ct[1] * s_ + ct[2] * np.power(s_, 2) + ct[3] * np.power(s_, 3) + ct[4] * np.power(s_, 4) + ct[
            #     5] * np.power(s_, 5)
            yaw += ct[i] * np.power(s_, i)
        return yaw

    def cal_path_use_theta(self, start, yaw, res):
        x = np.cumsum(res * np.cos(yaw))
        y = np.cumsum(res * np.sin(yaw))
        return x + start[0], y + start[1]

def plot_all(s_, res, x, y, yaw_, x_, y_, ref_traj):
    #       expand ref path
    cal_path = continuous_interpolate(ref_traj[:2, :], ref_traj[2, :], res)
    expanded_path = np.delete(cal_path, 0, 1)
    expanded_path = expanded_path[:, :len(yaw_)]

    f = plt.figure()
    ax = plt.subplot(311)
    ax.plot(s_, np.rad2deg(yaw_), "-r", label="yaw from s")
    ax.plot(s_, np.rad2deg(expanded_path[2, :]), "-g", label="yaw_ref")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("line length[m]")
    ax.set_ylabel("yaw angle[deg]")

    ax = plt.subplot(312)
    ax.plot(s_, x_, "-r", label="x s")
    ax.plot(s_, expanded_path[0, :], "-g", label="ref x")
    ax.plot(s_, x, "-b", label="x from theta")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("line length[m]")
    ax.set_ylabel("x [m]")

    ax = plt.subplot(313)
    ax.plot(s_, y_, "-r", label="y s")
    ax.plot(s_, expanded_path[1, :], "-g", label="ref y")
    ax.plot(s_, y, "-b", label="y from theta")
    ax.grid(True)
    ax.legend()
    ax.set_xlabel("line length[m]")
    ax.set_ylabel("y [m]")

    f2 = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_, y_, "-r", label="curve from s")
    ax.plot(x, y, "-b", label="curve from theta")
    ax.plot(ref_traj[0, :], ref_traj[1, :], "-g", label="refpath")
    ax.grid(True)
    ax.legend()
    plt.axis("equal")
    plt.show()


def main():
    res = 0.05
    load = np.load("../data/saved_hybrid_a_star.npz")
    ref_traj = load["saved_traj"]
    # load = np.load("../data/smoothed_traj.npz")
    # ref_traj = load["refpath"]

    # np.savetxt("path.txt", ref_traj[:2, :].T)

    #       cal coeff for all
    calco = Curve_Fitting()
    coeffx, coeffy, coefft, s0 = calco.cal_coefficient(ref_traj)

    #       generate curve
    yaw_ = calco.get_curve(coefft, res, s0[-1])
    x_ = calco.get_curve(coeffx, res, s0[-1])
    y_ = calco.get_curve(coeffy, res, s0[-1])
    x, y = calco.cal_path_use_theta(ref_traj[:2, 0], yaw_, res)

    s_ = np.arange(0, s0[-1], res)
    print("ct:", coefft, "\ncx:", coeffx, "\ncy:", coeffy)
    plot_all(s_, res, x, y, yaw_, x_, y_, ref_traj)


if __name__ == '__main__':
    main()
