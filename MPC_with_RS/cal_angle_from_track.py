import numpy as np
import matplotlib.pyplot as plt


class CalAngleFromTracks():

    def mod_theta(self, theta, t_last):
        if abs(theta - t_last) > np.pi / 2:
            if np.sign(theta) > 0:
                return theta - np.pi
            else:
                return theta + np.pi

        return (theta + np.pi) % (2 * np.pi) - np.pi

    def generate_orientation(self, track, starter):
        l_track = len(track.T)

        vlast = starter[:2]
        vec_list = []
        th_list = []
        t_last = starter[2]

        if l_track > 4:
            AppendNum = 3
        else:
            AppendNum = 1

        for i in range(AppendNum):
            vec_list.append(starter)
            th_list.append(starter[2])

        for i in range(AppendNum, l_track):
            vec = self.get_vector(track[:, i - 1], track[:, i], "+")
            if self.check_flip(vlast, vec):
                sign = "-"
            # print("flipped!")
            else:
                sign = "+"

            vec_ = self.get_vector(np.array([0, 0]), vec, sign)
            theta = np.arctan2(vec_[1], vec_[0])
            theta_mod = self.mod_theta(theta, t_last)
            th_list.append(theta_mod)
            vec_list.append(np.hstack((vec_, [theta])))
            vlast = vec_
            t_last = theta_mod

        return np.array(vec_list).T, np.array(th_list)

    def check_flip(self, vlast, v_cur):
        ip = vlast[0] * v_cur[0] + vlast[1] * v_cur[1]
        if ip < 0:
            return True
        else:
            return False

    def get_vector(self, p1, p2, sign):
        if sign == "+":
            return p2 - p1
        else:
            return p1 - p2


if __name__ == '__main__':
    test_track = np.array([[0, 1, 1.5, 1., 0.],
                           [0, 1, 1.1, 1.9, 1.5]])
    diff_test_track = np.diff(test_track, axis=1)

    test_traj = np.load("./raw_MPC/saved_traj.npz", allow_pickle=True)
    trackx = test_traj["traj"][0]
    tracky = test_traj["traj"][1]
    trackth = np.array(test_traj["traj"][2])
    start = np.array([trackx[0], tracky[1], trackth[2]])
    track = np.vstack((trackx, tracky))
    caft = CalAngleFromTracks()
    vec_cal, theta_cal = caft.generate_orientation(track, start)

    fig = plt.figure()
    ax1 = plt.subplot(211)
    ax1.plot(theta_cal * 180 / np.pi, label="theta_cal")
    ax1.plot(trackth * 180 / np.pi, label="ref_theta")
    ax1.grid(True)
    ax1.legend()

    ax1 = plt.subplot(212)
    ax1.plot((theta_cal - trackth) * 180 / np.pi, label="theta_error")
    ax1.legend()
    ax1.grid(True)
    plt.show()
    print(1)
