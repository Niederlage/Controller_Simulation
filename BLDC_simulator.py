import numpy as np


class Motor_Simulator():
    def __init__(self):
        self.L = 0.097e-3  # H
        self.R = 0.194  # Ohm
        self.I = 0.  # A
        self.U = 24.  # V
        self.p = 7
        self.K1 = 1 / 24.48
        self.K2 = 0.3
        self.Psi_PM = 0.02
        self.T_last = 0.2

    def field_schwaech(self, U_abs, U_s_max):
        pass

    def park_transformation(self, theta, i_vec3):
        P = np.zeros((3, 3))
        c1 = np.cos(theta)
        c2 = np.cos(theta - 2 * np.pi / 3)
        c3 = np.cos(theta + 2 * np.pi / 3)
        s1 = np.sin(theta)
        s2 = np.sin(theta - 2 * np.pi / 3)
        s3 = np.sin(theta + 2 * np.pi / 3)
        P[0, :] = np.array([c1, c2, c3])
        P[1, :] = np.array([s1, s2, s3])
        P[2, :] = np.array([0.5, 0.5, 0.5])
        i_vec2 = 2 / 3 * P @ i_vec3
        return i_vec2[:2]

    def strom_regler(self, i_sdq):
        pass

    def Spannungs_Sollwert_Modell(self, i_soll, omega):
        pass

    def Spannungsvektor(self, u_sdq, theta):
        pass

    def Raumzeiger_Steuerung(self, U_vec):
        pass
    def Spannungs_Istwertmodell(self,u_sdq, i_sdq):
        pass

    def motor_model(self, U, i_in, n_in):
        di = (U - i_in * self.R - self.K1 * n_in) / self.L
        dn = self.get_Torque(i_in) - self.T_last
        return di, dn

    def get_Torque(self, i_vec2):
        T = 3 * self.p * i_vec2[1] * self.Psi_PM



i_0 = np.array([1., 0., 0.])
theta = np.pi/2

a = np.ones((2,2))
w, e = np.linalg.eig(a)
e_ = np.linalg.inv(e)
l = e_.dot(a).dot(e)
print(1)