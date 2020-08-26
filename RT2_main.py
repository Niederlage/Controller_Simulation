import math
import time

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


class Initializing():
    g = 9.80665
    m1 = 1.4337
    m2 = 0.345  # Masse der Last
    rho = 0.03  # Durchmesser der Trommel
    J = 0.012477  # wird für die Berechnung der Trommelreibung benötigt
    dt = 0.01
    T = 20

    # # # # # # # # # # # # # # # Begrenzungen: # # # # # # # # # # # # # # #

    # Begrenzungen für Seillänge
    lmax = 0.7
    lmin = 0.23
    l_m = 0.5  # mittlere Seillänge

    # Begrenzungenfür die Wagenposition
    xmax = 0.55
    xmin = -0.55

    # Maximale Antriebskraft des Wagens, maximales Moment des Trommelmotors
    Fmax = 22.5
    M_Tmax = 3.75 * rho * 13

    # Reibparameter der Trommel
    mu_T = 0.0097254
    # Reibparameter des Wagens (positive und negative Bewegungsrichtung)
    mu_W_pos = 2.9194
    mu_W_neg = 3.0754

    # statex = sio.loadmat('initialData/statex.mat')
    A = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [-2.44137580e+01, 0.0, 0.0, 4.05594406e+00, -5.03496503e-02, 0.0, ],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [2.40022902e+00, 0.0, 0.0, -2.02797203e+00, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                  [2.97787080e-04, 0.0, 0.0, 0.0, 1.26047441e-10, -7.28501689e-01]])
    B = np.array([[0., 0.],
                  [-1.3986014, 0.],
                  [0., 0.],
                  [0.6993007, 0.],
                  [0., 0.],
                  [0., -2.25309801]])
    C = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])

    Ae = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                   [-24.41375805, 0., 0., 4.05594406, 0., 0., - 1.3986014, 0.],
                   [0., 0., 0., 1., 0., 0., 0., 0.],
                   [2.40022902, 0., 0., - 2.02797203, 0., 0., 0.6993007, 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., -0.72850169, 0., - 2.25309801],
                   [0., 0., 0., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 0., 0.]])
    Be = np.array([[0, 0], [-1.3986, 0], [0, 0], [0.6993, 0], [0, 0], [0, -2.2531], [0, 0], [0, 0]])
    Ce = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0]])

    R = 0.0031 * np.eye(2)
    Q = np.diag([1, 60, 5, 60, 2, 30])
    pB = np.array([-52, -53, -54, -51, -55, -56, -57, -50])
    # L = control.place(Ae.T,Ce.T,pB)
    # K, S, E = control.lqr(A, B, Q, R)
    # K from lqr
    K = np.array([[-497.409434908136, -81.1311085819819, 40.1608674320254, 150.658373189884, -1.22931785150275,
                   -0.00556508808891539],
                  [-1.94442726621085, 0.173666734963667, 0.0891791141002383, 0.365263795151701, -25.4049973884183,
                   -98.1656256301973]])

    # L from place
    L = np.array([[228.381781202783, -39.1718793344587, -0.876912968252177],
                  [16809.8565099534, -6544.26498081226, -140.978964163650],
                  [-28.9095652346727, 190.921412870443, 0.638020944703494],
                  [-4833.42819964506, 10498.7357324117, 87.8801032881437],
                  [2.72856599775791, 11.2685272524123, 253.940332203992],
                  [464.729131419997, 1899.19784506670, 21248.0126339501],
                  [-291722.429444774, 218241.458538312, 3983.56622962423],
                  [-8862.91789724334, -35331.1024847313, -264597.638832013]])


init = Initializing()
J = init.J
m1 = init.m1
m2 = init.m2
g = init.g
rho = init.rho
Fmax = init.Fmax
MTmax = init.M_Tmax
xmax = init.xmax
lmax = init.lmax
lmin = init.lmin
mu_W_pos = init.mu_W_pos
mu_W_neg = init.mu_W_neg


def flachtrajektorie(z0, k, T):
    za = z0[0]
    zb = z0[1]
    sum = za
    t = sp.symbols('t')

    for i in range(k + 1, 2 * k + 1 + 1):
        sum += (zb - za) * p_factorial(i, k) * (t / T) ** i
    return sum


def p_factorial(i, k):
    up = (-1) ** (i - k - 1) * math.factorial(2 * k + 1)
    down = i * math.factorial(k) * math.factorial(i - k - 1) * math.factorial(2 * k + 1 - i)
    return up / down


def x_derivative(x0, order, T):
    X = []
    t = sp.symbols('t')

    Xsoll = flachtrajektorie(x0, 4, T)
    x_t = Xsoll
    X.append(sp.lambdify(t, Xsoll, 'numpy'))

    for i in range(order):
        Xsoll = sp.diff(Xsoll, t)
        X.append(sp.lambdify(t, Xsoll, 'numpy'))
    return X, x_t


def y_derivative(x_t, f_x, order):
    Y = []
    t = sp.symbols('t')
    Ysoll = f_x.subs('x', x_t)
    Y.append(sp.lambdify(t, Ysoll, 'numpy'))

    for i in range(order):
        Ysoll = sp.diff(Ysoll, t)
        Y.append(sp.lambdify(t, Ysoll, 'numpy'))
    return Y


def solltrajctory(total_step, x_vec, f_x, dt, T_total):
    t = 0
    soll_state = np.zeros((10, total_step))
    Xlist, x_t = x_derivative(x_vec, 4, T_total)
    Ylist = y_derivative(x_t, f_x, 4)

    start = time.time()
    for step in range(total_step):
        X = Xlist[0](t)
        dX = Xlist[1](t)
        d2X = Xlist[2](t)
        d3X = Xlist[3](t)
        d4X = Xlist[4](t)

        Y = Ylist[0](t)
        dY = Ylist[1](t)
        d2Y = Ylist[2](t)
        d3Y = Ylist[3](t)
        d4Y = Ylist[4](t)

        t += dt

        soll_state[0, step] = X
        soll_state[1, step] = dX
        soll_state[2, step] = d2X
        soll_state[3, step] = d3X
        soll_state[4, step] = d4X
        soll_state[5, step] = Y
        soll_state[6, step] = dY
        soll_state[7, step] = d2Y
        soll_state[8, step] = d3Y
        soll_state[9, step] = d4Y
    print('solltraj:', time.time() - start)
    return soll_state


def visualize_solltrajctory(solltraj, schritt):
    timeline = np.arange(schritt)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(timeline, solltraj[0, :], '#9900CC', label='X')
    ax.plot(timeline, solltraj[1, :], '#CC0033', label='dX')
    ax.plot(timeline, solltraj[2, :], '#993366', label='d2X')
    ax.plot(timeline, solltraj[3, :], '#FF0066', label='d3X')
    ax.plot(timeline, solltraj[4, :], '#FF99CC', label='d4X')
    ax.plot(timeline, solltraj[5, :], '#003333', label='Y')
    ax.plot(timeline, solltraj[6, :], '#006633', label='dY')
    ax.plot(timeline, solltraj[7, :], '#00FF33', label='d2Y')
    ax.plot(timeline, solltraj[8, :], '#00CC66', label='d3Y')
    ax.plot(timeline, solltraj[9, :], '#00FFCC', label='d4Y')
    legend = fig.legend(loc='upper center', shadow=True, ncol=10)
    ax.set_xlabel('t/s')
    ax.grid(True)
    plt.show()


def streckenmodell(u, xi):
    dxi = np.zeros((6, 1)).flatten()
    xi1 = xi[0]
    xi2 = xi[1]
    xi3 = xi[2]
    xi4 = xi[3]
    xi5 = xi[4]
    xi6 = xi[5]
    F = u[0]
    MT = u[1]

    if xi4 >= 0.:
        FR = 2.9 * xi4 + 4.8 * np.sign(xi4) + (5.7 - 4.8) * np.sign(xi4) * np.exp(-(xi4 / 0.032) ** 2)
    else:
        FR = 3.1 * xi4 + 3.9 * np.sign(xi4) + (4.9 - 3.9) * np.sign(xi4) * np.exp(-(xi4 / 0.022) ** 2)
    omega = -xi6 / rho
    MRP = xi5 * (0.0076 * xi2 + 0.027 * np.sign(xi2) + (0.012 - 0.027) * np.sign(xi2) * np.exp(-0.073 * abs(xi2)))
    MR = 0.0097 * omega + 0.6 * np.sign(omega) + (0.41 - 0.6) * np.sign(omega) * np.exp(-(omega / 0.015) ** 2)

    dxi[0] = xi2
    dxi[1] = ((2 * J * xi2 * xi5 * xi6 - MRP * rho ** 2) * m2 ** 2 * np.cos(xi1) ** 2 - xi5 * m2 *
              ((xi5 * xi2 ** 2 * J - (-MT + MR) * rho) * m2 * np.sin(xi1) + (m2 * rho ** 2 + J) * (F - FR)) * np.cos(
                xi1)
              - (2 * (m1 * rho ** 2 + J) * m2 + 2 * J * m1) * (xi5 * xi2 * xi6 * m2 + m2 * g * xi5 * np.sin(xi1) / 2 +
                                                               MRP / 2)) / (
                     xi5 ** 2 * (-J * np.cos(xi1) ** 2 * m2 + (m1 * rho ** 2 + J) * m2 + J * m1) * m2)

    dxi[2] = xi4  # TODO IF NECESSARY BOUNDARY RESTRICTION BETWEEN +-0.5
    dxi[3] = ((xi5 * np.sin(xi1) * J * g * m2 + MRP * (m2 * rho ** 2 + J)) * np.cos(xi1) + xi5 * (
            (xi5 * xi2 ** 2 * J - (-MT + MR) * rho) * m2 * np.sin(xi1)
            + (m2 * rho ** 2 + J) * (F - FR))) / (
                     xi5 * (-J * np.cos(xi1) ** 2 * m2 + (m1 * rho ** 2 + J) * m2 + J * m1))
    dxi[4] = xi6
    dxi[5] = -rho * (
            xi5 * m2 * (-MT + MR) * np.cos(xi1) ** 2 - rho * m2 * (xi5 * g * m1 - MRP * np.sin(xi1)) * np.cos(xi1)
            + (rho * m2 * (F - FR) * np.sin(xi1) + (-m1 * rho * xi2 ** 2 * xi5 - MR + MT) * m2 - m1 * (-MT + MR))
            * xi5) / (xi5 * (-J * np.cos(xi1) ** 2 * m2 + (m1 * rho ** 2 + J) * m2 + J * m1))

    return dxi


def RungeKutta_integration(xk, fxk, u, ymess, dt):
    if len(xk) == 6:
        k1 = fxk
        k2 = streckenmodell(u, xk + dt / 2 * k1)
        k3 = streckenmodell(u, xk + dt / 2 * k2)
        k4 = streckenmodell(u, xk + dt * k3)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    elif len(xk) == 8:
        k1 = fxk
        k2 = linear_observer(u, xk + dt / 2 * k1, ymess)
        k3 = linear_observer(u, xk + dt / 2 * k2, ymess)
        k4 = linear_observer(u, xk + dt * k3, ymess)
        return xk + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def system_simulation(us, xs, x, schritt, dt):
    x_sequence = np.zeros((6, schritt))
    u_sequence = np.zeros((2, schritt))
    dx_sequence = np.zeros((6, schritt))
    e_u = [0, 0]
    e_int = [0, 0]
    for step in range(schritt):
        x_sequence[:, step] = x
        e_x = xs[:, step] - x
        u_correct, e_u, e_int = np.array(full_controller(us[:, step], e_x, e_u, e_int, None))
        u_correct = u_correct.flatten()
        u_sequence[:, step] = u_correct
        # u_correct = us[:, step]
        dx = streckenmodell(u_correct, x)
        dx_sequence[:, step] = dx
        x = RungeKutta_integration(x, dx, u_correct, None, dt)

    return x_sequence, u_sequence, dx_sequence


def visualize_system_simualtion(x_sq, uk, us, dt, T_total):
    timeline = np.arange(0, T_total, dt)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(timeline, x_sq[0, :], 'g', label='phi')
    ax1.plot(timeline, x_sq[1, :], '#0033FF', label='dphi')
    ax1.plot(timeline, x_sq[2, :], 'r', label='x')
    ax1.plot(timeline, x_sq[3, :], '#6699FF', label='dx')
    ax1.plot(timeline, x_sq[4, :], '#33CCFF', label='l')
    ax1.plot(timeline, x_sq[5, :], '#99CCFF', label='dl')
    ax1.legend(loc='best', shadow=True, ncol=2)
    ax1.set_xlabel('t/s')
    ax1.grid(True)

    ax2 = fig.add_subplot(122)
    ax2.plot(timeline, uk[0, :], 'b', label='Fk')
    ax2.plot(timeline, uk[1, :], 'r', label='MTk')
    ax2.plot(timeline, us[0, :], label='Fksoll')
    ax2.plot(timeline, us[1, :], 'y', label='MTksoll')
    ax2.legend(loc='best', shadow=True)
    ax2.set_xlabel('t/s')
    ax2.grid(True)

    plt.show()


def linear_observer(u, xi_hat, y_mess):
    dx_plus = np.zeros((8, 1)).flatten()
    y_hat = np.dot(init.Ce, xi_hat)
    deltaxi = np.dot(init.L, (y_mess - y_hat)).flatten()

    xi1 = xi_hat[0]
    xi2 = xi_hat[1]
    xi3 = xi_hat[2]
    xi4 = xi_hat[3]
    xi5 = xi_hat[4]
    xi6 = xi_hat[5]
    deltaz1 = xi_hat[6]
    deltaz2 = xi_hat[7]
    F = u[0]
    MT = u[1]

    if xi4 >= 0.:
        FR = 2.9 * xi4 + 4.8 * np.sign(xi4) + (5.7 - 4.8) * np.sign(xi4) * np.exp(-(xi4 / 0.032) ** 2) - deltaz1
    else:
        FR = 3.1 * xi4 + 3.9 * np.sign(xi4) + (4.9 - 3.9) * np.sign(xi4) * np.exp(-(xi4 / 0.022) ** 2) - deltaz1
    omega = -xi6 / rho
    MRP = xi5 * (0.0076 * xi2 + 0.027 * np.sign(xi2) + (0.012 - 0.027) * np.sign(xi2) * np.exp(-0.073 * abs(xi2)))
    MR = 0.0097 * omega + 0.6 * np.sign(omega) + (0.41 - 0.6) * np.sign(omega) * np.exp(-(omega / 0.015) ** 2) - deltaz2

    dx_plus[0] = deltaxi[0] + xi2
    dx_plus[1] = deltaxi[1] + ((2 * J * xi2 * xi5 * xi6 - MRP * rho ** 2) * m2 ** 2 * np.cos(xi1) ** 2 - xi5 * m2 *
                               ((xi5 * xi2 ** 2 * J - (-MT + MR) * rho) * m2 * np.sin(xi1) + (m2 * rho ** 2 + J) * (
                                       F - FR)) * np.cos(
                xi1) - (2 * (m1 * rho ** 2 + J) * m2 + 2 * J * m1) * (
                                       xi5 * xi2 * xi6 * m2 + m2 * g * xi5 * np.sin(xi1) / 2
                                       + MRP / 2)) / (
                         xi5 ** 2 * (-J * np.cos(xi1) ** 2 * m2 + (m1 * rho ** 2 + J) * m2 + J * m1) * m2)

    dx_plus[2] = deltaxi[2] + xi4  # TODO IF NECESSARY BOUNDARY RESTRICTION BETWEEN +-0.5
    dx_plus[3] = deltaxi[3] + ((xi5 * np.sin(xi1) * J * g * m2 + MRP * (m2 * rho ** 2 + J)) * np.cos(xi1) + xi5 * (
            (xi5 * xi2 ** 2 * J - (-MT + MR) * rho) * m2 * np.sin(xi1)
            + (m2 * rho ** 2 + J) * (F - FR))) / (
                         xi5 * (-J * np.cos(xi1) ** 2 * m2 + (m1 * rho ** 2 + J) * m2 + J * m1))
    dx_plus[4] = deltaxi[4] + xi6
    dx_plus[5] = deltaxi[5] - rho * (
            xi5 * m2 * (-MT + MR) * np.cos(xi1) ** 2 - rho * m2 * (xi5 * g * m1 - MRP * np.sin(xi1)) * np.cos(xi1)
            + (rho * m2 * (F - FR) * np.sin(xi1) + (-m1 * rho * xi2 ** 2 * xi5 - MR + MT) * m2 - m1 * (-MT + MR))
            * xi5) / (xi5 * (-J * np.cos(xi1) ** 2 * m2 + (m1 * rho ** 2 + J) * m2 + J * m1))
    dx_plus[6] = deltaxi[6]
    dx_plus[7] = deltaxi[7]

    return dx_plus


def simulation_with_observation(us, xs, x0, schritt, dt):
    # xhat_sequence = np.zeros((8, schritt))
    x_real_sequence = np.zeros((6, schritt))
    u_sequence = np.zeros((2, schritt))
    v_sequence = np.zeros((2, schritt))
    xhat = np.append([0., 0., -0.45, 0., 0.65, 0.], [0, 0])
    x_real = x0
    u_full = us[:, 0]
    e_u = [0, 0]
    e_int = [0, 0]

    start = time.time()
    for step in range(schritt):
        u_sequence[:, step] = u_full
        x_real_sequence[:, step] = x_real
        # xhat_sequence[:, step] = xhat
        e_x = xs[:, step] - xhat[:6]
        u_full, e_u, e_int = full_controller(us[:, step], e_x, e_u, e_int, xhat[6:])
        v_sequence[:, step] = e_int
        y_mess = np.dot(init.C, x_real)
        dx_real = streckenmodell(u_full, x_real)
        x_real = RungeKutta_integration(x_real, dx_real, u_full, None, dt)

        dxi = linear_observer(u_full, xhat, y_mess)
        xhat = RungeKutta_integration(xhat, dxi, u_full, y_mess, dt)

    print('observe time:', time.time() - start)
    return u_sequence, x_real_sequence, v_sequence


def visualize_system_simualtion_with_observer(u_sq, xi_sq, us, xs, dt, T_total):
    timeline = np.arange(0, T_total, dt)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(timeline, xi_sq[0, :], 'g', label='phi')
    ax1.plot(timeline, xi_sq[1, :], '#0033FF', label='dphi')
    ax1.plot(timeline, xi_sq[2, :], 'r', label='x')
    ax1.plot(timeline, xi_sq[3, :], '#6699FF', label='dx')
    ax1.plot(timeline, xi_sq[4, :], '#33CCFF', label='l')
    ax1.plot(timeline, xi_sq[5, :], '#99CCFF', label='dl')

    ax1.plot(timeline, xs[0, :], 'Black', label='phi_soll')
    ax1.plot(timeline, xs[1, :], 'Black', label='dphi_soll')
    ax1.plot(timeline, xs[2, :], 'Black', label='x_soll')
    ax1.plot(timeline, xs[3, :], 'Black', label='dx_soll')
    ax1.plot(timeline, xs[4, :], 'Black', label='l_soll')
    ax1.plot(timeline, xs[5, :], 'Black', label='dl_soll')
    ax1.legend(loc='best', shadow=True, ncol=2)
    ax1.set_xlabel('t/s')
    ax1.grid(True)

    ax2 = fig.add_subplot(122)
    ax2.plot(timeline, u_sq[0, :], 'b', label='Fk')
    ax2.plot(timeline, u_sq[1, :], 'r', label='MTk')
    ax2.plot(timeline, us[0, :], label='Fsoll')
    ax2.plot(timeline, us[1, :], 'y', label='MTsoll')
    ax2.legend(loc='best', shadow=True)
    ax2.set_xlabel('t/s')
    ax2.grid(True)

    plt.show()


def flachsystem(statesoll):
    X = statesoll[0]
    dX = statesoll[1]
    d2X = statesoll[2]
    d3X = statesoll[3]
    d4X = statesoll[4]
    Y = statesoll[5]
    dY = statesoll[6]
    d2Y = statesoll[7]
    d3Y = statesoll[8]
    d4Y = statesoll[9]

    usoll = [0, 0]
    xsoll = [0, 0, 0, 0, 0, 0]

    xsoll[0] = np.arctan(d2X / (d2Y - g))
    xsoll[2] = X - Y * d2X / (d2Y - g)
    xsoll[4] = Y * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2)
    xsoll[3] = dX - dY * d2X / (d2Y - g) - Y * d3X / (d2Y - g) + Y * d2X * d3Y / (d2Y - g) ** 2
    xsoll[5] = dY * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2) + Y * (2 * d2X * d3X / (d2Y - g) ** 2 - 2 * d2X ** 2 * d3Y
                                                                  / (d2Y - g) ** 3) / (
                       2 * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2))
    xsoll[1] = (d3X / (d2Y - g) - d2X * d3Y / (d2Y - g) ** 2) / (1 + d2X ** 2 / (d2Y - g) ** 2)

    usoll[0] = m1 * (d2X - d2Y * d2X / (d2Y - g) - 2 * dY * d3X / (d2Y - g) + 2 * dY * d2X * d3Y / (
            d2Y - g) ** 2 - Y * d4X / (d2Y - g)
                     + 2 * Y * d3X * d3Y / (d2Y - g) ** 2 - 2 * Y * d2X * d3Y ** 2 / (d2Y - g) ** 3 + Y * d2X * d4Y / (
                             d2Y - g) ** 2) + m2 * d2X
    usoll[1] = -m2 * (d2Y - g) * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2) * rho - J * (
            d2Y * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2)
            + dY * (2 * d2X * d3X / (d2Y - g) ** 2 - 2 * d2X ** 2 * d3Y / (d2Y - g) ** 3) / np.sqrt(
        1 + d2X ** 2 / (d2Y - g) ** 2)
            - Y * (2 * d2X * d3X / (d2Y - g) ** 2 - 2 * d2X ** 2 * d3Y / (d2Y - g) ** 3) ** 2 / (
                    4 * (1 + d2X ** 2 / (d2Y - g) ** 2) ** (3 / 2))
            + Y * (2 * d3X ** 2 / (d2Y - g) ** 2 - 8 * d2X * d3X * d3Y / (d2Y - g) ** 3 + 2 * d2X * d4X / (
            d2Y - g) ** 2
                   + 6 * d2X ** 2 * d3Y ** 2 / (d2Y - g) ** 4 - 2 * d2X ** 2 * d4Y / (d2Y - g) ** 3) / (
                    2 * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2))) / rho

    FRKLpos = 2.9 * dX - 2.9 * dY * d2X / (d2Y - g) - 2.9 * Y * d3X / (d2Y - g) + 2.9 * Y * d2X * d3Y / (d2Y - g) ** 2 \
              + 4.8 * np.sign(dX - dY * d2X / (d2Y - g) - Y * d3X / (d2Y - g) + Y * d2X * d3Y / (d2Y - g) ** 2) + .9 \
              * np.sign(dX - dY * d2X / (d2Y - g) - Y * d3X / (d2Y - g) + Y * d2X * d3Y / (d2Y - g) ** 2) * np.exp(
        -976.5625000
        * (dX - dY * d2X / (d2Y - g) - Y * d3X / (d2Y - g) + Y * d2X * d3Y / (d2Y - g) ** 2) ** 2)
    FRKLneg = 3.1 * dX - 3.1 * dY * d2X / (d2Y - g) - 3.1 * Y * d3X / (d2Y - g) + 3.1 * Y * d2X * d3Y / (
            d2Y - g) ** 2 + 3.9 \
              * np.sign(
        dX - dY * d2X / (d2Y - g) - Y * d3X / (d2Y - g) + Y * d2X * d3Y / (d2Y - g) ** 2) + 1.0 * np.sign(dX - dY
                                                                                                          * d2X / (
                                                                                                                  d2Y - g) - Y * d3X / (
                                                                                                                  d2Y - g) + Y * d2X * d3Y / (
                                                                                                                  d2Y - g) ** 2) * np.exp(
        -2066.115702 * (dX - dY * d2X / (d2Y - g)
                        - Y * d3X / (d2Y - g) + Y * d2X * d3Y / (d2Y - g) ** 2) ** 2)
    MRKL = -(0.97e-2 * (dY * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2) + Y * (2 * d2X * d3X / (d2Y - g) ** 2 - 2 * d2X ** 2
                                                                           * d3Y / (d2Y - g) ** 3) / (
                                2 * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2)))) / rho + .60 * np.sign(-(
            dY * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2) + Y * (
            2 * d2X * d3X / (d2Y - g) ** 2 - 2 * d2X ** 2 * d3Y / (d2Y - g) ** 3)
            / (2 * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2))) / rho) - .19 * np.sign(
        -(dY * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2)
          + Y * (2 * d2X * d3X / (d2Y - g) ** 2 - 2 * d2X ** 2 * d3Y / (d2Y - g) ** 3) / (
                  2 * np.sqrt(1 + d2X ** 2 / (d2Y
                                              - g) ** 2))) / rho) * np.exp(
        -4444.444445 * (dY * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2) + Y * (2 * d2X * d3X /
                                                                           (d2Y - g) ** 2 - 2 * d2X ** 2 * d3Y / (
                                                                                   d2Y - g) ** 3) / (
                                2 * np.sqrt(1 + d2X ** 2 / (d2Y - g) ** 2))) ** 2 / rho ** 2)

    if xsoll[3] >= 0.:
        usoll[0] = usoll[0] + FRKLpos
    else:
        usoll[0] = usoll[0] + FRKLneg

    usoll[1] = usoll[1] + MRKL

    return xsoll, usoll


def flachvorsteuerung(state_sequenze, schritt):
    usoll_sequence = np.zeros((2, schritt))
    xsoll_sequence = np.zeros((6, schritt))
    for step in range(schritt):
        xsoll_sequence[:, step], usoll_sequence[:, step] = flachsystem(state_sequenze[:, step])

    return xsoll_sequence, usoll_sequence


def visualize_flachvorsteuerung(x, u, dt, T_total):
    timeline = np.arange(0, T_total, dt)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(timeline, x[0, :], '#0000FF', label='phi_soll')
    ax1.plot(timeline, x[1, :], '#0033FF', label='dphi_soll')
    ax1.plot(timeline, x[2, :], '#3333FF', label='x_soll')
    ax1.plot(timeline, x[3, :], '#6699FF', label='dx_soll')
    ax1.plot(timeline, x[4, :], '#33CCFF', label='l_soll')
    ax1.plot(timeline, x[5, :], '#99CCFF', label='dl_soll')
    ax1.legend(loc='upper center', shadow=True)
    ax1.set_xlabel('t/s')
    ax1.grid(True)

    ax2 = fig.add_subplot(122)
    ax2.plot(timeline, u[0, :], '#330000', label='F')
    ax2.plot(timeline, u[1, :], '#FFFF33', label='MT')
    ax2.legend(loc='upper center', shadow=True)
    ax2.set_xlabel('t/s')
    ax2.grid(True)

    plt.show()


def full_controller(us, e_x, e_u, e_int, deltaz):
    if deltaz is not True:
        deltaz = np.zeros((2, 1)).flatten()

    v = anti_windup(e_u, e_x.flatten(), e_int)
    ur = np.dot(init.K, e_x).flatten() + v
    u_correct = us + ur - deltaz
    u_sat = np.copy(u_correct)
    # saturation
    if abs(u_correct[0]) >= Fmax:
        u_sat[0] = np.sign(u_correct[0]) * Fmax

    elif abs(u_correct[1]) >= MTmax:
        u_sat[1] = np.sign(u_correct[1]) * MTmax

    eu = u_sat - u_correct
    # print('overshoot = ', eu)
    return u_sat, eu, v


def anti_windup(e_u, e_x, v_pre):
    dt = 0.005
    Ki2 = 0.2
    dv = np.dot(init.K, e_x) + np.dot(Ki2, e_u)
    v = v_pre + dv * dt
    return v


def visualize_x_y_function(state_sq, x_vec, f_x, dt):
    x_is = state_sq[2][:]
    y_is = state_sq[4][:]
    y_sq = []
    timeline = np.arange(x_vec[0], x_vec[1], dt)
    x = sp.symbols('x')
    ysoll = sp.lambdify(x, f_x, 'numpy')
    for xsoll in timeline:
        y_sq.append(ysoll(xsoll))
    fig = plt.figure()
    ax1 = fig.add_subplot()
    ax1.plot(timeline, y_sq, '#0000FF', label='y_soll')
    ax1.plot(x_is, y_is, 'r', label='y_ist')
    ax1.legend(loc='best', shadow=True)
    ax1.set_xlabel('x')
    ax1.set_xlabel('y')
    ax1.grid(True)

    plt.show()


if __name__ == '__main__':
    pass
    dt, T_total = 0.005, 18
    schritt = int(T_total / dt)
    x0 = np.array([0., -1., -0.45, 0., 0.65, 0.])
    start = time.perf_counter()
    x_vec = [x0[2], 0.45]
    x, t = sp.symbols('x t')
    y = 1.728395 * x ** 2 + 0.3
    # y = cos(0.5*x)
    y = sp.sympify(y)
    sollstate = solltrajctory(schritt, x_vec, y, dt, T_total)
    # visualize_solltrajctory(sollstate, schritt)
    x_soll, u_soll = flachvorsteuerung(sollstate, schritt)
    # visualize_flachvorsteuerung(x_soll, u_soll, dt, T_total)
    x_real_seq, u_seq, dx_seq = system_simulation(u_soll, x_soll, x0, schritt, dt)
    # visualize_system_simualtion(x_real_seq, u_seq, u_soll, dt, T_total)
    u_sq, x_sq, v_sq = simulation_with_observation(u_soll, x_soll, x0, schritt, dt)
    visualize_system_simualtion_with_observer(u_sq, x_sq, u_soll, x_soll, dt, T_total)

    # #
    visualize_x_y_function(x_real_seq, x_vec, y, dt)
    # # end = time.perf_counter() - start
    # # print('elapsed time:{}s'.format(end))
