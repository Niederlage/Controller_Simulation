import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import random

def get_f_val(x,arg):
    x1 = x[0]
    x2 = x[1]
    if arg == 'f':
        f = 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2
    if arg == 'grad':
        f = np.array([[-400 * x1 * (-x1 ** 2 + x2) + 2 * x1 - 2], [-200 * x1 ** 2 + 200 * x2]])
    if arg == 'hesse':
        f = np.array([[1200 * x1 ** 2 - 400 * x2 + 2, -400 * x1], [-400 * x1, 200]])
    return f

def get_step_size_backtracing(x, alpha0, s,):
    alpha = alpha0
    rho = 0.5
    c = 0.1
    while True:
        alpha *= rho
        fs = np.dot(get_f_val(x, 'grad').T,s).reshape(1)
        if get_f_val(x + alpha * s,'f') <= get_f_val(x,'f') + c * alpha * fs:
            break
    # print(alpha)
    return alpha

def get_step_size_goldenratio(alpha_l0, alpha_r0, x, s):
    alpha_r = alpha_r0
    alpha_l = alpha_l0
    r = 0.618
    epsilon_alpha = 1e-4
    epsilon_f = 1e-4
    a_l = alpha_l + (1 - r) * (alpha_r - alpha_l)
    a_r = alpha_l + r * (alpha_r - alpha_l)
    while True:
        if get_f_val(x + a_l*s,'f') < get_f_val(x + a_r*s,'f'):
            alpha_r = a_r
            a_r = a_l
            a_l = alpha_l + (1 - r) * (alpha_r - alpha_l)
        else:
            alpha_l = a_l
            a_l = a_r
            a_r = alpha_l + r * (alpha_r - alpha_l)

        error = abs(get_f_val(x + a_l * s, 'f') - get_f_val(x + a_r * s, 'f'))
        # if error < 1:
        #     print(error)
        #     print('a_l=', a_l)
        #     print('a_r=',a_r)
        if abs(alpha_r - alpha_l) <= epsilon_alpha:
            break
        if  error <= epsilon_f:
            break
    return alpha_r+alpha_l/2

def gradient_desent(x0):
    # 初始化
    search_line = []
    search_line.append(x0)
    x_0 = x0
    epsilon_x = 1e-10
    epsilon_f = 1e-10
    # epsilon_x, epsilon_f 为算法终止判定
    x_pre = x_0
    x_post = x_0
    k = 0
    alpha = 1
    alpha_l = 0
    alpha_r = alpha
    while True:
        s = - get_f_val(x_pre,'grad').reshape(2)
        # using backtracing
        alpha = get_step_size_backtracing(x_pre, alpha, s)
        # using golden ratio
        # alpha, alpha_l, alpha_r = get_step_size_goldenratio(alpha_l, alpha_r, x_pre, s)
        x_pre = x_post
        # print(alpha)
        x_post = x_pre + alpha * s
        # print(x_post)
        search_line.append(x_post)
        k = k + 1
        error = get_f_val(x_post, 'f') - get_f_val(x_pre, 'f')
        # print(error)
        if  np.linalg.norm(x_post - x_pre) <= epsilon_x:
            print('step small enough')
            break
        elif abs(error) <= epsilon_f:
            print('error small enough')
            break
    # print(search_line[-1])
    search_line = np.array(search_line).T
    print('gradient_discent_cost:', k, 'step')

    return search_line

def gradient_desent_moment(x0):
    # 初始化
    search_line = []
    search_line.append(x0)
    x_0 = x0
    epsilon_x = 1e-5
    epsilon_f = 1e-10
    # epsilon_x, epsilon_f 为算法终止判定
    x_pre = x_0
    x_post = x_0
    k = 0
    eta = 0.0001
    gamma = 0.88
    v = np.array([0.,0.09])
    v_trace = [v]
    while True:
        s = - get_f_val(x_pre,'grad').reshape(2)
        v = gamma * v + eta * s
        v_trace.append(v)
        x_pre = x_post
        # print(alpha)
        x_post = x_pre + v
        # print(x_post)
        search_line.append(x_post)
        k = k + 1
        error = get_f_val(x_post, 'f') - get_f_val(x_pre, 'f')
        # print(error)
        if  np.linalg.norm(x_post - x_pre) <= epsilon_x:
            print('step small enough')
            break
        elif abs(error) <= epsilon_f:
            print('error small enough')
            break

    print(search_line[-1])
    search_line = np.array(search_line).T
    print('gradient_discent_cost:', k, 'step')

    return search_line, v_trace

def conjugate_gradient_method(x0):
    # 初始化
    search_line = []
    search_line.append(x0)
    x_0 = x0
    epsilon_x = 1e-15
    epsilon_f = 1e-15
    # epsilon_x, epsilon_f 为算法终止判定
    x_pre = x_0
    x_post = x_0
    k = 0
    alpha = 1
    alpha_l = 0
    alpha_r = alpha
    s = - get_f_val(x_pre, 'grad').reshape(2)

    while True:

        beta0 = np.linalg.norm(get_f_val(x_post, 'grad')) / np.linalg.norm(get_f_val(x_pre, 'grad'))
        beta = beta0 ** 2
        s = - get_f_val(x_pre, 'grad').reshape(2) + beta * s
        alpha = get_step_size_backtracing(x_pre, alpha, s)
        # print(alpha)
        # using golden ratio
        # alpha, alpha_l, alpha_r = get_step_size_goldenratio(alpha_l, alpha_r, x_pre, s)
        x_pre = x_post
        x_post = x_pre + alpha * s
        search_line.append(x_post)
        k = k + 1
        error = get_f_val(x_post, 'f') - get_f_val(x_pre, 'f')
        # print(error)
        if np.linalg.norm(x_post - x_pre) <= epsilon_x:
            print('step small enough')
            break
        elif abs(error) <= epsilon_f:
            print('error small enough')
            break
    print(search_line[-1])
    search_line = np.array(search_line).T
    print('conjugate gradient_discent_cost:', k, 'step')
    # print(search_line)
    return search_line

def newton_method(x0):
    # 初始化
    search_line = []
    search_line.append(x0)
    x_0 = x0
    epsilon_x = 1e-4
    epsilon_f = 1e-10
    # epsilon_x, epsilon_f 为算法终止判定
    x_pre = np.array(x_0)
    x_post = 0 * np.array(x_0)
    k = 0
    alpha = 0.001
    alpha_l = 0
    alpha_r = alpha
    B = np.eye(2)
    while True:
        # # Newton Method
        # B = np.linalg.inv(get_f_val(x_pre, 'hesse'))
        # s = - np.dot(B, get_f_val(x_pre, 'grad')).reshape(2)
        #quasi Newton method
        d = x_post - x_pre
        y = get_f_val(x_post, 'grad') - get_f_val(x_pre, 'grad')
        y = y.flatten()
        rho = 1/(np.dot(y.T, d))
        A_vor = np.dot(np.eye(2) - rho * np.dot(d, y.T), B)
        B = np.dot(A_vor, np.eye(2) - rho * np.dot(y, d.T)) + rho * d * d.T
        s = - np.dot(B, get_f_val(x_pre, 'grad')).flatten()
        # alpha = get_step_size_backtracing(x_pre, alpha, s)
        # print(alpha)
        # using golden ratio
        alpha = 0.9 * alpha
        # alpha = get_step_size_goldenratio(alpha_l, alpha_r, x_pre, s)
        # print('alpha = ',alpha)
        # print(x_pre)
        x_pre = x_post
        x_post = x_pre + alpha * s

        search_line.append(x_post)
        k = k + 1
        error = get_f_val(x_post, 'f') - get_f_val(x_pre, 'f')
        # print(error)
        if np.linalg.norm(x_post - x_pre) <= epsilon_x:
            print('step small enough')
            break
        elif abs(error) <= epsilon_f:
            print('error small enough')
            break
        if abs(error)>1e2:
            print('error big enough')
            break
    print(search_line[-1])
    search_line = np.array(search_line).T
    print('newton_cost:', k, 'step')
    # print(search_line)
    return search_line

def m_s_approximate(x, s):
    dfs = np.dot(get_f_val(x, 'grad').T, s)
    sd2f = 0.5 * np.dot(s, get_f_val(x, 'hesse'))
    m_s = get_f_val(x, 'f') + dfs + np.dot(sd2f, s)
    return m_s

def trust_region_method(x0, delta0):
    # 初始化
    search_line = []
    search_line.append(x0)
    x = x0
    s0 = [0,0]
    delta = delta0
    delta_gr = 20
    eta = 0.1
    k = 0
    epsilon_x = 1e-10
    epsilon_df = 1e-10
    while True:
        B = np.linalg.inv(get_f_val(x, 'hesse'))
        s = - np.dot(B, get_f_val(x, 'grad')).reshape(2)
        # print(s)
        while np.linalg.norm(s) > delta:
            s = 0.99* s

        m0 = m_s_approximate(x, s0)
        m_s = m_s_approximate(x, s)
        rho = (get_f_val(x, 'f') - get_f_val(x + s,'f'))/(m0 - m_s)

        if rho < 0.25:
            delta *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(s) - delta) < 1e-3:
            delta = min(2*delta, delta_gr)
        else:
            delta = delta

        if rho > eta:
            x = x + s
        else:
            x = x

        search_line.append(x)
        k += 1
        if np.linalg.norm(get_f_val(x,'grad')) <= epsilon_df:
            print('error small enough')
            break
        if np.linalg.norm(x) <= epsilon_x:
            print('step small enough')
            break
    print(search_line[-1])
    search_line = np.array(search_line).T
    # print(search_line)
    print('trust_region_method_cost:', k, 'step')
    return search_line

def nelder_mead_method(x0):
    epsilon_x = 1e-10
    epsilon_f = 1e-10
    search_line = []
    search_line.append(x0)
    search_triangle = []
    k = 0
    # generate a random x_list to initialize Startsimplex
    x_list = []
    x_list.append(x0)

    # x1 = np.array(x0) + 3*np.array([1,0])
    # x2 = np.array(x0) + 7.58*np.array([0, 1])
    # x_list.append(x1)
    # x_list.append(x2)
    for i in range(2):
        temp = np.array([random.random(), random.random()])
        x_list.append(4*temp -2)
    x_list = np.array(x_list)

    #generate unordered (x,f(x)) list
    y_list = []
    for i in range(3):
        y_list.append(get_f_val(x_list[i, :],'f'))

    y_list = np.array(y_list)
    y_list = np.concatenate((x_list,y_list.reshape(3,1)),axis=1)
    # print(y_list)

    while True:
        # TODO Sorting
        y = np.argsort(y_list,axis=0)
        y_order = np.zeros((3,3))
        for i in range(3):
            y_order[i,:] = y_list[y[i,2],:]
        search_triangle.append(y_order)
        # generate x_bar
        x_list = y_order[:,:2]
        x_bar = np.sum(x_list, axis=0) / 3
        search_line.append(x_bar)
        x_r = x_bar + (x_bar - y_order[0,:2])
        f_r = get_f_val(x_r, 'f')

        # TODO Reflection
        if y_order[0,2]<= f_r and f_r <= y_order[1,2]:
            print('Reflection')
            y_order[2,:2] = x_r

        # TODO Expansion
        elif f_r < y_order[0, 2]:
            x_e = x_bar + (x_bar - y_order[0,:2]) * 2
            f_e = get_f_val(x_e, 'f')
            print('Expansion')
            if f_e < f_r:
                y_order[2, :2] = x_e
            else:
                y_order[2, :2] = x_r
        # TODO Outer Contraction
        elif y_order[1,2] <= f_r < y_order[2, 2]:
            x_c = x_bar + (x_bar - y_order[0, :2]) * 0.5
            f_c = get_f_val(x_c, 'f')
            print('Outer Contraction')
            if f_c <= f_r:
                y_order[2, :2] = x_c
            else:
                y_order[1, :2] = (y_order[0, :2] + y_order[1, :2] )/2 # TODO Shrink
                print('Shrink')
        # TODO inner Contraction
        elif f_r >= y_order[2, 2]:
            x_c = x_bar - (x_bar - y_order[0, :2]) * 0.5
            f_c = get_f_val(x_c, 'f')
            print('inner Contraction')
            if f_c < y_order[2, 2]:
                y_order[2, :2] = x_c
            else:
                y_order[1, :2] = (y_order[0, :2] + y_order[1, :2]) / 2  # TODO Shrink
                print('Shrink')

        # TODO Update f(x)
        for i in range(3):
            y_order[i,2] = get_f_val(y_order[i,:2],'f')
        y_list = y_order
        x_ref = np.tile(y_list[0,:2], (3,1))
        dist = np.linalg.norm(y_list[:,:2] - x_ref, axis=1)
        error = abs(y_list[2, 2] - y_list[0, 2])
        k += 1
        # TODO Stop kriterium
        if  error <= epsilon_f:
            print('error small enough')
            break
        if max(dist) <= epsilon_x:
            print('step small enough')
            break
    print(search_line[-1])
    search_line = np.array(search_line).T
    print('nelder_mead_method_cost:', k, 'step')
    return search_line, search_triangle

def plot_function(line):

    X = np.arange(-1.5, 1.5, 0.15)
    Y = np.arange(-1.5, 1.5, 0.15)
    X,Y = np.meshgrid(X, Y, sparse=True)
    Z = 100 * (Y - X ** 2) ** 2 + (X - 1) ** 2
    zline = 100 * (line[1,:] - line[0,:] ** 2) ** 2 + (line[0,:] - 1) ** 2

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha = 0.5)
    ax.plot(line[0,:], line[1,:], zline, label='gradient descent', color = 'r', marker='o')
    ax.legend()#画一条空间曲线
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_NM_function(line, triangle):

    X = np.arange(-1.5, 1.5, 0.15)
    Y = np.arange(-1.5, 1.5, 0.15)
    X,Y = np.meshgrid(X, Y, sparse=True)
    Z = 100 * (Y - X ** 2) ** 2 + (X - 1) ** 2
    zline = 100 * (line[1,:] - line[0,:] ** 2) ** 2 + (line[0,:] - 1) ** 2

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(X,Y,Z,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,alpha = 0.5)
    ax.plot(line[0,:], line[1,:], zline, label='gradient descent', color = 'r', marker='o')
    for j in range(len(triangle)):
        chrome0 = '#0CC'
        dreieck_x = []
        for i in range(4):
            if i == 3:
                dreieck_x.append(triangle[j][0, 0])
            else:
                dreieck_x.append(triangle[j][i, 0])
        dreieck_y = []
        for i in range(4):
            if i == 3:
                dreieck_y.append(triangle[j][0, 1])
            else:
                dreieck_y.append(triangle[j][i, 1])
        dreieck_z = []
        for i in range(4):
            if i == 3:
                dreieck_z.append(triangle[j][0, 2])
            else:
                dreieck_z.append(triangle[j][i, 2])
        # generate different colror
        if 10*j < 10:
            chrome = chrome0 + '000'
        elif 10*j < 100 and 10*j >= 10:
            chrome = chrome0 + str(0) + str(10 * j)
        ax.plot(dreieck_x, dreieck_y, dreieck_z, color=chrome)

    ax.legend()#画一条空间曲线
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

if __name__ == '__main__':
    x0 = [-1., -1.]
    # line = gradient_desent(x0)
    # line = conjugate_gradient_method(x0)
    line, v_trace = gradient_desent_moment(x0)
    # line = newton_method(x0)
    # line = trust_region_method(x0, delta0 = 10)
    # line, triangle = nelder_mead_method(x0)
    plot_function(line)
    # plot_NM_function(line, triangle)
    # plt.plot(v_trace)
    # plt.show()