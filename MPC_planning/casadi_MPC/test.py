import os
import time
import scipy.sparse as ss
import numpy as np

# Define problem data
P = ss.csc_matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = ss.csc_matrix([[1, 1], [1, 0], [0, 1]])
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])
# # Create an OSQP object
# prob = osqp.OSQP()
#
# # Setup workspace and change alpha parameter
# prob.setup(P, q, A, l, u, alpha=1.0)
#
# # Solve problem
# res = prob.solve()

#
# from multiprocessing import Process
#
#
# def run_proc(name):
#     print('子进程运行中，name%s,pin=%d...' % (name, os.getpid()))
#
#     time.sleep(10)
#     print('子进程已经结束')
#
#
# if __name__ == '__main__':
#     print('父进程%d.' % os.getpid())
#     p = Process(target=run_proc, args=('test',))
#     print('子进程将要执行')
#     p.start()
