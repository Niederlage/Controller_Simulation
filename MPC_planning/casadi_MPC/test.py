import os
import time

from multiprocessing import Process


def run_proc(name):
    print('子进程运行中，name%s,pin=%d...' % (name, os.getpid()))

    time.sleep(10)
    print('子进程已经结束')


if __name__ == '__main__':
    print('父进程%d.' % os.getpid())
    p = Process(target=run_proc, args=('test',))
    print('子进程将要执行')
    p.start()
