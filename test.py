# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/6 11:30 PM
@Auth ： LiuYun ZhaoYing
@File ：test.py
@IDE ：PyCharm Community Edition

"""
import numpy as np
import matplotlib.pyplot as plt


# def sigmoid(x):
#     return 1.0 / (1 + np.exp(-x))
#
#
# sigmoid_inputs = np.arange(-10, 10, 0.1)
# sigmoid_outputs = sigmoid(sigmoid_inputs)
# print("Sigmoid Function Input :: {}".format(sigmoid_inputs))
# print("Sigmoid Function Output :: {}".format(sigmoid_outputs))
#
# plt.plot(sigmoid_inputs, sigmoid_outputs)
# plt.xlabel("Sigmoid Inputs")
# plt.ylabel("Sigmoid Outputs")
# plt.show()
import time

def outer(func):
    def inner():
        print("记录日志开始")
        func() # 业务函数
        print("记录日志结束")
    return inner

def outer2(func):
    print("记录日志开始")
    return func  # 业务函数
    print("记录日志结束")

def time_calc(func):
    def wrapper(*args, **kargs):
        start_time = time.time()
        f = func(*args,**kargs)
        exec_time = time.time() - start_time
        print(exec_time)
        return f
    return wrapper

@time_calc
def foo():
    print("foo")

@time_calc
def add(a, b):
    return a + b
# foo = outer(foo)
# foo()


foo()