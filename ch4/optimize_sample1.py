#coding: utf-8
import numpy as np
from scipy import optimize

"""
Conjugate Gradientによるコスト関数を最小化する
パラメータの求め方
"""

def J(theta, *args):
    """最小化を目指すコスト関数を返す
    xはパラメータリスト"""
    theta1, theta2 = theta
    return (theta1 - 5) ** 2 + (theta2 - 5) ** 2

def gradient(theta, *args):
    """コスト関数の偏微分を返す
    各パラメータで偏微分した関数リストを返す"""
    theta1, theta2 = theta
    # Jをtheta1で偏微分した関数
    gt1 = 2 * (theta1 - 5)
    # Jをtheta2で偏微分した関数
    gt2 = 2 * (theta2 - 5)
    return np.array((gt1, gt2))

if __name__ == "__main__":
    # パラメータの初期値
    initial_theta = np.array([0, 0])

    # Conjugate Gradientによるコスト関数最小化
    theta = optimize.fmin_cg(J, initial_theta, fprime=gradient)
    print "conjugate gradient:", theta

    # BFGSによるコスト関数最小化
    theta = optimize.fmin_bfgs(J, initial_theta, fprime=gradient)
    print "BFGS:", theta
