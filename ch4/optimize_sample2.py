#coding: utf-8
import numpy as np
from scipy import optimize

"""
Conjugate Gradientによるコスト関数を最小化する
パラメータの求め方
"""

def J(param, *args):
    """最小化を目指すコスト関数を返す"""
    u, v = param
    # パラメータ以外のデータはargsを通して渡す
    a, b, c, d, e, f = args
    return a*u**2 + b*u*v + c*v**2 + d*u + e*v + f

def gradient(param, *args):
    """コスト関数の偏微分を返す
    各パラメータで偏微分した関数リストを返す"""
    u, v = param
    a, b, c, d, e, f = args
    gu = 2*a*u + b*v + d
    gv = b*u + 2*c*v + e
    return np.array((gu, gv))

if __name__ == "__main__":
    # パラメータの初期値
    initial_param = np.array([0, 0])
    args = (2, 3, 7, 8, 9, 10)

    # Conjugate Gradientによるコスト関数最小化
    param = optimize.fmin_cg(J, initial_param, fprime=gradient, args=args)
    print "conjugate gradient:", param

    # BFGSによるコスト関数最小化
    param = optimize.fmin_bfgs(J, initial_param, fprime=gradient, args=args)
    print "BFGS:", param
