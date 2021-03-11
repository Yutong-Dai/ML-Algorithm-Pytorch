'''
File: svrg.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-10 21:46
Last Modified: 2021-03-11 17:02
--------------------------------------------
Description:
'''
import sys
import os
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.Algorithm.SVRG.SVRG import SVRG
from src.Algorithm.SVRG.params import params
from src.Lossfunctions.LeastSquares import LeastSquares
import torch

torch.manual_seed(0)
print("[Least Squares] case 1: m > n")
m, n = 100000, 1000
if torch.cuda.is_available():
    A = torch.randn(m, n).cuda()
    b = A@(torch.randn(n, 1).cuda())
    x0 = torch.randn(n, 1).cuda().requires_grad_()
else:
    A = torch.randn(m, n)
    b = A@(torch.randn(n, 1))
    x0 = torch.randn(n, 1).requires_grad_()

prob = LeastSquares(A, b)
svrg = SVRG(prob)
params['batchsize'] = m // 2
lambdamax = torch.linalg.norm(torch.matmul(A.T, A), 2) / m
params['stepsize'] = 1 / lambdamax
params['maxepoch'] = 100
result = svrg.solve(x0, params)

print("[Least Squares] case 2: m < n")
m, n = 100, 1000
if torch.cuda.is_available():
    A = torch.randn(m, n).cuda()
    b = torch.randn(m, 1).cuda()
    x0 = torch.randn(n, 1).cuda().requires_grad_()
else:
    A = torch.randn(m, n)
    b = torch.randn(m, 1)
    x0 = torch.randn(n, 1).requires_grad_()

prob = LeastSquares(A, b)
svrg = SVRG(prob)
params['batchsize'] = m // 2
lambdamax = torch.linalg.norm(torch.matmul(A.T, A), 2) / m
params['stepsize'] = 1 / lambdamax
params['maxepoch'] = 100
result = svrg.solve(x0, params)
