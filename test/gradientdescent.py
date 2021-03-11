'''
File: test_gradientdescent.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-10 00:39
Last Modified: 2021-03-11 17:01
--------------------------------------------
Description:
'''
import sys
import os
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from src.Algorithm.GradientDescent import GradientDescent
from src.Lossfunctions.LeastSquares import LeastSquares
import torch

torch.manual_seed(0)
print("[Gradient Descent] case 1: m > n")
m, n = 100000, 1000
if torch.cuda.is_available():
    A = torch.randn(m, n).cuda()
    b = torch.randn(m, 1).cuda()
    x0 = torch.randn(n, 1).cuda().requires_grad_()
else:
    A = torch.randn(m, n)
    b = torch.randn(m, 1)
    x0 = torch.randn(n, 1).requires_grad_()

lambdamax = torch.linalg.norm(torch.matmul(A.T, A), 2) / m
prob = LeastSquares(A, b)
params = {}
params['stepsize'] = 1 / lambdamax
params['tol'] = 1e-6
params['maxiter'] = 100
gd = GradientDescent(prob)
result = gd.solve(x0, params)

print("[Gradient Descent] case 2: m < n")
m, n = 100, 1000
if torch.cuda.is_available():
    A = torch.randn(m, n).cuda()
    b = torch.randn(m, 1).cuda()
    x0 = torch.randn(n, 1).cuda().requires_grad_()
else:
    A = torch.randn(m, n)
    b = torch.randn(m, 1)
    x0 = torch.randn(n, 1).requires_grad_()

lambdamax = torch.linalg.norm(torch.matmul(A.T, A), 2) / m
prob = LeastSquares(A, b)
params = {}
params['stepsize'] = 1 / lambdamax
params['tol'] = 1e-6
params['maxiter'] = 100
gd = GradientDescent(prob)
result = gd.solve(x0, params)

# torch.manual_seed(0)
# print("[Gradient Descent] case 1: m > n")
# m, n = 10, 3
# if torch.cuda.is_available():
#     A = torch.randn(m, n).cuda()
#     b = A@(torch.randn(n, 1).cuda())
#     x0 = torch.randn(n, 1).cuda().requires_grad_()
# else:
#     A = torch.randn(m, n)
#     b = A@(torch.randn(n, 1))
#     x0 = torch.randn(n, 1).requires_grad_()

# lambdamax = torch.linalg.norm(torch.matmul(A.T, A), 2) / m
# prob = LeastSquares(A, b)
# params = {}
# params['stepsize'] = 1 / lambdamax
# params['tol'] = 1e-6
# params['maxiter'] = 100
# gd = GradientDescent(prob)
# result = gd.solve(x0, params)
