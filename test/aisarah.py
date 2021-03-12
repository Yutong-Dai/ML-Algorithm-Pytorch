'''
File: aisarah.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-10 21:46
Last Modified: 2021-03-12 02:17
--------------------------------------------
Description:
'''
import sys
import os
PACKAGE_PARENT = '../'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
if True:
    from src.Algorithm.AISARAH.AISARAH import AISARAH
else:
    from src.Algorithm.AISARAH.AISARAHdebug import AISARAH
from src.Algorithm.AISARAH.params import params
from src.Lossfunctions.LeastSquares import LeastSquares
import torch

torch.manual_seed(0)
print("[Least Squares] sanity-check 1")
m, n = 100, 10
if torch.cuda.is_available():
    A = torch.randn(m, n).cuda()
    b = A @ (torch.randn(n, 1).cuda())
    x0 = torch.randn(n, 1).cuda().requires_grad_()
else:
    A = torch.randn(m, n)
    b = A @ (torch.randn(n, 1))
    x0 = torch.randn(n, 1).requires_grad_()

prob = LeastSquares(A, b)
aisarah = AISARAH(prob)
params['batchsize'] = min(16, m // 2)
params['maxepoch'] = 30
result = aisarah.solve(x0, params)

print("[Least Squares] sanity-check 2")
m, n = 10, 3
if torch.cuda.is_available():
    A = torch.randn(m, n).cuda()
    b = torch.randn(m, 1).cuda()
    x0 = torch.randn(n, 1).cuda().requires_grad_()
else:
    A = torch.randn(m, n)
    b = torch.randn(m, 1)
    x0 = torch.randn(n, 1).requires_grad_()

prob = LeastSquares(A, b)
aisarah = AISARAH(prob)
params['batchsize'] = min(16, m // 2)
params['maxepoch'] = 30
result = aisarah.solve(x0, params)

# torch.manual_seed(0)
# print("case 1: m > n")
# # m, n = 100000, 1000
# m, n = 10, 3
# if torch.cuda.is_available():
#     A = torch.randn(m, n).cuda()
#     b = torch.randn(m, 1).cuda()
#     x0 = torch.randn(n, 1).cuda().requires_grad_()
# else:
#     A = torch.randn(m, n)
#     b = torch.randn(m, 1)
#     x0 = torch.randn(n, 1).requires_grad_()

# prob = LeastSquares(A, b)
# aisarah = AISARAH(prob)
# params['batchsize'] = m // 2
# params['maxepoch'] = 20
# result = aisarah.solve(x0, params)
