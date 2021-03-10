'''
File: LeastSquares.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-10 00:33
Last Modified: 2021-03-10 01:18
--------------------------------------------
Description:
'''
import torch


class LeastSquares:
    def __init__(self, A, b):
        """
          @input:
            A: torch.tensor of shape (m, n)
            b: torch.tensor of shape (m, 1)
        """
        self.m, self.n = A.shape
        self.A, self.b = A, b

    def forward(self, x):
        """
          @input:
            @     x: torch tensor of shape (n, 1)
          @return: torch tensor; the function value evaluated a the point x
        """
        self.AxMinusb = torch.matmul(self.A, x) - self.b
        self.loss = torch.sum(self.AxMinusb ** 2) / (self.m * 2)
        return self.loss

    def grad(self, x):
        """
          @input:
            @     x: torch tensor of shape (n, 1)
          @return: torch tensor; the gradient value evaluated a the point x
        """
        self.loss.backward()
        return x.grad

    def _mgrad(self, x):
        """
          Just for sanity check purpose. Won't be used later.
          @input:
            @     x: torch tensor of shape (n, 1)
          @return: torch tensor; the gradient value evaluated a the point x
        """
        return torch.matmul(self.A.T, self.AxMinusb) / self.m


if __name__ == "__main__":
    torch.manual_seed(0)
    m, n = 5, 3
    if torch.cuda.is_available():
        A = torch.randn(m, n).cuda()
        b = torch.randn(m, 1).cuda()
        x = torch.randn(n, 1).cuda().requires_grad_()
    else:
        A = torch.randn(m, n)
        b = torch.randn(m, 1)
        x = torch.randn(n, 1).requires_grad_()
    ls = LeastSquares(A, b)
    fval = ls.forward(x)
    gradm = ls._mgrad(x)
    grada = ls.grad(x)
    print(f'autodiff:{grada.detach().view(1,-1)} | manualdiff:{gradm.detach().view(1,-1)}')
