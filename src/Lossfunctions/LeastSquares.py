'''
File: LeastSquares.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-10 00:33
Last Modified: 2021-03-11 11:43
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
        if A.device.type == 'cuda':
            self.x = self.x.cuda()
        self.x = torch.zeros(1)
        self.minibatch = None

    def forward(self, x, minibatch=None):
        """
          @input:
            @     x: torch tensor of shape (n, 1)
          @return: torch tensor; the function value evaluated a the point x
        """
        self.x = x
        self.minibatch = minibatch
        if minibatch is None:
            self.AxMinusb = torch.matmul(self.A, x) - self.b
            self.loss = torch.sum(self.AxMinusb ** 2) / (self.m * 2)
            self.loss_on_minibacth = False
            self.minibatch_id = None
        else:
            self.AxMinusb = torch.matmul(self.A[minibatch, :], x) - self.b[minibatch, :]
            self.loss = torch.sum(self.AxMinusb ** 2) / (len(minibatch) * 2)
            self.loss_on_minibacth = True
        return self.loss

    def grad(self, x, minibatch=None):
        """
          @input:
            @     x: torch tensor of shape (n, 1)
          @return: torch tensor; the gradient value evaluated a the point x
        """
        if (not torch.equal(self.x, x)) or (self.minibatch != minibatch):
            self.forward(x, minibatch)
        # clear gradient
        if x.grad is not None:
            x.grad.data.zero_()
        if (minibatch is None) and (self.loss_on_minibacth == False):
            self.loss.backward()
        elif (minibatch is not None) and (self.loss_on_minibacth == True):
            self.loss.backward()
        else:
            if self.loss_on_minibacth:
                fplace = 'mini-batch'
            else:
                fplace = 'full-batch'
            if minibatch is None:
                gplace = 'full-bacth'
            else:
                gplace = 'mini-batch'
            raise ValueError(f'Inconsistency: function is evaluated on {fplace} while attempting to evaluate gradient on {gplace}!')
        return x.grad

    def _mgrad(self, x, minibatch=None):
        """
          Just for sanity check purpose. Won't be used later.
          @input:
            @     x: torch tensor of shape (n, 1)
          @return: torch tensor; the gradient value evaluated a the point x
        """
        if minibatch is None:
            return torch.matmul(self.A.T, self.AxMinusb) / self.m
        else:
            return torch.matmul(self.A[minibatch, :].T, self.AxMinusb) / len(minibatch)


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
    # fval = ls.forward(x)
    grada = ls.grad(x)
    gradm = ls._mgrad(x)
    print(f'autodiff:{grada.detach().view(1,-1)} | manualdiff:{gradm.detach().view(1,-1)}')

    print(f' test minibatch ...')
    minibatch = [0, 1, 4]
    # fval_minibacth = ls.forward(x, minibatch)
    grada_minibacth = ls.grad(x, minibatch)
    gradm_minibacth = ls._mgrad(x, minibatch)

    print(f'autodiff:{grada_minibacth.detach().view(1,-1)} | manualdiff:{gradm_minibacth.detach().view(1,-1)}')
