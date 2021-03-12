'''
File: AISARAH.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-10 14:17
Last Modified: 2021-03-12 02:33
--------------------------------------------
Description:
'''

import torch
import numpy as np
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AISARAH:
    def __init__(self, prob):
        self.prob = prob
        self.nSamples = prob.m
        self.dim = prob.n

    def _cal_delta(self, alpha, iteration, beta=0.999):
        if iteration == 0:
            self.old = 1 / alpha
        else:
            ans = self.old * beta + (1 - beta) * 1 / alpha
            self.old = ans
        return self.old

    def solve(self, x0, params):
        x = x0
        alpha_max = np.inf
        print(f"Computation is done on: {x0.device.type}")
        if not x.requires_grad:
            raise ValueError('Input x0 must be a tensor requires gradient')
        epoch = 0
        samples = [i for i in range(self.nSamples)]
        totalBatches = int(np.ceil(self.nSamples / params['batchsize']))
        flag = 'Reach the maximum number of iterations.'
        alpha = torch.zeros(1, requires_grad=True).to(device)
        fseq = []
        if params['printlevel'] > 0:
            print(f'*******************************************************************')
            print(f'                    AI-SARAH. Version: (03/11/2021)                ')
            print(f' Algorithm parameters')
            for k, v in params.items():
                print(f' params: {k} | value:{v}')
            print(f'*******************************************************************')
        while epoch <= params['maxepoch']:
            # outter loop: evaluate full gradient
            # clear gradient
            if x.grad is not None:
                x.grad.data.zero_()
            gradfx_full = self.prob.grad(x) + 0.0
            gradfx_full_norm = torch.linalg.norm(gradfx_full)
            if epoch == 0:
                gradfx0_norm = gradfx_full_norm
            if gradfx_full_norm <= params['tol'] * gradfx0_norm:
                flag = 'Find the optimal solution with the desired accuracy.'
                break
            # print(f'epoch:{epoch} | fval:{self.prob.loss} | grad:{gradfx_full_norm}')
            if epoch % params['printevery'] == 0:
                print(f' epoch       f         |grad|  | iters    pass      |v|')
            print(f'{epoch:5d}  {self.prob.loss:3.4e}   {gradfx_full_norm:3.4e}', end='')
            fseq.append(self.prob.loss.item())
            v = gradfx_full
            v_norm = gradfx_full_norm
            iteration = 0
            effective_pass = 0
            # inner iteration
            while v_norm >= params['gamma'] * gradfx_full_norm:
                # print(v_norm, params['gamma'] * gradfx_full_norm)
                counter = np.mod(iteration, totalBatches)
                start, end = counter * params['batchsize'], (counter + 1) * params['batchsize']
                if start == 0:
                    effective_pass += 1
                    np.random.shuffle(samples)
                    if effective_pass > params['effective_pass']:
                        break
                minibatch = samples[start:end]
                # print(minibatch)
                # construct optimality measure xi_alpha
                gradfx_minibacth = self.prob.grad(x, minibatch) + 0.0
                x_trial = x - alpha * v
                # gradfx_trial_minibatch is a function of alpha
                fx_trial_minibacth = self.prob.forward(x_trial, minibatch)
                gradfx_trial_minibatch = torch.autograd.grad(fx_trial_minibacth, x_trial, retain_graph=True, create_graph=True)
                xi_alpha = torch.linalg.norm(gradfx_trial_minibatch[0] - gradfx_minibacth + v) ** 2
                # perform one newton step to find an approximate minimizer of xi_alpha
                # this operation will not set value to alpha.grad; no need to zero out gradient
                grad_alpha = torch.autograd.grad(xi_alpha, alpha, retain_graph=True, create_graph=True)
                hess_alpha = torch.autograd.grad(grad_alpha, alpha)
                alpha_approx = alpha - grad_alpha[0] / hess_alpha[0]
                stepsize = min(alpha_approx.data, alpha_max)
                delta = self._cal_delta(alpha_approx, iteration, beta=0.999).data
                alpha_max = 1 / delta
                # print(alpha_approx.data, stepsize, alpha_max)

                # perform update
                with torch.no_grad():
                    x.sub_(stepsize * v)
                v.add_(self.prob.grad(x, minibatch) - gradfx_minibacth)
                v_norm = torch.linalg.norm(v)
                iteration += 1
            # print(f'effective pass:{effective_pass-1} | acc:{v_norm}')
            print(f' | {iteration:5d}  {effective_pass:5d}   {v_norm:3.4e}')
            epoch += 1
        print(f'-------------------------------------------------------------------')
        print(f'Exit: {flag}')
        result = {'x': x.detach(), 'fx': fseq[-1], 'gradNorm': gradfx_full_norm, 'fseq': fseq}
        return result
