'''
File: SVRG.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-10 14:17
Last Modified: 2021-03-12 01:27
--------------------------------------------
Description:
'''

import torch
import numpy as np
np.random.seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SVRG:
    def __init__(self, prob):
        self.prob = prob
        self.nSamples = prob.m
        self.dim = prob.n

    def solve(self, x0, params):
        x = x0
        print(f"Computation is done on: {x0.device.type}")
        if not x.requires_grad:
            raise ValueError('Input x0 must be a tensor requires gradient')
        epoch = 0
        samples = [i for i in range(self.nSamples)]
        totalBatches = int(np.ceil(self.nSamples / params['batchsize']))
        flag = 'Reach the maximum number of iterations.'
        fseq = []
        if params['printlevel'] > 0:
            print(f'*******************************************************************')
            print(f'                    SVRG      Version: (03/11/2021)                ')
            print(f' Algorithm parameters')
            for k, v in params.items():
                print(f' params: {k} | value:{v}')
            print(f'*******************************************************************')
        while epoch <= 2:
            # outter loop: evaluate full gradient
            # clear gradient
            if x.grad is not None:
                x.grad.data.zero_()
            gradfx_full = self.prob.grad(x) + 0.0
            print(f'x:{x} | gfull:{gradfx_full}')
            gradfx_full_norm = torch.linalg.norm(gradfx_full)
            if epoch % params['printevery'] == 0:
                print(f' epoch       f         |grad| ')
            print(f'{epoch:5d}  {self.prob.loss:3.4e}   {gradfx_full_norm:3.4e}')
            fseq.append(self.prob.loss.item())

            # check termination
            if epoch == 0:
                gradfx0_norm = gradfx_full_norm
            if gradfx_full_norm <= params['tol'] * gradfx0_norm:
                flag = 'Find the optimal solution with the desired accuracy.'
                break

            # inner iteration
            x_trial = x.clone().detach().requires_grad_(True)
            for j in range(2):
                np.random.shuffle(samples)
                for i in range(2):
                    start, end = i * params['batchsize'], (i + 1) * params['batchsize']
                    minibatch = samples[start:end]
                    # if j == 1:
                    # print(f'xt:{x_trial}, x:{x}')
                    gradfx_minibacth = self.prob.grad(x, minibatch)
                    if i == 0 and j == 0:
                        gradfx_trial_minibacth = gradfx_minibacth
                    else:
                        gradfx_trial_minibacth = self.prob.grad(x_trial, minibatch)
                    v = gradfx_trial_minibacth - gradfx_minibacth + gradfx_full
                    print(f'bacth:{minibatch} | gxt:{gradfx_trial_minibacth} | gx:{gradfx_minibacth} | v:{v}')
                    with torch.no_grad():
                        x_trial.sub_(params['stepsize'] * v)
                    print('==')
                print('=========')
            x = x_trial.clone().detach().requires_grad_(True)
            epoch += 1
        print(f'-------------------------------------------------------------------')
        print(f'Exit: {flag}')
        result = {'x': x.detach(), 'fx': fseq[-1], 'gradNorm': gradfx_full_norm, 'fseq': fseq}
        return result
