'''
File: GradientDescent.py
Author: Yutong Dai (yutongdai95@gmail.com)
File Created: 2021-03-10 00:36
Last Modified: 2021-03-10 01:38
--------------------------------------------
Description:
'''
import torch

"""Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value. True for success, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """


class GradientDescent:
    """
    Perform gradient descent with a constant stepsize rule.
    """

    def __init__(self, prob):
        """Constructor,
        Args:
            prob (object): a object that provides two methods:
                            prob.forward(x): return the function value evaluate a the point x
                            prob.geval(x): return the gradient value evaluate a the point x
        Returns:
            None
        """
        self.prob = prob

    def solve(self, x0, params):
        """Run gradient descent.
        Args:
            x0 (torch.tensor): starting point; requires_grad = True
            params (dict): a dictionary contains all parameters for the algorithm:
                            'stepsize' (float): a scalar used in the gradient descent; default to 1;
                            'tol' (float): desired precision; default to 1e-4;
                            'maxiter': max number iterations to run; default to 100.
                            'printlevel': control the verbosity of the output; default to 1;
                                            0: print termination final status; 
                                            1: print per iteration information
                            'printevery': print the header for 'printevery' iterations; default to 10.
                                          only takes effect if printlevel is set to 1.

        Returns:
            result (dict): a dictionary reporting the final results.
                            'x' (torch.tensor): final solution
                            'fx' (torch.tensor): final objective value
                            'gradNorm' (torch.tensor): the gradient of the objective at the returned point
                            'fseq' (list): a history of function value sequence 
        """
        if not params:
            params = {'stepsize': 1.0, 'tol': 1e-4, 'maxiter': 100,
                      'printlevel': 1, 'printevery': 10}

        if 'stepsize' not in params.keys():
            self.stepsize = 1
        else:
            self.stepsize = params['stepsize']
        if 'tol' not in params.keys():
            self.tol = 1e-4
        else:
            self.tol = params['tol']
        if 'maxiter' not in params.keys():
            self.maxiter = 100
        else:
            self.maxiter = params['maxiter']
        if 'printlevel' not in params.keys():
            self.printlevel = 1
        else:
            self.printlevel = params['printlevel']
        if 'printevery' not in params.keys():
            self.printevery = 10
        else:
            self.printevery = params['printevery']
        x = x0
        print(f"Computation is done on: {x0.device.type}")
        if not x.requires_grad:
            raise ValueError('Input x0 must be a tensor requires gradient')
        fx = self.prob.forward(x)
        gradfx = self.prob.grad(x)
        gradNorm = torch.linalg.norm(gradfx, 2)
        iter = 0
        fseq = [fx.item()]
        if self.printlevel > 0:
            print(f'*******************************************************************')
            print(f'Gradient Descent with constant stepsize rule. Version: (03/10/2021)')
            print(f' Algorithm parameters')
            for k, v in params.items():
                print(f' params: {k} | value:{v}')
            print(f'*******************************************************************')
            print(f' iter        f         |grad|')
            print(f'--------------------------------')
            print(f'{iter:5d}  {fx:3.4e}   {gradNorm:3.4e}')
        reltol = self.tol * max(1, gradNorm)
        while gradNorm > reltol and iter < self.maxiter:
            x.data -= self.stepsize * gradfx
            iter += 1
            x.grad.data.zero_()
            fx = self.prob.forward(x)
            fseq.append(fx.item())
            gradfx = self.prob.grad(x)
            gradNorm = torch.linalg.norm(gradfx, 2)
            if iter % self.printevery == 0:
                print(f' iter        f         |grad|')
            print(f'{iter:5d}  {fx:3.4e}   {gradNorm:3.4e}')
        print(f'--------------------------------')
        if gradNorm <= reltol:
            print("Exit: Find the optimal solution with the desired tolerance.")
        else:
            print("Exit: Reach the maximum number of iterations.")
        result = {'x': x.detach(), 'fx': fx, 'gradNorm': gradNorm, 'fseq': fseq}
        return result
