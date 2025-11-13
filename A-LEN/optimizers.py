import numpy as np 
import time 
import math 
import torch 

def CRN_step(x, Q, U, g, M, lamb_guess, args):
    # def func(lamb):
    #     return M * np.linalg.norm(Q @ np.diag(1/(U + lamb)) @ Q.T @ g) - lamb
    # lamb = fsolve(func, lamb_guess, xtol=args.eps)
    # x = x - Q @ np.diag(1/(U + lamb)) @ Q.T @ g

    # Univariate Newton method to implement the CRN oracle
    n = g.shape[0]
    r = lamb_guess / M
    for j in range(args.K):
        w = 1/(U+M*r)
        diag = w.reshape(n,1)
        diag_cubic = (w**3).reshape(n,1)
        Qg = Q.T @ g
        u = Q @ (diag * Qg)
        phi = r - torch.norm(u) 
        if torch.abs(phi) < args.eps:
            break 
        d_phi = 1 + M / torch.norm(u) * Qg.T @ (diag_cubic * Qg) 
        r = r - phi / d_phi[0,0]
    
    lamb = M * r 
    x = x - u
    return x, lamb

## Accelerated gradient descent
def run_AGD(args, gradient, function_value, x0):
    time_lst_AGD = []
    gap_lst_AGD = []
    ellapse_time = 0.0
    x = x0
    x_pre = x0
    y = x0
    i = 0
    gamma = args.gamma
    while True:
        gap = function_value(x)
        gap_lst_AGD.append(gap)
        time_lst_AGD.append(ellapse_time)
        
        if i % 10 == 0:
            print('AGD: %d/%d  Epoch %d | Loss %.4f' % (int(ellapse_time), int(args.training_time), i, gap))
        start = time.time()
        y = x + gamma * ( x -x_pre)
        x_pre = x
        x = y - args.eta * gradient(y)
        end = time.time()
        ellapse_time += end - start

        if ellapse_time > args.training_time:
            break 
        i = i+1 
    return time_lst_AGD, gap_lst_AGD

def run_LazyCRN(args, gradient, function_value, Hessian, x0, M, m, lamb=1):
    time_lst_CRN = []
    gap_lst_CRN = []

    i =  0
    x = x0
    ellapse_time = 0.0
    while True:
        time_lst_CRN.append(ellapse_time)
        gap = function_value(x)
        gap_lst_CRN.append(gap)

        if i % 10 == 0:
            print('CRN-%d: %d/%d Epoch %d | Loss %.4f' % (m, int(ellapse_time), int(args.training_time),  i, gap))

        start = time.time()
        ## Snapshot
        if i % m == 0:
            H = Hessian(x)
            U, Q = torch.linalg.eigh(H)
        
        # Lazy CRN update
        g =  gradient(x)
        x, lamb = CRN_step(x, Q, U, g, M=M, lamb_guess=lamb, args=args) 

        end = time.time()
        ellapse_time += end - start

        if ellapse_time > args.training_time:
            break 
        i = i+1 
    
    return time_lst_CRN, gap_lst_CRN

def run_MS(args, gradient, function_value, x0, MS, name, alpha=2, lamb_guess=1):
    x_best = x0 
    f_best = function_value(x0)

    def update_best_point(x, f, x_best, f_best):
        if f < f_best:
            x_best = x
            f_best = f
        return x_best, f_best
    
    v0 = x0
    A = 0
    
    time_lst_MS = []
    gap_lst_MS = []

    ellapse_time = 0.0 
    start = time.time()
    tilde_x, lamb, y_best, f_y = MS(x0, lamb_guess=lamb_guess)
    x_best, f_best = update_best_point(y_best, f_y, x_best, f_best)
    end = time.time()
    ellapse_time += end - start
    time_lst_MS.append(ellapse_time)
    gap_lst_MS.append(f_best)

    lamb_prime = lamb_guess 
    x = x0
    v = v0 
    i = 0
    while True:

        time_lst_MS.append(ellapse_time)
        f_x = function_value(x)
        x_best, f_best = update_best_point(x, f_x, x_best, f_best)
        gap_lst_MS.append(f_best)
        if i % 10 == 0:
            print('%s: %d/%d Epoch %d | Loss %.4f' % (name, int(ellapse_time), int(args.training_time), i, f_best))

        start = time.time()

        a_prime = (1 + math.sqrt(1 + 4 * lamb_prime * A)) / (2* lamb_prime)
        A_prime = A + a_prime 
        bar_x = A / A_prime * x + a_prime / A_prime * v 
        if i > 0:
            tilde_x, lamb, y_best, f_y = MS(bar_x, lamb_prime)
            x_best, f_best = update_best_point(y_best, f_y, x_best, f_best)

        ##  Search free Newton step
        if lamb < lamb_prime:
            a = a_prime
            A_next = A_prime 
            x_next = tilde_x
            lamb_prime /= alpha 
        else:
            gamma = lamb_prime / lamb 
            a = gamma * a_prime 
            A_next = A + a 
            x_next = (1-gamma) * A / A_next * x + gamma * A_prime / A_next * tilde_x 
           
            lamb_prime *= alpha 
        
        ## Extra Gradient step 

        v_next = v - a * gradient(tilde_x)

        # Update
        x, v, A = x_next, v_next, A_next

        # a_prime = (1 / (2 * lamb_prime)) * (1 + math.sqrt(1 + 4 * A * lamb_prime))
        # A_next = A + a_prime

        # # momentum 
        # y = (A / A_next) * x + (a_prime / A_next) * v

        # # MS oracle 
        # w_next, lamb  = MS(y, lamb_prime)
        # grad = gradient(w_next)

        # # first iteration
        # if i == 0:
        #     lamb_prime = lamb 
        #     a_prime = (1 / (2 * lamb_prime)) * (1 + math.sqrt(1 + 4 * A * lamb_prime))
        #     A_next = A + a_prime
        
        # A_prime = A_next
        # gamma = min(1, lamb_prime / lamb)
        # a = gamma * a_prime
        # A_next = A + a
        # v_next = v - a * grad
        # x_next = (1 - gamma) * A * x / A_next + gamma * A_prime * w_next / A_next
        
        end = time.time()
        ellapse_time += end - start

        if ellapse_time > args.training_time:
            break 

        i = i+1 
    return time_lst_MS, gap_lst_MS

## A-NPE
def run_ANPE(args, gradient, function_value, Hessian, x0, M):
    def MS(x, lamb_guess):
        U, Q = torch.linalg.eigh(Hessian(x))
        g = gradient(x)
        x, lamb =  CRN_step(x, Q, U, g, M=M, lamb_guess=lamb_guess, args=args)
        return x, lamb, x, function_value(x)
    
    time_lst, gap_lst =  run_MS(args, gradient, function_value, x0, MS, name='A-NPE')
    return time_lst, gap_lst 

def run_ALEN(args, gradient, function_value, Hessian, x0, M, m, gamma=None):
    if gamma is None:
        gamma = M / m / m

    def MS(x, lamb_guess):
        y = x
        n = x.shape[0]
        lamb = lamb_guess
        ## Lazy-CRN method on the second-order proximal point sub-problem

        y_best  = y
        f_best = function_value(y)

        K = m 
        for i in range(K):
            ## Snapshot
            if i % m == 0:
                yx = y - x
                diff = torch.norm(yx)
                H = Hessian(y)
                if diff >= 1e-10:
                    H = H  + 2 * gamma * (diff * torch.eye(n) + yx * yx.reshape((1,-1)) / diff)
                U, Q = torch.linalg.eigh(H)

            # Lazy CRN update
            g =  gradient(y) + gamma * torch.norm(y-x) * (y-x)
            y, lamb = CRN_step(y, Q, U, g, M=M+m*gamma, lamb_guess=lamb, args=args) 

            # record the best y 
            f = function_value(y)
            if f < f_best:
                f_best = f 
                y_best = y

        lamb = gamma * torch.norm(y-x)
        return y, lamb, y_best, f_best
    
    time_lst, gap_lst=  run_MS(args, gradient, function_value, x0, MS, name='A-LEN-' + str(m))
    return time_lst, gap_lst