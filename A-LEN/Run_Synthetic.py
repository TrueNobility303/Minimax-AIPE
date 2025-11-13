import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import scipy 
import math
from scipy.optimize import fsolve
import time 
from optimizers import * 
import torch 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--training_time',  type=float, default=100.0, help='total training time')
parser.add_argument('--n', type=int, default=100, help='size of problem')
parser.add_argument('--seed',  type=int, default=42, help='seed of random number')
parser.add_argument('--eta', type=float, default=1e-3, help='stepsize of accelerated gradient descent')
parser.add_argument('--eps',  type=float, default=1e-6, help='precision for cubic subsolver')
parser.add_argument('--K', type=int, default=20, help='numbers of inner loop for cubic susolver')
parser.add_argument('--gamma', type=float, default=0.9, help='momemtum in AGD')

args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

set_seed(args.seed)

# def QR(A, max_iterations=10, tolerance=1e-10):
#     A_k = A.copy()
#     for _ in range(max_iterations):
#         Q, R = np.linalg.qr(A_k)
#         A_k = R @ Q

#         off_diagonal_sum = np.sum(np.abs(A_k - np.diag(np.diagonal(A_k))))
#         if off_diagonal_sum < tolerance:
#             break
            
#     U = np.diagonal(A_k)
#     return U, Q
    
def generate_matrix(n):
    matrix = torch.zeros((n, n))
    matrix.fill_diagonal_(1)
    for i in range(n - 1):
        matrix[i][i+1] = -1
    return matrix

## Define the worse case function
A = generate_matrix(args.n)
AT = A.T 
def grad_pho(x):
    return x * torch.abs(x)

def gradient(x):
    g = AT @ grad_pho(A @ x) 
    g[0,0] -= 1
    return g 

def Hess_rho(x):
    return 2 * torch.abs(x[:,0]).reshape(args.n,1)

def Hessian(x):
    return AT @ (Hess_rho(A @ x) * A) 

def rho(x):
    y = torch.abs(x)
    return torch.sum(y*y*y) / 3

def function_value(x):
    return rho(A @ x) - x[0,0]

x0 = torch.zeros((args.n,1))
L = 2**3.5 

import pickle 
# AGD: Optimal first-order method
time_lst_AGD, gap_lst_AGD = run_AGD(args, gradient, function_value, x0)
with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_AGD.pkl', 'wb') as f:
    pickle.dump(time_lst_AGD, f)
with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_AGD.pkl', 'wb') as f:
    pickle.dump(gap_lst_AGD, f)

# # A-NPE: Optimal second-order method and A-LEN which improves its computational cost 
# time_lst_ANPE, gap_lst_ANPE = run_ANPE(args, gradient, function_value, Hessian, x0, L)
# with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_ANPE.pkl', 'wb') as f:
#     pickle.dump(time_lst_ANPE, f)
# with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_ANPE.pkl', 'wb') as f:
#     pickle.dump(gap_lst_ANPE, f)

# time_lst_ALEN_2, gap_lst_ALEN_2 = run_ALEN(args, gradient, function_value, Hessian, x0, L, m=2)
# with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_ALEN_2.pkl', 'wb') as f:
#     pickle.dump(time_lst_ALEN_2, f)
# with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_ALEN_2.pkl', 'wb') as f:
#     pickle.dump(gap_lst_ALEN_2, f)

# time_lst_ALEN_10, gap_lst_ALEN_10 = run_ALEN(args, gradient, function_value, Hessian, x0, L, m=10)
# with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_ALEN_10.pkl', 'wb') as f:
#     pickle.dump(time_lst_ALEN_10, f)
# with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_ALEN_10.pkl', 'wb') as f:
#     pickle.dump(gap_lst_ALEN_10, f)

# time_lst_ALEN_100, gap_lst_ALEN_100 = run_ALEN(args, gradient, function_value, Hessian, x0, L, m=100)
# with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_ALEN_100.pkl', 'wb') as f:
#     pickle.dump(time_lst_ALEN_100, f)
# with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_ALEN_100.pkl', 'wb') as f:
#     pickle.dump(gap_lst_ALEN_100, f)

# CRN by Nesterov and its lazy version by Doikov et al.
# time_lst_CRN, gap_lst_CRN = run_LazyCRN(args, gradient, function_value, Hessian, x0, L, m=1)
# with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_CRN.pkl', 'wb') as f:
#     pickle.dump(time_lst_CRN, f)
# with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_CRN.pkl', 'wb') as f:
#     pickle.dump(gap_lst_CRN, f)

# time_lst_CRN_2, gap_lst_CRN_2 = run_LazyCRN(args, gradient, function_value, Hessian, x0, L, m=2)
# with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_CRN_2.pkl', 'wb') as f:
#     pickle.dump(time_lst_CRN_2, f)
# with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_CRN_2.pkl', 'wb') as f:
#     pickle.dump(gap_lst_CRN_2, f)

# time_lst_CRN_10, gap_lst_CRN_10 = run_LazyCRN(args, gradient, function_value, Hessian, x0, L, m=10)
# with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_CRN_10.pkl', 'wb') as f:
#     pickle.dump(time_lst_CRN_10, f)
# with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_CRN_10.pkl', 'wb') as f:
#     pickle.dump(gap_lst_CRN_10, f)

# time_lst_CRN_100, gap_lst_CRN_100 = run_LazyCRN(args, gradient, function_value, Hessian, x0, L, m=100)
# with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_CRN_100.pkl', 'wb') as f:
#     pickle.dump(time_lst_CRN_100, f)
# with open('./result/'+  'Toy.' + str(args.n)  + '.gap_lst_CRN_100.pkl', 'wb') as f:
#     pickle.dump(gap_lst_CRN_100, f)

## Plot the figure for comparison 

# # time vs. gap 
# plt.rcParams['pdf.fonttype'] = 42
# plt.rcParams['ps.fonttype'] = 42
# plt.rc('font', size=21)
# plt.figure()
# plt.grid()
# plt.yscale('log')
# plt.plot(gap_lst_ANPE, ':r', label='A-NPE', linewidth=3)
# # plt.plot(time_lst_AGD, gap_lst_AGD, '-.b', label='AGD', linewidth=3)
# plt.plot(gap_lst_ALEN_2, '-g',  label='m=2', linewidth=3)
# plt.plot(gap_lst_ALEN_5, '-', color='#A9A9A9',  label='m=5', linewidth=3)
# plt.plot(gap_lst_CRN, '-k',  label='Lazy-CRN', linewidth=3)
# # plt.plot(time_lst_LEN_d, gnorm_lst_LEN_d, '-',   label='LEN(m=10d)', linewidth=3)
# # plt.plot(time_lst_ALEN, gap_lst_ALEN, '-k', label='A-LEN', linewidth=3)
# plt.legend(fontsize=20, loc='lower right')
# plt.tick_params('x',labelsize=21)
# plt.tick_params('y',labelsize=21)
# plt.ylabel('grad. norm')
# plt.xlabel('time (s)')
# plt.tight_layout()
# plt.savefig('./img/'+  'Synthetic.' +  str(args.n) + '.gap.png')
# plt.savefig('./img/'+  'Synthetic.' +  str(args.n) + '.gap.pdf', format = 'pdf')


