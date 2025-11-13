import argparse
import numpy as np 
import matplotlib.pyplot as plt 
import random 
import scipy 
import math
from scipy.optimize import fsolve
import time 
import torch
from sklearn.datasets import load_svmlight_file 
import sys 
from scipy.io import loadmat, savemat

parser = argparse.ArgumentParser(description='')
parser.add_argument('--training_time',  type=float, default=100.0, help='total training time')
parser.add_argument('--seed',  type=int, default=42, help='seed of random number')
parser.add_argument('--dataset',  type=str, default='a9a', help='dataset')
parser.add_argument('--eps',  type=float, default=1e-6, help='precision for cubic subsolver')
parser.add_argument('--K', type=int, default=20, help='numbers of inner loop for cubic susolver')
parser.add_argument('--m', type=int, default=100, help='frequency of lazy Hessian updates')
parser.add_argument('--gamma', type=float, default=0.9, help='momemtum in AGD')
args = parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

set_seed(args.seed)

def min_max_normalize(X):
    # Get the minimum and maximum values of the matrix
    min_val = X.min(axis=0)  # Minimum value for each column
    max_val = X.max(axis=0)  # Maximum value for each column
    
    # Normalize the matrix
    X_normalized = (X - min_val) / (max_val - min_val)
    
    return X_normalized

# LIBSVM datasets
# Before running the codes, you should download the dataset from "https://www.csie.ntu.edu.tw/~cjlin/libsvm/" and put a txt file in './Data'

if args.dataset == 'lawschool':
    m = loadmat('./Data/LSTUDENT_DATA1.mat')
    X=np.array(m['A']).astype("float")
    y=np.array(m['b']).astype("float")
    n,d = X.shape
    X = min_max_normalize(X)

elif args.dataset != 'synthetic':
    source = './Data/' + args.dataset + '.txt'
    data = load_svmlight_file(source)
    ## We define the global varaiables X,y, n,d
    X = scipy.sparse.csr_matrix(data[0]).toarray().astype("double")
    n,d = X.shape

    X = min_max_normalize(X)
    y = np.array(data[1]).reshape(n,1).astype("double")

if args.dataset in ['a9a' , 'w8a', 'splice', 'madelon', 'sonar', 'heart', 'ijcnn1']:
    y = (y+1)/2
elif args.dataset in ['mushrooms', 'covtype']:
    y = y-1 
elif args.dataset == 'synthetic':
    n = 1000 
    d = 200
    def sample_from_sphere(radius, num_samples):
        vec = np.random.randn(num_samples, d) 
        vec /= np.linalg.norm(vec, axis=1, keepdims=True) 
        return vec * radius  
    
    mean1 = sample_from_sphere(0.5, 1)[0]
    mean2 = sample_from_sphere(0.5, 1)[0]

    data1 = np.random.normal(loc=mean1, scale=1, size=(n // 2, d))
    data2 = np.random.normal(loc=mean2, scale=1, size=(n // 2, d))

    synthetic_data = np.vstack((data1, data2))
    labels = np.array([0] * (n // 2) + [1] * (n // 2))

    indices = np.arange(n)
    np.random.shuffle(indices)
    X = synthetic_data[indices]
    X = min_max_normalize(X)
    y = labels[indices].reshape(n,1)

X = torch.tensor(X)
y = torch.tensor(y)
XT = X.T 
lamb = 1/n 

def gradient(x):
    h = 1 / (1 + torch.exp(-X @ x)) 
    gradient = XT @ (h - y) + lamb * x 
    return gradient 

def Hessian(x):
    h = 1 / (1 + torch.exp(-X @ x)) + lamb
    hessian = XT @ (h * X) 
    return hessian 

def function_value(x):
    h = 1 / (1 + torch.exp(-X @ x)) 
    objective_value = -torch.sum(y * torch.log(h + 1e-10) + (1 - y) * torch.log(1 - h + 1e-10))  / n + lamb / 2 * torch.sum(x*x)
    return objective_value

x0 =np.zeros((d,1)). astype('float')
x0 = torch.tensor(x0)

L1 = L2 = torch.linalg.norm(XT @ X, ord=2).item()
args.eta = 1 / L1

# For simplicity we fix M=L2. Tuning parameter M would definitely lead to better performances.
M = L2

import pickle 
from optimizers import * 

time_lst_ALEN, gap_lst_ALEN = run_ALEN(args, gradient, function_value, Hessian, x0, M, m=args.m)
with open('./result/'+  'Logistic.' + args.dataset + '.time_lst_ALEN' + str(args.m) + '.pkl', 'wb') as f:
    pickle.dump(time_lst_ALEN, f)
with open('./result/'+  'Logistic.' + args.dataset  + '.gap_lst_ALEN' + str(args.m) + '.pkl', 'wb') as f:
    pickle.dump(gap_lst_ALEN, f)

# AGD: Optimal first-order method
time_lst_AGD, gap_lst_AGD = run_AGD(args, gradient, function_value, x0)
with open('./result/' + 'Logistic.' + args.dataset + '.time_lst_AGD' + str(args.gamma) + '.pkl', 'wb') as f:
    pickle.dump(time_lst_AGD, f)
with open('./result/' + 'Logistic.' + args.dataset + '.gap_lst_AGD' + str(args.gamma) + '.pkl', 'wb') as f:
    pickle.dump(gap_lst_AGD, f)

# For the synthetic dataset, we found that M LCRN should be larger to ensure convergence
if args.dataset == 'synthetic':
    M_LCRN = args.m * L2 
else:
    M_LCRN = L2
time_lst_LCRN, gap_lst_LCRN = run_LazyCRN(args, gradient, function_value, Hessian, x0, M_LCRN, m=args.m)
with open('./result/'+  'Logistic.' + args.dataset + '.time_lst_CRN_' + str(args.m) + '.pkl', 'wb') as f:
    pickle.dump(time_lst_LCRN, f)
with open('./result/'+  'Logistic.' + args.dataset  + '.gap_lst_CRN_' + str(args.m) + '.pkl', 'wb') as f:
    pickle.dump(gap_lst_LCRN, f)

# A-NPE: Optimal second-order method and A-LEN which improves its computational cost 
time_lst_ANPE, gap_lst_ANPE = run_ANPE(args, gradient, function_value, Hessian, x0, M)
with open('./result/'+  'Logistic.' + args.dataset + '.time_lst_ANPE.pkl', 'wb') as f:
    pickle.dump(time_lst_ANPE, f)
with open('./result/'+  'Logistic.' + args.dataset  + '.gap_lst_ANPE.pkl', 'wb') as f:
    pickle.dump(gap_lst_ANPE, f)

# CRN by Nesterov and its lazy version by Doikov et al.
time_lst_CRN, gap_lst_CRN = run_LazyCRN(args, gradient, function_value, Hessian, x0, M, m=1)
with open('./result/'+  'Logistic.' + args.dataset + '.time_lst_CRN.pkl', 'wb') as f:
    pickle.dump(time_lst_CRN, f)
with open('./result/'+  'Logistic.' + args.dataset  + '.gap_lst_CRN.pkl', 'wb') as f:
    pickle.dump(gap_lst_CRN, f)


