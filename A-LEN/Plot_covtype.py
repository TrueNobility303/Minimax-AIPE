import pickle 
import matplotlib.pyplot as plt 
import argparse 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset', type=str, default='covtype', help='name of dataset')
args = parser.parse_args()

with open('./result/'+  'Logistic.' + args.dataset + '.gap_lst_AGD.pkl', 'rb') as f:
    gap_lst_AGD = pickle.load(f)
with open('./result/'+  'Logistic.' + args.dataset + '.time_lst_AGD.pkl', 'rb') as f:
    time_lst_AGD = pickle.load(f)

with open('./result/'+ 'Logistic.' + args.dataset + '.gap_lst_ALEN_10.pkl', 'rb') as f:
    gap_lst_ALEN = pickle.load(f)
with open('./result/'+ 'Logistic.' + args.dataset + '.time_lst_ALEN_10.pkl', 'rb') as f:
    time_lst_ALEN = pickle.load(f)

with open('./result/'+  'Logistic.' + args.dataset + '.gap_lst_CRN_10.pkl', 'rb') as f:
    gap_lst_LCRN = pickle.load(f)
with open('./result/'+  'Logistic.' + args.dataset + '.time_lst_CRN_10.pkl', 'rb') as f:
    time_lst_LCRN = pickle.load(f)

with open('./result/'+  'Logistic.' + args.dataset + '.gap_lst_ANPE.pkl', 'rb') as f:
    gap_lst_ANPE = pickle.load(f)
with open('./result/'+  'Logistic.' + args.dataset + '.time_lst_ANPE.pkl', 'rb') as f:
    time_lst_ANPE = pickle.load(f)

with open('./result/'+  'Logistic.' + args.dataset + '.gap_lst_CRN.pkl', 'rb') as f:
    gap_lst_CRN = pickle.load(f)
with open('./result/'+  'Logistic.' + args.dataset + '.time_lst_CRN.pkl', 'rb') as f:
    time_lst_CRN = pickle.load(f)


opt = min(min(min(gap_lst_ANPE[-1], gap_lst_ALEN[-1]), gap_lst_CRN[-1]), gap_lst_LCRN[-1]) 
ini = max(max(max(gap_lst_ANPE[-1], gap_lst_ALEN[-1]), gap_lst_CRN[-1]), gap_lst_LCRN[-1]) 
gap_lst_ANPE = [f-opt for f in gap_lst_ANPE]
gap_lst_AGD = [f-opt for f in gap_lst_AGD]
gap_lst_ALEN_100 = [f-opt for f in gap_lst_ALEN]
gap_lst_CRN = [f-opt for f in gap_lst_CRN]
gap_lst_CRN_100 = [f-opt for f in gap_lst_LCRN]

# time vs. gap 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font', size=21)
plt.figure()
plt.grid()
plt.xlim((0,600))
plt.ylim((1e-5,ini))
plt.yscale('log')


plt.plot(time_lst_AGD, gap_lst_AGD, ':b', label='AGD', linewidth=3)
plt.plot(time_lst_CRN, gap_lst_CRN, '-m^',   label='CRN', linewidth=3, markersize=12, markevery=int(len(gap_lst_CRN)/10))
plt.plot(time_lst_LCRN, gap_lst_CRN_100, '-gX',   label='L-CRN', linewidth=3, markersize=12, markevery=int(len(gap_lst_CRN_100)/10))
plt.plot(time_lst_ANPE, gap_lst_ANPE, '-ro', label='A-NPE', linewidth=3, markersize=12, markevery=int(len(gap_lst_ANPE)/10))
plt.plot(time_lst_ALEN, gap_lst_ALEN_100, '-k',  label='A-LEN', linewidth=3)
plt.legend(fontsize=18, loc='lower left')
plt.tick_params('x',labelsize=21)
plt.tick_params('y',labelsize=21)
plt.ylabel('gap')
plt.xlabel('time (s)')
plt.tight_layout()
plt.savefig('./img/Logistic.'+ args.dataset + '.gap.png')
plt.savefig('./img/Logistic.'+ args.dataset + '.gap.pdf', format = 'pdf')
