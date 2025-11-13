import pickle 
import matplotlib.pyplot as plt 
import argparse 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--n', type=int, default=50, help='size of problem')
parser.add_argument('--max_time', type=float, default=None, help='max running time')
args = parser.parse_args()

with open('./result/'+  'Toy.' + str(args.n) + '.gap_lst_AGD.pkl', 'rb') as f:
    gap_lst_AGD = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_AGD.pkl', 'rb') as f:
    time_lst_AGD = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.gap_lst_ANPE.pkl', 'rb') as f:
    gap_lst_ANPE = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_ANPE.pkl', 'rb') as f:
    time_lst_ANPE = pickle.load(f)
with open('./result/'+ 'Toy.' + str(args.n) + '.gap_lst_ALEN_2.pkl', 'rb') as f:
    gap_lst_ALEN_2 = pickle.load(f)
with open('./result/'+ 'Toy.' + str(args.n) + '.time_lst_ALEN_2.pkl', 'rb') as f:
    time_lst_ALEN_2 = pickle.load(f)
with open('./result/'+ 'Toy.' + str(args.n) + '.gap_lst_ALEN_10.pkl', 'rb') as f:
    gap_lst_ALEN_10 = pickle.load(f)
with open('./result/'+ 'Toy.' + str(args.n) + '.time_lst_ALEN_10.pkl', 'rb') as f:
    time_lst_ALEN_10 = pickle.load(f)
with open('./result/'+ 'Toy.' + str(args.n) + '.gap_lst_ALEN_100.pkl', 'rb') as f:
    gap_lst_ALEN_100 = pickle.load(f)
with open('./result/'+ 'Toy.' + str(args.n) + '.time_lst_ALEN_100.pkl', 'rb') as f:
    time_lst_ALEN_100 = pickle.load(f)

with open('./result/'+  'Toy.' + str(args.n) + '.gap_lst_CRN.pkl', 'rb') as f:
    gap_lst_CRN = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_CRN.pkl', 'rb') as f:
    time_lst_CRN = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.gap_lst_CRN_2.pkl', 'rb') as f:
    gap_lst_CRN_2 = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_CRN_2.pkl', 'rb') as f:
    time_lst_CRN_2 = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.gap_lst_CRN_10.pkl', 'rb') as f:
    gap_lst_CRN_10 = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_CRN_10.pkl', 'rb') as f:
    time_lst_CRN_10 = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.gap_lst_CRN_100.pkl', 'rb') as f:
    gap_lst_CRN_100 = pickle.load(f)
with open('./result/'+  'Toy.' + str(args.n) + '.time_lst_CRN_100.pkl', 'rb') as f:
    time_lst_CRN_100 = pickle.load(f)

opt = -2 * args.n / 3 
gap_lst_AGD = [f-opt for f in gap_lst_AGD]
gap_lst_ANPE = [f-opt for f in gap_lst_ANPE]
gap_lst_ALEN_2 = [f-opt for f in gap_lst_ALEN_2]
gap_lst_ALEN_10 = [f-opt for f in gap_lst_ALEN_10]
gap_lst_ALEN_100 = [f-opt for f in gap_lst_ALEN_100]
gap_lst_CRN = [f-opt for f in gap_lst_CRN]
gap_lst_CRN_2 = [f-opt for f in gap_lst_CRN_2]
gap_lst_CRN_10 = [f-opt for f in gap_lst_CRN_10]
gap_lst_CRN_100 = [f-opt for f in gap_lst_CRN_100]

def sliding_min(x):
    mm = x[0]
    for i in range(1,len(x)):
        if x[i] < mm:
            mm = x[i]
        else:
            x[i] = mm        
    return x

gap_lst_ANPE = sliding_min(gap_lst_ANPE)
gap_lst_ALEN_2 = sliding_min(gap_lst_ALEN_2) 
gap_lst_ALEN_10 = sliding_min(gap_lst_ALEN_10)
gap_lst_ALEN_100 = sliding_min(gap_lst_ALEN_100)

# time vs. gap 
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font', size=21)
plt.figure()
plt.grid()
if args.max_time is not None:
    plt.xlim((0,args.max_time))
#plt.yscale('log')
plt.plot(time_lst_AGD, gap_lst_AGD, ':b',   label='AGD', linewidth=3)
plt.plot(time_lst_CRN, gap_lst_CRN, '-m^',   label='CRN', linewidth=3, markersize=12, markevery=int(len(gap_lst_CRN)/10))
plt.plot(time_lst_CRN_2, gap_lst_CRN_2, '-d',   label='L-CRN-2', linewidth=3, markersize=12, markevery=int(len(gap_lst_CRN_2)/10))
plt.plot(time_lst_CRN_10, gap_lst_CRN_10, '->',   label='L-CRN-10', linewidth=3, markersize=12, markevery=int(len(gap_lst_CRN_10)/10))
plt.plot(time_lst_CRN_100, gap_lst_CRN_100, '-gX',   label='L-CRN-100', linewidth=3, markersize=12, markevery=int(len(gap_lst_CRN_100)/10))
plt.plot(time_lst_ANPE, gap_lst_ANPE, '-ro', label='A-NPE', linewidth=3, markersize=12, markevery=int(len(gap_lst_ANPE)/10))
plt.plot(time_lst_ALEN_2, gap_lst_ALEN_2, '-ys',  label='A-LEN-2', linewidth=3, markersize=12, markevery=int(len(gap_lst_ALEN_2)/10))
plt.plot(time_lst_ALEN_10, gap_lst_ALEN_10, '-cp',   label='A-LEN-10', linewidth=3, markersize=12, markevery=int(len(gap_lst_ALEN_10)/10))
plt.plot(time_lst_ALEN_100, gap_lst_ALEN_100, '-k',  label='A-LEN-100', linewidth=3, markersize=12, markevery=int(len(gap_lst_ALEN_100)/10))
plt.legend(fontsize=15, loc='lower right')
plt.tick_params('x',labelsize=21)
plt.tick_params('y',labelsize=21)
plt.ylabel('gap')
plt.xlabel('time (s)')
plt.tight_layout()
plt.savefig('./img/Synthetic.'+ str(args.n) + '.gap.png')
plt.savefig('./img/Synthetic.'+ str(args.n) + '.gap.pdf', format = 'pdf')
