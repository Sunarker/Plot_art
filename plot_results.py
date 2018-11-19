import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rc('xtick', labelsize=22)
plt.rc('ytick', labelsize=22)

#dataname = 'column'
#dataname = 'tridiagonal'
dataname = 'block'

methods = ['NOISY','F-correction','S-adaptation','MASKING','CLEAN']

raw_data = np.genfromtxt('%s_results/%s_all.csv'%(dataname,dataname),
                     delimiter=',',skip_header=0)

data = {}
for e in xrange(len(methods)):
    tmp = raw_data[:,(e*2):((e+1)*2)]
    data[methods[e]] = tmp[~np.isnan(tmp).any(axis=1)]
    #print(data[methods[e]])
    #print(data[methods[e]].shape)

# sample an subset for MASKING so that the plot will be more clear
length = data[methods[3]].shape[0]
length_base = data[methods[0]].shape[0]
data[methods[3]] = data[methods[3]][range(0,length,int(length/length_base)),:]

mycolor = np.array([[224,32,32],
                    [255,192,0],
                    [32,160,64],
                    [48,96,192],
                    [192,48,192]])/255.0
mymarker = ['o','s','*','H','D']
mylinewidth = 3
mymarkersize = 10

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 10))

plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)

m0_, = ax.plot(data[methods[0]][:,0],data[methods[0]][:,1],'-',
                 color=mycolor[0,:],marker=mymarker[0],linewidth=mylinewidth,markersize=mymarkersize,label=methods[0])
m1_, = ax.plot(data[methods[1]][:,0],data[methods[1]][:,1],'-',
                 color=mycolor[1,:],marker=mymarker[1],linewidth=mylinewidth,markersize=mymarkersize,label=methods[1])
m2_, = ax.plot(data[methods[2]][:,0],data[methods[2]][:,1],'-',
                 color=mycolor[2,:],marker=mymarker[2],linewidth=mylinewidth,markersize=mymarkersize,label=methods[2])
m3_, = ax.plot(data[methods[3]][:,0],data[methods[3]][:,1],'-',
                 color=mycolor[3,:],marker=mymarker[3],linewidth=mylinewidth,markersize=mymarkersize,label=methods[3])
m4_, = ax.plot(data[methods[4]][:,0],data[methods[4]][:,1],'-',
                 color=mycolor[4,:],marker=mymarker[4],linewidth=mylinewidth,markersize=mymarkersize,label=methods[4])


# ax.fill_between(epochs,lowers[epochs,5],uppers[epochs,5],
#                 color=mycolor[3,:],alpha=0.2)
# ax.fill_between(epochs,lowers[epochs,4],uppers[epochs,4],
#                 color=mycolor[3,:],alpha=0.2)
# ax.fill_between(epochs,lowers[epochs,3],uppers[epochs,3],
#                 color=mycolor[0,:],alpha=0.2)
# ax.fill_between(epochs,lowers[epochs,2],uppers[epochs,2],
#                 color=mycolor[0,:],alpha=0.2)
# ax.fill_between(epochs,lowers[epochs,1],uppers[epochs,1],
#                 color=mycolor[1,:],alpha=0.4)
# ax.fill_between(epochs,lowers[epochs,0],uppers[epochs,0],
#                 color=mycolor[1,:],alpha=0.4)

# ax.axhline(0,linestyle='-.',color='black')

ax.set_xlabel('Iteration',fontsize=30,fontweight='bold')
ax.set_ylabel('Accuracy',fontsize=30,fontweight='bold')
ax.set_xlim(0,150000)
if dataname == 'column':
    ax.set_ylim(0.4,0.9)
    #ax.legend(handles=[m0_,m1_,m2_,m3_,m4_],loc='lower right',fontsize=14)
elif dataname == 'tridiagonal':
    ax.set_ylim(0.4,0.9)
    #ax.legend(handles=[m0_,m1_,m2_,m3_,m4_],loc='lower right',fontsize=14)
elif dataname == 'block':
    ax.set_ylim(0.3,0.6)
    ax.legend(handles=[m0_,m1_,m2_,m3_,m4_],loc='lower right')
    legend = plt.legend(fontsize=26)
    frame = legend.get_frame()
    frame.set_alpha(1)
    frame.set_facecolor('none')

#plt.show()
fig.savefig(dataname+'_results.pdf',dpi=100,bbox_inches='tight')
