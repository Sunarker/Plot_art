import numpy as np
import matplotlib.pyplot as plt

dataname = 'mnist'
dataname = 'epsilon'
dataname = '20news'
dataname = 'cifar10'

listmis = ['0.8','0.9','1.0','1.1','1.2']

mycolor = np.array([[224,32,32],
                    [255,192,0],
                    [32,160,64],
                    [48,96,192],
                    [192,48,192]])/255.0
mylinewidth = 3

numepochs = 200
epochs = np.arange(1,numepochs+1)

def get_data(mid):    
    name = 'misspecified/'+dataname+'_'+listmis[mid]+'.csv'
    data = np.genfromtxt(name,delimiter=',',skip_header=1)
    m = []
    for i in xrange(1,numepochs+1):
        tmp = data[data[:,1]==i,3]
        m.append(np.mean(tmp))
#    t = [m[0]]*2+m+[m[-1]]*2
#    for i in xrange(numepochs):
#        m[i] = np.mean(t[i:i+5])
    return m

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8,6))

nnpu = [0]*5
m_risk = 1
for i in xrange(5):
    m = get_data(i)
#    m = [np.mean(m[j:j+5]) for j in xrange(0,numepochs,5)]
#    nnpu[i], = ax.plot(epochs[4:numepochs:5],m,
#        '-',color=mycolor[i,:],linewidth=mylinewidth,label=listmis[i])
#    if m_risk>np.min(m):
#        m_risk,m_idx,m_mis = np.min(m),np.argmin(m)*5+5,i
    nnpu[i], = ax.plot(epochs,m,
        '-',color=mycolor[i,:],linewidth=mylinewidth,label=listmis[i])
    if m_risk>np.min(m):
        m_risk,m_idx,m_mis = np.min(m),np.argmin(m)+1,i
#ax.plot(m_idx,m_risk,'o',color=mycolor[m_mis,:],markersize=14)
ax.axvline(m_idx,linestyle='-.',color=mycolor[m_mis,:])
ax.axhline(m_risk,linestyle='-.',color=mycolor[m_mis,:])

ax.set_xlabel('Epoch',fontsize=16)
ax.set_ylabel('Risk w.r.t. zero-one loss',fontsize=16)
ax.set_xlim(0,numepochs+1)
if dataname == 'mnist':
#    ax.set_ylim(0.045,0.215)
    ax.set_ylim(0.04,0.3)
    ax.legend(handles=nnpu,fontsize=14,loc='upper right')
elif dataname == 'epsilon':
#    ax.set_ylim(0.295,0.405)
    ax.set_ylim(0.28,0.46)
elif dataname == '20news':
#    ax.set_ylim(0.205,0.26)
    ax.set_ylim(0.2,0.3)
elif dataname == 'cifar10':
#    ax.set_ylim(0.18,0.24)
    ax.set_ylim(0.17,0.3)

plt.show()
fig.savefig('misspecified_'+dataname+'.pdf',bbox_inches='tight')
