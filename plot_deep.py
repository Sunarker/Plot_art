import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

dataname = 'mnist'
dataname = 'epsilon'
dataname = '20news'
dataname = 'cifar10'

data = np.genfromtxt('misspecified/'+dataname+'_1.0.csv',
                     delimiter=',',skip_header=1)

mycolor = np.array([[224,32,32],
                    [255,192,0],
                    [32,160,64],
                    [48,96,192],
                    [192,48,192]])/255.0
mylinewidth = 3

numepochs = 200
epochs = np.arange(1,numepochs+1)

means = np.zeros([numepochs+1,6])
uppers = np.zeros([numepochs+1,6])
lowers = np.zeros([numepochs+1,6])
for i in xrange(1,numepochs+1):
    tmp = data[data[:,1]==i,2:8]
    means[i] = np.mean(tmp,0)
    uppers[i] = means[i]+np.std(tmp,0)
    lowers[i] = means[i]-np.std(tmp,0)

plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(8,6.5))

pn_te, = ax.plot(epochs,means[epochs,5],'-',
                 color=mycolor[3,:],linewidth=mylinewidth,label='PN test')
pn_tr, = ax.plot(epochs,means[epochs,4],'--',
                 color=mycolor[3,:],linewidth=mylinewidth,label='PN train')
upu_te, = ax.plot(epochs,means[epochs,3],'-',
                 color=mycolor[0,:],linewidth=mylinewidth,label='uPU test')
upu_tr, = ax.plot(epochs,means[epochs,2],'--',
                 color=mycolor[0,:],linewidth=mylinewidth,label='uPU train')
nnpu_te, = ax.plot(epochs,means[epochs,1],'-',
                   color=mycolor[1,:],linewidth=mylinewidth,label='nnPU test')
nnpu_tr, = ax.plot(epochs,means[epochs,0],'--',
                   color=mycolor[1,:],linewidth=mylinewidth,label='nnPU train')

ax.fill_between(epochs,lowers[epochs,5],uppers[epochs,5],
                color=mycolor[3,:],alpha=0.2)
ax.fill_between(epochs,lowers[epochs,4],uppers[epochs,4],
                color=mycolor[3,:],alpha=0.2)
ax.fill_between(epochs,lowers[epochs,3],uppers[epochs,3],
                color=mycolor[0,:],alpha=0.2)
ax.fill_between(epochs,lowers[epochs,2],uppers[epochs,2],
                color=mycolor[0,:],alpha=0.2)
ax.fill_between(epochs,lowers[epochs,1],uppers[epochs,1],
                color=mycolor[1,:],alpha=0.4)
ax.fill_between(epochs,lowers[epochs,0],uppers[epochs,0],
                color=mycolor[1,:],alpha=0.4)

ax.axhline(0,linestyle='-.',color='black')

ax.set_xlabel('Epoch',fontsize=16)
ax.set_ylabel('Risk w.r.t. zero-one loss',fontsize=16)
ax.set_xlim(0,numepochs+1)
loc = ticker.MultipleLocator(base=0.1)
if dataname == 'mnist':
    ax.set_ylim(-0.4,0.5)
    ax.legend(handles=[pn_te,pn_tr,upu_te,upu_tr,nnpu_te,nnpu_tr],fontsize=14)
elif dataname == 'epsilon':
    ax.set_ylim(-0.025,0.5)
    loc = ticker.MultipleLocator(base=0.05)
elif dataname == '20news':
    ax.set_ylim(-0.1,0.4)
elif dataname == 'cifar10':
    ax.set_ylim(-0.4,0.5)
ax.yaxis.set_major_locator(loc)

plt.show()
fig.savefig('deep_'+dataname+'.pdf',bbox_inches='tight')
