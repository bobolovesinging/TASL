"""Five-seed final-policy utility and uncertainty, with raw curves in CSV."""
from pathlib import Path
import csv,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
fig,axes=plt.subplots(1,2,figsize=(6.5,2.65),sharey=True,layout='constrained')
export=[];names=['fedavg','fltrust','tasl_full','lasa'];labels=['FedAvg','FLTrust','TASL','LASA'];colors=['#666666','#386cb0','#c17c16','#73529c']
for ax,attack,title in zip(axes,['none','label_flip'],['(a) Clean','(b) 40% label flipping']):
 for method,label,color in zip(names,labels,colors):
  paths=sorted((ROOT/'results/c6_repeated').glob(f'{method}_{attack}_seed*/metrics.json'));assert len(paths)==5
  values=np.array([json.loads(p.read_text())['acc_history'] for p in paths]);assert values.shape==(5,50)
  mean=values.mean(0);sd=values.std(0,ddof=1);x=np.arange(1,51)
  ax.plot(x,mean,label=label,color=color,lw=1.2);ax.fill_between(x,mean-sd,mean+sd,color=color,alpha=.14,lw=0)
  for p,row in zip(paths,values):
   seed=int(p.parent.name.split('seed')[-1])
   export.extend([dict(condition=attack,method=label,seed=seed,round=int(r),accuracy=float(a)) for r,a in zip(x,row)])
 ax.set(xlim=(1,50),ylim=(0,100),xlabel='Communication round',title=title);ax.grid(axis='y',alpha=.15,lw=.5)
axes[0].set_ylabel('Test accuracy (%)');axes[1].legend(loc='lower right',ncol=2,fontsize=7)
out=ROOT/'results';fig.savefig(out/'r2_final_learning_curves.pdf');fig.savefig(out/'r2_final_learning_curves.svg');fig.savefig(out/'r2_final_learning_curves.png',dpi=300)
with (out/'learning_curves.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
