"""Illustrate horizontal MPI partitions, full-height ownership and one-cell halos."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
colors=['#a8cce4','#f2c79b','#c4dbad','#d9bfdf','#f3dca2','#b7dcd4']
fig=plt.figure(figsize=(11,4.3),layout='constrained')
ax=fig.add_subplot(121)
for j in range(2):
    for i in range(3):
        rank=j*3+i
        ax.add_patch(Rectangle((i*6,j*6),6,6,facecolor=colors[rank],edgecolor='#34495e',lw=2))
        ax.text(i*6+3,j*6+4.7,f'Rank {rank}',ha='center',fontsize=10,weight='bold',zorder=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
for x,y,w,h in [(2,1,2,2),(8,3,2,2),(14,1,2,3),(4,7,2,2),(9,7,3,2),(15,8,2,2)]:
    ax.add_patch(Rectangle((x,y),w,h,facecolor='#515b65',zorder=4))
for x in range(19): ax.plot([x,x],[0,12],color='white',lw=.45,zorder=2)
for y in range(13): ax.plot([0,18],[y,y],color='white',lw=.45,zorder=2)
for x,y,w,h in [(5,0,1,6),(12,0,1,6),(6,6,6,1)]:
    ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#bf3145',ls='--',lw=1.8,zorder=5))
ax.annotate('',xy=(6.8,2),xytext=(5.2,2),arrowprops=dict(arrowstyle='<->',color='#bf3145',lw=2),zorder=6)
ax.annotate('',xy=(12.8,2),xytext=(11.2,2),arrowprops=dict(arrowstyle='<->',color='#bf3145',lw=2),zorder=6)
ax.set(xlim=(-1.5,19),ylim=(-.2,13),xlabel='x cells',ylabel='y cells',title='(a) Top view: block decomposition')
ax.set_aspect('equal');ax.set_xticks([0,6,12,18]);ax.set_yticks([0,6,12])
ax.legend(handles=[Patch(facecolor='#515b65',label='Solid cells: zero normal flux'),Patch(facecolor='none',edgecolor='#bf3145',linestyle='--',label='Rank 1: neighbouring face halos')],loc='upper center',bbox_to_anchor=(.5,-.14),frameon=False,fontsize=8)
bx=fig.add_subplot(122,projection='3d')
for j in range(2):
    for i in range(3):
        x,y,z=i*2,j*2,0;w,h,d=2,2,3
        faces=[[(x,y,z),(x+w,y,z),(x+w,y,z+d),(x,y,z+d)],[(x,y,z+d),(x+w,y,z+d),(x+w,y+h,z+d),(x,y+h,z+d)],[(x+w,y,z),(x+w,y+h,z),(x+w,y+h,z+d),(x+w,y,z+d)]]
        bx.add_collection3d(Poly3DCollection(faces,facecolors=colors[j*3+i],edgecolors='#34495e',linewidths=.7,alpha=.65))
        bx.text(x+1,y+1,z+d+.12,str(j*3+i),ha='center',fontsize=10)
bx.set(xlim=(0,6),ylim=(0,4),zlim=(0,3.6),xlabel='x',ylabel='y',zlabel='z',title='(b) Every rank owns the full vertical extent')
bx.set_xticks([0,2,4,6]);bx.set_yticks([0,2,4]);bx.set_zticks([0,3]);bx.set_zticklabels(['bottom','top']);bx.view_init(25,-55);bx.set_box_aspect((6,4,3))
fig.suptitle('MPI exchanges boundary concentrations; Kokkos updates cells within each rank',fontsize=12)
fig.supxlabel('Simplified example geometry. Each rank exchanges a one-cell halo with its face neighbours.',fontsize=9)
out=Path(__file__).resolve().parents[1]/'report/figures'
for ext in ('pdf','png'):fig.savefig(out/f'grid_partition.{ext}',dpi=180)
