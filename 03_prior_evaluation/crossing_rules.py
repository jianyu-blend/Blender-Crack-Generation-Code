"""The in-brick traversal rule applied to every complete opposite-side crossing.

A transverse crossing is accepted when its in-brick run length L and the brick
height H satisfy 0.7 <= L/H <= 1.4, the interval measured on the real masks.
Oblique complete crossings are held to the same interval; same-side excursions
are recorded separately. A partial brick path that ends at the supplied
endpoint is not a complete crossing.
"""
import math
import numpy as np
from numba import njit

# The eight-neighbour offsets every walk and rule shares.
DIRECTIONS = np.array([[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]], dtype=np.int32)
D = DIRECTIONS


def centres_for(task):
    centres=np.zeros(len(task['heights']),np.float64)
    for label in range(1,len(centres)):
        coords=np.argwhere(task['labels']==label)
        projection=coords@task['axes'][label]
        centres[label]=.5*(projection.min()+projection.max())
    return centres


@njit(cache=True)
def transition(labels,heights,axes,centres,path,n,entry,label,length,yy,xx):
    y,x=path[n-1];new_entry,new_label,new_length=entry,label,length
    step=math.hypot(yy-y,xx-x)
    if labels[y,x]==0 and labels[yy,xx]>0:
        new_entry=n-1;new_label=labels[yy,xx];new_length=step
    elif entry>=0: new_length+=step
    crossing=0
    if new_entry>=0 and labels[yy,xx]==0:
        ey,ex=path[new_entry];dy,dx=yy-ey,xx-ex
        alignment=abs((dy*axes[new_label,0]+dx*axes[new_label,1])/max(math.hypot(dy,dx),1e-12))
        before=ey*axes[new_label,0]+ex*axes[new_label,1]-centres[new_label]
        after=yy*axes[new_label,0]+xx*axes[new_label,1]-centres[new_label]
        if before*after<=0:
            if not .7*heights[new_label]<=new_length<=1.4*heights[new_label]:
                return False,new_entry,new_label,new_length,0
            crossing=1
        new_entry=-1;new_label=0;new_length=0.
    return True,new_entry,new_label,new_length,crossing


@njit(cache=True)
def audit_and_nll(path,prob,labels,heights,axes,goal,gamma,inertia,centres):
    h,w=prob.shape;positions=np.full((h,w),-1,np.int32);positions[path[0,0],path[0,1]]=0
    entry=-1;label=0;length=0.;nll=0.;crossings=0
    for j in range(1,len(path)):
        y,x=path[j-1];ny,nx=path[j]
        if max(abs(ny-y),abs(nx-x))!=1 or positions[ny,nx]>=0: return False,math.inf,crossings
        if j>=2: py,px=float(y-path[j-2,0]),float(x-path[j-2,1])
        else: py,px=float(goal[0]-y),float(goal[1]-x)
        pn=max(math.hypot(py,px),1e-12);distance=math.hypot(y-goal[0],x-goal[1]);total=0.;selected=0.
        for k in range(8):
            yy,xx=y+D[k,0],x+D[k,1]
            if yy<0 or xx<0 or yy>=h or xx>=w: continue
            valid,_,_,_,_=transition(labels,heights,axes,centres,path,j,entry,label,length,yy,xx)
            if not valid: continue
            cosine=(D[k,0]*py+D[k,1]*px)/(math.hypot(D[k,0],D[k,1])*pn)
            weight=max(prob[yy,xx],1e-6)*math.exp(gamma*(distance-math.hypot(yy-goal[0],xx-goal[1])))*(1+inertia*cosine)*(.01 if positions[yy,xx]>=0 else 1.)
            total+=weight
            if yy==ny and xx==nx: selected=weight
        if selected<=0 or total<=0: return False,math.inf,crossings
        nll-=math.log(selected/total)
        valid,entry,label,length,crossing=transition(labels,heights,axes,centres,path,j,entry,label,length,ny,nx)
        if not valid: return False,math.inf,crossings
        crossings+=crossing;positions[ny,nx]=j
    return True,nll,crossings


def crossing_records(path,task):
    centres=centres_for(task);records=[];entry=None;length=0.;label=0
    for j in range(1,len(path)):
        a,b=path[j-1],path[j];old=task['labels'][tuple(a)];new=task['labels'][tuple(b)]
        step=float(np.linalg.norm(b-a))
        if old==0 and new>0: entry=j-1;label=int(new);length=step
        elif entry is not None: length+=step
        if entry is not None and new==0:
            vector=b-path[entry];alignment=abs(float(vector@task['axes'][label]))/max(float(np.linalg.norm(vector)),1e-12)
            before=float(path[entry]@task['axes'][label]-centres[label]);after=float(b@task['axes'][label]-centres[label])
            opposite=before*after<=0;transverse=opposite
            ratio=length/task['heights'][label];accepted=not transverse or .7<=ratio<=1.4
            assert accepted,(entry,j,ratio)
            records.append(dict(entry_xy=path[entry,::-1].tolist(),exit_xy=b[::-1].tolist(),brick_id=label,
                length_px=length,height_px=float(task['heights'][label]),ratio=float(ratio),transverse=bool(transverse),accepted=bool(accepted),
                direction_aligned=bool(alignment>=np.sqrt(.5)),opposite_sides=bool(opposite),entry_projection=before,exit_projection=after))
            entry=None;length=0.;label=0
    return records
