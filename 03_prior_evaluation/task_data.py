"""Build the per-image task: layout, reference centreline and brick geometry.

The supplied reference endpoints are used exactly and may lie inside a brick.
The intermediate reference path is kept by the evaluator only and is never
passed to the sampler.
"""
import heapq
import math

import numpy as np
import cv2

from workspace import load_mask


def dominant_path(skeleton):
    _,labels,stats,_=cv2.connectedComponentsWithStats(skeleton.astype(np.uint8),8)
    if len(stats)<=1:
        return np.empty((0,2),dtype=np.int32)
    comp=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    coords=np.argwhere(labels==comp); points={tuple(v) for v in coords.tolist()}
    if len(points)<2:
        return coords
    def sweep(start):
        distances={start:0.0}; previous={}; heap=[(0.0,start)]
        while heap:
            dist,point=heapq.heappop(heap)
            if dist!=distances[point]: continue
            y,x=point
            for dy in (-1,0,1):
                for dx in (-1,0,1):
                    q=(y+dy,x+dx)
                    if not (dy or dx) or q not in points: continue
                    nd=dist+math.hypot(dy,dx)
                    if nd<distances.get(q,math.inf):
                        distances[q]=nd;previous[q]=point;heapq.heappush(heap,(nd,q))
        end=max(distances,key=distances.get)
        return end,previous
    start,_=sweep(min(points));end,previous=sweep(start)
    path=[end]
    while path[-1]!=start: path.append(previous[path[-1]])
    return np.asarray(path[::-1],dtype=np.int32)


def task_data(data,name):
    brick=load_mask(data,'brick_binary',name)
    gt=dominant_path(load_mask(data,'skeleton',name))
    if len(gt)<2 or np.linalg.norm(gt[-1]-gt[0])<20 or np.all(brick):
        return None
    count,labels,stats,_=cv2.connectedComponentsWithStats(brick.astype(np.uint8),8)
    heights=np.ones(count,dtype=np.float64)*51.2
    axes=np.zeros((count,2),dtype=np.float64);axes[:,0]=1
    hs=[]
    for label in range(1,count):
        ys,xs=np.where(labels==label)
        if len(xs)<4: continue
        rect=cv2.minAreaRect(np.column_stack([xs,ys]).astype(np.float32))
        box=cv2.boxPoints(rect)
        vectors=np.roll(box,-1,axis=0)-box
        lens=np.linalg.norm(vectors,axis=1)
        ix=int(np.argmin(lens))
        heights[label]=max(float(lens[ix]),2.0)
        axes[label]=vectors[ix,::-1]/max(float(lens[ix]),1e-12)
        if len(xs)>=25: hs.append(heights[label])
    height=float(np.median(hs)) if hs else 51.2
    start,goal=gt[0].copy(),gt[-1].copy()
    if np.linalg.norm(goal-start)<20: return None
    gtmask=np.zeros(brick.shape,dtype=np.uint8);gtmask[gt[:,0],gt[:,1]]=1
    dt=cv2.distanceTransform(1-gtmask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    return dict(brick=brick,labels=labels.astype(np.int32),heights=heights,axes=axes,
                gt=gt,gtmask=gtmask,gt_distance=dt,height=height,start=start.astype(np.int32),goal=goal.astype(np.int32),
                start_shift=float(np.linalg.norm(start-gt[0])),goal_shift=float(np.linalg.norm(goal-gt[-1])))
