"""Scoring for a generated centreline against the reference centreline.

`fixed_metrics` uses the fixed pixel tolerances. `score` adds the
width-adaptive agreement built from the reference crack width, which is a
region-agreement score and not a claim of sub-width localisation accuracy.
Reference widths are read from the target mask and never reach the sampler.
"""
import numpy as np
import cv2
import itertools

from workspace import SEEDS


def path_metrics(path,task,success):
    pmask=np.zeros(task['brick'].shape,dtype=np.uint8);pmask[path[:,0],path[:,1]]=1
    dist=cv2.distanceTransform(1-pmask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    p_to_t=task['gt_distance'][pmask>0];t_to_p=dist[task['gtmask']>0]
    distance=float((p_to_t.mean()+t_to_p.mean())/2)
    precision=float(np.mean(p_to_t<=2));recall=float(np.mean(t_to_p<=2))
    f1=2*precision*recall/max(precision+recall,1e-12)
    length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
    chord=max(float(np.linalg.norm(path[-1]-path[0])),1)
    true_length=float(np.linalg.norm(np.diff(task['gt'],axis=0),axis=1).sum())
    return dict(success=int(success),centreline_distance_px=distance,centreline_distance_over_height=distance/task['height'],
                centreline_f1_at_2px=f1,generated_tortuosity=length/chord,
                reference_tortuosity=true_length/max(float(np.linalg.norm(task['gt'][-1]-task['gt'][0])),1),
                generated_in_brick_fraction=float(task['brick'][pmask>0].mean()),
                reference_in_brick_fraction=float(task['brick'][task['gtmask']>0].mean()),
                selection_score=distance/task['height']+int(not success),
                endpoint_error_px=float(np.linalg.norm(path[-1]-task['goal'])))


def fixed_metrics(path, task, success):
    m = path_metrics(path,task,success)
    mask = np.zeros(task['brick'].shape,np.uint8)
    mask[path[:,0],path[:,1]] = 1
    distance = cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    p = float(np.mean(task['gt_distance'][mask>0]<=4))
    r = float(np.mean(distance[task['gtmask']>0]<=4))
    m.update(centreline_precision_at_4px=p,centreline_recall_at_4px=r,
             centreline_f1_at_4px=2*p*r/max(p+r,1e-12))
    return m


def reference_geometry(target,task):
    # Padding treats outside the raster as background, including edge cracks.
    inside=cv2.distanceTransform(np.pad(target.astype(np.uint8),1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
    gt=task['gt']
    radius=np.maximum(1.0,inside[gt[:,0],gt[:,1]]-.5)
    bands={}
    h,w=target.shape
    for scale in [.5,1.,1.5]:
        band=np.zeros((h,w),bool)
        radii=np.maximum(1.,radius*scale)
        for (y,x),r in zip(gt,radii):
            limit=int(np.ceil(r));y0=max(0,y-limit);y1=min(h,y+limit+1)
            x0=max(0,x-limit);x1=min(w,x+limit+1)
            yy,xx=np.ogrid[y0:y1,x0:x1]
            band[y0:y1,x0:x1]|=(yy-y)**2+(xx-x)**2<=r*r+1e-9
        bands[scale]=band
    return dict(radius=radius,bands=bands,median_width_px=float(2*np.median(radius)),
                radius_min=float(radius.min()),radius_max=float(radius.max()))


def score(path,task,success,geometry):
    m=fixed_metrics(path,task,success)
    mask=np.zeros(task['brick'].shape,np.uint8);mask[path[:,0],path[:,1]]=1
    distance=cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    gt=task['gt']
    errors=distance[gt[:,0],gt[:,1]]
    for scale,tag in [(.5,'half'),(1.,'width'),(1.5,'one_half')]:
        precision=float(geometry['bands'][scale][mask>0].mean())
        recall=float(np.mean(errors<=np.maximum(1.,geometry['radius']*scale)))
        m[f'adaptive_precision_{tag}']=precision
        m[f'adaptive_recall_{tag}']=recall
        m[f'adaptive_f1_{tag}']=2*precision*recall/max(precision+recall,1e-12)
    m['median_reference_width_px']=geometry['median_width_px']
    return m


def morphology(path,task):
    delta=np.diff(path,axis=0).astype(float)
    if len(delta)<2: turn=0.
    else:
        cosine=np.sum(delta[:-1]*delta[1:],axis=1)/(np.linalg.norm(delta[:-1],axis=1)*np.linalg.norm(delta[1:],axis=1))
        turn=float(np.mean(cosine<1-1e-9))
    return dict(direction_change_fraction=turn,path_pixels=len(path))


def diversity(paths,shape,height):
    distances=[];masks=[]
    for path in paths:
        mask=np.zeros(shape,np.uint8);mask[path[:,0],path[:,1]]=1;masks.append(mask)
        distances.append(cv2.distanceTransform(1-mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
    values=[]
    for i in range(len(paths)):
        for j in range(i+1,len(paths)):
            values.append(float((distances[i][masks[j]>0].mean()+distances[j][masks[i]>0].mean())/(2*height)))
    return dict(unique_seed_paths=len({a.tobytes() for a in paths}),seed_pairs=len(values),
                mean_pairwise_distance_over_height=float(np.mean(values)))


def geometry(task):
    return {k:task[k] for k in ['brick','labels','heights','axes','start','goal','height']}


def aggregate(rows):
    per=[];summary=[]
    for cid in dict.fromkeys(r['config'] for r in rows):
        for method in ['unet','no_unet']:
            selected=[r for r in rows if r['config']==cid and r['method']==method]
            for name in sorted({r['filename'] for r in selected}):
                rs=[r for r in selected if r['filename']==name]
                assert sorted(int(r['seed']) for r in rs)==SEEDS
                f=[float(r['centreline_f1_at_4px']) for r in rs]
                per.append(dict(config=cid,method=method,filename=name,mean_f1=float(np.mean(f)),max_f1=max(f),
                    completion=float(np.mean([float(r['success']) for r in rs])),
                    recovery_fraction=float(np.mean([float(r['recovery_used']) for r in rs])),
                    unique_paths=int(rs[0]['unique_paths']),
                    best_seed=int(rs[int(np.argmax(f))]['seed'])))
            ps=[r for r in per if r['config']==cid and r['method']==method]
            summary.append(dict(config=cid,method=method,images=len(ps),
                mean_f1=float(np.mean([r['mean_f1'] for r in ps])),mean_max_f1=float(np.mean([r['max_f1'] for r in ps])),
                completion=float(np.mean([r['completion'] for r in ps])),
                recovery_fraction=float(np.mean([r['recovery_fraction'] for r in ps])),
                five_distinct_fraction=float(np.mean([r['unique_paths']==5 for r in ps]))))
    return per,summary


def spread(paths, shape, height):
    distances=[]
    for path in paths:
        mask=np.ones(shape,np.uint8);mask[path[:,0],path[:,1]]=0
        distances.append(cv2.distanceTransform(mask,cv2.DIST_L2,cv2.DIST_MASK_PRECISE))
    means=[];p90=[];separated=[]
    for a,b in itertools.combinations(range(len(paths)),2):
        ds=np.concatenate([distances[a][paths[b][:,0],paths[b][:,1]],distances[b][paths[a][:,0],paths[a][:,1]]])
        means.append(float(ds.mean()/height));p90.append(float(np.percentile(ds,90)/height))
        separated.append(float(np.mean(ds>max(4.,.15*height))))
    return dict(pair_mean_distance_over_height=float(np.mean(means)), pair_p90_distance_over_height=float(np.mean(p90)),
        separated_point_fraction=float(np.mean(separated)))
