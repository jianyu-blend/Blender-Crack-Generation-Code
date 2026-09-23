"""The probability-guided stochastic crack-path sampler.

`generate` is the entry point. It composes five stages, applied in this order
to one call, exactly as the reported configuration does:

1. `fuse_views`        Blend the four aligned flip predictions into one map,
                       with seed-specific Dirichlet weights.
2. `_temper_prior`     Raise the image-relative map to a seed-specific
                       exponent, so the strength of the learned guidance
                       varies between samples.
3. `_prepare_material` Add the material-preference random field.
4. `_prepare_field`    Build the reverse-Dijkstra remaining-cost field that
                       makes every retained step descend towards the endpoint.
5. `_walk_lookahead`   Draw each pixel from the locally normalised weights,
                       with the look-ahead regret factor.

Every candidate pixel must lower the remaining cost and pass the in-brick
crossing rule in `crossing_rules`. The reference centreline is never read.
Strict cost descent makes the retained path acyclic, so the 0.01 revisit factor
is inactive on it.
"""
import heapq
import math
import itertools

import cv2
import numpy as np
from numba import njit

from crossing_rules import DIRECTIONS, D, transition, centres_for, audit_and_nll


# ---------------------------------------------------------------
# Reverse remaining-cost field and the geometry-guided walk
# ---------------------------------------------------------------
@njit(cache=True)
def reverse_field(cost,goal):
    h,w=cost.shape;field=np.full((h,w),np.inf);field[goal[0],goal[1]]=0.
    heap=[(0.,np.int64(goal[0]*w+goal[1]))]
    while heap:
        distance,pixel=heapq.heappop(heap);y,x=pixel//w,pixel%w
        if distance!=field[y,x]: continue
        for k in range(8):
            yy,xx=y+D[k,0],x+D[k,1]
            if yy<0 or xx<0 or yy>=h or xx>=w: continue
            edge=math.hypot(D[k,0],D[k,1])*.5*(cost[y,x]+cost[yy,xx])
            candidate=distance+edge
            if candidate<field[yy,xx]:
                field[yy,xx]=candidate;heapq.heappush(heap,(candidate,np.int64(yy*w+xx)))
    return field


@njit(cache=True)
def _walk_field(prob,field,labels,heights,axes,centres,start,goal,gamma,inertia,seed,budget):
    np.random.seed(seed);h,w=prob.shape
    path=np.empty((budget+1,2),np.int32);path[0]=start;n=1
    blocked=np.zeros((budget+1,8),np.uint8)
    entries=np.full(budget+1,-1,np.int32);ids=np.zeros(budget+1,np.int32);lengths=np.zeros(budget+1)
    rejected=backs=rollback=0
    for attempted in range(budget):
        y,x=path[n-1]
        if y==goal[0] and x==goal[1]: return path[:n].copy(),True,attempted,rejected,backs,rollback
        if n>=2: py,px=float(y-path[n-2,0]),float(x-path[n-2,1])
        else: py,px=float(goal[0]-y),float(goal[1]-x)
        pn=max(math.hypot(py,px),1e-12);distance=math.hypot(y-goal[0],x-goal[1])
        weights=np.zeros(8);total=0.
        for k in range(8):
            yy,xx=y+D[k,0],x+D[k,1]
            if yy<0 or xx<0 or yy>=h or xx>=w or blocked[n-1,k]: continue
            if not field[yy,xx]<field[y,x]: continue
            valid,_,_,_,_=transition(labels,heights,axes,centres,path,n,entries[n-1],ids[n-1],lengths[n-1],yy,xx)
            if not valid: blocked[n-1,k]=1;rejected+=1;continue
            cosine=(D[k,0]*py+D[k,1]*px)/(math.hypot(D[k,0],D[k,1])*pn)
            # Strict field descent makes revisit impossible on the retained path;
            # the existing 0.01 revisit coefficient is therefore inactive here.
            weights[k]=max(prob[yy,xx],1e-6)*math.exp(gamma*(distance-math.hypot(yy-goal[0],xx-goal[1])))*(1+inertia*cosine)
            total+=weights[k]
        if total<=0:
            if n==1: break
            # Reject the failed in-brick route at its entry, not the entire wall.
            # Initial cropped-brick partial fragments have no mortar entry.
            anchor=entries[n-1] if entries[n-1]>=0 else n-2
            if entries[n-1]>=0: rollback+=1
            for k in range(8):
                if path[anchor,0]+D[k,0]==path[anchor+1,0] and path[anchor,1]+D[k,1]==path[anchor+1,1]: blocked[anchor,k]=1
            n=anchor+1;backs+=1;continue
        u=np.random.random()*total;cumulative=0.;chosen=-1
        for k in range(8):
            cumulative+=weights[k]
            if weights[k]>0: chosen=k
            if u<cumulative: break
        yy,xx=y+D[chosen,0],x+D[chosen,1]
        _,entry,label,length,_=transition(labels,heights,axes,centres,path,n,entries[n-1],ids[n-1],lengths[n-1],yy,xx)
        path[n,0],path[n,1]=yy,xx;entries[n]=entry;ids[n]=label;lengths[n]=length
        blocked[n,:]=0;n+=1
    ok=np.array_equal(path[n-1],goal)
    return path[:n].copy(),ok,attempted+1,rejected,backs,rollback


def _prepare_field(prob,geometry,brick_penalty):
    prob=np.ascontiguousarray(prob,dtype=np.float64)
    if prob.shape!=geometry['brick'].shape or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):
        raise ValueError('Invalid probability map')
    clipped=np.maximum(prob,1e-6)
    # Image-relative probability removes a constant-map scale effect. With a
    # spatially constant map, any constant in (0,1] gives the same guiding cost.
    cost=1.+float(brick_penalty)*geometry['brick']-np.log(clipped/clipped.max())
    return dict(probability=prob.copy(),potential=reverse_field(cost,geometry['goal']),centres=centres_for(geometry),
        goal=geometry['goal'].copy(),brick=geometry['brick'].copy())


def _generate_field(prob,geometry,seed,config,attempts=8,budget=16384,endpoint_mode='cropped_mask'):
    if endpoint_mode not in ['cropped_mask','full_wall']: raise ValueError('Unknown endpoint mode')
    if endpoint_mode=='full_wall' and (geometry['brick'][tuple(geometry['start'])] or geometry['brick'][tuple(geometry['goal'])]):
        raise ValueError('Full-wall generation requires caller-selected mortar endpoints; endpoints are never silently moved')
    key=config['brick_penalty']
    cached=geometry.get('_cached_fields')
    if (geometry.get('_cached_key')!=key or cached is None or not np.array_equal(cached['probability'],prob)
        or not np.array_equal(cached['goal'],geometry['goal']) or not np.array_equal(cached['brick'],geometry['brick'])):
        geometry['_cached_fields']=_prepare_field(prob,geometry,config['brick_penalty']);geometry['_cached_key']=key
    fields=geometry['_cached_fields'];first=None;totals=np.zeros(4,np.int64)
    for trial in range(attempts):
        path,ok,steps,rejected,backs,rollback=_walk_field(fields['probability'],fields['potential'],geometry['labels'],geometry['heights'],geometry['axes'],fields['centres'],
            geometry['start'],geometry['goal'],config['gamma'],config['inertia'],int(seed)+100003*trial,budget)
        if first is None: first=path
        totals+=np.array([steps,rejected,backs,rollback])
        if ok: break
    if not ok: path=first
    valid,_,crossings=audit_and_nll(path,fields['probability'],geometry['labels'],geometry['heights'],geometry['axes'],geometry['goal'],config['gamma'],config['inertia'],fields['centres'])
    assert valid and np.all(np.diff(fields['potential'][path[:,0],path[:,1]])<0)
    return path,dict(success=bool(ok),primary_success=int(ok),recovery_used=0,
        mode='geometry_guided_random' if ok else 'incomplete_random',attempts_used=trial+1,
        selected_attempt=trial+1 if ok else 1,attempted_steps=int(totals[0]),rejected_exits=int(totals[1]),
        backtracks=int(totals[2]),brick_entry_rollbacks=int(totals[3]),accepted_transverse_crossings=int(crossings),
        endpoint_mode=endpoint_mode,generated_by='categorical_pixel_draws_on_descending_material_probability_field',posthoc_smoothing=False)


# ---------------------------------------------------------------
# Material-preference random field
# ---------------------------------------------------------------
def _prepare_material(prob, geometry, seed, config):
    prob = np.ascontiguousarray(prob, dtype=np.float64)
    if prob.shape != geometry['brick'].shape or not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)):
        raise ValueError('Invalid probability map')
    # Separate random stream: paired methods receive exactly the same latent
    # material preference, independent of pixel draw/rejection counts.
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 12012]))
    z = float(np.clip(rng.normal(), -2, 2))
    penalty = float(config['brick_penalty'] * np.exp(config['route_sigma'] * z))
    h, w = prob.shape
    spacing = max(8., float(geometry['height']))
    coarse = rng.normal(size=(max(2, int(np.ceil(h / spacing)) + 1), max(2, int(np.ceil(w / spacing)) + 1)))
    noise = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = np.clip((noise - noise.mean()) / max(noise.std(), 1e-12), -2, 2)
    clipped = np.maximum(prob, 1e-6)
    cost = (1. + penalty * geometry['brick']) * np.exp(config['field_sigma'] * noise) - np.log(clipped / clipped.max())
    return dict(probability=prob, potential=reverse_field(cost, geometry['goal']), centres=centres_for(geometry),
                effective_brick_penalty=penalty, noise=noise)


def _generate_material(prob, geometry, seed, config, attempts=8, budget=16384, endpoint_mode='cropped_mask'):
    if endpoint_mode not in ['cropped_mask', 'full_wall']:
        raise ValueError('Unknown endpoint mode')
    if endpoint_mode == 'full_wall' and (geometry['brick'][tuple(geometry['start'])] or geometry['brick'][tuple(geometry['goal'])]):
        raise ValueError('Full-wall generation requires supplied mortar endpoints; endpoints are never moved')
    fields = _prepare_material(prob, geometry, seed, config)
    first = None
    totals = np.zeros(4, np.int64)
    for trial in range(attempts):
        path, ok, steps, rejected, backs, rollback = walk(fields['probability'], fields['potential'], geometry['labels'],
            geometry['heights'], geometry['axes'], fields['centres'], geometry['start'], geometry['goal'],
            config['gamma'], config['inertia'], int(seed) + 100003 * trial, budget)
        if first is None:
            first = path
        totals += np.array([steps, rejected, backs, rollback])
        if ok:
            break
    if not ok:
        path = first
    valid, _, crossings = audit_and_nll(path, fields['probability'], geometry['labels'], geometry['heights'], geometry['axes'],
        geometry['goal'], config['gamma'], config['inertia'], fields['centres'])
    assert valid and np.all(np.diff(fields['potential'][path[:, 0], path[:, 1]]) < 0)
    return path, dict(success=bool(ok), primary_success=int(ok), recovery_used=0,
        mode='random_material_and_pixel' if ok else 'incomplete_random', attempts_used=trial+1,
        selected_attempt=trial+1 if ok else 1, attempted_steps=int(totals[0]), rejected_exits=int(totals[1]),
        backtracks=int(totals[2]), brick_entry_rollbacks=int(totals[3]), accepted_transverse_crossings=int(crossings),
        endpoint_mode=endpoint_mode, effective_brick_penalty=fields['effective_brick_penalty'],
        latent_seed=int(seed), generated_by='seed_specific_material_field_then_categorical_pixel_draws', posthoc_smoothing=False)


# ---------------------------------------------------------------
# Seed-specific learned-prior exponent
# ---------------------------------------------------------------
def _temper_prior(prob,seed,config):
    prob=np.asarray(prob,dtype=np.float64)
    if not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):
        raise ValueError('Invalid probability map')
    rng=np.random.default_rng(np.random.SeedSequence([int(seed),13013]))
    exponent=float(config['prior_power']*np.exp(config['prior_sigma']*np.clip(rng.normal(),-2,2)))
    relative=np.maximum(prob,1e-6);relative/=relative.max()
    return relative**exponent,exponent


def _prepare_tempered(prob,geometry,seed,config):
    transformed,exponent=_temper_prior(prob,seed,config)
    fields=_prepare_material(transformed,geometry,seed,config)
    fields['effective_prior_power']=exponent
    return fields


def _generate_tempered(prob,geometry,seed,config,attempts=8,budget=16384,endpoint_mode='cropped_mask'):
    transformed,exponent=_temper_prior(prob,seed,config)
    path,meta=_generate_material(transformed,geometry,seed,config,attempts,budget,endpoint_mode)
    meta.update(effective_prior_power=exponent,mode='random_prior_material_pixels' if meta['success'] else 'incomplete_random',
        generated_by='seed_specific_prior_power_material_field_and_categorical_pixel_draws')
    return path,meta


# ---------------------------------------------------------------
# Cost-aware look-ahead walk
# ---------------------------------------------------------------
def _prepare_lookahead(prob,geometry,seed,config):
    fields=_prepare_tempered(prob,geometry,seed,config)
    clipped=np.maximum(fields['probability'],1e-6)
    fields['cost']=(1.+fields['effective_brick_penalty']*geometry['brick'])*np.exp(config['field_sigma']*fields['noise'])-np.log(clipped/clipped.max())
    return fields


@njit(cache=True)
def _walk_lookahead(prob,field,cost,labels,heights,axes,centres,start,goal,gamma,inertia,lookahead,seed,budget):
    np.random.seed(seed);h,w=prob.shape
    path=np.empty((budget+1,2),np.int32);path[0]=start;n=1
    blocked=np.zeros((budget+1,8),np.uint8)
    entries=np.full(budget+1,-1,np.int32);ids=np.zeros(budget+1,np.int32);lengths=np.zeros(budget+1)
    rejected=backs=rollback=0
    for attempted in range(budget):
        y,x=path[n-1]
        if y==goal[0] and x==goal[1]: return path[:n].copy(),True,attempted,rejected,backs,rollback
        if n>=2: py,px=float(y-path[n-2,0]),float(x-path[n-2,1])
        else: py,px=float(goal[0]-y),float(goal[1]-x)
        pn=max(math.hypot(py,px),1e-12);distance=math.hypot(y-goal[0],x-goal[1])
        weights=np.zeros(8);total=0.
        for k in range(8):
            yy,xx=y+D[k,0],x+D[k,1]
            if yy<0 or xx<0 or yy>=h or xx>=w or blocked[n-1,k]: continue
            if not field[yy,xx]<field[y,x]: continue
            valid,_,_,_,_=transition(labels,heights,axes,centres,path,n,entries[n-1],ids[n-1],lengths[n-1],yy,xx)
            if not valid: blocked[n-1,k]=1;rejected+=1;continue
            cosine=(D[k,0]*py+D[k,1]*px)/(math.hypot(D[k,0],D[k,1])*pn)
            # Strict field descent makes revisit impossible on the retained path;
            # the existing 0.01 revisit coefficient is therefore inactive here.
            weights[k]=max(prob[yy,xx],1e-6)*math.exp(gamma*(distance-math.hypot(yy-goal[0],xx-goal[1])))*(1+inertia*cosine)
            edge=math.hypot(D[k,0],D[k,1])*.5*(cost[y,x]+cost[yy,xx])
            regret=max(0.,edge+field[yy,xx]-field[y,x])/max(edge,1e-12)
            weights[k]*=math.exp(-lookahead*regret)
            total+=weights[k]
        if total<=0:
            if n==1: break
            # Reject the failed in-brick route at its entry, not the entire wall.
            # Initial cropped-brick partial fragments have no mortar entry.
            anchor=entries[n-1] if entries[n-1]>=0 else n-2
            if entries[n-1]>=0: rollback+=1
            for k in range(8):
                if path[anchor,0]+D[k,0]==path[anchor+1,0] and path[anchor,1]+D[k,1]==path[anchor+1,1]: blocked[anchor,k]=1
            n=anchor+1;backs+=1;continue
        u=np.random.random()*total;cumulative=0.;chosen=-1
        for k in range(8):
            cumulative+=weights[k]
            if weights[k]>0: chosen=k
            if u<cumulative: break
        yy,xx=y+D[chosen,0],x+D[chosen,1]
        _,entry,label,length,_=transition(labels,heights,axes,centres,path,n,entries[n-1],ids[n-1],lengths[n-1],yy,xx)
        path[n,0],path[n,1]=yy,xx;entries[n]=entry;ids[n]=label;lengths[n]=length
        blocked[n,:]=0;n+=1
    ok=np.array_equal(path[n-1],goal)
    return path[:n].copy(),ok,attempted+1,rejected,backs,rollback


def _generate_lookahead(prob,geometry,seed,config,attempts=8,budget=16384,endpoint_mode='cropped_mask'):
    if endpoint_mode not in ['cropped_mask','full_wall']:raise ValueError('Unknown endpoint mode')
    if endpoint_mode=='full_wall' and (geometry['brick'][tuple(geometry['start'])] or geometry['brick'][tuple(geometry['goal'])]):
        raise ValueError('Full-wall generation requires supplied mortar endpoints; endpoints are never moved')
    fields=_prepare_lookahead(prob,geometry,seed,config);first=None;totals=np.zeros(4,np.int64)
    for trial in range(attempts):
        path,ok,steps,rejected,backs,rollback=_walk_lookahead(fields['probability'],fields['potential'],fields['cost'],geometry['labels'],
            geometry['heights'],geometry['axes'],fields['centres'],geometry['start'],geometry['goal'],config['gamma'],config['inertia'],
            config['lookahead'],int(seed)+100003*trial,budget)
        if first is None:first=path
        totals+=np.array([steps,rejected,backs,rollback])
        if ok:break
    if not ok:path=first
    valid,_,crossings=audit_and_nll(path,fields['probability'],geometry['labels'],geometry['heights'],geometry['axes'],
        geometry['goal'],config['gamma'],config['inertia'],fields['centres'])
    assert valid and np.all(np.diff(fields['potential'][path[:,0],path[:,1]])<0)
    return path,dict(success=bool(ok),primary_success=int(ok),recovery_used=0,
        mode='cost_aware_stochastic_pixels' if ok else 'incomplete_random',attempts_used=trial+1,selected_attempt=trial+1 if ok else 1,
        attempted_steps=int(totals[0]),rejected_exits=int(totals[1]),backtracks=int(totals[2]),brick_entry_rollbacks=int(totals[3]),
        accepted_transverse_crossings=int(crossings),endpoint_mode=endpoint_mode,
        effective_brick_penalty=fields['effective_brick_penalty'],effective_prior_power=fields['effective_prior_power'],
        latent_seed=int(seed),generated_by='categorical_pixel_draws_with_normalised_cost_regret',posthoc_smoothing=False)


# ---------------------------------------------------------------
# Four-view probability fusion; this is the public entry point
# ---------------------------------------------------------------
def fuse_views(prob,seed,config):
    prob=np.asarray(prob,dtype=np.float64)
    if prob.ndim!=3 or prob.shape[0]!=4 or not np.isfinite(prob).all() or np.any((prob<0)|(prob>1)):
        raise ValueError('Expected four aligned probability maps')
    if config['fusion']=='half_mean':weights=np.array([.625,.125,.125,.125])
    elif config['fusion']=='mean':weights=np.full(4,.25)
    elif config['fusion']=='random':
        weights=np.random.default_rng(np.random.SeedSequence([int(seed),15015])).dirichlet(np.full(4,.5))
    else:raise ValueError('Unknown fusion')
    return np.clip(np.einsum('i,ijk->jk',weights,prob),0,1),weights


def prepare_fields(prob,geometry,seed,config):
    merged,weights=fuse_views(prob,seed,config);fields=_prepare_lookahead(merged,geometry,seed,config)
    fields['flip_weights']=weights.tolist();return fields


def generate(prob,geometry,seed,config,attempts=8,budget=16384,endpoint_mode='cropped_mask'):
    merged,weights=fuse_views(prob,seed,config)
    path,details=_generate_lookahead(merged,geometry,seed,config,attempts,budget,endpoint_mode)
    details.update(flip_weights=weights.tolist(),mode='flip_prior_cost_aware_pixels' if details['success'] else 'incomplete_random')
    return path,details
