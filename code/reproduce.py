"""Reproduce the 10-of-40 retinal synchrony comparison from ISTA's released spike bins."""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from urllib.request import Request, urlopen

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / 'protocol.json').read_text())
SEED = PROTOCOL['seed']
TEMPERATURES = np.array([1.0, .92, .80, .65, .48, .30, .12, 0.0], dtype='float64')
UINT_COUNT = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare_data(cache):
    cache.mkdir(parents=True, exist_ok=True)
    bundled = ROOT / 'data' / 'IST-2017-61-v1+1_bint_fishmovie32_100.mat'
    source = bundled if bundled.exists() else cache / 'data.mat'
    if not source.exists():
        with urlopen(Request(PROTOCOL['mat_url'], headers={'User-Agent':'retina-reproduction/1.0'}), timeout=60) as result:
            content = result.read()
        if hashlib.sha256(content).hexdigest() != PROTOCOL['sha256']:
            raise ValueError('The download differs from the frozen dataset.')
        source.write_bytes(content)
    if hashlib.sha256(source.read_bytes()).hexdigest() != PROTOCOL['sha256']:
        raise ValueError('The cached MAT file differs from the frozen dataset.')
    dest = cache / ('binary-' + PROTOCOL['sha256'][:14] + '.npy')
    # Build the binary cache afresh from the verified source on every invocation.
    matrix = loadmat(source, variable_names=[PROTOCOL['data_key']])[PROTOCOL['data_key']]
    if list(matrix.shape) != PROTOCOL['expected_shape'] or not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError('Unexpected source data shape or nonbinary input.')
    np.save(dest, matrix.astype(np.uint8))
    return dest


def unpack(words, n):
    return ((words[:,None] >> np.arange(n, dtype=np.uint64)) & 1).astype(bool)


def pack(samples):
    byte_matrix = np.packbits(samples, axis=1, bitorder='little')
    padded = np.zeros((len(samples), 8), dtype=np.uint8)
    padded[:,:byte_matrix.shape[1]] = byte_matrix
    return np.ascontiguousarray(padded).view('<u8').ravel()


def counts_from_words(words):
    return UINT_COUNT[np.ascontiguousarray(words).view(np.uint8).reshape(-1,8)].sum(1).astype(int)


def features(x):
    """Sparse sufficient statistics in [single cells, upper-triangle pairs] order."""
    n = x.shape[1]
    row, col = np.nonzero(x)
    rows, cols = [row], [col]
    start = n
    for i in range(n-1):
        row, offset = np.nonzero(x[:,i,None] & x[:,i+1:])
        rows.append(row)
        cols.append(start + offset)
        start += n-i-1
    row, col = np.concatenate(rows), np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(row)), (row,col)), shape=(len(x), start))


def conditional_tables(theta, n):
    J = np.zeros((n,n))
    ix = np.triu_indices(n,1)
    J[ix] = theta[n:]
    J[(ix[1],ix[0])] = theta[n:]
    return np.ascontiguousarray(theta[:n]), J


def time_split(cube, group):
    cells = PROTOCOL['groups_zero_based'][group-1]
    trials = np.ascontiguousarray(cube[:,cells,:].transpose(0,2,1))
    test = np.arange(trials.shape[0]) % 3 == 2
    return trials[~test], trials[test]


def poisson_binomial(p):
    """The coefficient of z^k in the product of (1-p_i+p_i*z). Also works per frame."""
    p = np.asarray(p)
    n = p.shape[-1]
    mass = np.zeros(p.shape[:-1] + (n+1,))
    mass[...,0] = 1.
    for i in range(n):
        q = p[...,i,None]
        mass[...,1:i+2] = mass[...,1:i+2]*(1.-q) + mass[...,:i+1]*q
        mass[...,0] *= 1.-p[...,i]
    return mass


def baselines(cube_file, results):
    cube = np.load(cube_file, mmap_mode='r')
    output = []
    for group in range(1,31):
        train, test = time_split(cube,group)
        n = train.shape[-1]
        conditional_p = train.mean(axis=0)
        independent = poisson_binomial(conditional_p.mean(0))
        conditioned = poisson_binomial(conditional_p).mean(0)
        held_counts = test.sum(axis=2)
        held_hist = np.bincount(held_counts.ravel().astype(int), minlength=n+1)
        observed = held_hist / held_hist.sum()
        per_repeat = (held_counts==10).mean(axis=1)
        rng = np.random.default_rng(np.random.SeedSequence([SEED,group,73]))
        indices = rng.integers(len(test), size=(2000,len(test)))
        ci = np.quantile(per_repeat[indices].mean(1),[.025,.975])
        p_obs,p_ind,p_time = observed[10],independent[10],conditioned[10]
        output.append({'group':group, 'test_windows':int(held_hist.sum()), 'test_events':int(held_hist[10]),
                       'observed':p_obs, 'observed_ci95':ci.tolist(), 'independent':p_ind,
                       'movie_conditioned':p_time, 'obs_over_independent':p_obs/p_ind,
                       'obs_over_movie_conditioned':p_obs/p_time,
                       'observed_histogram':observed.tolist(), 'independent_histogram':independent.tolist(),
                       'movie_conditioned_histogram':conditioned.tolist()})
    atomic_json(results/'baselines.json', output)
    return output


def compile_sampler(cache):
    source = Path(__file__).with_name('ising_sampler.cpp')
    name = 'gibbs-' + platform.machine() + '-' + hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    lib = cache/(name + ('.dylib' if platform.system()=='Darwin' else '.so'))
    if not lib.exists():
        link = '-dynamiclib' if platform.system()=='Darwin' else '-shared'
        subprocess.run([os.environ.get('CXX','c++'),'-O3','-std=c++17','-fPIC',link,str(source),'-o',str(lib)],check=True)
    return lib


def sample(lib_path,theta,n,draws,seed,dense,burn=1500):
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.sample_network
    def ptr(dtype): return np.ctypeslib.ndpointer(dtype,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int,ptr('float64'),ptr('float64'),ctypes.c_int,ptr('float64'),ctypes.c_int,
                 ctypes.c_int,ctypes.c_uint64,ctypes.c_int,ptr('uint64'),ptr('int64'),ptr('int64')]
    fn.restype=ctypes.c_int
    h,J=conditional_tables(theta,n)
    out=np.empty(draws,dtype=np.uint64); yes=np.zeros(len(TEMPERATURES)-1,np.int64); tries=np.zeros_like(yes)
    status=fn(n,h,J,len(TEMPERATURES),TEMPERATURES,burn,draws,int(seed),int(dense),out,yes,tries)
    if status: raise RuntimeError('Gibbs sampler rejected its configuration.')
    return out,yes/np.maximum(tries,1)


def verify_sampler(lib_path):
    n=8; rnd=np.random.default_rng(2048)
    theta=np.r_[rnd.uniform(-2.2,-.4,n),rnd.normal(0,.65,n*(n-1)//2)]
    states=np.arange(2**n,dtype=np.uint64)
    f=features(unpack(states,n)); log_weight=f@theta; probs=np.exp(log_weight-logsumexp(log_weight))
    truth_hist=np.bincount(counts_from_words(states),weights=probs,minlength=n+1)
    w,swap=sample(lib_path,theta,n,90_000,72449,True)
    hist=np.bincount(counts_from_words(w),minlength=n+1)/len(w)
    empirical_probs=np.bincount(w.astype(int),minlength=len(states))/len(w)
    moment_error=float(np.max(np.abs(f.T@(empirical_probs-probs))))
    histogram_error=float(np.max(np.abs(truth_hist-hist)))
    if max(moment_error,histogram_error)>.012: raise RuntimeError('Sampler disagrees with exact enumeration.')
    return {'cells':n,'exact_states':2**n,'draws':len(w),'max_abs_moment_error':moment_error,
            'max_abs_count_probability_error':histogram_error,'min_swap_acceptance':float(min(swap))}


def pl_start(x, weights, target, n, prior):
    rows,cols=np.triu_indices(n,1)
    reference=np.zeros(len(target)); q=target[:n]
    reference[:n]=np.log(q/(1-q))
    scale=1/np.sqrt(np.maximum(target*(1-target),3e-5))
    x=np.asarray(x,dtype=float,order='C')
    def objective(z):
        theta=reference+scale*z
        h,J=conditional_tables(theta,n)
        logits=x@J+h
        loss=weights@np.sum(np.logaddexp(0,logits)-x*logits,axis=1) + np.dot(prior,theta*theta)
        residual=(expit(logits)-x)*weights[:,None]
        gradJ=x.T@residual
        gradient=np.r_[residual.sum(axis=0),(gradJ+gradJ.T)[rows,cols]] + 2*prior*theta
        return float(loss),scale*gradient
    opt=minimize(objective,np.zeros_like(target),jac=True,method='L-BFGS-B',
                 options={'maxiter':260,'ftol':5e-11,'gtol':1e-6})
    return reference+scale*opt.x, {'iterations':int(opt.nit),'success':bool(opt.success),'message':str(opt.message)}


def monte_carlo_update(theta,words,target,prior,n,step):
    patterns,num=np.unique(words,return_counts=True); f=features(unpack(patterns,n))
    w=num/num.sum(); log_w=np.log(w); expected=f.T@w
    sd=np.sqrt(np.maximum(target*(1-target),2e-5))
    # The quadratic trust term limits changes when the finite reference run contains few rare states.
    trust=.22
    def objective(u):
        delta=u/sd; log_ratio=f@delta; log_norm=logsumexp(log_w+log_ratio)
        new_w=np.exp(log_w+log_ratio-log_norm)
        value=log_norm-np.dot(target,delta)+.5*np.dot(prior,(theta+delta)**2)+.5*trust*(u@u)
        gradient=(f.T@new_w-target+prior*(theta+delta))/sd+trust*u
        return float(value),gradient
    limit=.65*sd
    fit=minimize(objective,np.zeros_like(theta),jac=True,method='L-BFGS-B',bounds=list(zip(-limit,limit)),
                 options={'maxiter':70,'ftol':2e-12,'gtol':1e-6})
    delta=step*fit.x/sd
    def ess(d):
        lr=f@d
        return float(np.exp(2*logsumexp(log_w+lr)-logsumexp(log_w+2*lr)))
    while ess(delta)<.5:delta*=.5
    return theta+delta,{'single_rms':float(np.sqrt(np.mean((expected[:n]-target[:n])**2))),
                         'pair_rms':float(np.sqrt(np.mean((expected[n:]-target[n:])**2))),
                         'reweight_ess_fraction':ess(delta),'histogram_patterns':len(patterns)}


def seed_for(group, stage, chain):
    return int(np.random.SeedSequence([SEED,group,stage,chain]).generate_state(1,dtype=np.uint64)[0])


def balance_aggregate_pairs(lib_path,theta,n,target,group,stage,draws):
    """Line-search the likelihood along a common coupling using training pair moments only.

    A conditional-likelihood start can assign almost all stationary mass to a dense mode.
    The derivative along a common coupling is E[sum of pair indicators] minus the
    observed sum. It is monotonic because its derivative is a variance.
    """
    goal=float(target[n:].sum())
    def gap(shift,index):
        candidate=theta.copy(); candidate[n:]+=shift
        states,_=sample(lib_path,candidate,n,draws,seed_for(group,200+stage,index),index%2==0,burn=2200)
        count=counts_from_words(states)
        return float((count*(count-1)/2).mean()-goal)
    original_gap=gap(0.,0)
    if abs(original_gap)<max(.018,.015*goal):
        return theta,{'coupling_shift':0.,'pair_sum_error_before':original_gap}
    sign=-1 if original_gap>0 else 1
    step=.0125
    for i in range(10):
        edge=sign*step; value=gap(edge,i+1)
        if original_gap*value<=0:break
        step*=2
    else:
        raise RuntimeError('Unable to bracket the sum of the training pair moments.')
    low,high=sorted((0.,edge))
    best=(min(abs(original_gap),abs(value)),0. if abs(original_gap)<abs(value) else edge)
    for j in range(8):
        mid=(low+high)/2;value=gap(mid,20+j)
        if abs(value)<best[0]: best=(abs(value),mid)
        if value>0:high=mid
        else:low=mid
    result=theta.copy();result[n:]+=best[1]
    return result,{'coupling_shift':best[1],'pair_sum_error_before':original_gap,'pair_sum_error_after_short_chain':best[0]}


def fit_group(job):
    group,cube_file,lib_path,results=job
    with threadpool_limits(limits=1):
        cube=np.load(cube_file,mmap_mode='r'); train,_=time_split(cube,group); n=train.shape[-1]
        train=train.reshape(-1,n); ntrain=len(train)
        patterns,counts=np.unique(pack(train),return_counts=True)
        train_patterns=unpack(patterns,n); train_w=counts/ntrain
        target=np.asarray(features(train_patterns).T@train_w)
        prior=np.r_[np.full(n,1/(100*ntrain)),np.full(len(target)-n,1/(4*ntrain))]
        theta,initializer=pl_start(train_patterns,train_w,target,n,prior)
        print(f'Group {group}: initialized from {ntrain} training windows, {len(patterns)} distinct patterns.',flush=True)
        trace=[]
        theta,initial_balance=balance_aggregate_pairs(lib_path,theta,n,target,group,0,40_000)
        print(f'Group {group}: training pair-sum initialization {initial_balance}.',flush=True)
        stages=((90_000,.85),(140_000,.8),(200_000,.75),(260_000,.7),(360_000,.65),(460_000,.6),(600_000,.5))
        for stage,(total_draws,step) in enumerate(stages):
            batches=[sample(lib_path,theta,n,total_draws//2,seed_for(group,stage,c),bool(c%2))[0] for c in range(2)]
            theta,diagnostics=monte_carlo_update(theta,np.concatenate(batches),target,prior,n,step)
            theta,diagnostics['pair_sum_line_search']=balance_aggregate_pairs(lib_path,theta,n,target,group,stage+1,32_000 if stage<4 else 60_000)
            trace.append(diagnostics)
            print(f'Group {group}: training refinement {stage+1}, single RMS {diagnostics["single_rms"]:.5f}, pair RMS {diagnostics["pair_rms"]:.5f}.',flush=True)
        # Independent, longer simulations assess the frozen fit. Nothing below changes theta.
        chains=[]; swaps=[]; histograms=[]; block_rates=[]
        for c in range(4):
            words,swap=sample(lib_path,theta,n,250_000,seed_for(group,100,c),bool(c%2),burn=3000)
            chains.append(words);swaps.append(swap)
            cnt=counts_from_words(words)
            histograms.append(np.bincount(cnt,minlength=n+1)/len(cnt))
            block_rates.extend((cnt==10).reshape(-1,5000).mean(1))
        predicted=np.mean(histograms,axis=0); rates=np.array(histograms)[:,10]
        allwords=np.concatenate(chains); states,num=np.unique(allwords,return_counts=True)
        moments=features(unpack(states,n)).T@(num/num.sum()); diff=moments-target
        health={'train_single_rms':float(np.sqrt(np.mean(diff[:n]**2))),
                'train_pair_rms':float(np.sqrt(np.mean(diff[n:]**2))),
                'max_abs_single_error':float(max(abs(diff[:n]))),'max_abs_pair_error':float(max(abs(diff[n:]))),
                'mean_sim_minus_train_k':float(moments[:n].sum()-target[:n].sum()),
                'min_replica_swap_acceptance':float(np.min(swaps)),
                'max_simulated_k':int(np.flatnonzero(predicted)[-1])}
        block_se=np.std(block_rates,ddof=1)/np.sqrt(len(block_rates))
        chain_se=np.std(rates,ddof=1)/np.sqrt(len(rates))
        result={'group':group,'train_windows':ntrain,'parameter_count':len(theta),'initializer':initializer,
                'regularization':'Gaussian penalties, SD 10 for h and 2 for J, scaled by training window count',
                'initial_pair_sum_line_search':initial_balance,'refinement':trace,'pairwise':float(predicted[10]),'pairwise_histogram':predicted.tolist(),
                'mc_se':float(max(block_se,chain_se)),'chain_k10':rates.tolist(),'draws_per_chain':250000,
                'chain_count':4,'burn_final':3000,'health':health}
        atomic_json(results/f'pairwise-{group:02d}.json',result)
        h,J=conditional_tables(theta,n)
        np.savez_compressed(results/f'weights-{group:02d}.npz',h=h,J=J,cells=PROTOCOL['groups_zero_based'][group-1],
                            train_moments=target,simulated_moments=moments)
        print(f'Group {group}: pairwise simulation complete. Diagnostics {health}.',flush=True)
        return group


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'.cache')
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--mode',choices=('all','baselines','pairwise','check'),default='all')
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--groups',nargs='+',type=int,default=PROTOCOL['pairwise_groups_one_based'])
    opts=parser.parse_args();opts.cache=opts.cache.resolve();opts.results=opts.results.resolve()
    opts.cache.mkdir(parents=True,exist_ok=True);opts.results.mkdir(parents=True,exist_ok=True)
    if not set(opts.groups).issubset(PROTOCOL['pairwise_groups_one_based']):
        raise ValueError('Requested fit groups must be in the frozen protocol.')
    if opts.mode!='baselines':
        lib=compile_sampler(opts.cache)
        check=verify_sampler(lib);atomic_json(opts.results/'sampler-check.json',check);print('Exact enumeration check:',check,flush=True)
        if opts.mode=='check':return
    cube=prepare_data(opts.cache)
    if opts.mode in ('all','baselines'):
        rows=baselines(cube,opts.results)
        print('Frozen groups:',len(rows),'; held-out repeats:',99,flush=True)
    if opts.mode in ('all','pairwise'):
        jobs=[(g,cube,lib,opts.results) for g in opts.groups]
        if opts.jobs==1:
            for job in jobs:fit_group(job)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(opts.jobs,len(jobs))) as pool:
                list(pool.map(fit_group,jobs))


if __name__=='__main__':main()
