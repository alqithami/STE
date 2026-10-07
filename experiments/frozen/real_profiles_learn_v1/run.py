#!/usr/bin/env python3
"""Exploratory, whole-source-heldout training on actual preference ballots."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'frozen'))
import torch
from torch.nn import functional as F
from ste.common import atomic_json, atomic_npz, configure_torch, seed_for, sha
from ste.data import parse_profile, ballot_counts, nested_subsamples
from ste.models import PairModel, pair_loss
from ste.operators import hard_core, hard_core_independent, soft_uc
from ste.baselines import count_methods
from ste.metrics import decide, select_threshold, set_metrics, thresholds


def emit(**record):
    print(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**record},sort_keys=True),flush=True)


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: raise RuntimeError('Empty table: '+str(path))
    keys=list(dict.fromkeys(k for row in rows for k in row))
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    temporary.replace(path)


def source_digest():
    names=['run.py','config.json','PROTOCOL.md','data/ELIGIBLE_PROFILES.csv']
    names+= [str(p.relative_to(ROOT)) for p in sorted((ROOT/'frozen').rglob('*.py'))]
    return {name:sha(ROOT/name) for name in names}


def load_config(smoke=False):
    c=json.loads((ROOT/'config.json').read_text())
    if smoke:
        c.update(initializations=1,subsample_replicates=2,system_budget_seconds=1.,posterior_draws=64,candidate_checkpoints=1)
    c['software_smoke']=smoke
    return c


def rows_manifest():
    with (ROOT/'data/ELIGIBLE_PROFILES.csv').open() as f: rows=list(csv.DictReader(f))
    for row in rows:
        for k in ('n','voters','uc_size'):row[k]=int(row[k])
    return rows


def inventory(c):
    rows=rows_manifest(); seen=set(); sources=set(); verified=[]
    for row in rows:
        path=ROOT/row['relative_path']
        if sha(path)!=row['sha256']:raise RuntimeError('Raw input checksum mismatch: '+row['file'])
        p=parse_profile(path); W,T=ballot_counts(p,p['multiplicity']); n=p['n']; ii,jj=np.triu_indices(n,1)
        if not (4<=n<=48 and np.all((W+W.T)[ii,jj]>0) and np.all(W[ii,jj]!=W[jj,ii])):
            raise RuntimeError('Strict-profile eligibility changed: '+row['file'])
        y=hard_core(W>W.T,'uc')
        if not np.array_equal(y,hard_core_independent(W>W.T,'uc')):raise RuntimeError('Independent UC oracle mismatch')
        if (n,p['voters'],int(y.sum()))!=(row['n'],row['voters'],row['uc_size']):raise RuntimeError('Frozen inventory changed')
        digest=hashlib.sha256(W.tobytes()+T.tobytes()).hexdigest()
        if digest in seen:raise RuntimeError('Duplicate full pair counts across profiles')
        seen.add(digest);sources.add(row['source']);verified.append(row)
    if len(rows)!=c['expected_profiles'] or sorted(sources)!=c['sources']:
        raise RuntimeError('Frozen real cohort count/source mismatch')
    if sum(1<r['uc_size']<r['n'] for r in rows)!=c['expected_selective_profiles']:
        raise RuntimeError('Frozen selective count mismatch')
    return verified


def folds(c):
    sources=c['sources']
    return [{'fold':i,'test_source':s,'development_source':sources[(i+1)%len(sources)],
             'training_sources':[t for t in sources if t not in (s,sources[(i+1)%len(sources)])]}
            for i,s in enumerate(sources)]


def prepare(c,out):
    out.mkdir(parents=True,exist_ok=True)
    lock={'configuration':c,'sources':source_digest(),'raw_input_hashes':{r['file']:r['sha256'] for r in inventory(c)}}
    lp=out/'RUN_LOCK.json'
    if lp.exists() and json.loads(lp.read_text())!=lock:raise RuntimeError('Refusing changed configuration/code/inputs in existing output directory')
    atomic_json(lp,lock);atomic_json(out/'FOLDS_BEFORE_LEARNING.json',folds(c))
    atomic_json(out/'COHORT_SCOPE.json',{'eligible_profiles':36,'sources':6,'selective_profiles':3,
      'reference':'hard UC of full archived empirical strict majority relation',
      'input':'actual ballot subsamples without replacement; unranked alternatives unobserved',
      'strictness_is_full_reference_only':True,'old_profile_results_already_seen':True,
      'cohort_was_preselected_in_earlier_metadata_screen':True,'human_sources_only':True,
      'excluded_constituency_and_sports_sources':True,'population_labels_available':False,
      'source_series_independence_not_asserted':True,'bootstrap_cases_not_independent_experimental_units':True})
    observation_manifest=out/'OBSERVATION_HASHES.json'
    retained=json.loads(observation_manifest.read_text()) if observation_manifest.exists() else None
    expected={'observations/'+r['file']+'/data.npz' for r in inventory(c)}
    if retained is not None:
        if set(retained)!=expected:raise RuntimeError('Recorded observation manifest membership changed')
        for rel,digest in retained.items():
            if not (out/rel).exists() or sha(out/rel)!=digest:raise RuntimeError('Retained observation hash failed: '+rel)
    for r in inventory(c):
        path=out/'observations'/r['file']/'data.npz'
        if retained is not None:continue
        # Without a completed observation hash manifest, regenerate every case
        # deterministically from verified raw records instead of trusting files.
        p=parse_profile(ROOT/r['relative_path']);fullW,fullT=ballot_counts(p,p['multiplicity'])
        ys=hard_core(fullW>fullW.T,'uc'); Ws=[];Ts=[];mults=[];fs=[];rs=[]
        for rep in range(c['subsample_replicates']):
            rng=np.random.default_rng(seed_for(c['seed'],'real-ballot-subsample',r['file'],rep))
            for frac,(W,T,mult) in nested_subsamples(p,c['fractions'],rng).items():
                if np.any(mult>p['multiplicity']) or mult.sum()!=max(1,int(np.floor(frac*p['voters']))):raise RuntimeError('Invalid without-replacement sample')
                Ws.append(W);Ts.append(T);mults.append(mult);fs.append(frac);rs.append(rep)
        atomic_npz(path,W=np.asarray(Ws),T=np.asarray(Ts),ballot_multiplicity=np.asarray(mults),fraction=np.asarray(fs),
                   replicate=np.asarray(rs),y=ys,full_W=fullW,full_T=fullT,ranks=p['ranks'],original_multiplicity=p['multiplicity'])
    if retained is None:
        atomic_json(observation_manifest,{str(p.relative_to(out)):sha(p) for p in sorted((out/'observations').rglob('*.npz'))})
    return inventory(c)


def load_profiles(out, rows):
    loaded=[]
    for row in rows:
        with np.load(out/'observations'/row['file']/'data.npz') as z:data={k:z[k] for k in z.files}
        loaded.append({'metadata':row,'data':data})
    return loaded


@torch.no_grad()
def predict(model,profiles,c,device,only_primary=False):
    model.eval(); result=[]
    for profile in profiles:
        d=profile['data']; idx=np.flatnonzero(np.isclose(d['fraction'],c['primary_fraction'])) if only_primary else np.arange(len(d['W']))
        W=torch.as_tensor(d['W'][idx],dtype=torch.float32,device=device);T=torch.as_tensor(d['T'][idx],dtype=torch.float32,device=device)
        P,direct=model(W,T);structural=soft_uc(P,c['temperature'],c['temperature'])
        if not torch.isfinite(P).all() or not torch.isfinite(direct).all() or not torch.isfinite(structural).all():raise RuntimeError('Nonfinite model prediction')
        ps=P.cpu().numpy();hard=np.stack([hard_core(p>.5,'uc') for p in ps])
        result.append({'profile':profile,'indices':idx,'P':ps,'direct':direct.cpu().numpy(),'structural':structural.cpu().numpy(),'hard':hard})
    return result


def select_from_predictions(preds,c,readout):
    if readout=='hard':
        value=float(np.mean([np.mean([set_metrics(p['profile']['data']['y'],v)['f1'] for v in p['hard']]) for p in preds]))
        return {'readout':'hard','threshold':None,'development_macro_profile_f1':value}
    xs=[]; ys=[]; weights=[]
    for p in preds:
        values=p[readout];y=p['profile']['data']['y']
        xs.extend(values);ys.extend([y]*len(values));weights.extend([1/len(values)]*len(values))
    cfg={'decisions':{'threshold_step':c['threshold_step']}}
    h,f=select_threshold(xs,ys,thresholds(cfg),'absolute',weights)
    return {'readout':readout,'threshold':h,'development_macro_profile_f1':f}


def candidate(c,fold,init,system,weight,budget,train,dev,path,device):
    path.mkdir(parents=True,exist_ok=True)
    marker=path/'COMPLETE.json'
    if marker.exists():
        mark=json.loads(marker.read_text())
        for name,digest in mark['files'].items():
            if sha(path/name)!=digest:raise RuntimeError('Completed candidate artifact changed')
        return json.loads((path/'selection.json').read_text()),json.loads((path/'budget.json').read_text())
    # Incomplete candidates restart from the same initialization and receive their full budget.
    # Completed candidates are reused only after hash verification.
    seed=seed_for(c['seed'],'real-model-initialization',fold['fold'],init)
    configure_torch(seed,device);rng=np.random.default_rng(seed_for(c['seed'],'real-training-order',fold['fold'],init))
    model=PairModel(c['hidden'],system=='relational_aux').to(device);optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};history=[]; selections=[];steps=0
    source_groups={s:[p for p in train if p['metadata']['source']==s] for s in fold['training_sources']}
    if any(not ps for ps in source_groups.values()):raise RuntimeError('Empty training source')
    tensors=[]
    for profile in train:
        d=profile['data'];profile['tensors']={k:torch.as_tensor(d[k],dtype=torch.float32,device=device) for k in ('W','T','y')}
    if device.type=='cuda' and any(p.device.type!='cuda' for p in model.parameters()):raise RuntimeError('Model is not on GPU')
    atomic_json(path/'DEVICE_PROOF.json',{'model_parameter_devices':sorted({str(p.device) for p in model.parameters()}),
        'input_devices':sorted({str(p['tensors']['W'].device) for p in train}),'label_devices':sorted({str(p['tensors']['y'].device) for p in train}),
        'fold':fold,'initialization':init,'system':system,'weight':weight,'smoke':c['software_smoke']})
    if device.type=='cuda':torch.cuda.synchronize(device)
    start=time.perf_counter();checkpoint=0;next_at=budget/c['candidate_checkpoints']
    last_loss=None;last_norm=None
    while True:
        elapsed=time.perf_counter()-start
        if steps and elapsed>=next_at:
            checkpoint+=1
            preds=predict(model,dev,c,device,only_primary=True)
            state_path=path/f'checkpoint_{checkpoint:02d}.pt'
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'optimizer_steps':steps,
                        'system':system,'weight':weight,'seed':seed},state_path)
            readouts=['hard','structural']+(['direct'] if system in ('aux','relational_aux') else [])
            for readout in readouts:
                selections.append({**select_from_predictions(preds,c,readout),'checkpoint':str(state_path.relative_to(path.parent.parent)),
                                   'checkpoint_index':checkpoint,'optimizer_steps':steps,'weight':weight,'system':system})
            for p in preds:
                atomic_npz(path/'development'/f"c{checkpoint:02d}_{p['profile']['metadata']['file']}.npz",
                    P=p['P'],direct=p['direct'],structural=p['structural'],hard=p['hard'],y=p['profile']['data']['y'],indices=p['indices'])
            if device.type=='cuda':torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-start
            history.append({'checkpoint':checkpoint,'optimizer_steps':steps,'elapsed_training_and_development_seconds':elapsed,
                            'last_total_loss':last_loss,'last_gradient_norm_pre_clip':last_norm})
            emit(stage='real_candidate_checkpoint',fold=fold['fold'],init=init,system=system,weight=weight,checkpoint=checkpoint,
                 optimizer_steps=steps,elapsed_seconds=elapsed,budget_seconds=budget,device=str(device))
            if elapsed>=budget or checkpoint>=c['candidate_checkpoints']:break
            next_at=min(budget,(checkpoint+1)*budget/c['candidate_checkpoints'])
        source=fold['training_sources'][int(rng.integers(len(fold['training_sources'])))]; pool=source_groups[source]
        p=pool[int(rng.integers(len(pool)))];d=p['data'];t=p['tensors']
        frac=c['fractions'][int(rng.integers(len(c['fractions'])))];indices=np.flatnonzero(np.isclose(d['fraction'],frac))
        indices=rng.choice(indices,size=c['batch_size'],replace=True);ix=torch.as_tensor(indices,device=device)
        model.train();optimizer.zero_grad(set_to_none=True);P,direct=model(t['W'][ix],t['T'][ix]);pair=pair_loss(P,t['W'][ix])
        y=t['y'][None].expand(len(indices),-1);core=torch.zeros((),device=device)
        if system in ('aux','relational_aux'):core=F.binary_cross_entropy(direct.clamp(1e-6,1-1e-6),y)
        elif system=='ste':core=F.binary_cross_entropy(soft_uc(P,c['temperature'],c['temperature']).clamp(1e-6,1-1e-6),y)
        loss=pair+weight*core
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite real training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['gradient_clip'])
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite real training gradient')
        optimizer.step();steps+=1;last_loss=float(loss.detach());last_norm=float(norm)
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v.detach().cpu(),initial[k])]
    if not changed or not any(k.startswith('encoder') for k in changed) or not steps:raise RuntimeError('Training failed to update encoder weights')
    if device.type=='cuda':torch.cuda.synchronize(device)
    audit={'allocated_seconds':budget,'measured_training_and_development_seconds':time.perf_counter()-start,'optimizer_steps':steps,
           'changed_parameter_names':changed,'candidate_restart_policy':'incomplete candidate restarts; completed candidates checksum-verified',
           'includes_development_and_checkpoint_writing':True,'checkpoint_count':checkpoint,'system':system,'weight':weight,'initialization_seed':seed}
    atomic_json(path/'selection.json',selections);atomic_json(path/'budget.json',audit);write_csv(path/'history.csv',history)
    atomic_json(marker,{'files':{str(p.relative_to(path)):sha(p) for p in sorted(path.rglob('*')) if p.is_file() and p!=marker}})
    return selections,audit


def choice_key(x):
    return (x['development_macro_profile_f1'],-x['checkpoint_index'],-x['weight'],
            -abs((x['threshold'] if x['threshold'] is not None else .5)-.5),-(x['threshold'] if x['threshold'] is not None else .5))


def baseline_config(c):
    return {'learning':{'temperature':c['temperature']},'counts':{k:c[k] for k in ('posterior_draws','btl_l2','btl_iterations')}}


def count_predictions(c,out,profiles,device):
    cache={}
    for profile in profiles:
        row=profile['metadata'];path=out/'counts'/row['file']/'scores.npz';d=profile['data']
        if not path.exists() or not (path.parent/'COMPLETE.json').exists():
            scores=[];fixed=[];joints=[];diagnostics=[]
            for i,(W,T) in enumerate(zip(d['W'],d['T'])):
                sc,fx,joint,meta=count_methods(W,T,baseline_config(c),device,seed_for(c['seed'],'real-posterior',row['file'],int(d['replicate'][i]),float(d['fraction'][i])))
                scores.append(sc['uc']);fixed.append(fx['uc']);joints.append(joint['uc']);diagnostics.append({'case':i,**meta})
            arrays={k:np.stack([s[k] for s in scores]) for k in scores[0]}
            arrays.update({'fixed_'+k:np.stack([s[k] for s in fixed]) for k in fixed[0]});arrays['joint_uc']=np.stack(joints)
            atomic_npz(path,**arrays);write_csv(path.parent/'diagnostics.csv',diagnostics)
            atomic_json(path.parent/'COMPLETE.json',{'scores_sha256':sha(path),'diagnostics_sha256':sha(path.parent/'diagnostics.csv')})
            emit(stage='real_count_profile_complete',file=row['file'],cases=len(d['W']),posterior_draws=c['posterior_draws'])
        mark=json.loads((path.parent/'COMPLETE.json').read_text())
        if sha(path)!=mark['scores_sha256'] or sha(path.parent/'diagnostics.csv')!=mark['diagnostics_sha256']:raise RuntimeError('Count baseline cache changed')
        with np.load(path) as z:cache[row['file']]={k:z[k] for k in z.files}
    return cache


def test_rows(c,fold,init,method,predictions,threshold):
    rows=[]
    for p in predictions:
        d=p['profile']['data'];meta=p['profile']['metadata']
        for j,index in enumerate(p['indices']):
            v=p['values'][j];score=None if threshold is None else v;selected=v if threshold is None else decide(v,'absolute',threshold)
            row={'fold':fold['fold'],'source':meta['source'],'source_name':meta['source_name'],'file':meta['file'],'n':meta['n'],
                 'initialization':init,'method':method,'fraction':float(d['fraction'][index]),'replicate':int(d['replicate'][index]),
                 'threshold':threshold,**set_metrics(d['y'],selected,score)}
            rows.append(row)
    return rows


def run_fold(c,out,fold,profiles,cache,device):
    fp=out/'folds'/f"fold_{fold['fold']:02d}";fp.mkdir(parents=True,exist_ok=True)
    completed=fp/'COMPLETE.json'
    if completed.exists():
        m=json.loads(completed.read_text())
        for name,digest in m['files'].items():
            if sha(fp/name)!=digest:raise RuntimeError('Completed fold artifact changed')
        emit(stage='real_fold_resume_verified',fold=fold['fold']);return
    train=[p for p in profiles if p['metadata']['source'] in fold['training_sources']]
    dev=[p for p in profiles if p['metadata']['source']==fold['development_source']]
    test=[p for p in profiles if p['metadata']['source']==fold['test_source']]
    if set(p['metadata']['file'] for p in train)&set(p['metadata']['file'] for p in dev+test):raise RuntimeError('Profile split leakage')
    rows=[];budgets=[];choices=[]
    # Baseline cutoffs are calibrated using only the completely held-out development source.
    baseline_choices=[]
    for name in ('ste_lse','post','copeland','smooth_copeland','winrate','btl','hodge','rank_centrality'):
        preds=[]
        for p in dev:
            ix=np.flatnonzero(np.isclose(p['data']['fraction'],c['primary_fraction']))
            preds.append({'profile':p,'indices':ix,'score':cache[p['metadata']['file']][name][ix]})
        sel=select_from_predictions(preds,c,'score');baseline_choices.append({'method':name,**sel})
    atomic_json(fp/'BASELINE_CHOICES_BEFORE_TEST.json',{'fold':fold,'choices':baseline_choices})
    for selected in baseline_choices:
        preds=[{'profile':p,'indices':np.arange(len(p['data']['W'])),'values':cache[p['metadata']['file']][selected['method']]} for p in test]
        rows.extend(test_rows(c,fold,-1,'count/'+selected['method'],preds,selected['threshold']))
    for name in ('hard','post_half','post_gfm','all','none'):
        preds=[{'profile':p,'indices':np.arange(len(p['data']['W'])),'values':cache[p['metadata']['file']]['fixed_'+name]} for p in test]
        rows.extend(test_rows(c,fold,-1,'count/'+name,preds,None))
    for init in range(c['initializations']):
        ip=fp/f'init_{init:02d}';ip.mkdir(exist_ok=True);selected={}
        for system in c['systems']:
            weights=[0.] if system=='pair' else c['core_weights'];candidates=[]
            for weight in weights:
                cp=ip/f'{system}_lambda_{weight:g}'
                sel,audit=candidate(c,fold,init,system,weight,c['system_budget_seconds']/len(weights),train,dev,cp,device)
                candidates.extend(sel);budgets.append({'fold':fold['fold'],'initialization':init,**audit})
            readouts=['hard','structural']+(['direct'] if system in ('aux','relational_aux') else [])
            for readout in readouts:
                choice=max([x for x in candidates if x['readout']==readout],key=choice_key)
                selected[system+'/'+readout]=choice;choices.append({'fold':fold['fold'],'initialization':init,**choice})
        # All neural choices for this fold/initialization are frozen before any test model prediction.
        atomic_json(ip/'CHOICES_BEFORE_TEST.json',{'fold':fold,'choices':selected,'test_source_used_for_selection':False})
        loaded={}
        for method,choice in selected.items():
            checkpoint=fp/choice['checkpoint']
            if str(checkpoint) not in loaded:
                state=torch.load(checkpoint,map_location=device,weights_only=True)
                model=PairModel(c['hidden'],choice['system']=='relational_aux').to(device);model.load_state_dict(state['model'])
                loaded[str(checkpoint)]=predict(model,test,c,device)
            preds=loaded[str(checkpoint)];readout=choice['readout'];evaluated=[]
            for p in preds:
                pp=ip/'test_predictions'/method.replace('/','_')/(p['profile']['metadata']['file']+'.npz')
                atomic_npz(pp,P=p['P'],direct=p['direct'],structural=p['structural'],hard=p['hard'],y=p['profile']['data']['y'],indices=p['indices'])
                evaluated.append({**p,'values':p[readout]})
            rows.extend(test_rows(c,fold,init,method,evaluated,choice['threshold']))
        if device.type=='cuda':torch.cuda.empty_cache()
        emit(stage='real_fold_initialization_complete',fold=fold['fold'],initialization=init,expected_initializations=c['initializations'])
    write_csv(fp/'PER_CASE_METRICS.csv',rows);write_csv(fp/'MODEL_SELECTION.csv',choices);write_csv(fp/'BUDGET_AUDIT.csv',budgets)
    atomic_json(completed,{'files':{str(p.relative_to(fp)):sha(p) for p in sorted(fp.rglob('*')) if p.is_file() and p!=completed}})
    emit(stage='real_fold_complete',fold=fold['fold'],test_source=fold['test_source'])


def aggregate(c,out):
    import pandas as pd
    paths=sorted((out/'folds').glob('fold_*/PER_CASE_METRICS.csv'))
    if len(paths)!=len(c['sources']):raise RuntimeError('Cannot analyze incomplete source folds')
    df=pd.concat([pd.read_csv(p,dtype={'source':str,'file':str}) for p in paths],ignore_index=True)
    df.to_csv(out/'PER_CASE_METRICS.csv',index=False)
    metrics=['f1','exact','precision','recall','selected_size','target_size','ap','auc']
    # Resamples and initialization are averaged before treating profiles/source series as units.
    profile=df.groupby(['source','source_name','file','n','method','fraction','core_class'],dropna=False)[metrics].mean().reset_index()
    profile.to_csv(out/'PROFILE_MEANS.csv',index=False)
    source=profile.groupby(['source','source_name','method','fraction'],dropna=False)[metrics].mean().reset_index()
    source.to_csv(out/'SOURCE_MEANS.csv',index=False)
    summary=source.groupby(['method','fraction'],dropna=False)[metrics].mean().reset_index()
    summary['source_series']=6;summary.to_csv(out/'SOURCE_MACRO_SUMMARY.csv',index=False)
    primary=summary[np.isclose(summary.fraction,c['primary_fraction'])];primary.to_csv(out/'DESCRIPTIVE_PRIMARY_SUMMARY.csv',index=False)
    selective=profile[profile.core_class=='selective_non_singleton'];selective.to_csv(out/'THREE_SELECTIVE_PROFILE_MEANS.csv',index=False)
    # Descriptive point differences only: shared source populations and tiny selective cohort do not justify p values.
    pivot=source[np.isclose(source.fraction,c['primary_fraction'])].pivot(index='source',columns='method',values='f1')
    diffs=[]
    for baseline in ('aux/direct','relational_aux/direct','count/post','count/post_gfm','count/hard'):
        for src in pivot.index:diffs.append({'source':src,'contrast':'ste/hard minus '+baseline,'difference':float(pivot.loc[src,'ste/hard']-pivot.loc[src,baseline])})
    write_csv(out/'DESCRIPTIVE_SOURCE_DIFFERENCES.csv',diffs)
    atomic_json(out/'INTERPRETATION_LIMITS.json',{'exploratory':True,'previous_test_profile_outcomes_seen':True,
      'learning_uses_genuine_ballots':True,'profile_or_source_holdout_not_ballot_holdout_only':True,
      'target_is_archived_finite_record_not_population':True,'reference_strictness_does_not_imply_subsample_strictness':True,
      'sampled_tournaments_are_only_posterior_computation_not_synthetic_learning_data':True,
      'all_class_source_macro_is_descriptive_primary':True,'selective_count':3,'selective_sources':2,
      'no_pvalues_generated':True,'no_automatic_superiority_claim':True,'protocol_does_not_replace_existing_paper_primary_test':True})
    emit(stage='real_analysis_complete',profiles=len(profile.file.unique()),source_series=6,selective_profiles=3)


def environment(c,device):
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'cuda_runtime':torch.version.cuda,'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
            'smoke':c['software_smoke'],'torch_threads':torch.get_num_threads(),'cuda_device_count':torch.cuda.device_count(),
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'tf32':torch.backends.cuda.matmul.allow_tf32}


def finalize(c,out):
    required=['RUN_LOCK.json','OBSERVATION_HASHES.json','FOLDS_BEFORE_LEARNING.json','COHORT_SCOPE.json','ENVIRONMENT.json',
              'PER_CASE_METRICS.csv','PROFILE_MEANS.csv','SOURCE_MEANS.csv','SOURCE_MACRO_SUMMARY.csv','DESCRIPTIVE_PRIMARY_SUMMARY.csv',
              'THREE_SELECTIVE_PROFILE_MEANS.csv','DESCRIPTIVE_SOURCE_DIFFERENCES.csv','INTERPRETATION_LIMITS.json']
    files={name:sha(out/name) for name in required}
    files.update({str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('COMPLETE.json')) if p!=out/'COMPLETE.json'})
    atomic_json(out/'COMPLETE.json',{'classification':'software smoke only' if c['software_smoke'] else 'completed exploratory real-profile learning',
                                   'files':files,'production_results':not c['software_smoke']})


def verify(out):
    marker=out/'COMPLETE.json'
    if not marker.exists():raise RuntimeError('Run is not complete: COMPLETE.json missing')
    for name,digest in json.loads(marker.read_text())['files'].items():
        if sha(out/name)!=digest:raise RuntimeError('Run output hash failed: '+name)
    for mp in sorted(out.rglob('COMPLETE.json')):
        data=json.loads(mp.read_text())
        for name,digest in data.get('files',{}).items():
            if sha(mp.parent/name)!=digest:raise RuntimeError('Nested output hash failed: '+str(mp.parent/name))
        if 'scores_sha256' in data:
            if sha(mp.parent/'scores.npz')!=data['scores_sha256'] or sha(mp.parent/'diagnostics.csv')!=data['diagnostics_sha256']:
                raise RuntimeError('Count baseline output hash failed: '+str(mp.parent))
    for rel,digest in json.loads((out/'OBSERVATION_HASHES.json').read_text()).items():
        if sha(out/rel)!=digest:raise RuntimeError('Observation output hash failed: '+rel)
    emit(stage='real_all_results_verified',classification=json.loads(marker.read_text())['classification'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results');p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--smoke',action='store_true');p.add_argument('--stage',choices=['preflight','prepare','run','status','verify'],default='run');args=p.parse_args()
    c=load_config(args.smoke);out=args.out.resolve()
    if args.stage=='status':
        completed=out/'COMPLETE.json'
        emit(stage='status',complete=completed.exists(),completed_source_folds=len(list((out/'folds').glob('fold_*/COMPLETE.json'))),expected_source_folds=6)
        if completed.exists():verify(out)
        return
    if args.stage=='verify':verify(out);return
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but not available; refusing CPU fallback')
    if args.stage=='run' and not args.smoke and args.device!='cuda':raise RuntimeError('Production requires CUDA. CPU is allowed only for software smoke/preparation.')
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu');configure_torch(c['seed'],device)
    rows=inventory(c)
    if args.stage=='preflight':
        emit(stage='preflight_pass',profiles=len(rows),selective_profiles=sum(1<r['uc_size']<r['n'] for r in rows),sources=c['sources'],environment=environment(c,device));return
    rows=prepare(c,out);atomic_json(out/'ENVIRONMENT.json',environment(c,device))
    if args.stage=='prepare':emit(stage='real_observations_prepared',profiles=len(rows));return
    profiles=load_profiles(out,rows);cache=count_predictions(c,out,profiles,device)
    for fold in folds(c):run_fold(c,out,fold,profiles,cache,device)
    aggregate(c,out);finalize(c,out);verify(out)
    emit(stage='REAL_LEARNING_COMPLETE',production_results=not args.smoke,out=str(out))


if __name__=='__main__':
    try:main()
    except Exception as e:
        emit(stage='FAILED',error=repr(e));raise
