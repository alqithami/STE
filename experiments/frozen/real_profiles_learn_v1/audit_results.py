#!/usr/bin/env python3
"""Independent data/selected-weight/aggregation audit for a completed run."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import run

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    run.verify(out);lock=json.loads((out/'RUN_LOCK.json').read_text());c=lock['configuration']
    run.configure_torch(c['seed'],torch.device('cpu'))
    profiles=run.load_profiles(out,run.rows_manifest())
    mapping={p['metadata']['file']:p for p in profiles};data_cases=0;replay_files=0;replay_max=0.;selection_checks=0
    for profile in profiles:
        d=profile['data'];parsed=run.parse_profile(run.ROOT/profile['metadata']['relative_path'])
        fullW,fullT=run.ballot_counts(parsed,parsed['multiplicity'])
        assert np.array_equal(fullW,d['full_W']) and np.array_equal(fullT,d['full_T'])
        assert np.array_equal(run.hard_core_independent(fullW>fullW.T,'uc'),d['y'])
        for i,mult in enumerate(d['ballot_multiplicity']):
            W,T=run.ballot_counts(parsed,mult)
            assert np.array_equal(W,d['W'][i]) and np.array_equal(T,d['T'][i])
            assert mult.sum()==max(1,int(np.floor(float(d['fraction'][i])*parsed['voters'])))
            assert np.all(mult<=parsed['multiplicity']);data_cases+=1
    for fold in run.folds(c):
        fp=out/'folds'/f"fold_{fold['fold']:02d}"
        assert fold['test_source'] not in fold['training_sources']
        assert fold['development_source'] not in fold['training_sources']
        assert fold['test_source']!=fold['development_source']
        for ip in sorted(fp.glob('init_*')):
            choices=json.loads((ip/'CHOICES_BEFORE_TEST.json').read_text())['choices'];candidate_records=[]
            for cp in sorted(ip.glob('*_lambda_*')):candidate_records.extend(json.loads((cp/'selection.json').read_text()))
            models={}
            for method,choice in choices.items():
                valid=[x for x in candidate_records if x['system']==choice['system'] and x['readout']==choice['readout']]
                assert choice==max(valid,key=run.choice_key);selection_checks+=1
                checkpoint=fp/choice['checkpoint']
                if str(checkpoint) not in models:
                    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    model=run.PairModel(c['hidden'],choice['system']=='relational_aux');model.load_state_dict(state['model']);model.eval();models[str(checkpoint)]=model
                model=models[str(checkpoint)]
                for saved in sorted((ip/'test_predictions'/method.replace('/','_')).glob('*.npz')):
                    filename=saved.name[:-4];p=mapping[filename]
                    assert p['metadata']['source']==fold['test_source']
                    with np.load(saved) as z:old={k:z[k] for k in z.files}
                    pred=run.predict(model,[p],c,torch.device('cpu'))[0]
                    for name in ('P','direct','structural'):
                        error=float(np.max(np.abs(pred[name]-old[name])));replay_max=max(replay_max,error)
                        if not np.allclose(pred[name],old[name],atol=2e-5,rtol=2e-5):raise RuntimeError('Selected checkpoint replay mismatch: '+str(saved)+'/'+name)
                    # CPU/GPU near-half numerical differences, if any, must be disclosed, not silently hidden.
                    if not np.array_equal(pred['hard'],old['hard']):raise RuntimeError('Hard UC differs on selected-weight CPU replay: '+str(saved))
                    assert np.array_equal(pred['profile']['data']['y'],old['y']);replay_files+=1
    df=pd.read_csv(out/'PER_CASE_METRICS.csv',dtype={'source':str,'file':str})
    primary=df[np.isclose(df.fraction,c['primary_fraction'])]
    recomputed=primary.groupby(['source','file','method']).f1.mean().groupby(['source','method']).mean().groupby('method').mean()
    reported=pd.read_csv(out/'DESCRIPTIVE_PRIMARY_SUMMARY.csv').set_index('method').f1
    if not np.allclose(recomputed.sort_index(),reported.sort_index(),atol=1e-12,rtol=0):raise RuntimeError('Source/profile/resample aggregation mismatch')
    result={'classification':'software smoke audit' if c['software_smoke'] else 'completed production exploratory audit',
            'raw_ballot_reconstruction_cases':data_cases,'selected_weight_prediction_files_replayed':replay_files,
            'development_best_choice_checks':selection_checks,'max_cpu_replay_absolute_error':replay_max,
            'source_macro_aggregation_recomputed':True,'profiles':36,'selective_profiles':3,'no_new_outcomes_from_smoke':c['software_smoke']}
    run.atomic_json(out/'REPLAY_AUDIT.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
