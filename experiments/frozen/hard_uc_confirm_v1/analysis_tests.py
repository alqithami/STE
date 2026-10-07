#!/usr/bin/env python3
"""Independent software tests for inference, budget gates and transport integrity.

Synthetic fixtures here test rejection boundaries and arithmetic only. They are
not experimental measurements and must never enter the manuscript results.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from analyze import budget_audit, primary_analysis, collection_means, METRICS, atomic_json
from package_results import split_archive
from verify_archive import verify_parts, safe_name
from ste.common import finish_unit

HERE = Path(__file__).resolve().parent


class BudgetGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = json.loads((HERE / 'config.json').read_text())
        self.cfg['learning']['collections'] = 1
        self.cfg['learning']['initializations'] = 1
        self.paths = []
        for system in self.cfg['learning']['objectives']:
            for weight in ([0.] if system == 'pair-hard' else self.cfg['learning']['core_weights']):
                path = self.root / 'learning/collection_00/uc/init_00' / f'{system}_lambda_{weight:g}'
                path.mkdir(parents=True)
                budget = 220. if system == 'pair-hard' else 110.
                timing = {'budget_seconds':budget, 'charged_seconds':budget, 'last_quantum_seconds':.2,
                          'checkpoint_records':[{'no_training_since_previous_checkpoint':False} for _ in range(5)]}
                for index in range(1,6): (path / f'checkpoint_{index:02d}.pt').write_bytes(b'QA fixture; not model weights')
                atomic_json(path / 'timing.json', timing)
                self.seal(path)
                self.paths.append(path)
        self.cost_path = self.root / 'learning/collection_00/uc/init_00/SELECTION_COST.json'
        atomic_json(self.cost_path, {'systems':{system:{'winner_scan_seconds':0., 'shared_freeze_seconds':0., 'charged_seconds':0.} for system in self.cfg['learning']['objectives']},
                                    'charged_seconds_total':0., 'complete':True, 'included_in_complete_system_budget':True})
    def tearDown(self): self.temp.cleanup()
    def seal(self, path): finish_unit(path, [p for p in path.iterdir() if p.name != 'COMPLETE.json'])
    def change(self, path, **updates):
        timing = json.loads((path / 'timing.json').read_text()); timing.update(updates)
        atomic_json(path / 'timing.json', timing); self.seal(path)
    def test_complete_equal_allocations(self):
        candidates, systems, valid = budget_audit(self.cfg, self.root)
        self.assertTrue(valid); self.assertEqual(len(candidates),11); self.assertEqual(len(systems),6)
        self.assertTrue(np.array_equal(systems.budget_seconds.to_numpy(),np.full(6,220.)))
    def test_fixed_two_percent_boundary(self):
        self.change(self.paths[0], charged_seconds=224.4)
        self.assertTrue(budget_audit(self.cfg, self.root)[2])
        self.change(self.paths[0], charged_seconds=224.40001)
        self.assertFalse(budget_audit(self.cfg, self.root)[2])
    def test_undershoot_is_invalid(self):
        self.change(self.paths[1], charged_seconds=107.79)
        self.assertFalse(budget_audit(self.cfg, self.root)[2])
    def test_skipped_training_checkpoint_is_invalid(self):
        records = [{'no_training_since_previous_checkpoint':i == 2} for i in range(5)]
        self.change(self.paths[3], checkpoint_records=records)
        self.assertFalse(budget_audit(self.cfg, self.root)[2])
    def test_fixed_tolerance_cannot_be_relaxed(self):
        cfg=copy.deepcopy(self.cfg); cfg['budget']['relative_tolerance']=.05
        with self.assertRaisesRegex(RuntimeError,'exactly 0.02'): budget_audit(cfg,self.root)
    def test_unequal_system_allocation_is_rejected(self):
        self.change(self.paths[1], budget_seconds=120., charged_seconds=120.)
        with self.assertRaisesRegex(RuntimeError,'prospective configuration'): budget_audit(self.cfg,self.root)
    def test_missing_checkpoint_is_rejected(self):
        (self.paths[1]/'checkpoint_03.pt').unlink(); self.seal(self.paths[1])
        with self.assertRaisesRegex(RuntimeError,'retained checkpoints'): budget_audit(self.cfg,self.root)
    def test_missing_selection_cost_is_rejected(self):
        self.cost_path.unlink()
        with self.assertRaisesRegex(RuntimeError,'Missing charged development-selection cost'): budget_audit(self.cfg,self.root)
    def test_final_selection_cost_can_invalidate_complete_system_budget(self):
        cost=json.loads(self.cost_path.read_text())
        cost['systems']['ste-hard'].update(winner_scan_seconds=4.401,charged_seconds=4.401)
        cost['charged_seconds_total']=4.401; atomic_json(self.cost_path,cost)
        candidates,systems,valid=budget_audit(self.cfg,self.root)
        self.assertTrue(candidates.candidate_budget_valid.all()); self.assertFalse(valid)
        self.assertFalse(bool(systems[systems.objective=='ste-hard'].iloc[0].system_budget_valid))


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg=json.loads((HERE/'config.json').read_text())
        rows=[]
        for c in range(12):
            for system,readout,rule,value in [('ste-hard','hard_uc','hard',.7),('aux-direct','direct','absolute',.6),('relational-direct','direct','absolute',.59)]:
                rows.append({'collection':c,'target':'uc','n':24,'objective':system,'readout':readout,'rule':rule,'core_class':'selective_non_singleton','f1':value})
        self.cores=pd.DataFrame(rows)
    def test_exact_twelve_unit_two_test_family(self):
        primary,differences=primary_analysis(self.cfg,self.cores,True)
        self.assertEqual(len(primary),2); self.assertEqual(len(differences),24)
        for row in primary:
            self.assertEqual(row['units'],12)
            self.assertAlmostEqual(row['p'],2/4096)
            self.assertAlmostEqual(row['p_holm'],4/4096)
            self.assertEqual(row['left_rule'],'hard'); self.assertEqual(row['right_rule'],'absolute')
    def test_invalid_budget_and_smoke_suppress_inference(self):
        rows,_=primary_analysis(self.cfg,self.cores,False)
        for row in rows:
            self.assertTrue(all(row[key] is None for key in ['p','p_holm','ci_low','ci_high']))
            self.assertFalse(row['formal_inference_valid'])
    def test_missing_collection_is_never_silently_dropped(self):
        with self.assertRaisesRegex(RuntimeError,'never silently drop'):
            primary_analysis(self.cfg,self.cores[~((self.cores.collection==11)&(self.cores.objective=='aux-direct'))],True)
    def test_initializations_are_averaged_before_collections(self):
        rows=[]
        for init,values in [(0,[0.,1.]),(1,[.2,.2]),(2,[.9,.9])]:
            for value in values:
                row={'collection':0,'target':'uc','n':24,'objective':'ste-hard','readout':'hard_uc','rule':'hard','init':init}
                row.update({metric:value for metric in METRICS}); rows.append(row)
        result=collection_means(pd.DataFrame(rows))
        self.assertEqual(len(result),1)
        self.assertAlmostEqual(float(result.iloc[0].f1),(.5+.2+.9)/3)


class TransportTests(unittest.TestCase):
    def test_ordered_parts_hash_and_tamper(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'review.zip'
            path.write_bytes(bytes(range(256))*37)
            split=split_archive(path,1000)
            manifest=path.parent/split['manifest_name']
            result=verify_parts(manifest)
            self.assertEqual(result['parts_verified'],10)
            first=path.parent/split['parts'][0]['name']
            data=first.read_bytes(); first.write_bytes(bytes([data[0]^1])+data[1:])
            with self.assertRaisesRegex(RuntimeError,'byte/hash mismatch'): verify_parts(manifest)
    def test_unsafe_or_reordered_parts_are_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'review.zip'; path.write_bytes(b'a'*3000)
            split=split_archive(path,1000); manifest=path.parent/split['manifest_name']
            value=json.loads(manifest.read_text()); value['parts'][0],value['parts'][1]=value['parts'][1],value['parts'][0]
            atomic_json(manifest,value)
            with self.assertRaisesRegex(RuntimeError,'out-of-order'): verify_parts(manifest)
        for name in ['/etc/passwd','../state.pt','run/../state.pt','run\\state.pt']:
            self.assertFalse(safe_name(name))


if __name__=='__main__': unittest.main(verbosity=2)
