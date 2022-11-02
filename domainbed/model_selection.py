# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import itertools
import numpy as np
from domainbed.lib.query import Q
scale_weight = 1
clf_weight = 0.2
first_idx = 1
second_idx = 20
NAME = 'clf_scale_3_clf_1_idx_3_10'

print('idx_align from ', first_idx, second_idx, 'clf_weight', clf_weight, 'scale weight', scale_weight)
def get_test_records(records):
    """Given records with a common test env, get the test records (i.e. the
    records with *only* that single test env and no other test envs)"""
    return records.filter(lambda r: len(r['args']['test_envs']) == 1)

class SelectionMethod:
    """Abstract class whose subclasses implement strategies for model
    selection across hparams and timesteps."""
    
    def __init__(self):
        raise TypeError

    @classmethod
    def run_acc(self, run_records):
        """
        Given records from a run, return a {val_acc, test_acc} dict representing
        the best val-acc and corresponding test-acc for that run.
        """
        raise NotImplementedError
    
    @classmethod
    def run_loss(self, run_records):
        """
        Given records from a run, return a {val_loss, test_acc} dict representing
        the best val-loss and corresponding test-acc for that run.
        """
        raise NotImplementedError
    
    @classmethod
    def hparams_accs(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_acc, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_acc(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_acc'])[::-1]
        )
    
    @classmethod
    def hparams_loss_mmd(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_loss, records) tuples.
        """
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_loss(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_mmd_clf_loss'])
        )
    @classmethod
    def hparams_loss_coral(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return a sorted list of (run_loss, records) tuples.
        """
        
        return (records.group('args.hparams_seed')
            .map(lambda _, run_records:
                (
                    self.run_loss(run_records),
                    run_records
                )
            ).filter(lambda x: x[0] is not None)
            .sorted(key=lambda x: x[0]['val_coral_clf_loss'])
        )


    @classmethod
    def sweep_acc(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_accs(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

    @classmethod
    def sweep_mmd_loss(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_loss_mmd(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None
    
    @classmethod
    def sweep_coral_loss(self, records):
        """
        Given all records from a single (dataset, algorithm, test env) pair,
        return the mean test acc of the k runs with the top val accs.
        """
        _hparams_accs = self.hparams_loss_coral(records)
        if len(_hparams_accs):
            return _hparams_accs[0][0]['test_acc']
        else:
            return None

        
class OracleSelectionMethod(SelectionMethod):
    """Like Selection method which picks argmax(test_out_acc) across all hparams
    and checkpoints, but instead of taking the argmax over all
    checkpoints, we pick the last checkpoint, i.e. no early stopping."""
    name = "test-domain validation set (oracle)"

    @classmethod
    def run_acc(self, run_records):
        run_records = run_records.filter(lambda r:
            len(r['args']['test_envs']) == 1)
        if not len(run_records):
            return None
        test_env = run_records[0]['args']['test_envs'][0]
        test_out_acc_key = 'env{}_out_acc'.format(test_env)
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        chosen_record = run_records.sorted(lambda r: r['step'])[-1]
        return {
            'val_acc':  chosen_record[test_out_acc_key],
            'test_acc': chosen_record[test_in_acc_key]
        }

class MMDSelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_sepcial_loss for env in train_envs))"""
    name = "training-domain mmd validation set"
    # print('clf_weight', alig_weight, 'idx is', first_idx, second_idx)
    @classmethod
    def _step_loss(self, record):
        """Given a single record, return a {val_clf_loss, val_mmd_clf_loss, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_loss')
        val_loss_mmd = record['val_mmd_loss']

        val_loss_clf = np.mean([record[key] for key in val_env_keys])
        if abs(val_loss_mmd) < 1e-5 or val_loss_mmd < 0:
            val_loss_mmd = 0
        if np.isnan(val_loss_mmd) or np.isnan(val_loss_clf):
            val_loss_clf = 1e+8
            val_loss_mmd = 1e+8

        val_loss =  scale_weight * val_loss_clf * clf_weight +  val_loss_mmd * (1-clf_weight) # important criteria
  
        return {
            'val_clf_loss': val_loss_clf,
            'val_mmd_loss': val_loss_mmd,
            'val_mmd_clf_loss': val_loss,
            'test_acc': record[test_in_acc_key]
        }   
    @classmethod
    def run_loss(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        
        mapped_test_records = test_records.map(self._step_loss).sorted(key=lambda x: x['val_mmd_loss'])[first_idx:second_idx]
        mapped_test_records = Q(mapped_test_records)

        try:
            return mapped_test_records.argmin('val_mmd_clf_loss')
        except:
            print('pass here')
            pass

class CoralSelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_sepcial_loss for env in train_envs))"""
    name = "training-domain coral validation set"

    @classmethod
    def _step_loss(self, record):
        """Given a single record, return a {val_loss, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_loss')

        val_loss_coral = record['val_coral_loss']
        val_loss_clf = np.sum([record[key] for key in val_env_keys])
        if abs(val_loss_coral) < 1e-5 or val_loss_coral < 0:
            val_loss_coral = 0
        if np.isnan(val_loss_coral) or np.isnan(val_loss_clf):
            val_loss_clf = 1e+8
            val_loss_coral = 1e+8
        val_loss =  scale_weight * val_loss_clf * clf_weight +   val_loss_coral * (1-clf_weight) 

        return {
            'val_clf_loss': val_loss_clf,
            'val_coral_loss': val_loss_coral,
            'val_coral_clf_loss': val_loss,
            'test_acc': record[test_in_acc_key]
        }   
    @classmethod
    def run_loss(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        
        mapped_test_records = test_records.map(self._step_loss).sorted(key=lambda x: x['val_coral_loss'])[first_idx:second_idx]
        mapped_test_records = Q(mapped_test_records)
        try:
            return mapped_test_records.argmin('val_coral_clf_loss')
        except:
            print('pass here')
            pass


class IIDAccuracySelectionMethod(SelectionMethod):
    """Picks argmax(mean(env_out_acc for env in train_envs))"""
    name = "training-domain validation set"

    @classmethod
    def _step_acc(self, record):
        """Given a single record, return a {val_acc, test_acc} dict."""
        test_env = record['args']['test_envs'][0]
        val_env_keys = []
        for i in itertools.count():
            if f'env{i}_out_acc' not in record:
                break
            if i != test_env:
                val_env_keys.append(f'env{i}_out_acc')
        test_in_acc_key = 'env{}_in_acc'.format(test_env)
        
        return {
            'val_acc': np.mean([record[key] for key in val_env_keys]),
            'test_acc': record[test_in_acc_key]
        }

    @classmethod
    def run_acc(self, run_records):
        test_records = get_test_records(run_records)
        if not len(test_records):
            return None
        return test_records.map(self._step_acc).argmax('val_acc')

class LeaveOneOutSelectionMethod(SelectionMethod):
    """Picks (hparams, step) by leave-one-out cross validation."""
    name = "leave-one-domain-out cross-validation"

    @classmethod
    def _step_acc(self, records):
        """Return the {val_acc, test_acc} for a group of records corresponding
        to a single step."""
        test_records = get_test_records(records)
        if len(test_records) != 1:
            return None

        test_env = test_records[0]['args']['test_envs'][0]
        n_envs = 0
        for i in itertools.count():
            if f'env{i}_out_acc' not in records[0]:
                break
            n_envs += 1
        val_accs = np.zeros(n_envs) - 1
        for r in records.filter(lambda r: len(r['args']['test_envs']) == 2):
            val_env = (set(r['args']['test_envs']) - set([test_env])).pop()
            val_accs[val_env] = r['env{}_in_acc'.format(val_env)]
        val_accs = list(val_accs[:test_env]) + list(val_accs[test_env+1:])
        if any([v==-1 for v in val_accs]):
            return None
        val_acc = np.sum(val_accs) / (n_envs-1)
        return {
            'val_acc': val_acc,
            'test_acc': test_records[0]['env{}_in_acc'.format(test_env)]
        }

    @classmethod
    def run_acc(self, records):
        step_accs = records.group('step').map(lambda step, step_records:
            self._step_acc(step_records)
        ).filter_not_none()
        if len(step_accs):
            return step_accs.argmax('val_acc')
        else:
            return None

