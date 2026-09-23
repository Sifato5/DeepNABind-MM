import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import DeepNABindMM
from train import ShardedNABPDataset, collate_fn, Trainer, MetricsCalculator, set_seed, load_dataset
from evaluate import TestSetEvaluator


def cross_validate(args):
    
    set_seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cv_output_dir = Path(args.output_dir) / f"cv_{timestamp}"
    cv_output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    manifest_path = data_dir / 'shard_manifest.json'
    splits_path = data_dir / 'cv_splits.json'
    
    if not splits_path.exists():
        print(f" CV splits not found at {splits_path}")
        print("   Please run the dataset preparation script first to generate splits.")
        return
    
    with open(splits_path, 'r') as f:
        cv_splits = json.load(f)
    
    with open(manifest_path, 'r') as f:
        shard_manifest = json.load(f)
    
    n_folds = len(cv_splits)
    print("\n" + "="*80)
    print(f"Running {n_folds}-Fold Cross-Validation")
    print("="*80)
    
    all_fold_results = []
    all_fold_models = []
    
    for fold in range(n_folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        fold_output_dir = cv_output_dir / f"fold_{fold}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
       
        split = cv_splits[fold]
        train_ids = split['train']
        val_ids = split['val']
        
        print(f"  Train samples: {len(train_ids)}")
        print(f"  Val samples: {len(val_ids)}")
        
       
        train_dataset = ShardedNABPDataset(
            args.data_dir, train_ids, shard_manifest, return_binding_labels=True
        )
        val_dataset = ShardedNABPDataset(
            args.data_dir, val_ids, shard_manifest, return_binding_labels=True
        )
        
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
       
        model = DeepNABindMM(
            hidden_dim=args.hidden_dim,
            num_classes=3,
            esm_dim=2560,
            dropout=args.dropout,
            temperature=args.temperature
        )
        
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=fold_output_dir,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            patience=args.patience,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3
        )
        
        
        history = trainer.train(num_epochs=args.epochs)
        
        final_metrics = trainer.validate()
        
        all_fold_results.append({
            'fold': fold,
            'train_size': len(train_ids),
            'val_size': len(val_ids),
            'best_val_f1': trainer.best_val_f1,
            'final_metrics': final_metrics,
            'training_history': history
        })
        
        model_path = fold_output_dir / 'best_model.pt'
        all_fold_models.append(str(model_path))
        
        print(f"\n  Fold {fold + 1} Summary:")
        print(f"    Best Val F1: {trainer.best_val_f1:.4f}")
        print(f"    Final Val Acc: {final_metrics['accuracy']:.4f}")
        print(f"    Final Val Macro F1: {final_metrics['macro_f1']:.4f}")
        print(f"    Final Val MCC: {final_metrics['mcc']:.4f}")
    
    metrics_keys = [
        'accuracy', 'macro_f1', 'weighted_f1', 'mcc', 'auc_ovr', 'auprc_ovr',
        'binding_auroc', 'binding_auprc', 'binding_precision',
        'binding_recall', 'binding_f1', 'binding_mcc'
    ]
    aggregated_metrics = defaultdict(list)
    
    for fold_result in all_fold_results:
        metrics = fold_result['final_metrics']
        for key in metrics_keys:
            if key in metrics:
                aggregated_metrics[key].append(metrics[key])
    
    cv_results = {}
    for key, values in aggregated_metrics.items():
        cv_results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    print("\nAggregated Metrics (Mean ± Std):")
    print(f"  Accuracy:    {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
    print(f"  Macro F1:    {cv_results['macro_f1']['mean']:.4f} ± {cv_results['macro_f1']['std']:.4f}")
    print(f"  Weighted F1: {cv_results['weighted_f1']['mean']:.4f} ± {cv_results['weighted_f1']['std']:.4f}")
    print(f"  MCC:         {cv_results['mcc']['mean']:.4f} ± {cv_results['mcc']['std']:.4f}")
    print(f"  AUC (OVR):   {cv_results['auc_ovr']['mean']:.4f} ± {cv_results['auc_ovr']['std']:.4f}")
    print(f"  AUPRC (OVR): {cv_results['auprc_ovr']['mean']:.4f} ± {cv_results['auprc_ovr']['std']:.4f}")
    for key, label in [
        ('binding_auroc', 'Binding AUROC'),
        ('binding_auprc', 'Binding AUPRC'),
        ('binding_f1', 'Binding F1'),
        ('binding_mcc', 'Binding MCC')
    ]:
        if key in cv_results:
            print(f"  {label:<14}: {cv_results[key]['mean']:.4f} ± {cv_results[key]['std']:.4f}")
   
    cv_summary = {
        'n_folds': n_folds,
        'aggregated_metrics': cv_results,
        'per_fold_results': all_fold_results,
        'config': vars(args),
        'model_paths': all_fold_models
    }
    
    cv_summary_file = cv_output_dir / 'cv_summary.json'
    with open(cv_summary_file, 'w') as f:
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable = convert_to_serializable(cv_summary)
        json.dump(serializable, f, indent=2)
    
    if args.train_final:
        print("\n" + "="*80)
        print("Training Final Model on All Data")
        print("="*80)
        
        final_output_dir = cv_output_dir / 'final_model'
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_ids = []
        for shard in shard_manifest['shards']:
            all_ids.extend(shard['proteins'])
        
        full_dataset = ShardedNABPDataset(
            args.data_dir, all_ids, shard_manifest, return_binding_labels=True
        )
        
        full_loader = DataLoader(
            full_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        from sklearn.model_selection import train_test_split
        train_ids_final, val_ids_final = train_test_split(
            all_ids, test_size=0.1, random_state=args.seed, stratify=None
        )
        
        train_dataset_final = ShardedNABPDataset(
            args.data_dir, train_ids_final, shard_manifest, return_binding_labels=True
        )
        val_dataset_final = ShardedNABPDataset(
            args.data_dir, val_ids_final, shard_manifest, return_binding_labels=True
        )
        
        train_loader_final = DataLoader(
            train_dataset_final,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        
        val_loader_final = DataLoader(
            val_dataset_final,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
        
        final_model = DeepNABindMM(
            hidden_dim=args.hidden_dim,
            num_classes=3,
            esm_dim=2560,
            dropout=args.dropout,
            temperature=args.temperature
        )
        
        final_trainer = Trainer(
            model=final_model,
            train_loader=train_loader_final,
            val_loader=val_loader_final,
            output_dir=final_output_dir,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            patience=args.patience,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3
        )
        
        final_trainer.train(num_epochs=args.epochs)
        
        final_model_path = final_output_dir / 'final_model.pt'
        final_trainer._save_model('final_model.pt', final_trainer.validate())
        
    
    return cv_results


def main():
    parser = argparse.ArgumentParser(description='Cross-validate DeepNABind-MM')
    
    parser.add_argument('--data_dir', type=str, 
                        default='/homeb/ali/second/data/DeepNABindMM_dataset',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./cv_results',
                        help='Output directory')
    
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for contrastive loss')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='Classification loss weight')
    parser.add_argument('--lambda2', type=float, default=0.3,
                        help='Binding loss weight')
    parser.add_argument('--lambda3', type=float, default=0.1,
                        help='Contrastive loss weight')
    
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--train_final', action='store_true',
                        help='Train final model on all data after CV')
    
    args = parser.parse_args()
    
    cross_validate(args)


if __name__ == "__main__":
    main()