import os
import sys
import json
import argparse
import pickle
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix
)

from model import DeepNABindMM

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ShardedNABPDataset(Dataset):
    
    def __init__(self, 
                 data_dir: str,
                 protein_ids: List[str],
                 shard_manifest: Dict,
                 return_binding_labels: bool = False):
        self.data_dir = Path(data_dir)
        self.protein_ids = protein_ids
        self.shard_manifest = shard_manifest
        self.return_binding_labels = return_binding_labels
        self.protein_to_location = {}
        self.shard_files = {}
        
        for shard in shard_manifest['shards']:
            shard_path = self.data_dir / 'graph_shards' / shard['file']
            self.shard_files[shard['file']] = shard_path
            
            for idx, pid in enumerate(shard['proteins']):
                if pid in protein_ids: 
                    self.protein_to_location[pid] = (shard['file'], idx)
        
        self.current_shard_file = None
        self.current_shard_data = None
        
        metadata_path = self.data_dir / 'proteins_metadata.json'
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
        
        self.metadata = {p['uniprot_id']: p for p in all_metadata}
        
    def _load_shard(self, shard_file: str):
        
        if self.current_shard_file != shard_file:
            shard_path = self.shard_files[shard_file]
            self.current_shard_data = torch.load(shard_path, map_location='cpu')
            self.current_shard_file = shard_file
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        shard_file, shard_idx = self.protein_to_location[protein_id]
        
        self._load_shard(shard_file)
        protein_data = self.current_shard_data[protein_id]
        
        graphs = protein_data['graphs']
        label = protein_data['label']
        
        atomic_graph = self._prepare_graph(graphs['atomic']) if graphs['atomic'] is not None else None
        residue_graph = self._prepare_graph(graphs['residue'])
        motif_graph = self._prepare_graph(graphs['motif']) if graphs['motif'] is not None else None
        
        seq_embedding = graphs['residue'].x  
        
        if self.return_binding_labels:
            binding_label_dir = self.data_dir / 'binding_labels'
            label_path = binding_label_dir / f"{protein_id}.npy"
            
            if not label_path.exists():
                raise FileNotFoundError(
                    f"Binding labels not found for {protein_id}: {label_path}\n"
                    "Create binding_labels/<UniProt_ID>.npy before training."
                )
            
            binding_labels = torch.tensor(
                np.load(label_path), dtype=torch.float32
            ).view(-1)
            
            if len(binding_labels) != len(seq_embedding):
                raise ValueError(
                    f"Binding-label length mismatch for {protein_id}: "
                    f"labels={len(binding_labels)}, sequence={len(seq_embedding)}"
                )
            
            if not torch.all((binding_labels == 0) | (binding_labels == 1)):
                raise ValueError(
                    f"Binding labels for {protein_id} must contain only 0/1 values."
                )
        else:
            binding_labels = None
        
        return {
            'protein_id': protein_id,
            'atomic_graph': atomic_graph,
            'residue_graph': residue_graph,
            'motif_graph': motif_graph,
            'seq_embedding': seq_embedding,
            'label': torch.tensor(label, dtype=torch.long),
            'binding_labels': binding_labels,
            'sequence_length': len(seq_embedding),
            'metadata': self.metadata.get(protein_id, {})
        }
    
    def _prepare_graph(self, graph):
        
        if graph is None:
            return None
        
        return graph


def collate_fn(batch):
    from torch_geometric.data import Batch
    
    atomic_graphs = [item['atomic_graph'] for item in batch if item['atomic_graph'] is not None]
    residue_graphs = [item['residue_graph'] for item in batch if item['residue_graph'] is not None]
    motif_graphs = [item['motif_graph'] for item in batch if item['motif_graph'] is not None]
    
    atomic_batch = Batch.from_data_list(atomic_graphs) if atomic_graphs else None
    residue_batch = Batch.from_data_list(residue_graphs) if residue_graphs else None
    motif_batch = Batch.from_data_list(motif_graphs) if motif_graphs else None
    
    seq_embeddings = [item['seq_embedding'] for item in batch]
    seq_lengths = [len(emb) for emb in seq_embeddings]
    max_len = max(seq_lengths)
    padded_embeddings = torch.zeros(len(batch), max_len, seq_embeddings[0].shape[1])
    
    for i, emb in enumerate(seq_embeddings):
        padded_embeddings[i, :len(emb)] = emb
    
    labels = torch.tensor([item['label'] if not torch.is_tensor(item['label']) else item['label'].item() 
                            for item in batch], dtype=torch.long)
    
    binding_labels = None
    binding_mask = None
    if batch[0]['binding_labels'] is not None:
        binding_labels = torch.zeros(len(batch), max_len, dtype=torch.float32)
        binding_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            L = item['binding_labels'].shape[0]
            binding_labels[i, :L] = item['binding_labels']
            binding_mask[i, :L] = True
    
    return {
        'protein_ids': [item['protein_id'] for item in batch],
        'atomic_graph': atomic_batch,
        'residue_graph': residue_batch,
        'motif_graph': motif_batch,
        'seq_embeddings': padded_embeddings,
        'seq_lengths': torch.tensor(seq_lengths),
        'labels': labels,
        'binding_labels': binding_labels,
        'binding_mask': binding_mask,
        'metadata': [item['metadata'] for item in batch]
    }

class MetricsCalculator:
    
    def __init__(self, num_classes: int = 3, class_names: List[str] = ['non-NABP', 'RBP', 'DBP']):
        self.num_classes = num_classes
        self.class_names = class_names
        
    def compute_all_metrics(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray,
                           y_proba: np.ndarray) -> Dict:
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.num_classes)
        )
        
        metrics['per_class'] = {
            self.class_names[i]: {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
            for i in range(self.num_classes)
        }
        
        metrics['macro_precision'] = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )[0]
        metrics['macro_recall'] = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )[1]
        metrics['macro_f1'] = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )[2]
        
        metrics['weighted_precision'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )[0]
        metrics['weighted_recall'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )[1]
        metrics['weighted_f1'] = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )[2]
        
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred, labels=range(self.num_classes)).tolist()
        
        try:
            metrics['auc_ovr'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='macro'
            )
            metrics['auprc_ovr'] = average_precision_score(
                y_true, y_proba, average='macro'
            )
        except Exception as e:
            metrics['auc_ovr'] = 0.0
            metrics['auprc_ovr'] = 0.0
        
        y_true_onehot = np.eye(self.num_classes)[y_true]
        metrics['per_class_auc'] = {}
        metrics['per_class_auprc'] = {}
        
        for i in range(self.num_classes):
            try:
                metrics['per_class_auc'][self.class_names[i]] = roc_auc_score(
                    y_true_onehot[:, i], y_proba[:, i]
                )
                metrics['per_class_auprc'][self.class_names[i]] = average_precision_score(
                    y_true_onehot[:, i], y_proba[:, i]
                )
            except:
                metrics['per_class_auc'][self.class_names[i]] = 0.0
                metrics['per_class_auprc'][self.class_names[i]] = 0.0
        
        return metrics
    
    @staticmethod
    def compute_binding_metrics(
        binding_labels: Optional[np.ndarray],
        binding_probs: Optional[np.ndarray],
        binding_mask: Optional[np.ndarray]
    ) -> Dict:
        """Compute residue-level binding-site metrics on valid (non-padding) residues."""
        if binding_labels is None or binding_probs is None or binding_mask is None:
            return {}
        
        y_true = np.asarray(binding_labels)[np.asarray(binding_mask, dtype=bool)]
        y_score = np.asarray(binding_probs)[np.asarray(binding_mask, dtype=bool)]
        
        if y_true.size == 0:
            return {}
        
        y_pred = (y_score >= 0.5).astype(np.int64)
        metrics = {
            'binding_positive_fraction': float(np.mean(y_true)),
            'binding_precision': 0.0,
            'binding_recall': 0.0,
            'binding_f1': 0.0,
            'binding_mcc': 0.0,
            'binding_auroc': float('nan'),
            'binding_auprc': float('nan')
        }
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        metrics['binding_precision'] = float(precision)
        metrics['binding_recall'] = float(recall)
        metrics['binding_f1'] = float(f1)
        metrics['binding_mcc'] = float(matthews_corrcoef(y_true, y_pred))
        
        if np.unique(y_true).size == 2:
            metrics['binding_auroc'] = float(roc_auc_score(y_true, y_score))
            metrics['binding_auprc'] = float(average_precision_score(y_true, y_score))
        
        return metrics
    
    def print_metrics(self, metrics: Dict, title: str = "Results"):
       
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  MCC:       {metrics['mcc']:.4f}")
        print(f"  AUC (OVR): {metrics['auc_ovr']:.4f}")
        print(f"  AUPRC (OVR): {metrics['auprc_ovr']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10} {'AUC':<10} {'AUPRC':<10}")
        print("-" * 80)
        for class_name, class_metrics in metrics['per_class'].items():
            auc = metrics['per_class_auc'].get(class_name, 0.0)
            auprc = metrics['per_class_auprc'].get(class_name, 0.0)
            print(f"{class_name:<15} {class_metrics['precision']:<12.4f} {class_metrics['recall']:<12.4f} "
                  f"{class_metrics['f1']:<12.4f} {class_metrics['support']:<10} {auc:<10.4f} {auprc:<10.4f}")
        
        print("\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(" " * 15 + "".join(f"{c:>10}" for c in metrics['per_class'].keys()))
        for i, class_name in enumerate(metrics['per_class'].keys()):
            print(f"{class_name:<15} " + "".join(f"{cm[i, j]:>10}" for j in range(cm.shape[1])))

class Trainer:
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 output_dir: str,
                 learning_rate: float = 5e-5,
                 weight_decay: float = 1e-2,
                 device: str = 'cuda',
                 patience: int = 15,
                 lambda1: float = 1.0,
                 lambda2: float = 0.3,
                 lambda3: float = 0.1):
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.patience = patience
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.metrics_calc = MetricsCalculator()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.train_history = []
        
    def train_epoch(self) -> Dict:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probas = []
        all_binding_labels = []
        all_binding_probs = []
        all_binding_masks = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch in pbar:
            batch = self._move_batch_to_device(batch)
            
            (
                logits, seq_feats, seq_residue_feats, struct_feats,
                binding_logits, binding_probs
            ) = self.model(
                batch['atomic_graph'],
                batch['residue_graph'],
                batch['motif_graph'],
                batch['seq_embeddings'],
                batch['seq_lengths']
            )
            
            losses = self.model.compute_loss(
                logits, batch['labels'],
                seq_feats, struct_feats,
                binding_logits, batch['binding_labels'], batch['binding_mask'],
                self.lambda1, self.lambda2, self.lambda3
            )
            
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += losses['total'].item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(batch['labels'].detach().cpu().numpy())
            all_probas.extend(F.softmax(logits, dim=1).detach().cpu().numpy())
            
            if batch['binding_labels'] is not None and batch['binding_mask'] is not None:
                all_binding_labels.append(batch['binding_labels'].detach().cpu().numpy())
                all_binding_probs.append(binding_probs.squeeze(-1).detach().cpu().numpy())
                all_binding_masks.append(batch['binding_mask'].detach().cpu().numpy())
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'cls': f"{losses['cls'].item():.4f}",
                'bind': f"{losses['bind'].item():.4f}",
                'contrast': f"{losses['contrast'].item():.4f}"
            })
        
        epoch_metrics = self.metrics_calc.compute_all_metrics(
            np.asarray(all_labels), np.asarray(all_preds), np.asarray(all_probas)
        )
        epoch_metrics['loss'] = total_loss / max(len(self.train_loader), 1)
        
        if all_binding_labels:
            epoch_metrics.update(self.metrics_calc.compute_binding_metrics(
                np.concatenate(all_binding_labels, axis=0),
                np.concatenate(all_binding_probs, axis=0),
                np.concatenate(all_binding_masks, axis=0)
            ))
        
        return epoch_metrics
    
    def validate(self) -> Dict:
        """Validate protein classification and residue-level binding prediction."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probas = []
        all_binding_labels = []
        all_binding_probs = []
        all_binding_masks = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1} [Val]"):
                batch = self._move_batch_to_device(batch)
                
                (
                    logits, seq_feats, seq_residue_feats, struct_feats,
                    binding_logits, binding_probs
                ) = self.model(
                    batch['atomic_graph'],
                    batch['residue_graph'],
                    batch['motif_graph'],
                    batch['seq_embeddings'],
                    batch['seq_lengths']
                )
                
                losses = self.model.compute_loss(
                    logits, batch['labels'],
                    seq_feats, struct_feats,
                    binding_logits, batch['binding_labels'], batch['binding_mask'],
                    self.lambda1, self.lambda2, self.lambda3
                )
                
                total_loss += losses['total'].item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_probas.extend(F.softmax(logits, dim=1).cpu().numpy())
                
                if batch['binding_labels'] is not None and batch['binding_mask'] is not None:
                    all_binding_labels.append(batch['binding_labels'].cpu().numpy())
                    all_binding_probs.append(binding_probs.squeeze(-1).cpu().numpy())
                    all_binding_masks.append(batch['binding_mask'].cpu().numpy())
        
        val_metrics = self.metrics_calc.compute_all_metrics(
            np.asarray(all_labels), np.asarray(all_preds), np.asarray(all_probas)
        )
        val_metrics['loss'] = total_loss / max(len(self.val_loader), 1)
        
        if all_binding_labels:
            val_metrics.update(self.metrics_calc.compute_binding_metrics(
                np.concatenate(all_binding_labels, axis=0),
                np.concatenate(all_binding_probs, axis=0),
                np.concatenate(all_binding_masks, axis=0)
            ))
        
        return val_metrics
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        
        batch['labels'] = batch['labels'].to(self.device)
        batch['seq_embeddings'] = batch['seq_embeddings'].to(self.device)
        batch['seq_lengths'] = batch['seq_lengths'].to(self.device)
        
        if batch['binding_labels'] is not None:
            batch['binding_labels'] = batch['binding_labels'].to(self.device)
        if batch['binding_mask'] is not None:
            batch['binding_mask'] = batch['binding_mask'].to(self.device)
        
        if batch['atomic_graph'] is not None:
            batch['atomic_graph'] = batch['atomic_graph'].to(self.device)
        if batch['residue_graph'] is not None:
            batch['residue_graph'] = batch['residue_graph'].to(self.device)
        if batch['motif_graph'] is not None:
            batch['motif_graph'] = batch['motif_graph'].to(self.device)
        
        return batch
    
    def train(self, num_epochs: int = 100, save_best: bool = True):
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            
            val_metrics = self.validate()
            
            self.scheduler.step(val_metrics['loss'])
            
            self._log_epoch_results(train_metrics, val_metrics)
            
            self._save_checkpoint(val_metrics, save_best)
            
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.patience_counter = 0
                
                self._save_model('best_model.pt', val_metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered after epoch {epoch+1}")
                    break
        
        self._save_training_history()
        
        return self.train_history
    
    def _log_epoch_results(self, train_metrics: Dict, val_metrics: Dict):
        
        print(f"\n{'='*60}")
        print(f"Epoch {self.current_epoch+1} Summary")
        print(f"{'='*60}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['macro_f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['macro_f1']:.4f}")
        if 'binding_auroc' in val_metrics:
            print(
                f"Binding - AUROC: {val_metrics['binding_auroc']:.4f}, "
                f"AUPRC: {val_metrics['binding_auprc']:.4f}, "
                f"F1: {val_metrics['binding_f1']:.4f}, "
                f"MCC: {val_metrics['binding_mcc']:.4f}"
            )
        print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}, Patience: {self.patience_counter}/{self.patience}")
        
        self.train_history.append({
            'epoch': self.current_epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'lr': self.optimizer.param_groups[0]['lr']
        })
    
    def _save_checkpoint(self, val_metrics: Dict, save_best: bool = True):
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'train_history': self.train_history,
            'val_metrics': val_metrics
        }
        
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{self.current_epoch+1}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pt'))
        for old_checkpoint in checkpoints[:-5]:
            old_checkpoint.unlink()
    
    def _save_model(self, filename: str, metrics: Dict = None):
        
        model_path = self.output_dir / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'hidden_dim': self.model.hidden_dim,
                'num_classes': self.model.num_classes,
                'esm_dim': 2560,
                'dropout': 0.3
            },
            'metrics': metrics,
            'training_config': {
                'lambda1': self.lambda1,
                'lambda2': self.lambda2,
                'lambda3': self.lambda3,
                'best_val_f1': self.best_val_f1
            }
        }, model_path)
        
    
    def _save_training_history(self):
       
        history_path = self.output_dir / 'training_history.json'
        
        serializable_history = []
        for epoch_data in self.train_history:
            serializable_epoch = {
                'epoch': epoch_data['epoch'],
                'lr': epoch_data['lr'],
                'train': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                         for k, v in epoch_data['train'].items()},
                'val': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in epoch_data['val'].items()}
            }
            serializable_history.append(serializable_epoch)
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
 
def load_dataset(data_dir: str, split_type: str = 'cv', fold: int = 0):
    
    data_dir = Path(data_dir)
    
    manifest_path = data_dir / 'shard_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Shard manifest not found at {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        shard_manifest = json.load(f)
    
    splits_path = data_dir / 'cv_splits.json'
    if splits_path.exists():
        with open(splits_path, 'r') as f:
            cv_splits = json.load(f)
        
        if fold < len(cv_splits):
            split = cv_splits[fold]
            train_ids = split['train']
            val_ids = split['val']
        else:
            
            train_ids = cv_splits[-1]['train']
            val_ids = cv_splits[-1]['val']
    else:
        
        all_ids = []
        for shard in shard_manifest['shards']:
            all_ids.extend(shard['proteins'])
        
        from sklearn.model_selection import train_test_split
        train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)
    
    train_dataset = ShardedNABPDataset(
        data_dir, train_ids, shard_manifest, return_binding_labels=True
    )
    val_dataset = ShardedNABPDataset(
        data_dir, val_ids, shard_manifest, return_binding_labels=True
    )
    
    return train_dataset, val_dataset, shard_manifest


def main():
    parser = argparse.ArgumentParser(description='Train DeepNABind-MM')
    parser.add_argument('--config', type=str, default=None, help='Path to config JSON file')
    parser.add_argument('--data_dir', type=str, default='/homeb/ali/second/data/DeepNABindMM_dataset',
                        help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for models and logs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--fold', type=int, default=0, help='CV fold to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    
    args = parser.parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
    
    set_seed(args.seed)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"run_{timestamp}_fold{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    train_dataset, val_dataset, shard_manifest = load_dataset(
        args.data_dir, split_type='cv', fold=args.fold
    )
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    model = DeepNABindMM(
        hidden_dim=args.hidden_dim,
        num_classes=3,
        esm_dim=2560,
        dropout=0.3,
        temperature=0.07
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        learning_rate=args.lr,
        weight_decay=1e-2,
        device=args.device,
        patience=15,
        lambda1=1.0,
        lambda2=0.3,
        lambda3=0.1
    )
    
    history = trainer.train(num_epochs=args.epochs)
    
    final_metrics = trainer.validate()
    trainer.metrics_calc.print_metrics(final_metrics, "Final Validation Results")
    
    with open(output_dir / 'final_metrics.json', 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()