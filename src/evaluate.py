import torch
import torch.nn.functional as F
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix
)
import argparse
import sys

sys.path.append('/homeb/ali/second')
from model import DeepNABindMM


class TestSetEvaluator:
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = device
        
        checkpoint = torch.load(model_path, map_location=device)
        
        model_config = checkpoint.get('model_config', {})
        self.model = DeepNABindMM(
            hidden_dim=model_config.get('hidden_dim', 512),
            num_classes=3,
            esm_dim=2560,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(device)
        self.model.eval()
        
        print(f" Model loaded from {model_path}")
    
    def evaluate_test_set(self, processed_test_dir: Path) -> dict:
        """Evaluate on a processed test set"""
       
        metadata_file = processed_test_dir / 'test_set_metadata.json'
        if not metadata_file.exists():
            print(f" Metadata not found: {metadata_file}")
            return None
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        graphs_dir = processed_test_dir / 'graphs'
        
        predictions = []
        
        for protein_info in tqdm(metadata['proteins'], desc="Inference"):
            if not protein_info['has_graph']:
                continue
            
            protein_id = protein_info['uniprot_id']
            true_label = protein_info['label']
            
            graph_file = graphs_dir / f"{protein_id}.pt"
            if not graph_file.exists():
                continue
            
            graph_data = torch.load(graph_file, map_location=self.device)
            
            with torch.no_grad():
                logits = self._run_inference(graph_data)
                probas = F.softmax(logits, dim=0).cpu().numpy()
                pred_label = torch.argmax(logits).item()
            
            predictions.append({
                'protein_id': protein_id,
                'true_label': true_label,
                'pred_label': pred_label,
                'prob_nonNABP': float(probas[0]),
                'prob_RBP': float(probas[1]),
                'prob_DBP': float(probas[2])
            })
        
        if not predictions:
            print("   No valid predictions")
            return None
        
        y_true = [p['true_label'] for p in predictions]
        y_pred = [p['pred_label'] for p in predictions]
        y_proba = [[p['prob_nonNABP'], p['prob_RBP'], p['prob_DBP']] for p in predictions]
        
        metrics = self._compute_metrics(y_true, y_pred, y_proba)
        
        df = pd.DataFrame(predictions)
        df.to_csv(processed_test_dir / 'predictions.csv', index=False)
        
        with open(processed_test_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        self._print_metrics(metrics, metadata['test_set_name'])
        
        return metrics
    
    def _run_inference(self, graph_data: dict) -> torch.Tensor:
        
        atomic_graph = graph_data.get('atomic')
        residue_graph = graph_data.get('residue')
        motif_graph = graph_data.get('motif')
        seq_embedding = residue_graph.x.unsqueeze(0) 
        seq_length = torch.tensor([len(seq_embedding[0])])
        
        if atomic_graph is not None:
            atomic_graph.batch = torch.zeros(len(atomic_graph.x), dtype=torch.long, device=self.device)
        
        residue_graph.batch = torch.zeros(len(residue_graph.x), dtype=torch.long, device=self.device)
        
        if motif_graph is not None:
            motif_graph.batch = torch.zeros(len(motif_graph.x), dtype=torch.long, device=self.device)
        
        seq_embedding = seq_embedding.to(self.device)
        seq_length = seq_length.to(self.device)
        residue_graph = residue_graph.to(self.device)
        
        if atomic_graph is not None:
            atomic_graph = atomic_graph.to(self.device)
        if motif_graph is not None:
            motif_graph = motif_graph.to(self.device)
        
        logits, _, _, _, _, _ = self.model(
            atomic_graph, residue_graph, motif_graph, seq_embedding, seq_length
        )
        
        return logits[0]  
    
    def _compute_metrics(self, y_true, y_pred, y_proba):
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=[0, 1, 2]
        )
        
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'per_class': {
                'non-NABP': {'precision': precision[0], 'recall': recall[0], 'f1': f1[0], 'support': int(support[0])},
                'RBP': {'precision': precision[1], 'recall': recall[1], 'f1': f1[1], 'support': int(support[1])},
                'DBP': {'precision': precision[2], 'recall': recall[2], 'f1': f1[2], 'support': int(support[2])}
            }
        }
        
        try:
            y_true_onehot = np.eye(3)[y_true]
            metrics['auc_ovr'] = roc_auc_score(y_true_onehot, y_proba, multi_class='ovr', average='macro')
            metrics['auprc_ovr'] = average_precision_score(y_true_onehot, y_proba, average='macro')
            
            metrics['per_class_auc'] = {}
            metrics['per_class_auprc'] = {}
            for i, name in enumerate(['non-NABP', 'RBP', 'DBP']):
                metrics['per_class_auc'][name] = roc_auc_score(y_true_onehot[:, i], [p[i] for p in y_proba])
                metrics['per_class_auprc'][name] = average_precision_score(y_true_onehot[:, i], [p[i] for p in y_proba])
        except:
            metrics['auc_ovr'] = 0.0
            metrics['auprc_ovr'] = 0.0
        
        return metrics
    
    def _print_metrics(self, metrics: dict, dataset_name: str):
        
        print(f"\n{dataset_name} Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  MCC:       {metrics['mcc']:.4f}")
        print(f"  AUC (OVR): {metrics['auc_ovr']:.4f}")
        print(f"  AUPRC (OVR): {metrics['auprc_ovr']:.4f}")
        
        print(f"\n  Per-Class:")
        for class_name, class_metrics in metrics['per_class'].items():
            auc = metrics['per_class_auc'].get(class_name, 0)
            print(f"    {class_name}: P={class_metrics['precision']:.3f}, R={class_metrics['recall']:.3f}, "
                  f"F1={class_metrics['f1']:.3f}, AUC={auc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_data_dir', type=str,
                       default='/homeb/ali/second/data/DeepNABindMM_dataset/test_processed',
                       help='Directory containing processed test sets')
    parser.add_argument('--test_sets', type=str, nargs='+',
                       default=['TEST474', 'Human', 'A_thaliana', 'S_cerevisiae'],
                       help='Test sets to evaluate')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    evaluator = TestSetEvaluator(args.model_path, args.device)
    
    results = {}
    for test_set in args.test_sets:
        test_path = Path(args.test_data_dir) / test_set
        if test_path.exists():
            metrics = evaluator.evaluate_test_set(test_path)
            if metrics:
                results[test_set] = metrics
    
    if results:
        print("\n" + "="*80)
        print("SUMMARY OF ALL TEST SETS")
        print("="*80)
        print(f"{'Test Set':<20} {'Accuracy':<10} {'Macro F1':<10} {'MCC':<10} {'AUC':<10}")
        print("-" * 60)
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['macro_f1']:<10.4f} "
                  f"{metrics['mcc']:<10.4f} {metrics['auc_ovr']:<10.4f}")
        
        summary_file = Path(args.model_path).parent / 'test_set_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        print(f"\n Summary saved to {summary_file}")


if __name__ == "__main__":
    main()