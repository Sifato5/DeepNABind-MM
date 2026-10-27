import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             matthews_corrcoef, roc_auc_score)
from model import DeepNABindMM
from train import ShardedNABPDataset, collate_fn, Trainer, set_seed

NAMES = ["non-NABP", "RBP", "DBP"]

def aligned_collate_fn(items):
    batch = collate_fn(items)
    for key in ("atomic_graph", "residue_graph", "motif_graph"):
        graph = batch[key]
        if graph is None:
            if key == "residue_graph":
                raise ValueError("All proteins must have a residue graph")
            continue
        indices = [i for i, item in enumerate(items) if item[key] is not None]
        graph.original_graph_indices = torch.tensor(indices, dtype=torch.long)
        if key == "residue_graph" and len(indices) != len(items):
            raise ValueError("Residue graph missing for a protein")
    return batch


def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [safe(v) for v in x]
    if isinstance(x, np.ndarray): return safe(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    return x

def read_dataset_index(root):
    with open(root / "shard_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    ids = list(dict.fromkeys(pid for shard in manifest["shards"] for pid in shard["proteins"]))
    wanted, labels = set(ids), {}
    for shard in manifest["shards"]:
        pids = [p for p in shard["proteins"] if p in wanted]
        if not pids: continue
        path = root / "graph_shards" / shard["file"]
        try: data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError: data = torch.load(path, map_location="cpu")
        for pid in pids:
            value = data[pid]["label"]
            labels[pid] = int(value.item()) if torch.is_tensor(value) else int(value)
    missing = [p for p in ids if p not in labels]
    if missing: raise ValueError(f"Missing class labels for proteins: {missing[:5]}")
    y = np.array([labels[p] for p in ids], dtype=np.int64)
    if not set(np.unique(y)).issubset({0, 1, 2}):
        raise ValueError(f"Expected class labels 0,1,2; found {np.unique(y).tolist()}")
    return manifest, ids, y

def cls_metrics(y, pred, prob):
    p, r, f, _ = precision_recall_fscore_support(
        y, pred, labels=[0,1,2], average="macro", zero_division=0)
    try: auc = roc_auc_score(y, prob, labels=[0,1,2], multi_class="ovr", average="macro")
    except ValueError: auc = np.nan
    return {"classification_ACC": float(accuracy_score(y,pred)),
            "classification_PRE_macro": float(p), "classification_REC_macro": float(r),
            "classification_F1_macro": float(f), "classification_MCC": float(matthews_corrcoef(y,pred)),
            "classification_AUROC_macro_ovr": float(auc)}

def bind_metrics(y, score):
    y, score = np.asarray(y, dtype=np.int64), np.asarray(score, dtype=float)
    pred = (score >= 0.5).astype(np.int64)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try: auc = float(roc_auc_score(y, score)) if np.unique(y).size == 2 else np.nan
    except ValueError: auc = np.nan
    return {"binding_PRE": float(p), "binding_REC": float(r), "binding_F1": float(f),
            "binding_MCC": float(matthews_corrcoef(y,pred)), "binding_AUROC": float(auc),
            "binding_threshold": 0.5, "binding_valid_residues": int(y.size),
            "binding_positive_fraction": float(np.mean(y)) if y.size else np.nan}

class CVTrainer(Trainer):
    def _epoch(self, training):
        self.model.train(training)
        loader = self.train_loader if training else self.val_loader
        total_loss, n_batches = 0.0, 0
        ys, preds, probs, bind_y, bind_p = [], [], [], [], []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                batch = self._move_batch_to_device(batch)
                if training: self.optimizer.zero_grad(set_to_none=True)
                out = self.model(batch["atomic_graph"], batch["residue_graph"],
                                batch["motif_graph"], batch["seq_embeddings"], batch["seq_lengths"])
                logits, seq_feats, _, struct_feats, bind_logits, bind_probs = out
                losses = self.model.compute_loss(
                    logits, batch["labels"], seq_feats, struct_feats, bind_logits,
                    batch["binding_labels"], batch["binding_mask"],
                    self.lambda1, self.lambda2, self.lambda3)
                if training:
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss += float(losses["total"].detach().cpu()); n_batches += 1
                ys.extend(batch["labels"].detach().cpu().numpy())
                preds.extend(torch.argmax(logits, 1).detach().cpu().numpy())
                probs.extend(F.softmax(logits, 1).detach().cpu().numpy())
                if batch["binding_labels"] is not None and batch["binding_mask"] is not None:
                    mask = batch["binding_mask"].bool()
                    bind_y.append(batch["binding_labels"][mask].detach().cpu().numpy())
                    bind_p.append(bind_probs.squeeze(-1)[mask].detach().cpu().numpy())
        metrics = cls_metrics(np.asarray(ys), np.asarray(preds), np.asarray(probs))
        
        metrics["accuracy"] = metrics["classification_ACC"]
        metrics["macro_f1"] = metrics["classification_F1_macro"]
        metrics["loss"] = total_loss / max(n_batches, 1)
        if bind_y:
            flat_y, flat_p = np.concatenate(bind_y), np.concatenate(bind_p)
            metrics.update(bind_metrics(flat_y, flat_p))
            
            from sklearn.metrics import average_precision_score
            try:
                metrics["binding_auprc"] = float(average_precision_score(flat_y, flat_p))
            except ValueError:
                metrics["binding_auprc"] = float("nan")
        return metrics
    def train_epoch(self): return self._epoch(True)
    def validate(self): return self._epoch(False)

def summarize(rows):
    keys = ["classification_ACC","classification_PRE_macro","classification_REC_macro",
            "classification_F1_macro","classification_MCC","classification_AUROC_macro_ovr",
            "binding_PRE","binding_REC","binding_F1","binding_MCC","binding_AUROC"]
    keys.append("binding_auprc")
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r and r[k] is not None and np.isfinite(r[k])], dtype=float)
        out[k] = {"mean": float(vals.mean()) if len(vals) else None,
                  "n_valid_folds": int(len(vals))}
    return out

def main():
    ap = argparse.ArgumentParser(description="Stratified K-fold CV for DeepNABind-MM.")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--output_dir", default="./results/cross_validation")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lambda_cls", type=float, default=1.0)
    ap.add_argument("--lambda_bind", type=float, default=0.3)
    ap.add_argument("--lambda_contrast", type=float, default=0.1)
    a = ap.parse_args()
    if a.folds < 2: ap.error("--folds must be >= 2")
    root, outdir = Path(a.data_dir).expanduser().resolve(), Path(a.output_dir).expanduser().resolve()
    if (outdir / "cross_validation_summary.json").exists():
        raise FileExistsError(f"Existing CV summary in {outdir}; choose a fresh --output_dir")
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)
    manifest, ids, y = read_dataset_index(root)
    counts = np.bincount(y, minlength=3)
    if counts.min() < a.folds:
        raise ValueError(f"Each class needs >= {a.folds} proteins. Counts (0,1,2): {counts.tolist()}")
    print(f"Proteins: {len(ids)}; class counts (non-NABP,RBP,DBP): {counts.tolist()}")
    splitter = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)
    rows = []
    for fold, (itr, iva) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        print(f"\\n{'='*60}\\nFOLD {fold}/{a.folds}\\n{'='*60}")
        set_seed(a.seed + fold - 1)
        fold_dir = outdir / f"fold_{fold}"
        if (fold_dir / "best_model.pt").exists():
            raise FileExistsError(f"Existing checkpoint in {fold_dir}; choose a fresh --output_dir")
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "split_ids.json", "w", encoding="utf-8") as f:
            json.dump({"train": [ids[i] for i in itr], "validation": [ids[i] for i in iva]}, f, indent=2)
        train_ds = ShardedNABPDataset(str(root), [ids[i] for i in itr], manifest, return_binding_labels=True)
        val_ds = ShardedNABPDataset(str(root), [ids[i] for i in iva], manifest, return_binding_labels=True)
        train_dl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, num_workers=0,
                              pin_memory=a.device.startswith("cuda"), collate_fn=aligned_collate_fn)
        val_dl = DataLoader(val_ds, batch_size=a.batch_size, shuffle=False, num_workers=0,
                            pin_memory=a.device.startswith("cuda"), collate_fn=aligned_collate_fn)
        model = DeepNABindMM(hidden_dim=a.hidden_dim, num_classes=3, esm_dim=2560,
                             dropout=0.3, temperature=0.07)
        trainer = CVTrainer(model, train_dl, val_dl, str(fold_dir), learning_rate=a.lr,
                            weight_decay=1e-2, device=a.device, patience=a.patience,
                            lambda1=a.lambda_cls, lambda2=a.lambda_bind, lambda3=a.lambda_contrast)
        trainer.best_val_f1 = float("-inf")
        trainer.train(num_epochs=a.epochs)
        best = fold_dir / "best_model.pt"
        if not best.exists():
            raise FileNotFoundError(f"No best checkpoint saved for fold {fold}: {best}")
        try: ckpt = torch.load(best, map_location=a.device, weights_only=False)
        except TypeError: ckpt = torch.load(best, map_location=a.device)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        metrics = trainer.validate()
        metrics.update({"fold": fold, "n_train_proteins": len(itr), "n_validation_proteins": len(iva)})
        rows.append(metrics)
        with open(fold_dir / "validation_metrics.json", "w", encoding="utf-8") as f:
            json.dump(safe(metrics), f, indent=2)
        for k,v in metrics.items():
            if isinstance(v, (int,float,np.integer,np.floating)):
                print(f"{k}: {float(v):.4f}" if np.isfinite(float(v)) else f"{k}: undefined")
    summary = {"n_folds": a.folds, "seed": a.seed,
               "class_counts": {NAMES[i]: int(counts[i]) for i in range(3)},
               "fold_results": rows, "mean_metrics": summarize(rows),
               "notes": {"classification_PRE_REC_F1": "macro averages",
                         "classification_AUROC": "macro one-vs-rest",
                         "binding_threshold": 0.5,
                         "binding_padding": "excluded using binding_mask",
                         "folding": "StratifiedKFold at protein level",
                         "selection": "early stopping uses the same validation fold used for reported metrics (selection optimism possible)"}}
    with open(outdir / "cross_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(safe(summary), f, indent=2)
    pd.DataFrame(rows).to_csv(outdir / "fold_metrics.csv", index=False)
    pd.DataFrame([{"metric":k, **v} for k,v in summary["mean_metrics"].items()]).to_csv(
        outdir / "mean_metrics.csv", index=False)
    print("\\nFIVE-FOLD SUMMARY (mean across folds)")
    for k,v in summary["mean_metrics"].items():
        print(f"{k:<34} " + ("undefined" if v["mean"] is None else f'{v["mean"]:.4f}'))
    print(f"\\nSaved results to {outdir}")

if __name__ == "__main__":
    main()