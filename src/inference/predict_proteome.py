import os
import sys
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("DeepNABind-MM")

CLASS_NAMES = {
    0: "non-NABP",
    1: "RBP",
    2: "DBP",
}

def set_seed(seed=42):
    """Set deterministic random seeds."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fasta(fasta_file):

    sequences = {}

    for record in SeqIO.parse(fasta_file, "fasta"):
        protein_id = record.id.split()[0]
        sequence = str(record.seq).upper()

        sequences[protein_id] = sequence

    logger.info(
        "Loaded %d protein sequences from %s",
        len(sequences),
        fasta_file
    )

    return sequences


def load_checkpoint(model, checkpoint_path, device):
    """
    Load DeepNABind-MM checkpoint.

    Supports checkpoints saved as:
        torch.save(model.state_dict(), ...)
    or
        torch.save({"model_state_dict": ...}, ...)
    """

    logger.info("Loading checkpoint: %s", checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model" in checkpoint and isinstance(
            checkpoint["model"], dict
        ):
            state_dict = checkpoint["model"]

        else:
            
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unsupported checkpoint format. "
            "Expected a PyTorch state_dict or checkpoint dictionary."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False
    )

    if missing:
        logger.warning(
            "Missing model parameters: %d",
            len(missing)
        )
        for name in missing[:20]:
            logger.warning("Missing: %s", name)

    if unexpected:
        logger.warning(
            "Unexpected checkpoint parameters: %d",
            len(unexpected)
        )
        for name in unexpected[:20]:
            logger.warning("Unexpected: %s", name)

    model.to(device)
    model.eval()

    logger.info("Checkpoint loaded successfully.")

    return model


def move_to_device(obj, device):
    """
    Recursively move tensors to device.
    """

    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {
            key: move_to_device(value, device)
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_device(value, device)
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_device(value, device)
            for value in obj
        )

    return obj


def find_sample_file(sample_dir, protein_id):
    """
    Find a preprocessed sample for a protein.

    Supported formats:
        protein_id.pt
        protein_id.pth
        protein_id.pt.gz is not directly supported
    """

    sample_dir = Path(sample_dir)

    candidates = [
        sample_dir / f"{protein_id}.pt",
        sample_dir / f"{protein_id}.pth",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_protein_sample(sample_path, device):

    sample = torch.load(
        sample_path,
        map_location="cpu",
        weights_only=False
    )

    sample = move_to_device(sample, device)

    return sample

def run_model(model, sample):

    if isinstance(sample, dict):

        model_inputs = {}

        possible_inputs = [
            "seq_embedding",
            "sequence_embedding",
            "embedding",
            "esm_embedding",
            "atomic_graph",
            "atom_graph",
            "residue_graph",
            "motif_graph",
            "sse_graph",
            "structure_graph",
        ]

        for key in possible_inputs:
            if key in sample:
                model_inputs[key] = sample[key]

        try:
            output = model(**model_inputs)

        except TypeError:

            seq_embedding = (
                sample.get("seq_embedding")
                or sample.get("sequence_embedding")
                or sample.get("embedding")
                or sample.get("esm_embedding")
            )

            atomic_graph = (
                sample.get("atomic_graph")
                or sample.get("atom_graph")
            )

            residue_graph = sample.get("residue_graph")

            motif_graph = (
                sample.get("motif_graph")
                or sample.get("sse_graph")
            )

            output = model(
                seq_embedding,
                atomic_graph,
                residue_graph,
                motif_graph
            )

    else:
        output = model(sample)

    return output


def unpack_model_output(output):

    logits = None
    binding_probs = None

    if isinstance(output, dict):

        logits = (
            output.get("logits")
            or output.get("classification_logits")
            or output.get("class_logits")
        )

        binding_probs = (
            output.get("binding_probs")
            or output.get("binding_probabilities")
        )

        if binding_probs is None and output.get("binding_logits") is not None:
            binding_probs = torch.sigmoid(
                output["binding_logits"]
            )

    elif isinstance(output, (tuple, list)):

        if len(output) >= 1:
            logits = output[0]

        if len(output) >= 5:
            binding_probs = output[4]

            if binding_probs is None:
                binding_probs = torch.sigmoid(output[3])

        elif len(output) >= 4:
            binding_probs = output[3]

    else:
        logits = output

    if logits is None:
        raise RuntimeError(
            "Could not identify classification logits "
            "from model output."
        )

    return logits, binding_probs

@torch.no_grad()
def predict_protein(model, sample, device):

    sample = move_to_device(sample, device)

    output = run_model(model, sample)

    logits, binding_probs = unpack_model_output(output)

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    class_probs = torch.softmax(logits, dim=-1)

    predicted_class = torch.argmax(
        class_probs,
        dim=-1
    ).item()

    probabilities = class_probs[0].detach().cpu().numpy()

    if binding_probs is not None:

        if binding_probs.ndim == 3:
            binding_probs = binding_probs.squeeze(-1)

        if binding_probs.ndim == 2:
            binding_probs = binding_probs[0]

        binding_probs = (
            binding_probs
            .detach()
            .cpu()
            .numpy()
        )

    return {
        "predicted_class": predicted_class,
        "class_probs": probabilities,
        "binding_probs": binding_probs,
    }


def save_binding_prediction(
    protein_id,
    sequence,
    binding_probs,
    output_dir,
    threshold=0.5
):

    if binding_probs is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    L = min(
        len(sequence),
        len(binding_probs)
    )

    positions = np.arange(1, L + 1)

    labels = (
        binding_probs[:L] >= threshold
    ).astype(int)

    residue_table = pd.DataFrame({
        "protein_id": [protein_id] * L,
        "position": positions,
        "residue": list(sequence[:L]),
        "binding_probability": binding_probs[:L],
        "predicted_binding": labels,
    })

    csv_file = (
        output_dir /
        f"{protein_id}_binding_sites.csv"
    )

    residue_table.to_csv(
        csv_file,
        index=False
    )

    npz_file = (
        output_dir /
        f"{protein_id}_binding_sites.npz"
    )

    np.savez_compressed(
        npz_file,
        sequence=np.array(list(sequence[:L])),
        binding_probability=binding_probs[:L],
        predicted_binding=labels,
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Proteome-wide inference using DeepNABind-MM"
        )
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Protein FASTA file."
    )

    parser.add_argument(
        "--sample_dir",
        required=True,
        help=(
            "Directory containing preprocessed "
            "DeepNABind-MM protein samples."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Trained DeepNABind-MM checkpoint."
    )

    parser.add_argument(
        "--output_dir",
        default="proteome_predictions",
        help="Output directory."
    )

    parser.add_argument(
        "--binding_threshold",
        type=float,
        default=0.5,
        help="Residue binding probability threshold."
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but unavailable. Using CPU."
        )
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    logger.info("Using device: %s", device)

    set_seed(args.seed)

    try:
        from model import DeepNABindMM
    except ImportError as exc:
        raise ImportError(
            "Could not import DeepNABindMM from model.py. "
            "Run this script from the project root or add src/ "
            "to PYTHONPATH."
        ) from exc

    model = DeepNABindMM()

    model = load_checkpoint(
        model,
        args.checkpoint,
        device
    )

    sequences = load_fasta(args.fasta)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    binding_dir = output_dir / "binding_sites"

    results = []

    total = len(sequences)

    for index, (protein_id, sequence) in enumerate(
        sequences.items(),
        start=1
    ):

        logger.info(
            "[%d/%d] Predicting %s",
            index,
            total,
            protein_id
        )

        sample_file = find_sample_file(
            args.sample_dir,
            protein_id
        )

        if sample_file is None:

            logger.warning(
                "Preprocessed sample not found for %s. "
                "Skipping.",
                protein_id
            )

            continue

        try:

            sample = load_protein_sample(
                sample_file,
                device
            )

            prediction = predict_protein(
                model,
                sample,
                device
            )

            predicted_class = prediction[
                "predicted_class"
            ]

            probs = prediction[
                "class_probs"
            ]

            binding_probs = prediction[
                "binding_probs"
            ]

            row = {
                "protein_id": protein_id,
                "length": len(sequence),
                "predicted_class": predicted_class,
                "predicted_label": CLASS_NAMES.get(
                    predicted_class,
                    str(predicted_class)
                ),
                "non_NABP_probability": (
                    float(probs[0])
                    if len(probs) > 0 else np.nan
                ),
                "RBP_probability": (
                    float(probs[1])
                    if len(probs) > 1 else np.nan
                ),
                "DBP_probability": (
                    float(probs[2])
                    if len(probs) > 2 else np.nan
                ),
            }

            results.append(row)

            save_binding_prediction(
                protein_id=protein_id,
                sequence=sequence,
                binding_probs=binding_probs,
                output_dir=binding_dir,
                threshold=args.binding_threshold
            )

        except Exception as exc:

            logger.exception(
                "Prediction failed for %s: %s",
                protein_id,
                exc
            )

    results_df = pd.DataFrame(results)

    results_file = (
        output_dir /
        "proteome_predictions.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    summary = {
        "input_fasta": str(args.fasta),
        "checkpoint": str(args.checkpoint),
        "number_of_input_proteins": len(sequences),
        "number_of_predictions": len(results_df),
        "class_distribution": (
            results_df["predicted_label"]
            .value_counts()
            .to_dict()
            if not results_df.empty
            else {}
        ),
        "binding_threshold": args.binding_threshold,
    }

    with open(
        output_dir / "prediction_summary.json",
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    logger.info(
        "Proteome prediction completed."
    )

    logger.info(
        "Protein-level results: %s",
        results_file
    )

    logger.info(
        "Binding-site results: %s",
        binding_dir
    )


if __name__ == "__main__":
    main()