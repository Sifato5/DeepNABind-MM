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

    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_fasta(fasta_file):

    sequences = {}

    for record in SeqIO.parse(
        fasta_file,
        "fasta"
    ):

        protein_id = record.id.split()[0]

        sequences[protein_id] = (
            str(record.seq).upper()
        )

    logger.info(
        "Loaded %d sequences.",
        len(sequences)
    )

    return sequences

def load_mutations(mutation_file):

    mutations = pd.read_csv(
        mutation_file
    )

    required_columns = {
        "protein_id",
        "position",
        "wild_type",
        "mutant",
    }

    missing = (
        required_columns -
        set(mutations.columns)
    )

    if missing:
        raise ValueError(
            "Mutation file is missing columns: "
            + ", ".join(sorted(missing))
        )

    mutations["position"] = (
        mutations["position"]
        .astype(int)
    )

    mutations["wild_type"] = (
        mutations["wild_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    mutations["mutant"] = (
        mutations["mutant"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return mutations


def apply_mutation(
    sequence,
    position,
    wild_type,
    mutant
):

    if position < 1 or position > len(sequence):
        raise ValueError(
            f"Position {position} is outside "
            f"sequence length {len(sequence)}."
        )

    index = position - 1

    observed_residue = sequence[index]

    if observed_residue != wild_type:

        raise ValueError(
            f"Wild-type mismatch at position {position}: "
            f"FASTA has '{observed_residue}', "
            f"but mutation file specifies '{wild_type}'."
        )

    if mutant == observed_residue:
        raise ValueError(
            f"Mutation {wild_type}{position}{mutant} "
            "does not change the residue."
        )

    mutant_sequence = (
        sequence[:index]
        + mutant
        + sequence[index + 1:]
    )

    return mutant_sequence

def load_checkpoint(
    model,
    checkpoint_path,
    device
):

    logger.info(
        "Loading checkpoint: %s",
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint[
                "state_dict"
            ]

        elif (
            "model" in checkpoint
            and isinstance(
                checkpoint["model"],
                dict
            )
        ):
            state_dict = checkpoint["model"]

        else:
            state_dict = checkpoint

    else:
        raise ValueError(
            "Unsupported checkpoint format."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned_state_dict[key] = value

    missing, unexpected = (
        model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )
    )

    if missing:
        logger.warning(
            "Missing parameters: %d",
            len(missing)
        )

    if unexpected:
        logger.warning(
            "Unexpected parameters: %d",
            len(unexpected)
        )

    model.to(device)
    model.eval()

    return model

def move_to_device(
    obj,
    device
):

    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {
            key: move_to_device(
                value,
                device
            )
            for key, value in obj.items()
        }

    if isinstance(obj, list):
        return [
            move_to_device(
                value,
                device
            )
            for value in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            move_to_device(
                value,
                device
            )
            for value in obj
        )

    return obj


def load_sample(
    sample_file,
    device
):

    sample = torch.load(
        sample_file,
        map_location="cpu",
        weights_only=False
    )

    return move_to_device(
        sample,
        device
    )

def run_model(
    model,
    sample
):

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

            output = model(
                **model_inputs
            )

        except TypeError:

            seq_embedding = (
                sample.get(
                    "seq_embedding"
                )
                or sample.get(
                    "sequence_embedding"
                )
                or sample.get(
                    "embedding"
                )
                or sample.get(
                    "esm_embedding"
                )
            )

            atomic_graph = (
                sample.get(
                    "atomic_graph"
                )
                or sample.get(
                    "atom_graph"
                )
            )

            residue_graph = (
                sample.get(
                    "residue_graph"
                )
            )

            motif_graph = (
                sample.get(
                    "motif_graph"
                )
                or sample.get(
                    "sse_graph"
                )
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


def unpack_output(
    output
):

    logits = None
    binding_probs = None

    if isinstance(output, dict):

        logits = (
            output.get("logits")
            or output.get(
                "classification_logits"
            )
            or output.get(
                "class_logits"
            )
        )

        binding_probs = (
            output.get(
                "binding_probs"
            )
            or output.get(
                "binding_probabilities"
            )
        )

        if (
            binding_probs is None
            and output.get(
                "binding_logits"
            ) is not None
        ):

            binding_probs = torch.sigmoid(
                output["binding_logits"]
            )

    elif isinstance(output, (tuple, list)):

        if len(output) >= 1:
            logits = output[0]

        if len(output) >= 5:

            binding_probs = output[4]

            if binding_probs is None:
                binding_probs = torch.sigmoid(
                    output[3]
                )

        elif len(output) >= 4:

            binding_probs = output[3]

    else:

        logits = output

    return logits, binding_probs

@torch.no_grad()
def predict(
    model,
    sample,
    device
):

    sample = move_to_device(
        sample,
        device
    )

    output = run_model(
        model,
        sample
    )

    logits, binding_probs = (
        unpack_output(output)
    )

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    class_probs = torch.softmax(
        logits,
        dim=-1
    )[0]

    predicted_class = torch.argmax(
        class_probs
    ).item()

    class_probs = (
        class_probs
        .detach()
        .cpu()
        .numpy()
    )

    if binding_probs is not None:

        if binding_probs.ndim == 3:
            binding_probs = (
                binding_probs.squeeze(-1)
            )

        if binding_probs.ndim == 2:
            binding_probs = (
                binding_probs[0]
            )

        binding_probs = (
            binding_probs
            .detach()
            .cpu()
            .numpy()
        )

    return {
        "predicted_class": predicted_class,
        "class_probs": class_probs,
        "binding_probs": binding_probs,
    }


def find_sample(
    sample_dir,
    protein_id
):

    sample_dir = Path(
        sample_dir
    )

    for extension in [
        ".pt",
        ".pth",
    ]:

        path = (
            sample_dir /
            f"{protein_id}{extension}"
        )

        if path.exists():
            return path

    return None

def main():

    parser = argparse.ArgumentParser(
        description=(
            "WT vs mutant prediction "
            "using DeepNABind-MM."
        )
    )

    parser.add_argument(
        "--fasta",
        required=True,
        help="Wild-type protein FASTA."
    )

    parser.add_argument(
        "--mutations",
        required=True,
        help=(
            "CSV file containing "
            "protein_id,position,wild_type,mutant."
        )
    )

    parser.add_argument(
        "--sample_dir",
        required=True,
        help=(
            "Directory containing "
            "preprocessed samples."
        )
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="DeepNABind-MM checkpoint."
    )

    parser.add_argument(
        "--output_dir",
        default="mutation_predictions",
        help="Output directory."
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="cuda or cpu."
    )

    parser.add_argument(
        "--binding_threshold",
        type=float,
        default=0.5,
        help="Binding probability threshold."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):

        logger.warning(
            "CUDA unavailable. Using CPU."
        )

        device = torch.device("cpu")

    else:

        device = torch.device(
            args.device
        )

    logger.info(
        "Using device: %s",
        device
    )

    set_seed(args.seed)

    try:

        from model import DeepNABindMM

    except ImportError as exc:

        raise ImportError(
            "Could not import DeepNABindMM "
            "from model.py."
        ) from exc

    model = DeepNABindMM()

    model = load_checkpoint(
        model,
        args.checkpoint,
        device
    )

    sequences = load_fasta(
        args.fasta
    )

    mutations = load_mutations(
        args.mutations
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = []

    for mutation_index, mutation in (
        mutations.iterrows()
    ):

        protein_id = str(
            mutation["protein_id"]
        )

        position = int(
            mutation["position"]
        )

        wild_type = str(
            mutation["wild_type"]
        ).upper()

        mutant = str(
            mutation["mutant"]
        ).upper()

        logger.info(
            "[%d/%d] %s %s%d%s",
            mutation_index + 1,
            len(mutations),
            protein_id,
            wild_type,
            position,
            mutant
        )

        if protein_id not in sequences:

            logger.warning(
                "Protein %s not found in FASTA. "
                "Skipping.",
                protein_id
            )

            continue

        wt_sequence = sequences[
            protein_id
        ]

        try:

            mutant_sequence = apply_mutation(
                wt_sequence,
                position,
                wild_type,
                mutant
            )

        except ValueError as exc:

            logger.warning(
                "Skipping %s: %s",
                protein_id,
                exc
            )

            continue

        wt_sample_file = find_sample(
            args.sample_dir,
            protein_id
        )

        if wt_sample_file is None:

            logger.warning(
                "WT sample not found for %s.",
                protein_id
            )

            continue

        mutation_name = (
            f"{protein_id}_"
            f"{wild_type}{position}{mutant}"
        )

        mutant_sample_file = find_sample(
            args.sample_dir,
            mutation_name
        )

        if mutant_sample_file is None:

            logger.warning(
                "Mutant sample not found: %s.pt. "
                "Generate the mutant ESM-2 embedding "
                "and mutant structural graph first.",
                mutation_name
            )

            continue

        try:

            wt_sample = load_sample(
                wt_sample_file,
                device
            )

            mutant_sample = load_sample(
                mutant_sample_file,
                device
            )

            wt_prediction = predict(
                model,
                wt_sample,
                device
            )

            mutant_prediction = predict(
                model,
                mutant_sample,
                device
            )

        except Exception as exc:

            logger.exception(
                "Prediction failed for %s: %s",
                mutation_name,
                exc
            )

            continue

        wt_probs = (
            wt_prediction["class_probs"]
        )

        mutant_probs = (
            mutant_prediction["class_probs"]
        )

        wt_binding = (
            wt_prediction["binding_probs"]
        )

        mutant_binding = (
            mutant_prediction["binding_probs"]
        )

        wt_binding_at_site = np.nan
        mutant_binding_at_site = np.nan
        delta_binding = np.nan

        if wt_binding is not None:

            if position <= len(wt_binding):

                wt_binding_at_site = float(
                    wt_binding[position - 1]
                )

        if mutant_binding is not None:

            if position <= len(mutant_binding):

                mutant_binding_at_site = float(
                    mutant_binding[position - 1]
                )

        if (
            not np.isnan(
                wt_binding_at_site
            )
            and not np.isnan(
                mutant_binding_at_site
            )
        ):

            delta_binding = (
                mutant_binding_at_site
                - wt_binding_at_site
            )

        wt_class = (
            wt_prediction[
                "predicted_class"
            ]
        )

        mutant_class = (
            mutant_prediction[
                "predicted_class"
            ]
        )

        result = {

            "protein_id":
                protein_id,

            "mutation":
                f"{wild_type}{position}{mutant}",

            "position":
                position,

            "wild_type":
                wild_type,

            "mutant":
                mutant,

            "wild_type_class":
                CLASS_NAMES.get(
                    wt_class,
                    str(wt_class)
                ),

            "mutant_class":
                CLASS_NAMES.get(
                    mutant_class,
                    str(mutant_class)
                ),

            "class_changed":
                bool(
                    wt_class != mutant_class
                ),

            "WT_non_NABP_probability":
                float(wt_probs[0]),

            "WT_RBP_probability":
                float(wt_probs[1]),

            "WT_DBP_probability":
                float(wt_probs[2]),

            "Mutant_non_NABP_probability":
                float(mutant_probs[0]),

            "Mutant_RBP_probability":
                float(mutant_probs[1]),

            "Mutant_DBP_probability":
                float(mutant_probs[2]),

            "Delta_non_NABP_probability":
                float(
                    mutant_probs[0]
                    - wt_probs[0]
                ),

            "Delta_RBP_probability":
                float(
                    mutant_probs[1]
                    - wt_probs[1]
                ),

            "Delta_DBP_probability":
                float(
                    mutant_probs[2]
                    - wt_probs[2]
                ),

            "WT_binding_probability":
                wt_binding_at_site,

            "Mutant_binding_probability":
                mutant_binding_at_site,

            "Delta_binding_probability":
                delta_binding,

            "WT_binding_predicted":
                (
                    int(
                        wt_binding_at_site
                        >= args.binding_threshold
                    )
                    if not np.isnan(
                        wt_binding_at_site
                    )
                    else np.nan
                ),

            "Mutant_binding_predicted":
                (
                    int(
                        mutant_binding_at_site
                        >= args.binding_threshold
                    )
                    if not np.isnan(
                        mutant_binding_at_site
                    )
                    else np.nan
                ),
        }

        results.append(result)

    results_df = pd.DataFrame(
        results
    )

    output_file = (
        output_dir /
        "mutation_predictions.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    summary = {

        "input_fasta":
            str(args.fasta),

        "mutation_file":
            str(args.mutations),

        "checkpoint":
            str(args.checkpoint),

        "number_of_mutations":
            len(mutations),

        "number_of_successful_predictions":
            len(results_df),

        "binding_threshold":
            args.binding_threshold,

        "note":
            (
                "Mutation predictions compare WT and "
                "mutant sequences using the same trained "
                "DeepNABind-MM model. Mutant structural "
                "representations should ideally be generated "
                "from mutant structures."
            ),
    }

    with open(
        output_dir /
        "mutation_prediction_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    logger.info(
        "Mutation prediction completed."
    )

    logger.info(
        "Results saved to: %s",
        output_file
    )


if __name__ == "__main__":
    main()