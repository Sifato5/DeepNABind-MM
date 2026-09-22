# DeepNABind-MM

**DeepNABind-MM** is a multimodal deep learning framework for **nucleic-acid-binding protein (NABP) prediction and residue-level nucleic-acid binding-site identification**.

The framework integrates:

* Protein sequence information from **ESM-2**
* AlphaFold2-predicted protein structures
* Multi-scale structural graphs
* Multimodal sequence–structure representation learning
* Protein-level NABP classification
* Residue-level binding-site prediction
* Cross-modal alignment/contrastive learning

The model classifies proteins into:

* **non-NABP**
* **RNA-binding protein (RBP)**
* **DNA-binding protein (DBP)**

---

## 1. Framework

```text
Protein Sequence
       │
       ├── ESM-2
       │      │
       │      └── Sequence Representation
       │
       └── AlphaFold2 Structure
              │
              ├── Atomic Graph
              ├── Residue Graph
              └── Secondary-Structure/Motif Graph
                       │
                       ▼
              Multimodal Deep Learning
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      Protein Classification   Binding-Site
             │                  Prediction
             ▼                   │
   non-NABP / RBP / DBP         ▼
                         Residue-level scores
```

---

## 2. Repository Structure

```text
DeepNABind-MM/
│
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── src/
│   ├── model.py
│   ├── train.py
│   ├── cross_validate.py
│   ├── evaluate.py
│   │
│   ├── data_utils.py
│   ├── final_data.py
│   ├── final_data_sharded.py
│   ├── embeddings.py
│   │
│   └── inference/
│       ├── predict_proteome.py
│       └── predict_mutations.py
│
├── configs/
│   └── config.json
│
├── data/
│   ├── raw/
│   ├── embeddings/
│   ├── structures/
│   ├── graphs/
│   └── binding_labels/
│
├── checkpoints/
│
├── examples/
│   ├── example.fasta
│   └── example_mutations.csv
│
└── results/
```

---

## 3. Installation

Clone the repository:

```bash
git clone https://github.com/Sifato5/DeepNABind-MM.git
cd DeepNABind-MM
```

Create an environment:

```bash
conda create -n deepnabind-mm python=3.10
conda activate deepnabind-mm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

or:

```bash
pip install -e .
```

A CUDA-enabled PyTorch installation is recommended. 

---

## 4. Dataset Preparation

The dataset preparation pipeline consists of:

```text
FASTA sequences
      │
      ▼
ESM-2 embeddings
      │
      ▼
AlphaFold2 structures
      │
      ▼
Structure processing
      │
      ▼
Multi-scale graph construction
      │
      ▼
Training-ready dataset
```

The main preprocessing scripts are:

```bash
python embeddings.py
```

and:

```bash
python final_data.py
```

For large datasets, use:

```bash
python final_data_sharded.py
```

The preprocessing pipeline generates:

* Protein metadata
* ESM-2 residue embeddings
* Atomic graphs
* Residue-level graphs
* Secondary-structure/motif graphs
* Cross-validation splits
* Dataset summary files

---

## 5. ESM-2 Embeddings

Generate residue-level ESM-2 embeddings using:

```bash
python embeddings.py \
    --data_dir data/raw \
    --output_dir data/embeddings \
    --device cuda
```

The exact ESM-2 model and layer should be kept consistent between preprocessing and inference.

---

## 6. Structural Graph Construction

DeepNABind-MM uses AlphaFold2-predicted structures as structural inputs.

The preprocessing pipeline constructs multiple graph representations:

```text
Atomic graph
     │
     ├── heavy atoms
     └── local atomic interactions

Residue graph
     │
     ├── Cα residues
     └── residue-level structural relationships

Motif/SSE graph
     │
     └── secondary-structure information
```

The structural preprocessing parameters should remain consistent between training and inference.

---

## 7. Binding-Site Labels

Residue-level nucleic-acid binding labels should be derived from **experimentally characterized protein–nucleic-acid complexes**, such as BioLiP/BioLiP2 annotations.

Recommended label format:

```text
0 = non-binding residue
1 = nucleic-acid-binding residue
```

Labels should be stored as:

```text
data/
└── binding_labels/
    ├── P12345.npy
    ├── P23456.npy
    └── ...
```

Each `.npy` file must contain a residue-level binary vector with exactly the same residue ordering as the sequence used by ESM-2.

---

## 8. Training

Train the model using:

```bash
python train.py
```

For cross-validation:

```bash
python cross_validate.py
```

The training procedure supports:

* Protein-level classification
* Residue-level binding-site prediction
* Sequence–structure multimodal learning
* Cross-modal alignment
* Validation and checkpointing

The best checkpoint should be saved under:

```text
checkpoints/
```

---

## 9. Evaluation

Evaluate a trained model using:

```bash
python evaluate.py
```

Protein-level evaluation includes:

* Accuracy
* Precision
* Recall
* F1-score
* MCC
* AUROC


For residue-level binding-site prediction, report:

* Binding-site AUROC
* Precision
* Recall
* F1-score
* MCC

Padding residues must be excluded from binding-site loss and evaluation.

---

## 10. Proteome-Wide Prediction

Use:

```bash
python predict_proteome.py \
    --fasta human_proteome.fasta \
    --sample_dir data/proteome_samples \
    --checkpoint checkpoints/deepnabind_mm_best.pt \
    --output_dir results/proteome \
    --device cuda
```

The output contains:

```text
results/proteome/
├── proteome_predictions.csv
├── prediction_summary.json
└── binding_sites/
```

Protein-level predictions include:

```text
non-NABP probability
RBP probability
DBP probability
predicted class
```

Residue-level predictions include:

```text
protein_id
position
residue
binding_probability
predicted_binding
```

---

## 11. Mutation Analysis

Mutation analysis compares a wild-type protein with its mutant using the **same trained DeepNABind-MM model**.

Mutation input:

```csv
protein_id,position,wild_type,mutant
P12345,45,K,R
P12345,120,R,A
P67890,87,G,D
```

Run:

```bash
python predict_mutations.py \
    --fasta human_proteome.fasta \
    --mutations mutations.csv \
    --sample_dir data/mutation_samples \
    --checkpoint checkpoints/deepnabind_mm_best.pt \
    --output_dir results/mutations \
    --device cuda
```

The output reports:

* WT predicted class
* Mutant predicted class
* WT class probabilities
* Mutant class probabilities
* Δ class probabilities
* WT binding probability at the mutation site
* Mutant binding probability at the mutation site
* Δ binding probability

### Important

For structural mutation analysis, the mutant should ideally have a **mutant-specific predicted structure**.

If the WT structure is reused for the mutant, the analysis should be interpreted as an approximation rather than a full structural mutation analysis.

---


## 12. Citation

If you use DeepNABind-MM in your research, please cite the associated publication:

## (Citation details will be provided here once available)

---

## 13. Contact

For questions regarding the code or methodology, please contact:

**Md. Sifat Ali**
School of Computer Science and Engineering,
Central South University, China.
Email: asasifat@csu.edu.cn



