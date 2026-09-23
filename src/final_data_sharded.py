import os
import sys
import json
import pickle
import time
import glob
import tempfile
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import MMCIFParser, PDBParser, DSSP
from Bio.PDB import PDBIO
from Bio.PDB.vectors import Vector

import esm

from sklearn.model_selection import StratifiedKFold, train_test_split

class LocalDataLoader:
    
    def __init__(self, base_dir: str = '.'):
        self.base_dir = Path(base_dir)
        
        self.fasta_files = {
            0: Path("/homeb/ali/second/data/non-NABP.fasta"),
            1: Path("/homeb/ali/second/data/RBP.fasta"),
            2: Path("/homeb/ali/second/data/DBP.fasta")
        }
        
        self.structure_folders = {
            0: Path("/homeb/ali/second/data/non-NABP"),
            1: Path("/homeb/ali/second/data/RBP"),
            2: Path("/homeb/ali/second/data/DBP")
        }
        
        self.class_names = {
            0: "non-NABP",
            1: "RBP",
            2: "DBP"
        }
    
    def load_all_data(self) -> Tuple[List[Dict], Dict[str, str]]:
        
        proteins = []
        structure_files = {}
        
        for label, fasta_path in self.fasta_files.items():
            class_name = self.class_names[label]
            struct_folder = self.structure_folders[label]
            
            if not fasta_path.exists():

                continue
            
            if not struct_folder.exists():
                
                continue
            
           
            sequences = list(SeqIO.parse(fasta_path, "fasta"))
            
            pdb_files = glob.glob(str(struct_folder / "*.pdb"))
            cif_files = glob.glob(str(struct_folder / "*.cif"))
            struct_files = pdb_files + cif_files
            
            available_structures = {}
            for struct_file in struct_files:
                filename = Path(struct_file).stem
                available_structures[filename] = struct_file
            
            matched = 0
            for record in sequences:
                seq_id = record.id
                sequence = str(record.seq)
                
                has_structure = False
                struct_file = None
                
                if seq_id in available_structures:
                    has_structure = True
                    struct_file = available_structures[seq_id]
                else:
                    base_id = seq_id.split('.')[0]
                    if base_id in available_structures:
                        has_structure = True
                        struct_file = available_structures[base_id]
                    else:
                        for avail_id in available_structures:
                            if avail_id.lower() == seq_id.lower():
                                has_structure = True
                                struct_file = available_structures[avail_id]
                                break
                
                protein_data = {
                    'uniprot_id': seq_id,
                    'entry_name': seq_id,
                    'sequence': sequence,
                    'length': len(sequence),
                    'organism': 'Unknown',
                    'taxon_id': None,
                    'go_annotation': class_name,
                    'go_terms': [],
                    'label': label,
                    'has_structure': has_structure,
                    'structure_file': struct_file,
                    'mean_plddt': None,
                    'dna_binding_type': 'dsDNA' if label == 2 else None
                }
                
                proteins.append(protein_data)
                
                if has_structure:
                    matched += 1
                    structure_files[seq_id] = struct_file
            
            print(f"  {matched}/{len(sequences)} have matching structure files")
        
        print(f"\nTotal proteins loaded: {len(proteins)}")
        print(f"Total structure files found: {len(structure_files)}")
        
        print("\nClass distribution:")
        for label, name in self.class_names.items():
            count = sum(1 for p in proteins if p['label'] == label)
            print(f"  {name}: {count}")
        
        return proteins, structure_files

class ShardedEmbeddingLoader:
    
    def __init__(self, embeddings_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.shard_files = None
        self.shard_index = None
        self._load_shard_index()
    
    def _load_shard_index(self):
        print(f"\nLoading sharded embeddings from: {self.embeddings_dir}")
        
        self.shard_files = sorted(glob.glob(str(self.embeddings_dir / "esm_embeddings_merged.pkl_shard_*.pkl")))
        
        if not self.shard_files:
            self.shard_files = sorted(glob.glob(str(self.embeddings_dir / "shard_*.pkl")))
        
        if not self.shard_files:
            raise FileNotFoundError(f"No shard files found in {self.embeddings_dir}")
        
        print(f"  Found {len(self.shard_files)} shard files")
        
        self.shard_index = {}
        self.shard_sizes = {}
        self.shard_data_cache = {}
        
        for shard_file in self.shard_files:
            shard_name = Path(shard_file).name
            try:
               
                with open(shard_file, 'rb') as f:
                    shard = pickle.load(f)
                    shard_size = len(shard)
                    self.shard_sizes[shard_name] = shard_size
                    
                    for protein_id in shard.keys():
                        self.shard_index[protein_id] = shard_file
                        
                print(f"    {shard_name}: {shard_size} proteins")
                
            except Exception as e:
                print(f"     Error loading {shard_name}: {e}")
        
        print(f"   Indexed {len(self.shard_index)} proteins across {len(self.shard_files)} shards")
    
    def get_embedding(self, protein_id: str) -> Optional[torch.Tensor]:
        
        shard_file = self.shard_index.get(protein_id)
        if shard_file is None:
            return None
        
        if shard_file in self.shard_data_cache:
            shard = self.shard_data_cache[shard_file]
        else:
            try:
                with open(shard_file, 'rb') as f:
                    shard = pickle.load(f)
                    if len(self.shard_data_cache) > 3:
                        oldest_key = next(iter(self.shard_data_cache))
                        del self.shard_data_cache[oldest_key]
                        gc.collect()
                    self.shard_data_cache[shard_file] = shard
            except Exception:
                return None
        
        return shard.get(protein_id)
    
    def get_embeddings_batch(self, protein_ids: List[str]) -> Dict[str, torch.Tensor]:
        
        shard_groups = {}
        for pid in protein_ids:
            shard_file = self.shard_index.get(pid)
            if shard_file:
                if shard_file not in shard_groups:
                    shard_groups[shard_file] = []
                shard_groups[shard_file].append(pid)
        
        embeddings = {}
        for shard_file, pids in shard_groups.items():
            try:
                with open(shard_file, 'rb') as f:
                    shard = pickle.load(f)
                    for pid in pids:
                        if pid in shard:
                            embeddings[pid] = shard[pid]
            except Exception as e:
                print(f"   Error loading from {Path(shard_file).name}: {e}")
        
        return embeddings
    
    def load_all_embeddings_sharded(self, protein_ids: List[str] = None):
        
        for shard_file in self.shard_files:
            try:
                with open(shard_file, 'rb') as f:
                    shard = pickle.load(f)
                    
                    if protein_ids is None:
                        yield shard
                    else:
                       
                        filtered_shard = {k: v for k, v in shard.items() if k in protein_ids}
                        if filtered_shard:
                            yield filtered_shard
            except Exception as e:
                print(f"   Error loading {Path(shard_file).name}: {e}")
    
    def get_num_embeddings(self) -> int:
        
        return len(self.shard_index)

class StructureParser:
    
    def __init__(self):
        self.mmcif_parser = MMCIFParser(QUIET=True)
        self.pdb_parser = PDBParser(QUIET=True)
        
        self.atom_map = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4,
            'F': 5, 'CL': 6, 'BR': 7, 'I': 8, 'MG': 9,
            'CA': 10, 'ZN': 11, 'FE': 12, 'CU': 13, 'MN': 14,
            'K': 15, 'NA': 16, 'CO': 17, 'SE': 18, 'H': 19
        }
    
    def parse_structure(self, file_path: str) -> Optional[Dict]:
        
        if not os.path.exists(file_path):
            return None
        
        if file_path.endswith('.cif'):
            parser = self.mmcif_parser
        else:
            parser = self.pdb_parser
        
        try:
            protein_id = Path(file_path).stem
            structure = parser.get_structure(protein_id, file_path)
            return self._extract_structure_data(structure, file_path, protein_id)
            
        except Exception as e:
            
            return None
    
    def _extract_structure_data(self, structure, file_path: str, protein_id: str) -> Dict:
        
        ca_coords = []
        plddt_scores = []
        all_atom_coords = []
        all_atom_types = []
        residue_indices = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    res_id = residue.get_id()[1]
                    residue_indices.append(res_id)
                    
                    if 'CA' in residue:
                        ca_atom = residue['CA']
                        ca_coords.append(ca_atom.get_coord())
                        plddt_scores.append(ca_atom.bfactor)
                    
                    for atom in residue:
                        if atom.element != 'H':
                            all_atom_coords.append(atom.get_coord())
                            atom_type = self.atom_map.get(atom.element.upper(), 36)
                            all_atom_types.append(atom_type)
        
        ca_coords = np.array(ca_coords) if ca_coords else np.array([])
        plddt_scores = np.array(plddt_scores) if plddt_scores else np.array([])
        all_atom_coords = np.array(all_atom_coords) if all_atom_coords else np.array([])
        all_atom_types = np.array(all_atom_types) if all_atom_types else np.array([])
        
        return {
            'uniprot_id': protein_id,
            'file_path': file_path,
            'ca_coords': ca_coords,
            'plddt': plddt_scores,
            'mean_plddt': np.mean(plddt_scores) if len(plddt_scores) > 0 else 0,
            'all_atom_coords': all_atom_coords,
            'all_atom_types': all_atom_types,
            'num_residues': len(ca_coords),
            'num_atoms': len(all_atom_coords),
            'residue_indices': residue_indices
        }

def assign_secondary_structures_simple(structure_path: str) -> Tuple[np.ndarray, List[Dict]]:
    return np.array([]), []  


def assign_secondary_structures(structure_path: str) -> Tuple[np.ndarray, List[Dict]]:
    
    import shutil
    dssp_path = shutil.which('mkdssp') or shutil.which('dssp')
    
    if dssp_path is None:
        return np.array([]), []
    
    try:
        
        if structure_path.endswith('.cif'):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)
        
        structure = parser.get_structure('protein', structure_path)
        
        pdb_file = None
        try:
            io = PDBIO()
            io.set_structure(structure)
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                pdb_file = f.name
                io.save(pdb_file)
            
            dssp = DSSP(structure[0], pdb_file, dssp=dssp_path)
            
            sse_labels = []
            sse_segments = []
            current_segment = []
            current_sse_type = None
            current_sse_name = None
            
            for residue_key in dssp.keys():
                dssp_data = dssp[residue_key]
                if not dssp_data:
                    continue
                
                sse = dssp_data[2]
                
                if sse in ['H', 'G', 'I']:
                    sse_code = 0
                    sse_name = 'HELIX'
                elif sse in ['E', 'B']:
                    sse_code = 1
                    sse_name = 'SHEET'
                else:
                    sse_code = 2
                    sse_name = 'COIL'
                
                sse_labels.append(sse_code)
                
                if sse_code == current_sse_type:
                    current_segment.append(residue_key[1])
                else:
                    if current_segment:
                        sse_segments.append({
                            'type': current_sse_type,
                            'name': current_sse_name,
                            'residues': current_segment,
                            'start': current_segment[0],
                            'end': current_segment[-1],
                            'length': len(current_segment)
                        })
                    current_segment = [residue_key[1]]
                    current_sse_type = sse_code
                    current_sse_name = sse_name
            
            if current_segment:
                sse_segments.append({
                    'type': current_sse_type,
                    'name': current_sse_name,
                    'residues': current_segment,
                    'start': current_segment[0],
                    'end': current_segment[-1],
                    'length': len(current_segment)
                })
            
            return np.array(sse_labels), sse_segments
            
        finally:
            if pdb_file and os.path.exists(pdb_file):
                os.unlink(pdb_file)
                
    except Exception:
        return np.array([]), []

class HierarchicalGraphBuilder:
    
    def __init__(self, 
                 atom_cutoff: float = 2.0,
                 residue_cutoff: float = 14.0,
                 sse_cutoff: float = 8.0):
        self.atom_cutoff = atom_cutoff
        self.residue_cutoff = residue_cutoff
        self.sse_cutoff = sse_cutoff
    
    def build_graphs(self, 
                    protein_data: Dict,
                    esm_embedding: torch.Tensor) -> Dict:
        
        atomic_graph = self.build_atomic_graph(
            protein_data.get('all_atom_coords', np.array([])),
            protein_data.get('all_atom_types', np.array([]))
        )
        
        residue_graph = self.build_residue_graph(
            protein_data.get('ca_coords', np.array([])),
            esm_embedding
        )
        
        sse_labels, sse_segments = assign_secondary_structures(
            protein_data['file_path']
        )
        
        if len(sse_labels) > 0:
            motif_graph = self.build_motif_graph(
                protein_data.get('ca_coords', np.array([])),
                torch.tensor(sse_labels),
                esm_embedding
            )
        else:
            motif_graph = None
        
        return {
            'atomic': atomic_graph,
            'residue': residue_graph,
            'motif': motif_graph,
            'sse_segments': sse_segments
        }
    
    def build_atomic_graph(self, coords: np.ndarray, atom_types: np.ndarray):
        
        from torch_geometric.data import Data
        
        if coords is None or len(coords) == 0:
            return None
        
        if len(coords) > 5000:
            
            indices = np.random.choice(len(coords), 5000, replace=False)
            coords = coords[indices]
            atom_types = atom_types[indices]
        
        coords = torch.tensor(coords, dtype=torch.float)
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        
        if len(coords) > 1000:
            
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 1)
        else:
            dist = torch.cdist(coords, coords)
            mask = (dist < self.atom_cutoff) & (dist > 1e-6)
            edge_index = mask.nonzero(as_tuple=False).t()
            
            if edge_index.shape[1] > 0:
                row, col = edge_index
                edge_attr = dist[row, col].unsqueeze(1)
            else:
                edge_attr = torch.empty(0, 1)
        
        x = F.one_hot(atom_types, num_classes=37).float()
        
        return Data(x=x, pos=coords, 
                   edge_index=edge_index, 
                   edge_attr=edge_attr)
    
    def build_residue_graph(self, ca_coords: np.ndarray, esm_embedding: torch.Tensor):
        
        from torch_geometric.data import Data
        
        if ca_coords is None or len(ca_coords) == 0:
            return None
        
        coords = torch.tensor(ca_coords, dtype=torch.float)
        n_res = len(coords)
        
        if esm_embedding.shape[0] != n_res:
            if esm_embedding.shape[0] > n_res:
                esm_embedding = esm_embedding[:n_res]
            else:
                pad = torch.zeros(n_res - esm_embedding.shape[0], esm_embedding.shape[1])
                esm_embedding = torch.cat([esm_embedding, pad], dim=0)
        
        edge_index = []
        edge_attr = []
        
        chunk_size = 500
        for i in range(0, n_res, chunk_size):
            end_i = min(i + chunk_size, n_res)
            coords_chunk = coords[i:end_i]
            dist_chunk = torch.cdist(coords_chunk, coords)
            
            for j in range(len(coords_chunk)):
                mask = (dist_chunk[j] < self.residue_cutoff) & (dist_chunk[j] > 1e-6)
                neighbors = mask.nonzero(as_tuple=False).squeeze()
                
                if neighbors.numel() > 0:
                    for neighbor in neighbors:
                        if neighbor.item() != (i + j):
                            edge_index.append([i + j, neighbor.item()])
                            edge_attr.append(dist_chunk[j, neighbor].item())
        
        if edge_index:
            edge_index = torch.tensor(edge_index).t().contiguous()
            edge_attr = torch.tensor(edge_attr).unsqueeze(1).float()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 1)
        
        return Data(
            x=esm_embedding.float(),
            pos=coords,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
    
    def build_motif_graph(self, ca_coords: np.ndarray, sse_labels: torch.Tensor, esm_embedding: torch.Tensor):
       
        from torch_geometric.data import Data
        
        if ca_coords is None or len(ca_coords) == 0 or len(sse_labels) == 0:
            return None
        
        coords = torch.tensor(ca_coords, dtype=torch.float)
        
        unique_sse = torch.unique(sse_labels)
        sse_coords = []
        sse_features = []
        residue_indices = []
        
        for sse_id in unique_sse:
            mask = (sse_labels == sse_id)
            if mask.sum() == 0:
                continue
            
            avg_coord = coords[mask].mean(dim=0)
            avg_feat = esm_embedding[mask].mean(dim=0)
            
            sse_coords.append(avg_coord)
            sse_features.append(avg_feat)
            residue_indices.append(mask.nonzero().squeeze())
        
        if not sse_coords:
            return None
        
        sse_coords = torch.stack(sse_coords)
        sse_features = torch.stack(sse_features)
        n_sse = len(sse_coords)
        
        edge_index = []
        edge_attr = []
        
        for i in range(n_sse):
            for j in range(i+1, n_sse):
                min_dist = float('inf')
                
                ri_indices = residue_indices[i]
                rj_indices = residue_indices[j]
                
                if ri_indices.dim() == 0:
                    ri_indices = ri_indices.unsqueeze(0)
                if rj_indices.dim() == 0:
                    rj_indices = rj_indices.unsqueeze(0)
                
                max_pairs = 100
                if len(ri_indices) * len(rj_indices) > max_pairs:
                    ri_indices = ri_indices[:max_pairs]
                    rj_indices = rj_indices[:max_pairs]
                
                for ri in ri_indices:
                    for rj in rj_indices:
                        dist = torch.norm(coords[ri] - coords[rj])
                        min_dist = min(min_dist, dist.item())
                
                if min_dist < self.sse_cutoff:
                    edge_index.extend([[i, j], [j, i]])
                    edge_attr.extend([min_dist, min_dist])
        
        if edge_index:
            edge_index = torch.tensor(edge_index).t()
            edge_attr = torch.tensor(edge_attr).unsqueeze(1).float()
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 1)
        
        return Data(
            x=sse_features,
            pos=sse_coords,
            edge_index=edge_index,
            edge_attr=edge_attr
        )

def build_graphs_sharded(
    proteins: List[Dict],
    structures: Dict[str, Dict],
    embedding_loader: ShardedEmbeddingLoader,
    graph_builder: HierarchicalGraphBuilder,
    output_dir: Path,
    batch_size: int = 10,
    shard_size: int = 500
) -> Dict:
   
    shards_dir = output_dir / 'graph_shards'
    shards_dir.mkdir(exist_ok=True)
    
    checkpoint_dir = output_dir / 'graph_checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    existing_shards = sorted(shards_dir.glob('graph_shard_*.pt'))
    processed_ids = set()
    
    print("\nLoading existing shards...")
    for shard_file in existing_shards:
        try:
            shard_data = torch.load(shard_file, map_location='cpu')
            processed_ids.update(shard_data.keys())
            print(f"   Loaded {shard_file.name}: {len(shard_data)} proteins")
        except Exception as e:
            print(f"   Error loading {shard_file.name}: {e}")
    
    print(f"\n  Total proteins already processed: {len(processed_ids)}")
    
    proteins_to_process = [p for p in proteins if p['uniprot_id'] not in processed_ids]
    print(f"  Remaining to process: {len(proteins_to_process)}")
    
    if not proteins_to_process:
        print("\n   All proteins already processed!")
       
        create_shard_manifest(shards_dir, output_dir)
        return {}
    
    current_shard = {}
    current_shard_idx = len(existing_shards)
    total_processed = len(processed_ids)
    failed_proteins_global = set()
    
    num_batches = (len(proteins_to_process) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
       
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(proteins_to_process))
        batch_proteins = proteins_to_process[start_idx:end_idx]
        
        print(f"\n  Processing batch {batch_idx + 1}/{num_batches} "
              f"({len(batch_proteins)} proteins)...")
        
        batch_protein_ids = [p['uniprot_id'] for p in batch_proteins]
        batch_embeddings = embedding_loader.get_embeddings_batch(batch_protein_ids)
        
        batch_success = 0
        batch_failed = 0
        
        for protein in tqdm(batch_proteins, desc=f"Batch {batch_idx + 1}"):
            uniprot_id = protein['uniprot_id']
            
            if uniprot_id in failed_proteins_global:
                continue
            
            struct = structures.get(uniprot_id)
            emb = batch_embeddings.get(uniprot_id)
            
            if struct is None or emb is None:
                failed_proteins_global.add(uniprot_id)
                batch_failed += 1
                continue
            
            try:
                
                protein_graphs = graph_builder.build_graphs(struct, emb)
                
                if protein_graphs['residue'] is not None:
                    current_shard[uniprot_id] = {
                        'protein': protein,
                        'structure': struct,
                        'embedding': emb,
                        'graphs': protein_graphs,
                        'label': protein['label'],
                        'dna_binding_type': protein.get('dna_binding_type', None)
                    }
                    batch_success += 1
                    
            except Exception as e:
                failed_proteins_global.add(uniprot_id)
                batch_failed += 1
                print(f"\n     Skipping {uniprot_id}: {str(e)[:80]}")
                
                failed_log = output_dir / 'failed_proteins.txt'
                with open(failed_log, 'a') as f:
                    f.write(f"{uniprot_id}\t{str(e)}\n")
                
                continue
        
        if len(current_shard) >= shard_size:
            shard_file = shards_dir / f'graph_shard_{current_shard_idx:04d}.pt'
            temp_file = shards_dir / f'graph_shard_{current_shard_idx:04d}.tmp'
            
            torch.save(current_shard, temp_file)
            temp_file.rename(shard_file)
            
            print(f"\n   Saved shard {current_shard_idx}: {shard_file.name} ({len(current_shard)} proteins)")
            
            total_processed += len(current_shard)
            current_shard = {}
            current_shard_idx += 1
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"    Batch results: {batch_success} success, {batch_failed} failed")
        print(f"    Current shard: {len(current_shard)}/{shard_size} proteins")
        print(f"    Total processed: {total_processed}")
    
    if current_shard:
        shard_file = shards_dir / f'graph_shard_{current_shard_idx:04d}.pt'
        temp_file = shards_dir / f'graph_shard_{current_shard_idx:04d}.tmp'
        
        torch.save(current_shard, temp_file)
        temp_file.rename(shard_file)
        
        print(f"\n   Saved final shard {current_shard_idx}: {shard_file.name} ({len(current_shard)} proteins)")
        total_processed += len(current_shard)
    
    create_shard_manifest(shards_dir, output_dir)
    return {}


def create_shard_manifest(shards_dir: Path, output_dir: Path):
    
    import json
    
    manifest = {
        'shards': [],
        'total_proteins': 0,
        'shard_size': None
    }
    
    shard_files = sorted(shards_dir.glob('graph_shard_*.pt'))
    
    for shard_file in shard_files:
        try:
            data = torch.load(shard_file, map_location='cpu')
            protein_ids = list(data.keys())
            manifest['shards'].append({
                'file': str(shard_file.name),
                'path': str(shard_file),
                'protein_count': len(protein_ids),
                'proteins': protein_ids
            })
            manifest['total_proteins'] += len(protein_ids)
            if manifest['shard_size'] is None:
                manifest['shard_size'] = len(protein_ids)
        except Exception as e:
            print(f"   Warning: Could not read {shard_file.name}: {e}")
    
    manifest_file = output_dir / 'shard_manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n   Created manifest: {manifest_file}")
    print(f"    Total shards: {len(manifest['shards'])}, Total proteins: {manifest['total_proteins']}")

class LocalNABPDatasetBuilder:
    
    def __init__(self, 
                 fasta_dir: str = '.',
                 structure_dir: str = '.',
                 output_dir: str = '/homeb/ali/second/data',
                 embeddings_dir: str = '/homeb/ali/second/data/embeddings'):
        self.fasta_dir = Path(fasta_dir)
        self.structure_dir = Path(structure_dir)
        self.output_dir = Path(output_dir)
        self.embeddings_dir = Path(embeddings_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = LocalDataLoader(base_dir=self.fasta_dir)
        self.structure_parser = StructureParser()
        self.graph_builder = HierarchicalGraphBuilder()
        self.embedding_loader = None
        self.embeddings = None  
        
        self.class_names = {
            0: "non-NABP",
            1: "RBP",
            2: "DBP"
        }
    
    def load_and_parse_structures(self, 
                                  proteins: List[Dict], 
                                  structure_files: Dict[str, str]) -> Dict[str, Dict]:
       
        structures = {}
        
        for protein in tqdm(proteins, desc="Parsing structures"):
            uniprot_id = protein['uniprot_id']
            struct_file = structure_files.get(uniprot_id)
            
            if struct_file and os.path.exists(struct_file):
                struct_data = self.structure_parser.parse_structure(struct_file)
                if struct_data and len(struct_data.get('ca_coords', [])) > 0:
                    structures[uniprot_id] = struct_data
                    protein['has_structure'] = True
                    protein['mean_plddt'] = struct_data['mean_plddt']
        
        print(f"   Successfully parsed {len(structures)} structures")
        return structures
    
    def filter_by_confidence(self, 
                            proteins: List[Dict], 
                            structures: Dict[str, Dict],
                            threshold: float = 70.0,
                            min_coverage: float = 0.8) -> Tuple[List[Dict], Dict[str, Dict]]:
        """Filter structures by pLDDT confidence"""
        
        print("\n" + "="*80)
        print(f"FILTERING BY CONFIDENCE (pLDDT > {threshold})")
        print("="*80)
        
        filtered_proteins = []
        filtered_structures = {}
        
        for protein in proteins:
            uniprot_id = protein['uniprot_id']
            struct = structures.get(uniprot_id)
            
            if struct is None:
                continue
            
            if len(struct.get('ca_coords', [])) == 0:
                continue
            
            mean_plddt = struct.get('mean_plddt', 0)
            if mean_plddt < threshold:
                continue
            
            plddt = struct.get('plddt', np.array([]))
            if len(plddt) > 0:
                high_conf_ratio = np.sum(plddt > 50) / len(plddt)
                if high_conf_ratio >= min_coverage:
                    filtered_proteins.append(protein)
                    filtered_structures[uniprot_id] = struct
        
        print(f"   {len(filtered_proteins)} proteins passed confidence filter")
        return filtered_proteins, filtered_structures
    
    def build_graphs_for_dataset(self, 
                                proteins: List[Dict],
                                structures: Dict[str, Dict]) -> Dict:
       
        self.embedding_loader = ShardedEmbeddingLoader(str(self.embeddings_dir))
        
        return build_graphs_sharded(
            proteins=proteins,
            structures=structures,
            embedding_loader=self.embedding_loader,
            graph_builder=self.graph_builder,
            output_dir=self.output_dir,
            batch_size=10,      
            shard_size=500      
        )
    
    def create_cv_splits(self, proteins: List[Dict], n_folds: int = 5) -> List[Dict]:
        """Create 5-fold cross-validation splits"""
        
        if not proteins:
            return []
        
        all_ids = [p['uniprot_id'] for p in proteins]
        all_labels = [p['label'] for p in proteins]
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(all_ids, all_labels)):
            train_ids = [all_ids[i] for i in train_idx]
            val_ids = [all_ids[i] for i in val_idx]
            splits.append({
                'fold': fold,
                'train': train_ids,
                'val': val_ids
            })
        
        return splits
    
    def save_dataset(self, 
                    proteins: List[Dict],
                    structures: Dict[str, Dict],
                    graphs: Dict,  
                    splits: List[Dict]):
        
        
        manifest_file = self.output_dir / 'shard_manifest.json'
        total_graphs = 0
        if manifest_file.exists():
            import json
            with open(manifest_file, 'r') as f:
                manifest = json.load(f)
                total_graphs = manifest['total_proteins']
        
        proteins_file = self.output_dir / 'proteins_metadata.json'
        with open(proteins_file, 'w') as f:
            proteins_serializable = []
            for p in proteins:
                p_copy = p.copy()
                p_copy.pop('go_terms', None)
                if 'sequence' in p_copy:
                    p_copy['sequence_length'] = len(p_copy['sequence'])
                    p_copy.pop('sequence', None)
                proteins_serializable.append(p_copy)
            json.dump(proteins_serializable, f, indent=2)
        print(f"   Saved proteins metadata: {proteins_file}")
        
        splits_file = self.output_dir / 'cv_splits.json'
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"   Saved CV splits: {splits_file}")
        
        summary = {
            'total_proteins_loaded': len(proteins),
            'total_structures': len(structures),
            'total_embeddings': self.embedding_loader.get_num_embeddings() if self.embedding_loader else 0,
            'total_graphs': total_graphs,
            'class_distribution': {
                self.class_names[0]: sum(1 for p in proteins if p['label'] == 0),
                self.class_names[1]: sum(1 for p in proteins if p['label'] == 1),
                self.class_names[2]: sum(1 for p in proteins if p['label'] == 2)
            },
            'cv_folds': len(splits),
            'embeddings_source': str(self.embeddings_dir),
            'embeddings_format': 'sharded',
            'graph_format': 'sharded'
        }
        
        summary_file = self.output_dir / 'dataset_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Saved dataset summary: {summary_file}")
        
        return summary

def main():
   
    CONFIG = {
        'fasta_dir': '.',
        'structure_dir': '.',
        'output_dir': '/homeb/ali/second/data/DeepNABindMM_dataset',
        'embeddings_dir': '/homeb/ali/second/data/embeddings',
        'confidence_filter': {
            'plddt_threshold': 70.0,
            'min_coverage': 0.8
        },
        'graphs': {
            'atom_cutoff': 2.0,
            'residue_cutoff': 14.0,
            'sse_cutoff': 8.0
        },
        'batch_size': 10,     
        'shard_size': 500     
    }
    
    builder = LocalNABPDatasetBuilder(
        fasta_dir=CONFIG['fasta_dir'],
        structure_dir=CONFIG['structure_dir'],
        output_dir=CONFIG['output_dir'],
        embeddings_dir=CONFIG['embeddings_dir']
    )
    
    proteins, structure_files = builder.data_loader.load_all_data()
    
    if not proteins:
        return
    
    all_structures = builder.load_and_parse_structures(proteins, structure_files)
    
    filtered_proteins, filtered_structures = builder.filter_by_confidence(
        proteins, 
        all_structures,
        threshold=CONFIG['confidence_filter']['plddt_threshold'],
        min_coverage=CONFIG['confidence_filter']['min_coverage']
    )
    
    if not filtered_proteins:
        print("\n No proteins passed confidence filter.")
        return
    
    graphs = builder.build_graphs_for_dataset(
        filtered_proteins, 
        filtered_structures
    )
    
    cv_splits = builder.create_cv_splits(filtered_proteins, n_folds=5)
    
    summary = builder.save_dataset(
        filtered_proteins, 
        filtered_structures, 
        graphs, 
        cv_splits
    )
    
    print("\n" + "="*80)
    print("DATASET PREPARATION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {builder.output_dir}")
    print("\nDataset Statistics:")
    print(f"  - Total proteins with complete data: {summary['total_graphs']}")
    print(f"  - Total structures: {summary['total_structures']}")
    print(f"  - Total embeddings (sharded): {summary['total_embeddings']}")
    print("\nClass Distribution:")
    for class_name, count in summary['class_distribution'].items():
        if count > 0:
            print(f"  - {class_name}: {count}")
    print("\n   Graphs saved as shards in: graph_shards/")
    print("  Use shard_manifest.json to load graphs during training")
    print("\n" + "="*80)


if __name__ == "__main__":
   
    required_packages = [
        "torch",
        "biopython",
        "fair-esm",
        "tqdm",
        "scikit-learn"
    ]
    
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            if package == "fair-esm":
                __import__("esm")
            else:
                __import__(package.replace("-", "_"))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    try:
        import torch_geometric
    except ImportError:
        print("Installing torch-geometric...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    
    main()