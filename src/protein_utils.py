import os
import torch
import numpy as np
from pathlib import Path
from Bio.PDB import MMCIFParser, PDBParser
import tempfile
import shutil


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
    
    def parse_structure(self, file_path: str):
        
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
    
    def _extract_structure_data(self, structure, file_path: str, protein_id: str):
        
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


def assign_secondary_structures(structure_path: str):
    import numpy as np
    return np.array([]), []


class HierarchicalGraphBuilder:
    
    def __init__(self, 
                 atom_cutoff: float = 2.0,
                 residue_cutoff: float = 14.0,
                 sse_cutoff: float = 8.0):
        self.atom_cutoff = atom_cutoff
        self.residue_cutoff = residue_cutoff
        self.sse_cutoff = sse_cutoff
    
    def build_graphs(self, protein_data: dict, esm_embedding: torch.Tensor):
        from torch_geometric.data import Data
        import torch
        
        residue_graph = self.build_residue_graph(
            protein_data.get('ca_coords', np.array([])),
            esm_embedding
        )
        
        atomic_graph = self.build_atomic_graph(
            protein_data.get('all_atom_coords', np.array([])),
            protein_data.get('all_atom_types', np.array([]))
        )
        
        motif_graph = None
        
        return {
            'atomic': atomic_graph,
            'residue': residue_graph,
            'motif': motif_graph
        }
    
    def build_residue_graph(self, ca_coords: np.ndarray, esm_embedding: torch.Tensor):
        """G_r: Node = Cα, Feature = ESM-2 embedding, Edge = <14.0Å"""
        from torch_geometric.data import Data
        import torch
        
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
        
        dist = torch.cdist(coords, coords)
        mask = (dist < self.residue_cutoff) & (dist > 1e-6)
        edge_index = mask.nonzero(as_tuple=False).t()
        
        if edge_index.shape[1] > 0:
            row, col = edge_index
            edge_attr = dist[row, col].unsqueeze(1)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 1)
        
        return Data(
            x=esm_embedding.float(),
            pos=coords,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
    
    def build_atomic_graph(self, coords: np.ndarray, atom_types: np.ndarray):
        """G_a: Node = heavy atoms, Edge = covalent or <2.0Å"""
        from torch_geometric.data import Data
        import torch
        
        if coords is None or len(coords) == 0:
            return None
        
        if len(coords) > 2000:
            indices = np.random.choice(len(coords), 2000, replace=False)
            coords = coords[indices]
            atom_types = atom_types[indices]
        
        coords = torch.tensor(coords, dtype=torch.float)
        atom_types = torch.tensor(atom_types, dtype=torch.long)
        
        dist = torch.cdist(coords, coords)
        mask = (dist < self.atom_cutoff) & (dist > 1e-6)
        edge_index = mask.nonzero(as_tuple=False).t()
        
        if edge_index.shape[1] > 0:
            row, col = edge_index
            edge_attr = dist[row, col].unsqueeze(1)
        else:
            edge_index = torch.empty(2, 0, dtype=torch.long)
            edge_attr = torch.empty(0, 1)
        
        x = torch.nn.functional.one_hot(atom_types, num_classes=37).float()
        
        return Data(x=x, pos=coords, edge_index=edge_index, edge_attr=edge_attr)