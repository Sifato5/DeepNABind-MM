import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax
import math
from typing import Tuple, Dict, List, Optional

class EGCL(MessagePassing):

    
    def __init__(self, hidden_dim: int, edge_dim: int = 1, aggr: str = 'add'):
        super(EGCL, self).__init__(aggr=aggr)
        
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        
        self.phi_e = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.phi_h = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.phi_x = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, pos, edge_index, edge_attr=None):
        
        out = self.propagate(edge_index, x=x, pos=pos, edge_attr=edge_attr)
        
        x = self.phi_h(torch.cat([x, out['node_update']], dim=-1))
        
        pos = pos + out['coord_update']
        
        return x, pos
    
    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        
        rel_pos = pos_i - pos_j
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        dist_clamped = torch.clamp(dist, min=1e-6)
        direction = rel_pos / dist_clamped
        
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)
        m = self.phi_e(m)
        
        coord_update = direction * self.phi_x(m)
        
        return {'node_update': m, 'coord_update': coord_update}
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        """Aggregate messages"""
        node_update = self.aggr_module(inputs['node_update'], index, dim_size=dim_size)
        coord_update = self.aggr_module(inputs['coord_update'], index, dim_size=dim_size)
        return {'node_update': node_update, 'coord_update': coord_update}

class HierarchicalGraphProcessor(nn.Module):
   
    def __init__(self, 
                 hidden_dim: int = 256,
                 atomic_dim: int = 37,  
                 residue_dim: int = 2560,  
                 motif_dim: int = 2560,
                 num_egcl_layers: int = 3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.atomic_proj = nn.Linear(atomic_dim, hidden_dim)
        self.residue_proj = nn.Linear(residue_dim, hidden_dim)
        self.motif_proj = nn.Linear(motif_dim, hidden_dim)
        
        self.atomic_layers = nn.ModuleList([
            EGCL(hidden_dim, edge_dim=1) for _ in range(num_egcl_layers)
        ])
        
        self.residue_layers = nn.ModuleList([
            EGCL(hidden_dim, edge_dim=1) for _ in range(num_egcl_layers)
        ])
        
        self.motif_layers = nn.ModuleList([
            EGCL(hidden_dim, edge_dim=1) for _ in range(num_egcl_layers)
        ])
       
        self.atom_to_residue_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.residue_to_motif_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
       
        self.atom_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.residue_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.motif_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                atomic_graph: Dict,
                residue_graph: Dict,
                motif_graph: Dict) -> torch.Tensor:
       
        batch_size = residue_graph.get('batch_size', 1)
        
        if atomic_graph is not None:
            x_a = self.atomic_proj(atomic_graph['x'])
            pos_a = atomic_graph['pos']
            edge_index_a = atomic_graph['edge_index']
            edge_attr_a = atomic_graph['edge_attr']
            
            for layer in self.atomic_layers:
                x_a, pos_a = layer(x_a, pos_a, edge_index_a, edge_attr_a)
            
            atomic_pooled = global_mean_pool(x_a, atomic_graph['batch'] if 'batch' in atomic_graph else torch.zeros(len(x_a), dtype=torch.long))
            atomic_feats = self.atom_pool(atomic_pooled)
        else:
            
            device = residue_graph['x'].device if 'x' in residue_graph else 'cpu'
            atomic_feats = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        x_r = self.residue_proj(residue_graph['x'])
        pos_r = residue_graph['pos']
        edge_index_r = residue_graph['edge_index']
        edge_attr_r = residue_graph['edge_attr']
        batch_r = residue_graph.get('batch', torch.zeros(len(x_r), dtype=torch.long, device=x_r.device))
        
        if atomic_graph is not None and atomic_feats is not None:
            
            num_residues = len(x_r)
            num_atoms = len(x_a)
            
            if num_residues > 0 and num_atoms > 0:
               
                dist = torch.cdist(pos_r, pos_a)  
                closest_atoms = torch.argmin(dist, dim=1)  
                
                atom_expanded = x_a[closest_atoms]  
                
                x_r_attended, _ = self.atom_to_residue_attn(
                    x_r.unsqueeze(0), 
                    atom_expanded.unsqueeze(0), 
                    atom_expanded.unsqueeze(0)
                )
                x_r = x_r + x_r_attended.squeeze(0)
        
        for layer in self.residue_layers:
            x_r, pos_r = layer(x_r, pos_r, edge_index_r, edge_attr_r)
        
        residue_pooled = global_mean_pool(x_r, batch_r)
        residue_feats = self.residue_pool(residue_pooled)
        
        if motif_graph is not None:
            x_s = self.motif_proj(motif_graph['x'])
            pos_s = motif_graph['pos']
            edge_index_s = motif_graph['edge_index']
            edge_attr_s = motif_graph['edge_attr']
            batch_s = motif_graph.get('batch', torch.zeros(len(x_s), dtype=torch.long, device=x_s.device))
            
            if len(x_r) > 0 and len(x_s) > 0:
                
                x_s_attended, _ = self.residue_to_motif_attn(
                    x_s.unsqueeze(0),
                    x_r.unsqueeze(0),
                    x_r.unsqueeze(0)
                )
                x_s = x_s + x_s_attended.squeeze(0)
            
            for layer in self.motif_layers:
                x_s, pos_s = layer(x_s, pos_s, edge_index_s, edge_attr_s)
            
            motif_pooled = global_mean_pool(x_s, batch_s)
            motif_feats = self.motif_pool(motif_pooled)
        else:
            motif_feats = torch.zeros(batch_size, self.hidden_dim, device=x_r.device)
       
        structural_feats = residue_feats + atomic_feats + motif_feats
        
        if structural_feats.dim() > 2:
            structural_feats = structural_feats.mean(dim=1)
        
        return structural_feats

class BindingSiteAwareSeqModel(nn.Module):
    
    def __init__(self, 
                 input_dim: int = 2560,
                 hidden_dim: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_heads: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.bilstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.binding_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self.mhsa = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, seq_embeddings: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        
        batch_size, L, _ = seq_embeddings.shape
        
        x = self.input_proj(seq_embeddings)
        
        if lengths is not None:
            
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (hidden, cell) = self.bilstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=L)
        else:
            x, _ = self.bilstm(x)
        
        binding_logits = self.binding_predictor(x)
        binding_probs = torch.sigmoid(binding_logits)
        
        binding_scores = binding_probs.squeeze(-1).clamp(min=1e-6, max=1.0)
        attention_bias = torch.sqrt(
            binding_scores.unsqueeze(-1) * binding_scores.unsqueeze(-2) + 1e-8
        )
        attention_bias = attention_bias * 10.0
        
        attn_mask = attention_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch_size * self.num_heads, L, L).log()
        
        key_padding_mask = None
        if lengths is not None:
            positions = torch.arange(L, device=seq_embeddings.device).unsqueeze(0)
            key_padding_mask = positions >= lengths.to(seq_embeddings.device).unsqueeze(1)
        
        x_attended, _ = self.mhsa(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )
        x = self.layer_norm(x + self.dropout(x_attended))
        
        binding_weights = binding_probs
        if key_padding_mask is not None:
            binding_weights = binding_weights.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        binding_weights = binding_weights / (binding_weights.sum(dim=1, keepdim=True) + 1e-8)
        global_seq_feats = (x * binding_weights).sum(dim=1)
        
        return global_seq_feats, x, binding_logits, binding_probs

class DynamicRoutingFusion(nn.Module):
   
    def __init__(self, feature_dim: int = 512, num_iterations: int = 3, num_output_capsules: int = 32):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.num_output_capsules = num_output_capsules
        
        self.seq_transform = nn.Linear(feature_dim, feature_dim)
        self.struct_transform = nn.Linear(feature_dim, feature_dim)
        
        self.W_seq = nn.Linear(feature_dim, feature_dim * num_output_capsules)
        self.W_struct = nn.Linear(feature_dim, feature_dim * num_output_capsules)
        
        self.cross_attn_seq = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        self.cross_attn_struct = nn.MultiheadAttention(feature_dim, num_heads=8, batch_first=True)
        
        self.layer_norm_seq = nn.LayerNorm(feature_dim)
        self.layer_norm_struct = nn.LayerNorm(feature_dim)
        
    def forward(self, seq_feats: torch.Tensor, struct_feats: torch.Tensor) -> torch.Tensor:
        
        batch_size = seq_feats.shape[0]
        
        seq_expanded = seq_feats.unsqueeze(1)  
        struct_expanded = struct_feats.unsqueeze(1)  
        
        seq_attended, _ = self.cross_attn_seq(seq_expanded, struct_expanded, struct_expanded)
        seq_feats = self.layer_norm_seq(seq_feats + seq_attended.squeeze(1))
        
        struct_attended, _ = self.cross_attn_struct(struct_expanded, seq_expanded, seq_expanded)
        struct_feats = self.layer_norm_struct(struct_feats + struct_attended.squeeze(1))
        
        seq_proj = self.seq_transform(seq_feats)
        struct_proj = self.struct_transform(struct_feats)
        
        seq_capsules = self.W_seq(seq_proj).view(batch_size, self.num_output_capsules, self.feature_dim)
        struct_capsules = self.W_struct(struct_proj).view(batch_size, self.num_output_capsules, self.feature_dim)
        
        primary_capsules = seq_capsules + struct_capsules
        
        b = torch.zeros(batch_size, self.num_output_capsules, self.num_output_capsules, device=seq_feats.device)
        
        for iteration in range(self.num_iterations):
           
            c = F.softmax(b, dim=-1)
            
            s = torch.einsum('bij,bjkd->bikd', c, primary_capsules.unsqueeze(1))
            
            v = self._squash(s)
            
            if iteration < self.num_iterations - 1:
                b = b + torch.einsum('bikd,bjkd->bij', s, primary_capsules.unsqueeze(1))
        
        fused_feats = v.mean(dim=1)  
        
        return fused_feats
    
    def _squash(self, x: torch.Tensor) -> torch.Tensor:
        
        norm = torch.norm(x, dim=-1, keepdim=True)
        scale = norm / (1 + norm**2)
        return scale * x

class DeepNABindMM(nn.Module):
   
    def __init__(self, 
                 hidden_dim: int = 512,
                 num_classes: int = 3,
                 esm_dim: int = 2560,
                 dropout: float = 0.3,
                 temperature: float = 0.07):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.temperature = temperature
        
        self.graph_processor = HierarchicalGraphProcessor(
            hidden_dim=hidden_dim,
            atomic_dim=37,
            residue_dim=esm_dim,
            motif_dim=esm_dim
        )
        
        self.seq_processor = BindingSiteAwareSeqModel(
            input_dim=esm_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            dropout=dropout
        )
        
        self.fusion = DynamicRoutingFusion(
            feature_dim=hidden_dim,
            num_iterations=3
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes)
        )
       
        self.binding_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, 
                atomic_graph: Dict,
                residue_graph: Dict,
                motif_graph: Optional[Dict],
                seq_embeddings: torch.Tensor,
                seq_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
       
        seq_feats, seq_residue_feats, binding_logits, binding_probs = self.seq_processor(
            seq_embeddings, seq_lengths
        )
        
        struct_feats = self.graph_processor(atomic_graph, residue_graph, motif_graph)
        
        fused_feats = self.fusion(seq_feats, struct_feats)
        
        if fused_feats.dim() > 2:
            fused_feats = fused_feats.mean(dim=1)
        
        logits = self.classifier(fused_feats)
        
        if logits.dim() > 2:
            logits = logits.mean(dim=1)
        
        return logits, seq_feats, seq_residue_feats, struct_feats, binding_logits, binding_probs
    
    def compute_loss(self,
                    logits: torch.Tensor,
                    labels: torch.Tensor,
                    seq_feats: torch.Tensor,
                    struct_feats: torch.Tensor,
                    binding_logits: Optional[torch.Tensor] = None,
                    binding_labels: Optional[torch.Tensor] = None,
                    binding_mask: Optional[torch.Tensor] = None,
                    lambda1: float = 1.0,
                    lambda2: float = 0.3,
                    lambda3: float = 0.1) -> Dict[str, torch.Tensor]:
        
        losses = {}
        
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, device=logits.device, dtype=torch.long)
        else:
           
            labels = labels.to(logits.device)
        
        if labels.dim() > 1:
            labels = labels.squeeze()
       
        if labels.numel() > labels.shape[0]:
            labels = labels.view(-1)
        
        assert logits.shape[0] == labels.shape[0], \
            f"Batch size mismatch: logits {logits.shape[0]}, labels {labels.shape[0]}"
        
        losses['cls'] = F.cross_entropy(logits, labels)
        
        if (
            binding_labels is not None
            and binding_logits is not None
            and binding_mask is not None
        ):
            binding_logits = binding_logits.squeeze(-1)
            binding_labels = binding_labels.to(logits.device, dtype=torch.float32)
            binding_mask = binding_mask.to(logits.device, dtype=torch.bool)
            
            valid_logits = binding_logits[binding_mask]
            valid_labels = binding_labels[binding_mask]
            
            if valid_logits.numel() == 0:
                losses['bind'] = torch.tensor(0.0, device=logits.device)
            else:
               
                n_pos = valid_labels.sum()
                n_neg = valid_labels.numel() - n_pos
                if n_pos > 0 and n_neg > 0:
                    pos_weight = (n_neg / n_pos).clamp(min=1.0, max=20.0)
                    losses['bind'] = F.binary_cross_entropy_with_logits(
                        valid_logits, valid_labels, pos_weight=pos_weight
                    )
                else:
                    losses['bind'] = F.binary_cross_entropy_with_logits(
                        valid_logits, valid_labels
                    )
        else:
            losses['bind'] = torch.tensor(0.0, device=logits.device)
        
        seq_feats_norm = F.normalize(seq_feats, dim=-1)
        struct_feats_norm = F.normalize(struct_feats, dim=-1)
        
        similarity = torch.matmul(seq_feats_norm, struct_feats_norm.T) / self.temperature
        eye_mask = torch.eye(seq_feats.shape[0], device=logits.device)
        
        pos_sim = similarity * eye_mask
        neg_sim = similarity * (1 - eye_mask)
        
        exp_pos = torch.exp(pos_sim.sum(dim=1))
        exp_neg = torch.exp(neg_sim).sum(dim=1)
        losses['contrast'] = -torch.log(exp_pos / (exp_pos + exp_neg + 1e-8)).mean()
        
        losses['total'] = lambda1 * losses['cls'] + lambda2 * losses['bind'] + lambda3 * losses['contrast']
        
        return losses