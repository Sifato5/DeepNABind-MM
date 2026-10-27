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
        
    @staticmethod
    def _graph_mapping(graph, batch_size, device):
        if graph is None:
            return None, None
        batch = graph.batch if hasattr(graph, 'batch') else torch.zeros(graph.x.size(0), dtype=torch.long, device=device)
        if hasattr(graph, 'original_graph_indices'):
            mapping = graph.original_graph_indices.to(device=device, dtype=torch.long)
        else:
            count = int(batch.max().item()) + 1 if batch.numel() else 0
            if count != batch_size:
                raise ValueError("Optional graphs are missing from a batch. Use aligned_collate_fn in cross_validate.py (or supply original_graph_indices).")
            mapping = torch.arange(batch_size, device=device)
        if mapping.numel() and (mapping.min() < 0 or mapping.max() >= batch_size):
            raise ValueError("Graph-to-protein mapping is out of range")
        if mapping.unique().numel() != mapping.numel():
            raise ValueError("Duplicate graph-to-protein indices")
        return batch, mapping

    def forward(self, atomic_graph, residue_graph, motif_graph):
        if residue_graph is None:
            raise ValueError("Residue graphs are required for every protein")
        x_r = self.residue_proj(residue_graph.x)
        pos_r = residue_graph.pos
        batch_r = residue_graph.batch if hasattr(residue_graph, 'batch') else torch.zeros(x_r.size(0), dtype=torch.long, device=x_r.device)
        batch_size = int(residue_graph.num_graphs) if hasattr(residue_graph, 'num_graphs') else 1
        if batch_size < 1:
            raise ValueError("Empty residue graph batch")
        if hasattr(residue_graph, 'original_graph_indices'):
            expected = torch.arange(batch_size, device=x_r.device)
            if not torch.equal(residue_graph.original_graph_indices.to(x_r.device), expected):
                raise ValueError("Residue graphs must occur in original protein order")
        atomic_feats = x_r.new_zeros(batch_size, self.hidden_dim)
        motif_feats = x_r.new_zeros(batch_size, self.hidden_dim)
        if atomic_graph is not None:
            batch_a, mapping_a = self._graph_mapping(atomic_graph, batch_size, x_r.device)
            x_a = self.atomic_proj(atomic_graph.x)
            pos_a = atomic_graph.pos
            for layer in self.atomic_layers:
                x_a, pos_a = layer(x_a, pos_a, atomic_graph.edge_index, atomic_graph.edge_attr)
            pooled_a = global_mean_pool(x_a, batch_a, size=len(mapping_a))
            atomic_feats = atomic_feats.index_copy(0, mapping_a, self.atom_pool(pooled_a))
           
            for local_id, original_id in enumerate(mapping_a.tolist()):
                r_idx = (batch_r == original_id).nonzero(as_tuple=True)[0]
                a_idx = (batch_a == local_id).nonzero(as_tuple=True)[0]
                if r_idx.numel() == 0 or a_idx.numel() == 0:
                    continue
                nearest = torch.cdist(pos_r[r_idx], pos_a[a_idx]).argmin(dim=1)
                atom_keys = x_a[a_idx[nearest]]
                attended, _ = self.atom_to_residue_attn(
                    x_r[r_idx].unsqueeze(0), atom_keys.unsqueeze(0), atom_keys.unsqueeze(0),
                    need_weights=False)
                x_r = x_r.index_add(0, r_idx, attended.squeeze(0))
        for layer in self.residue_layers:
            x_r, pos_r = layer(x_r, pos_r, residue_graph.edge_index, residue_graph.edge_attr)
        residue_feats = self.residue_pool(global_mean_pool(x_r, batch_r, size=batch_size))
        if motif_graph is not None:
            batch_s, mapping_s = self._graph_mapping(motif_graph, batch_size, x_r.device)
            x_s = self.motif_proj(motif_graph.x)
            pos_s = motif_graph.pos
            for local_id, original_id in enumerate(mapping_s.tolist()):
                r_idx = (batch_r == original_id).nonzero(as_tuple=True)[0]
                s_idx = (batch_s == local_id).nonzero(as_tuple=True)[0]
                if r_idx.numel() == 0 or s_idx.numel() == 0:
                    continue
                attended, _ = self.residue_to_motif_attn(
                    x_s[s_idx].unsqueeze(0), x_r[r_idx].unsqueeze(0), x_r[r_idx].unsqueeze(0),
                    need_weights=False)
                x_s = x_s.index_add(0, s_idx, attended.squeeze(0))
            for layer in self.motif_layers:
                x_s, pos_s = layer(x_s, pos_s, motif_graph.edge_index, motif_graph.edge_attr)
            pooled_s = global_mean_pool(x_s, batch_s, size=len(mapping_s))
            motif_feats = motif_feats.index_copy(0, mapping_s, self.motif_pool(pooled_s))
        return residue_feats + atomic_feats + motif_feats

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
    def __init__(
        self,
        feature_dim: int = 512,
        num_iterations: int = 3,
        num_output_capsules: int = 32
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_iterations = num_iterations
        self.num_output_capsules = num_output_capsules

        self.seq_transform = nn.Linear(feature_dim, feature_dim)
        self.struct_transform = nn.Linear(feature_dim, feature_dim)

        self.W_seq = nn.Linear(
            feature_dim,
            feature_dim * num_output_capsules
        )

        self.W_struct = nn.Linear(
            feature_dim,
            feature_dim * num_output_capsules
        )

        self.cross_attn_seq = nn.MultiheadAttention(
            feature_dim,
            num_heads=8,
            batch_first=True
        )

        self.cross_attn_struct = nn.MultiheadAttention(
            feature_dim,
            num_heads=8,
            batch_first=True
        )

        self.layer_norm_seq = nn.LayerNorm(feature_dim)
        self.layer_norm_struct = nn.LayerNorm(feature_dim)

    def forward(
        self,
        seq_feats: torch.Tensor,
        struct_feats: torch.Tensor
    ) -> torch.Tensor:

        B, D = seq_feats.shape

        assert struct_feats.shape == (B, D)
        assert D == self.feature_dim

        seq_token = seq_feats.unsqueeze(1)       
        struct_token = struct_feats.unsqueeze(1) 

        seq_attended, _ = self.cross_attn_seq(
            query=seq_token,
            key=struct_token,
            value=struct_token
        )

        struct_attended, _ = self.cross_attn_struct(
            query=struct_token,
            key=seq_token,
            value=seq_token
        )

        seq_feats = self.layer_norm_seq(
            seq_feats + seq_attended.squeeze(1)
        )

        struct_feats = self.layer_norm_struct(
            struct_feats + struct_attended.squeeze(1)
        )

        seq_proj = self.seq_transform(seq_feats)
        struct_proj = self.struct_transform(struct_feats)

        C = self.num_output_capsules

        seq_capsules = self.W_seq(seq_proj).reshape(B, C, D)
        struct_capsules = self.W_struct(struct_proj).reshape(B, C, D)

        primary_capsules = seq_capsules + struct_capsules

        b = primary_capsules.new_zeros(B, C, C)

        for iteration in range(self.num_iterations):
            c = F.softmax(b, dim=-1)

            s = torch.einsum(
                'bij,bid->bjd',
                c,
                primary_capsules
            )

            v = self._squash(s)

            if iteration < self.num_iterations - 1:

                agreement = torch.einsum(
                    'bid,bjd->bij',
                    primary_capsules,
                    v
                )

                b = b + agreement

        fused_feats = v.mean(dim=1)  

        return fused_feats

    @staticmethod
    def _squash(
        x: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:

        norm = torch.linalg.vector_norm(
            x,
            dim=-1,
            keepdim=True
        )

        scale = norm / (1.0 + norm.square())

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
        
        labels = torch.as_tensor(labels, device=logits.device, dtype=torch.long).reshape(-1)
        if labels.numel() != logits.shape[0]:
            raise ValueError(f"Expected {logits.shape[0]} labels, received {labels.numel()}")

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
        
        seq_feats_norm = F.normalize(
            seq_feats,
            p=2,
            dim=-1
        )

        struct_feats_norm = F.normalize(
            struct_feats,
            p=2,
            dim=-1
        )

        B = seq_feats_norm.shape[0]

        if B > 1:

            similarity = torch.matmul(
                seq_feats_norm,
                struct_feats_norm.transpose(0, 1)
            ) / self.temperature

            targets = torch.arange(
                B,
                device=similarity.device
            )

            loss_s2t = F.cross_entropy(
                similarity,
                targets
            )

            loss_t2s = F.cross_entropy(
                similarity.transpose(0, 1),
                targets
            )

            losses['contrast'] = 0.5 * (
                loss_s2t + loss_t2s
            )

        else:

            losses['contrast'] = (
                seq_feats_norm.sum()
                + struct_feats_norm.sum()
            ) * 0.0
        
        losses['total'] = lambda1 * losses['cls'] + lambda2 * losses['bind'] + lambda3 * losses['contrast']
        
        return losses