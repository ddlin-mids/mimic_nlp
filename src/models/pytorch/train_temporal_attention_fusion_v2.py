"""
Temporal Attention Fusion Model v2 for 30-day Readmission Prediction

This model implements sophisticated temporal attention over:
1. EHR sequences (day-level) - Transformer layers on top of raw features
2. Radiology reports (temporal) - Transformer with time-aware attention
3. Discharge summary - Static embedding

Key Features:
- 256-dim common projection space
- Learnable modality weights for interpretability
- Differential learning rates (1e-5 for reused, 1e-4 for new layers)
- Warmup + cosine decay LR schedule
- Ablation support for systematic evaluation

Architecture:
    EHR (days × 6240) → Project → Transformer → Attn Pool → 256-dim
    Radiology (reports × 768) → Project → Transformer → Attn Pool → 256-dim
    Discharge (768) → Project → 256-dim
         ↓
    Weighted Gated Fusion (learnable weights)
         ↓
    Classifier → Readmission

Usage:
    python train_temporal_attention_fusion_v2.py --ablation full
    python train_temporal_attention_fusion_v2.py --ablation ehr_only
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, precision_recall_curve
)
from transformers import get_cosine_schedule_with_warmup
import mlflow


# Configure logging
class FlushHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[FlushHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Attention Pooling
# =============================================================================
class AttentionPooling(nn.Module):
    """Learnable attention pooling over sequence dimension."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - 1 for valid, 0 for padding
        Returns:
            pooled: (batch, hidden_dim)
            weights: (batch, seq_len)
        """
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)  # (batch, seq_len)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, hidden_dim)
        return pooled, weights


# =============================================================================
# Modality Encoders
# =============================================================================
class EHRTemporalEncoder(nn.Module):
    """Temporal encoder for EHR sequences with transformer layers."""
    
    def __init__(
        self,
        input_dim: int,
        proj_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.ln_input = nn.LayerNorm(proj_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(proj_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.ln_input(h)
        h = self.dropout(h)
        key_padding_mask = (mask == 0)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        pooled, weights = self.attn_pool(h, mask)
        return pooled, weights


class RadiologyTemporalEncoder(nn.Module):
    """Temporal encoder for radiology reports with time-aware attention."""
    
    def __init__(
        self,
        input_dim: int = 768,
        proj_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_temporal_encoding: bool = True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.ln_input = nn.LayerNorm(proj_dim)
        self.use_temporal_encoding = use_temporal_encoding
        
        if use_temporal_encoding:
            self.time_proj = nn.Sequential(
                nn.Linear(1, proj_dim // 4),
                nn.GELU(),
                nn.Linear(proj_dim // 4, proj_dim),
            )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=proj_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(proj_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        hours_to_discharge: torch.Tensor,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.input_proj(x)
        h = self.ln_input(h)
        
        if self.use_temporal_encoding:
            hours_norm = hours_to_discharge.unsqueeze(-1) / 500.0
            time_embed = self.time_proj(hours_norm)
            h = h + time_embed
        
        h = self.dropout(h)
        key_padding_mask = (mask == 0)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        pooled, weights = self.attn_pool(h, mask)
        return pooled, weights


class DischargeEncoder(nn.Module):
    """Simple projection for discharge summary embedding."""
    
    def __init__(self, input_dim: int = 768, proj_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =============================================================================
# Gated Fusion with Learnable Weights
# =============================================================================
class WeightedGatedFusion(nn.Module):
    """Gated fusion with explicit learnable modality weights."""
    
    def __init__(self, proj_dim: int = 256, n_modalities: int = 3, dropout: float = 0.3):
        super().__init__()
        self.modality_weights_raw = nn.Parameter(torch.zeros(n_modalities))
        self.gate_ehr_rad = nn.Linear(proj_dim * 2, proj_dim)
        self.gate_combined_disch = nn.Linear(proj_dim * 2, proj_dim)
        self.ln = nn.LayerNorm(proj_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        h_ehr: torch.Tensor,
        h_rad: torch.Tensor,
        h_disch: torch.Tensor,
        has_radiology: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.modality_weights_raw, dim=0)
        w_ehr, w_rad, w_disch = weights[0], weights[1], weights[2]
        
        if has_radiology is not None:
            has_rad = has_radiology.float().unsqueeze(-1)
            eff_w_rad = w_rad * has_rad.squeeze(-1)
            total = w_ehr + eff_w_rad + w_disch
            weighted = (
                (w_ehr / total).unsqueeze(-1) * h_ehr +
                (eff_w_rad / total).unsqueeze(-1) * h_rad +
                (w_disch / total).unsqueeze(-1) * h_disch
            )
        else:
            weighted = w_ehr * h_ehr + w_rad * h_rad + w_disch * h_disch
        
        gate1 = torch.sigmoid(self.gate_ehr_rad(torch.cat([h_ehr, h_rad], dim=-1)))
        h_ehr_rad = gate1 * h_ehr + (1 - gate1) * h_rad
        
        gate2 = torch.sigmoid(self.gate_combined_disch(torch.cat([h_ehr_rad, h_disch], dim=-1)))
        h_gated = gate2 * h_ehr_rad + (1 - gate2) * h_disch
        
        fused = self.ln(weighted + h_gated)
        fused = self.dropout(fused)
        return fused, weights


# =============================================================================
# Full Model
# =============================================================================
class TemporalAttentionFusionV2(nn.Module):
    """Full temporal attention fusion model with ablation support."""
    
    def __init__(
        self,
        ehr_input_dim: int,
        rad_input_dim: int = 768,
        disch_input_dim: int = 768,
        proj_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_ehr: bool = True,
        use_radiology: bool = True,
        use_discharge: bool = True,
        use_temporal_ehr: bool = True,
        use_temporal_rad: bool = True,
    ):
        super().__init__()
        
        self.use_ehr = use_ehr
        self.use_radiology = use_radiology
        self.use_discharge = use_discharge
        self.use_temporal_ehr = use_temporal_ehr
        self.use_temporal_rad = use_temporal_rad
        self.proj_dim = proj_dim
        self.n_modalities = sum([use_ehr, use_radiology, use_discharge])
        
        if use_ehr:
            if use_temporal_ehr:
                self.ehr_encoder = EHRTemporalEncoder(
                    ehr_input_dim, proj_dim, num_heads, num_layers, dropout
                )
            else:
                self.ehr_encoder = nn.Sequential(
                    nn.Linear(ehr_input_dim, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
        
        if use_radiology:
            if use_temporal_rad:
                self.rad_encoder = RadiologyTemporalEncoder(
                    rad_input_dim, proj_dim, num_heads, num_layers, dropout
                )
            else:
                self.rad_encoder = nn.Sequential(
                    nn.Linear(rad_input_dim, proj_dim),
                    nn.LayerNorm(proj_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
        
        if use_discharge:
            self.disch_encoder = DischargeEncoder(disch_input_dim, proj_dim, dropout)
        
        if self.n_modalities > 1:
            self.fusion = WeightedGatedFusion(proj_dim, self.n_modalities, dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(proj_dim // 2, 1),
        )
    
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Forward pass with batch dict containing all inputs."""
        outputs: dict[str, Any] = {'attn_weights': {}}
        
        # Get device from any tensor in batch (skip non-tensor values like hadm_id list)
        tensor_vals = [v for v in batch.values() if isinstance(v, torch.Tensor)]
        device = tensor_vals[0].device
        batch_size = tensor_vals[0].shape[0]
        
        modality_embeddings = []
        
        # EHR
        if self.use_ehr:
            if 'ehr_seq' in batch and batch['ehr_seq'] is not None:
                ehr_seq = batch['ehr_seq']
                ehr_mask = batch['ehr_mask']
                if self.use_temporal_ehr:
                    h_ehr, ehr_attn = self.ehr_encoder(ehr_seq, ehr_mask)
                    outputs['attn_weights']['ehr'] = ehr_attn
                else:
                    mask_exp = ehr_mask.unsqueeze(-1).float()
                    ehr_pooled = (ehr_seq * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
                    h_ehr = self.ehr_encoder(ehr_pooled)
            else:
                h_ehr = torch.zeros(batch_size, self.proj_dim, device=device)
            modality_embeddings.append(h_ehr)
        
        # Radiology
        if self.use_radiology:
            if 'rad_embed' in batch and batch['rad_embed'] is not None:
                rad_embed = batch['rad_embed']
                rad_mask = batch['rad_mask']
                if self.use_temporal_rad:
                    rad_hours = batch['rad_hours']
                    h_rad, rad_attn = self.rad_encoder(rad_embed, rad_hours, rad_mask)
                    outputs['attn_weights']['radiology'] = rad_attn
                else:
                    mask_exp = rad_mask.unsqueeze(-1).float()
                    rad_pooled = (rad_embed * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1)
                    h_rad = self.rad_encoder(rad_pooled)
            else:
                h_rad = torch.zeros(batch_size, self.proj_dim, device=device)
            modality_embeddings.append(h_rad)
        
        # Discharge
        if self.use_discharge:
            if 'disch_embed' in batch and batch['disch_embed'] is not None:
                h_disch = self.disch_encoder(batch['disch_embed'])
            else:
                h_disch = torch.zeros(batch_size, self.proj_dim, device=device)
            modality_embeddings.append(h_disch)
        
        # Fusion
        if self.n_modalities > 1:
            if self.n_modalities == 3:
                has_rad = batch.get('has_radiology')
                fused, mod_weights = self.fusion(
                    modality_embeddings[0],  # h_ehr
                    modality_embeddings[1],  # h_rad
                    modality_embeddings[2],  # h_disch
                    has_rad
                )
            else:
                m1, m2 = modality_embeddings[0], modality_embeddings[1]
                gate = torch.sigmoid(self.fusion.gate_ehr_rad(torch.cat([m1, m2], dim=-1)))
                fused = gate * m1 + (1 - gate) * m2
                fused = self.fusion.ln(fused)
                mod_weights = F.softmax(self.fusion.modality_weights_raw[:2], dim=0)
            outputs['modality_weights'] = mod_weights
        else:
            fused = modality_embeddings[0]
            outputs['modality_weights'] = torch.tensor([1.0], device=device)
        
        outputs['logits'] = self.classifier(fused)
        return outputs


# =============================================================================
# Dataset
# =============================================================================
class TemporalFusionDataset(Dataset):
    """Dataset for temporal fusion model."""
    
    def __init__(
        self,
        hadm_ids: np.ndarray,
        labels: np.ndarray,
        ehr_data: Optional[dict] = None,
        rad_data: Optional[dict] = None,
        disch_data: Optional[dict] = None,
    ):
        self.hadm_ids = hadm_ids
        self.labels = labels
        self.ehr_data = ehr_data
        self.rad_data = rad_data
        self.disch_data = disch_data
        
        # Get feature dimensions
        self.ehr_feat_dim = ehr_data.get('feat_dim', 6240) if ehr_data else 6240
    
    def __len__(self) -> int:
        return len(self.hadm_ids)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        hadm_id = int(self.hadm_ids[idx])
        label = float(self.labels[idx])
        
        sample: dict[str, Any] = {
            'hadm_id': hadm_id,
            'label': torch.tensor(label, dtype=torch.float32),
        }
        
        # EHR
        if self.ehr_data is not None:
            if hadm_id in self.ehr_data:
                sample['ehr_seq'] = torch.tensor(
                    self.ehr_data[hadm_id]['sequence'], dtype=torch.float32
                )
                sample['ehr_mask'] = torch.tensor(
                    self.ehr_data[hadm_id]['mask'], dtype=torch.float32
                )
            else:
                sample['ehr_seq'] = torch.zeros(100, self.ehr_feat_dim, dtype=torch.float32)
                sample['ehr_mask'] = torch.zeros(100, dtype=torch.float32)
        
        # Radiology
        if self.rad_data is not None:
            if hadm_id in self.rad_data:
                sample['rad_embed'] = torch.tensor(
                    self.rad_data[hadm_id]['embeddings'], dtype=torch.float32
                )
                sample['rad_hours'] = torch.tensor(
                    self.rad_data[hadm_id]['hours'], dtype=torch.float32
                )
                sample['rad_mask'] = torch.tensor(
                    self.rad_data[hadm_id]['mask'], dtype=torch.float32
                )
                sample['has_radiology'] = torch.tensor(1.0, dtype=torch.float32)
            else:
                sample['rad_embed'] = torch.zeros(20, 768, dtype=torch.float32)
                sample['rad_hours'] = torch.zeros(20, dtype=torch.float32)
                sample['rad_mask'] = torch.zeros(20, dtype=torch.float32)
                sample['has_radiology'] = torch.tensor(0.0, dtype=torch.float32)
        
        # Discharge
        if self.disch_data is not None:
            if hadm_id in self.disch_data:
                sample['disch_embed'] = torch.tensor(
                    self.disch_data[hadm_id], dtype=torch.float32
                )
            else:
                sample['disch_embed'] = torch.zeros(768, dtype=torch.float32)
        
        return sample


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    """Custom collate to handle variable-length sequences."""
    result: dict[str, Any] = {}
    result['hadm_id'] = [s['hadm_id'] for s in batch]
    result['label'] = torch.stack([s['label'] for s in batch])
    
    if 'ehr_seq' in batch[0]:
        result['ehr_seq'] = torch.stack([s['ehr_seq'] for s in batch])
        result['ehr_mask'] = torch.stack([s['ehr_mask'] for s in batch])
    
    if 'rad_embed' in batch[0]:
        result['rad_embed'] = torch.stack([s['rad_embed'] for s in batch])
        result['rad_hours'] = torch.stack([s['rad_hours'] for s in batch])
        result['rad_mask'] = torch.stack([s['rad_mask'] for s in batch])
        result['has_radiology'] = torch.stack([s['has_radiology'] for s in batch])
    
    if 'disch_embed' in batch[0]:
        result['disch_embed'] = torch.stack([s['disch_embed'] for s in batch])
    
    return result


# =============================================================================
# Data Loading
# =============================================================================
class TemporalDataLoader:
    """Loads all data for temporal fusion model."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.cohort_path = self.base_dir / "data/interim/readmit_analysis/long_los_cohort.csv"
        self.ehr_temporal_path = self.base_dir / "data/interim/ehr_long_los/temporal/ehr.npz"
        self.rad_temporal_path = self.base_dir / "data/interim/ehr_long_los/temporal/radiology.npz"
        self.disch_path = self.base_dir / "data/interim/embeddings/notes/discharge_summary.npz"
        self.ehr_raw_pkl = self.base_dir / "data/interim/ehr_long_los/ehr_preprocessed_seq_by_day_cat_embedding.pkl"
    
    def load_cohort(self) -> pd.DataFrame:
        cohort = pd.read_csv(self.cohort_path)
        cohort = cohort[cohort['is_cardiorenal_long'] == True].copy()
        return cohort
    
    def load_ehr_data(self, cohort: pd.DataFrame) -> Optional[dict]:
        """Load EHR temporal sequences."""
        if self.ehr_temporal_path.exists():
            logger.info(f"Loading prepared temporal EHR from {self.ehr_temporal_path}")
            data = np.load(self.ehr_temporal_path)
            
            ehr_data: dict = {'feat_dim': int(data['feat_dim'])}
            hadm_ids = data['hadm_ids']
            sequences = data['raw_sequences']
            lengths = data['lengths']
            masks = data['mask']
            
            for i, hid in enumerate(hadm_ids):
                if lengths[i] > 0:
                    ehr_data[int(hid)] = {
                        'sequence': sequences[i],
                        'mask': masks[i],
                        'length': int(lengths[i]),
                    }
            
            logger.info(f"  Loaded {len(ehr_data) - 1} admissions with EHR data")
            return ehr_data
        
        elif self.ehr_raw_pkl.exists():
            logger.info(f"Loading raw EHR sequences from {self.ehr_raw_pkl}")
            with open(self.ehr_raw_pkl, 'rb') as f:
                seq_data = pickle.load(f)
            
            feat_dict = seq_data['feat_dict']
            sample_key = next(iter(feat_dict))
            feat_dim = feat_dict[sample_key].shape[1]
            
            ehr_data = {'feat_dim': feat_dim}
            cohort_copy = cohort.copy()
            cohort_copy['node_name'] = (
                cohort_copy['subject_id'].astype(str) + "_" + 
                cohort_copy['hadm_id'].astype(str)
            )
            name_to_hid = dict(zip(cohort_copy['node_name'], cohort_copy['hadm_id']))
            
            max_days = 100
            for node_name, seq in feat_dict.items():
                if node_name not in name_to_hid:
                    continue
                
                hadm_id = name_to_hid[node_name]
                
                try:
                    seq = np.array(seq, dtype=np.float32)
                except Exception:
                    continue
                
                seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
                
                if seq.shape[0] > max_days:
                    seq = seq[-max_days:]
                
                n_days = seq.shape[0]
                padded_seq = np.zeros((max_days, feat_dim), dtype=np.float32)
                padded_seq[:n_days] = seq
                
                mask = np.zeros(max_days, dtype=np.float32)
                mask[:n_days] = 1.0
                
                ehr_data[int(hadm_id)] = {
                    'sequence': padded_seq,
                    'mask': mask,
                    'length': n_days,
                }
            
            logger.info(f"  Loaded {len(ehr_data) - 1} admissions with EHR data")
            return ehr_data
        
        else:
            logger.warning("No EHR data found!")
            return None
    
    def load_radiology_data(self, cohort: pd.DataFrame) -> Optional[dict]:
        """Load radiology temporal sequences."""
        if self.rad_temporal_path.exists():
            logger.info(f"Loading prepared temporal radiology from {self.rad_temporal_path}")
            data = np.load(self.rad_temporal_path)
            
            rad_data: dict = {}
            hadm_ids = data['hadm_ids']
            embeddings = data['embeddings']
            hours = data['hours_to_discharge']
            masks = data['mask']
            
            for i, hid in enumerate(hadm_ids):
                rad_data[int(hid)] = {
                    'embeddings': embeddings[i],
                    'hours': hours[i],
                    'mask': masks[i],
                }
            
            logger.info(f"  Loaded {len(rad_data)} admissions with radiology")
            return rad_data
        
        logger.warning("Temporal radiology data not found")
        return None
    
    def load_discharge_data(self, cohort: pd.DataFrame) -> Optional[dict]:
        """Load discharge summary embeddings."""
        if not self.disch_path.exists():
            logger.warning(f"Discharge embeddings not found: {self.disch_path}")
            return None
        
        logger.info(f"Loading discharge embeddings from {self.disch_path}")
        data = np.load(self.disch_path)
        
        embeddings = data['embeddings']
        hadm_ids = data['hadm_ids']
        
        disch_data: dict = {}
        for hid, emb in zip(hadm_ids, embeddings):
            disch_data[int(hid)] = emb
        
        logger.info(f"  Loaded {len(disch_data)} discharge embeddings")
        return disch_data
    
    def load_all(self) -> tuple[pd.DataFrame, Optional[dict], Optional[dict], Optional[dict]]:
        """Load all data and return datasets."""
        cohort = self.load_cohort()
        ehr_data = self.load_ehr_data(cohort)
        rad_data = self.load_radiology_data(cohort)
        disch_data = self.load_discharge_data(cohort)
        return cohort, ehr_data, rad_data, disch_data


# =============================================================================
# Training Functions
# =============================================================================
def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Find threshold that maximizes F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0
    )
    best_idx = np.argmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in loader:
        # Move batch to device
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
        labels = batch_gpu['label'].unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(batch_gpu)
        loss = criterion(outputs['logits'], labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> dict[str, Any]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    for batch in loader:
        batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
        labels = batch_gpu['label'].unsqueeze(1)
        
        outputs = model(batch_gpu)
        loss = criterion(outputs['logits'], labels)
        
        probs = torch.sigmoid(outputs['logits']).cpu().numpy()
        
        total_loss += loss.item()
        all_preds.extend(probs.flatten())
        all_targets.extend(labels.cpu().numpy().flatten())
    
    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)
    
    auc = roc_auc_score(all_targets_arr, all_preds_arr)
    auprc = average_precision_score(all_targets_arr, all_preds_arr)
    preds_binary = (all_preds_arr > threshold).astype(int)
    
    return {
        'loss': total_loss / len(loader),
        'auc': float(auc),
        'auprc': float(auprc),
        'f1': float(f1_score(all_targets_arr, preds_binary, zero_division=0.0)),
        'precision': float(precision_score(all_targets_arr, preds_binary, zero_division=0.0)),
        'recall': float(recall_score(all_targets_arr, preds_binary, zero_division=0.0)),
        'accuracy': float(accuracy_score(all_targets_arr, preds_binary)),
        'preds': all_preds_arr,
        'targets': all_targets_arr,
    }


# =============================================================================
# Ablation Configurations
# =============================================================================
ABLATION_CONFIGS: dict[str, dict[str, bool]] = {
    'full': {
        'use_ehr': True, 'use_radiology': True, 'use_discharge': True,
        'use_temporal_ehr': True, 'use_temporal_rad': True,
    },
    'no_temporal_rad': {
        'use_ehr': True, 'use_radiology': True, 'use_discharge': True,
        'use_temporal_ehr': True, 'use_temporal_rad': False,
    },
    'no_temporal_ehr': {
        'use_ehr': True, 'use_radiology': True, 'use_discharge': True,
        'use_temporal_ehr': False, 'use_temporal_rad': True,
    },
    'no_temporal': {
        'use_ehr': True, 'use_radiology': True, 'use_discharge': True,
        'use_temporal_ehr': False, 'use_temporal_rad': False,
    },
    'ehr_only': {
        'use_ehr': True, 'use_radiology': False, 'use_discharge': False,
        'use_temporal_ehr': True, 'use_temporal_rad': False,
    },
    'rad_only': {
        'use_ehr': False, 'use_radiology': True, 'use_discharge': False,
        'use_temporal_ehr': False, 'use_temporal_rad': True,
    },
    'text_only': {
        'use_ehr': False, 'use_radiology': True, 'use_discharge': True,
        'use_temporal_ehr': False, 'use_temporal_rad': True,
    },
    'no_discharge': {
        'use_ehr': True, 'use_radiology': True, 'use_discharge': False,
        'use_temporal_ehr': True, 'use_temporal_rad': True,
    },
    'ehr_disch': {
        'use_ehr': True, 'use_radiology': False, 'use_discharge': True,
        'use_temporal_ehr': True, 'use_temporal_rad': False,
    },
    'ehr_rad': {
        'use_ehr': True, 'use_radiology': True, 'use_discharge': False,
        'use_temporal_ehr': True, 'use_temporal_rad': True,
    },
}


def main() -> None:
    print(">>> Starting Temporal Attention Fusion V2 Training...", flush=True)
    
    parser = argparse.ArgumentParser(description="Temporal Attention Fusion V2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", type=str, default="full", choices=list(ABLATION_CONFIGS.keys()))
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    
    # Clean MLflow state
    if "MLFLOW_RUN_ID" in os.environ:
        del os.environ["MLFLOW_RUN_ID"]
    if mlflow.active_run():
        mlflow.end_run()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Get ablation config
    ablation_cfg = ABLATION_CONFIGS[args.ablation]
    logger.info(f"Ablation: {args.ablation}")
    logger.info(f"Config: {ablation_cfg}")
    
    # Load data
    data_loader = TemporalDataLoader()
    cohort, ehr_data, rad_data, disch_data = data_loader.load_all()
    
    # Filter data based on ablation
    if not ablation_cfg['use_ehr']:
        ehr_data = None
    if not ablation_cfg['use_radiology']:
        rad_data = None
    if not ablation_cfg['use_discharge']:
        disch_data = None
    
    # Get dimensions
    ehr_dim = ehr_data.get('feat_dim', 6240) if ehr_data else 6240
    
    # Split data
    train_mask = cohort['split'] == 'train'
    val_mask = cohort['split'] == 'val'
    test_mask = cohort['split'] == 'test'
    
    train_ids = cohort.loc[train_mask, 'hadm_id'].values
    val_ids = cohort.loc[val_mask, 'hadm_id'].values
    test_ids = cohort.loc[test_mask, 'hadm_id'].values
    
    train_labels = cohort.loc[train_mask, 'readmitted_within_window'].values
    val_labels = cohort.loc[val_mask, 'readmitted_within_window'].values
    test_labels = cohort.loc[test_mask, 'readmitted_within_window'].values
    
    logger.info(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
    logger.info(f"Readmission rate: {train_labels.mean():.3f}")
    
    # Create datasets
    train_ds = TemporalFusionDataset(train_ids, train_labels, ehr_data, rad_data, disch_data)
    val_ds = TemporalFusionDataset(val_ids, val_labels, ehr_data, rad_data, disch_data)
    test_ds = TemporalFusionDataset(test_ids, test_labels, ehr_data, rad_data, disch_data)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Create model
    model = TemporalAttentionFusionV2(
        ehr_input_dim=ehr_dim,
        proj_dim=args.proj_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        **ablation_cfg,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Loss with class weighting
    pos_weight = torch.tensor([(1 - train_labels.mean()) / train_labels.mean()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Pos weight: {pos_weight.item():.2f}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # MLflow tracking
    mlflow.set_experiment("mimic_cardiorenal_temporal_fusion_v2")
    
    best_auc = 0.0
    best_state: Optional[dict] = None
    best_threshold = 0.5
    no_improve = 0
    
    with mlflow.start_run(run_name=f"ablation_{args.ablation}"):
        # Log tags and params
        mlflow.set_tag("ablation", args.ablation)
        mlflow.set_tag("model_type", "temporal_attention_fusion_v2")
        mlflow.log_params({
            "epochs": args.epochs,
            "lr": args.lr,
            "dropout": args.dropout,
            "proj_dim": args.proj_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
            "patience": args.patience,
            "n_params": n_params,
            "ehr_dim": ehr_dim,
        })
        mlflow.log_params({f"ablation_{k}": v for k, v in ablation_cfg.items()})
        
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            logger.info(
                f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | Val AUPRC: {val_metrics['auprc']:.4f}"
            )
            
            # Log metrics to MLflow
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_auc": val_metrics['auc'],
                "val_auprc": val_metrics['auprc'],
                "val_loss": val_metrics['loss'],
                "val_f1": val_metrics['f1'],
            }, step=epoch)
            
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_threshold, best_f1 = find_optimal_threshold(
                    val_metrics['targets'], val_metrics['preds']
                )
                logger.info(f"  ★ New best! Threshold: {best_threshold:.3f}, F1: {best_f1:.3f}")
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and evaluate on test
        if best_state:
            model.load_state_dict(best_state)
            model.to(device)
        
        test_metrics = evaluate(model, test_loader, criterion, device, threshold=best_threshold)
        
        # Log final results
        logger.info("=" * 60)
        logger.info(f"Final Test Results ({args.ablation}):")
        logger.info(f"  AUC:       {test_metrics['auc']:.4f}")
        logger.info(f"  AUPRC:     {test_metrics['auprc']:.4f}")
        logger.info(f"  F1:        {test_metrics['f1']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Threshold: {best_threshold:.3f}")
        logger.info("=" * 60)
        
        # Log final test metrics to MLflow (required metrics per AGENTS.md)
        mlflow.log_metrics({
            "test_auc": test_metrics['auc'],
            "test_auprc": test_metrics['auprc'],
            "test_f1": test_metrics['f1'],
            "test_precision": test_metrics['precision'],
            "test_recall": test_metrics['recall'],
            "test_acc": test_metrics['accuracy'],
            "optimal_threshold": best_threshold,
            "best_val_auc": best_auc,
        })


if __name__ == "__main__":
    main()
