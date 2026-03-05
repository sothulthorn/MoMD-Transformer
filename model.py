"""
MoMD Transformer: Mixture-of-Modality-Diagnosis Transformer.

Architecture:
  - Multi-modal embedding layer (Section 3.2.1)
  - Encoder with multi-channel FFN (Section 3.2.2)
  - Weight-shared classification head (Section 3.2.3)
  - Global Knowledge Transfer module (Section 3.3)
  - Masked Signal Modeling module (Section 3.4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    """Standard FFN: Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, embed_dim, mlp_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (Eq. 9)."""

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MoMDBlock(nn.Module):
    """
    Single MoMD Transformer block (Section 3.2.2).

    Shared MSA + multi-channel FFN (V-FFN, C-FFN, MM-FFN).
    Eqs. 13-14:
        z' = LayerNorm(z + MSA(z))
        z_out = LayerNorm(z' + FFN(z'))
    """

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Multi-channel FFN: three separate FFN paths
        self.v_ffn = FeedForwardNetwork(embed_dim, mlp_dim, dropout)
        self.c_ffn = FeedForwardNetwork(embed_dim, mlp_dim, dropout)
        self.mm_ffn = FeedForwardNetwork(embed_dim, mlp_dim, dropout)

    def forward(self, z, modality):
        """
        Args:
            z: (B, seq_len, embed_dim)
            modality: 'vibration', 'current', or 'both'
        """
        # MSA with pre-norm and residual
        z = z + self.attn(self.norm1(z))

        # Multi-channel FFN selection with pre-norm and residual
        z_normed = self.norm2(z)
        if modality == "vibration":
            z = z + self.v_ffn(z_normed)
        elif modality == "current":
            z = z + self.c_ffn(z_normed)
        else:  # 'both'
            z = z + self.mm_ffn(z_normed)

        return z


class MoMDTransformer(nn.Module):
    """
    Mixture-of-Modality-Diagnosis Transformer (Fig. 2).

    Supports three input modes:
      - Vibration only: x_vib provided, x_cur=None
      - Current only:   x_cur provided, x_vib=None
      - Multi-modal:    both x_vib and x_cur provided

    During training, GKT and MSM modules are activated:
      - GKT: aligns class embeddings from V-only and C-only passes (Eq. 17)
      - MSM: masks patches and reconstructs them (Eqs. 18-19)
    """

    def __init__(
        self,
        num_classes,
        signal_length=2048,
        segment_length=64,
        embed_dim=128,
        mlp_dim=512,
        num_heads=8,
        depth=3,
        dropout=0.2,
    ):
        super().__init__()
        self.signal_length = signal_length
        self.segment_length = segment_length
        self.embed_dim = embed_dim
        self.num_patches = signal_length // segment_length  # N
        self.depth = depth

        # --- Embedding layers (Section 3.2.1) ---
        # Projection matrices W_v, W_c
        self.vib_proj = nn.Linear(segment_length, embed_dim)
        self.cur_proj = nn.Linear(segment_length, embed_dim)

        # Class embeddings (learnable)
        self.cls_vib = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_cur = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embeddings (shared between modalities, Eq. 11)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        # Type embeddings (Eq. 12)
        self.type_embed_vib = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.type_embed_cur = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.embed_dropout = nn.Dropout(dropout)

        # --- Encoder blocks (Section 3.2.2) ---
        self.blocks = nn.ModuleList(
            [MoMDBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # --- Classification head (Section 3.2.3, Eq. 15) ---
        self.cls_head = nn.Linear(embed_dim, num_classes)

        # --- MSM reconstruction head (Section 3.4, Eq. 18) ---
        self.msm_head = nn.Linear(embed_dim, segment_length)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_vib, std=0.02)
        nn.init.trunc_normal_(self.cls_cur, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed_vib, std=0.02)
        nn.init.trunc_normal_(self.type_embed_cur, std=0.02)

        self.apply(self._init_module_weights)

    @staticmethod
    def _init_module_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _patchify(self, x):
        """Split 1D signal into patches. (B, L) -> (B, N, segment_length)"""
        return x.view(x.size(0), self.num_patches, self.segment_length)

    def _embed(self, x_vib=None, x_cur=None, mask_ratio=0.0):
        """
        Construct embedding sequence (Eqs. 10-12).

        Returns:
            z: (B, seq_len, embed_dim)
            mask_info: dict for MSM reconstruction (empty if mask_ratio=0)
        """
        mask_info = {}

        if x_vib is not None and x_cur is not None:
            # --- Multi-modal embedding ---
            B = x_vib.size(0)
            vib_patches = self._patchify(x_vib)  # (B, N, seg)
            cur_patches = self._patchify(x_cur)

            if mask_ratio > 0:
                mask_info = self._apply_mask(vib_patches, cur_patches, mask_ratio)
                vib_patches_input = mask_info["masked_vib"]
                cur_patches_input = mask_info["masked_cur"]
            else:
                vib_patches_input = vib_patches
                cur_patches_input = cur_patches

            # Project (Eq. 10)
            vib_embed = self.vib_proj(vib_patches_input)  # (B, N, d)
            cur_embed = self.cur_proj(cur_patches_input)

            # Prepend class tokens
            cls_v = self.cls_vib.expand(B, -1, -1)
            cls_c = self.cls_cur.expand(B, -1, -1)
            vib_seq = torch.cat([cls_v, vib_embed], dim=1)  # (B, N+1, d)
            cur_seq = torch.cat([cls_c, cur_embed], dim=1)

            # Add position embeddings (shared) + type embeddings (Eq. 12)
            vib_seq = vib_seq + self.pos_embed + self.type_embed_vib
            cur_seq = cur_seq + self.pos_embed + self.type_embed_cur

            z = torch.cat([vib_seq, cur_seq], dim=1)  # (B, 2(N+1), d)

        elif x_vib is not None:
            B = x_vib.size(0)
            vib_patches = self._patchify(x_vib)
            vib_embed = self.vib_proj(vib_patches)
            cls_v = self.cls_vib.expand(B, -1, -1)
            z = torch.cat([cls_v, vib_embed], dim=1)  # (B, N+1, d)
            z = z + self.pos_embed + self.type_embed_vib

        elif x_cur is not None:
            B = x_cur.size(0)
            cur_patches = self._patchify(x_cur)
            cur_embed = self.cur_proj(cur_patches)
            cls_c = self.cls_cur.expand(B, -1, -1)
            z = torch.cat([cls_c, cur_embed], dim=1)  # (B, N+1, d)
            z = z + self.pos_embed + self.type_embed_cur

        else:
            raise ValueError("At least one modality must be provided.")

        z = self.embed_dropout(z)
        return z, mask_info

    def _apply_mask(self, vib_patches, cur_patches, mask_ratio):
        """
        Randomly mask patches by setting them to zero (Section 3.4).

        Returns dict with masked patches, original patches, and mask indices.
        """
        B, N, _seg = vib_patches.shape
        num_mask = max(1, int(N * mask_ratio))
        device = vib_patches.device

        # Generate mask indices per sample
        vib_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        cur_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            v_idx = torch.randperm(N, device=device)[:num_mask]
            c_idx = torch.randperm(N, device=device)[:num_mask]
            vib_mask[b, v_idx] = True
            cur_mask[b, c_idx] = True

        # Apply mask (set to zero)
        masked_vib = vib_patches.clone()
        masked_cur = cur_patches.clone()
        masked_vib[vib_mask] = 0.0
        masked_cur[cur_mask] = 0.0

        return {
            "masked_vib": masked_vib,
            "masked_cur": masked_cur,
            "orig_vib": vib_patches,
            "orig_cur": cur_patches,
            "vib_mask": vib_mask,  # (B, N) bool
            "cur_mask": cur_mask,
        }

    def forward(self, x_vib=None, x_cur=None, mask_ratio=0.0, return_block_cls=False):
        """
        Forward pass.

        Args:
            x_vib: (B, signal_length) vibration signal or None
            x_cur: (B, signal_length) current signal or None
            mask_ratio: float, mask ratio for MSM (0 = no masking)
            return_block_cls: if True, return class embeddings from each block (for GKT)

        Returns:
            dict with:
                'logits': (B, num_classes) classification logits
                'block_cls': list of (B, embed_dim) per block (if return_block_cls)
                'msm_loss': scalar MSM reconstruction loss (if mask_ratio > 0)
        """
        # Determine modality
        if x_vib is not None and x_cur is not None:
            modality = "both"
        elif x_vib is not None:
            modality = "vibration"
        elif x_cur is not None:
            modality = "current"
        else:
            raise ValueError("At least one modality must be provided.")

        # Embedding
        z, mask_info = self._embed(x_vib, x_cur, mask_ratio)

        # Encoder blocks
        block_cls_list = []
        for block in self.blocks:
            z = block(z, modality)
            if return_block_cls:
                block_cls_list.append(z[:, 0])  # class embedding at position 0

        z = self.norm(z)

        # Classification (Eq. 15): use 0-th position (class embedding)
        logits = self.cls_head(z[:, 0])

        outputs = {"logits": logits}

        if return_block_cls:
            outputs["block_cls"] = block_cls_list

        # MSM reconstruction (Eq. 18-19)
        if mask_ratio > 0 and mask_info:
            msm_loss = self._compute_msm_loss(z, mask_info)
            outputs["msm_loss"] = msm_loss

        return outputs

    def _compute_msm_loss(self, z_final, mask_info):
        """
        Compute masked signal modeling loss (Eq. 19).

        Reconstructs masked patches from the final encoder output features
        at their respective positions and computes MSE.
        """
        N = self.num_patches
        vib_mask = mask_info["vib_mask"]  # (B, N)
        cur_mask = mask_info["cur_mask"]
        orig_vib = mask_info["orig_vib"]  # (B, N, seg)
        orig_cur = mask_info["orig_cur"]

        # In multi-modal sequence: [cls_v, v1..vN, cls_c, c1..cN]
        # Vibration patch positions: 1 to N
        # Current patch positions: N+2 to 2N+1
        vib_features = z_final[:, 1 : N + 1, :]       # (B, N, d)
        cur_features = z_final[:, N + 2 : 2 * N + 2, :]  # (B, N, d)

        # Reconstruct all patches, then compute loss only on masked ones
        vib_recon = self.msm_head(vib_features)  # (B, N, seg)
        cur_recon = self.msm_head(cur_features)

        # MSE on masked patches only
        vib_loss = F.mse_loss(
            vib_recon[vib_mask], orig_vib[vib_mask], reduction="mean"
        )
        cur_loss = F.mse_loss(
            cur_recon[cur_mask], orig_cur[cur_mask], reduction="mean"
        )

        return (vib_loss + cur_loss) / 2

    @staticmethod
    def compute_gkt_loss(block_cls_vib, block_cls_cur):
        """
        Global Knowledge Transfer loss (Eq. 17).

        L_gkt = (1/L) * sum_i ||z_V,i^0 - z_C,i^0||_2^2

        Args:
            block_cls_vib: list of L tensors, each (B, embed_dim)
            block_cls_cur: list of L tensors, each (B, embed_dim)
        """
        L = len(block_cls_vib)
        loss = 0.0
        for z_v, z_c in zip(block_cls_vib, block_cls_cur):
            loss = loss + torch.mean(torch.sum((z_v - z_c) ** 2, dim=-1))
        return loss / L
