"""
Lightweight per-residue transformer for backbone dihedral prediction.

Predicts, for each residue, the dominant conformer's (phi, psi) bin and omega
cis/trans class from neighbour-augmented per-residue features.

Design notes:
- **Ring size is injected as a global feature.** Backbone dihedrals — omega
  especially — are strongly ring-size dependent (strained cyclic tetrapeptides
  favour cis amides; larger rings favour trans). The model gets n_res (from the
  mask) appended to every residue's input.
- **No absolute positional encoding.** A head-to-tail macrocycle has no canonical
  start residue, so absolute positions are meaningless; local sequence order is
  carried by the cyclic neighbour augmentation, and longer-range context by
  attention over residue content. This makes the model equivariant to the
  (arbitrary) ring start, which is the correct symmetry.
"""

import torch
from torch import nn

from .residues import MAX_CHI, OMEGA_BINS, PHI_PSI_BINS

# Ring-size scale for the injected n_res feature (n_res / RING_NORM). This is a
# soft normaliser to keep the feature ~O(1), NOT a cap: larger macrocycles map to
# values >1 and are handled fine (the transformer has no fixed-length assumption).
# Set to comfortably cover the common large-macrocycle range (>10 residues).
RING_NORM = 30.0


class DihedralPredictor(nn.Module):
    """Transformer encoder over residues with phi/psi/omega classification heads."""

    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        phi_psi_bins: int = PHI_PSI_BINS,
        omega_bins: int = OMEGA_BINS,
    ):
        """
        Params:
            in_features: int : per-residue augmented feature dim (before ring size)
            d_model: int : transformer hidden width
            n_layers: int : number of encoder layers
            n_heads: int : attention heads
            dropout: float : dropout probability
            phi_psi_bins: int : number of phi/psi classification bins
            omega_bins: int : number of omega bins (cis/trans)
        Returns:
            None
        """
        super().__init__()
        self.input = nn.Linear(in_features + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.phi_head = nn.Linear(d_model, phi_psi_bins)
        self.psi_head = nn.Linear(d_model, phi_psi_bins)
        self.omega_head = nn.Linear(d_model, omega_bins)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Params:
            x: torch.Tensor : (B, L, F) augmented per-residue features
            mask: torch.Tensor : (B, L) bool, True for real residues
        Returns:
            tuple of logits (phi (B,L,phi_psi_bins), psi (...), omega (B,L,omega_bins))
        """
        ring = mask.sum(dim=1, keepdim=True).float() / RING_NORM  # (B, 1)
        ringfeat = ring.unsqueeze(-1).expand(-1, x.shape[1], 1)  # (B, L, 1)
        h = self.input(torch.cat([x, ringfeat], dim=-1))
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.phi_head(h), self.psi_head(h), self.omega_head(h)


class ChiPredictor(nn.Module):
    """Separate transformer for side-chain chi prediction (issue #20, Step 8).

    Deliberately a standalone model — NOT chi heads bolted onto DihedralPredictor —
    so training chi cannot regress backbone-prediction fidelity. Same architecture
    and per-residue inputs as the backbone model; predicts every chi slot
    (chi1..max_chi) per residue as an independent circular-bin classification.
    Residues with fewer chi (or none) are handled by the per-(residue, slot) mask
    at train/eval time. At seeding the predicted chi are set with SetDihedralDeg.
    """

    def __init__(
        self,
        in_features: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        phi_psi_bins: int = PHI_PSI_BINS,
        max_chi: int = MAX_CHI,
    ):
        """
        Params:
            in_features: int : per-residue augmented feature dim (before ring size)
            d_model: int : transformer hidden width
            n_layers: int : number of encoder layers
            n_heads: int : attention heads
            dropout: float : dropout probability
            phi_psi_bins: int : number of circular bins per chi
            max_chi: int : chi slots per residue
        Returns:
            None
        """
        super().__init__()
        self.max_chi = max_chi
        self.phi_psi_bins = phi_psi_bins
        self.input = nn.Linear(in_features + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.chi_head = nn.Linear(d_model, max_chi * phi_psi_bins)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Params:
            x: torch.Tensor : (B, L, F) augmented per-residue features
            mask: torch.Tensor : (B, L) bool, True for real residues
        Returns:
            chi logits, shape (B, L, max_chi, phi_psi_bins)
        """
        ring = mask.sum(dim=1, keepdim=True).float() / RING_NORM
        ringfeat = ring.unsqueeze(-1).expand(-1, x.shape[1], 1)
        h = self.input(torch.cat([x, ringfeat], dim=-1))
        h = self.encoder(h, src_key_padding_mask=~mask)
        b, n_len, _ = h.shape
        return self.chi_head(h).reshape(b, n_len, self.max_chi, self.phi_psi_bins)
