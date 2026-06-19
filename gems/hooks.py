"""GEMS hook implementations for forward-pass intervention.

Provides two primary hook classes:
- ``GEMSHook``: Full GEMS with norm-preserving weighted superposition,
  Gram-Schmidt orthogonalization, Gaussian envelope, and cosine decay.
- ``ActAddHook``: Pure ActAdd baseline (raw vector addition, no constraints).

Both support two operating modes:
- **Generation mode** (default): Operates on last token only, skips prefill.
- **Teacher-forcing mode**: Operates on all token positions.
"""

import math
import torch

from gems.utils import (
    DEFAULT_PEAK_LAYER,
    DEFAULT_SIGMA,
    DEFAULT_COSINE_DECAY_START,
    DEFAULT_COSINE_DECAY_SPAN,
)


class GEMSHook:
    """GEMS forward-pass hook with geometric constraints.

    Performs norm-preserving weighted superposition with optional Gram-Schmidt
    orthogonalization, Gaussian envelope, and cosine decay.

    Args:
        expert_vectors: List of unit expert direction vectors (on GPU).
        intensities: Per-expert intervention intensities.
        layer_idx: Current layer index (for envelope/decay computation).
        do_orthogonalize: Apply Gram-Schmidt orthogonalization.
        envelope_type: ``"gaussian"`` (default, paper Figure 2),
            ``"uniform"`` (E=1.0, no decay, full intervention at every layer).
        teacher_forcing: If True, operate on all token positions (for PPL
            evaluation). Otherwise only last token (generation).
    """

    def __init__(self, expert_vectors, intensities, layer_idx,
                 do_orthogonalize=True,
                 envelope_type="gaussian",
                 teacher_forcing=False):
        self.expert_vecs = [
            v.cuda() if v.device.type != "cuda" else v for v in expert_vectors
        ]
        self.intensities = intensities
        self.N = len(intensities)
        self.do_ortho = do_orthogonalize
        self.teacher_forcing = teacher_forcing

        # Gaussian envelope
        if envelope_type == "gaussian":
            self.E = math.exp(
                -((layer_idx - DEFAULT_PEAK_LAYER) ** 2) / (2 * DEFAULT_SIGMA ** 2)
            )
        elif envelope_type == "uniform":
            self.E = 1.0
        else:
            self.E = 0.0

        # Cosine decay (disabled for uniform envelope)
        if envelope_type == "uniform":
            self.tau = 1.0
        elif layer_idx <= DEFAULT_COSINE_DECAY_START:
            self.tau = 1.0
        else:
            self.tau = math.cos(
                (math.pi / 2.0)
                * ((layer_idx - DEFAULT_COSINE_DECAY_START) / DEFAULT_COSINE_DECAY_SPAN)
            )

    def __call__(self, module, inputs, output):
        dH = output[0] if isinstance(output, tuple) else output
        orig_dtype = dH.dtype

        # Generation mode: skip prefill
        if not self.teacher_forcing and (dH.shape[1] > 1 or self.N == 0):
            return output

        B, S, D = dH.shape
        dH_f = dH.float()

        # NaN guard on input (teacher-forcing)
        if self.teacher_forcing and (torch.isnan(dH_f).any() or torch.isinf(dH_f).any()):
            dH_f = torch.nan_to_num(dH_f, nan=0.0, posinf=0.0, neginf=0.0)

        norm_d = torch.norm(dH_f, dim=-1, keepdim=True) + 1e-6

        u = dH_f / norm_d

        # Flatten for teacher-forcing (process all B*S positions)
        if self.teacher_forcing:
            u_flat = u.reshape(-1, D)
            dH_ref = dH_f.reshape(-1, D)
            norm_ref = norm_d.reshape(-1, 1)
        else:
            u_flat = u
            dH_ref = dH_f
            norm_ref = norm_d

        # Gram-Schmidt orthogonalization
        if self.do_ortho:
            dirs = []
            for v in self.expert_vecs:
                vp = v.clone()
                vp = vp - torch.sum(v * u_flat, dim=-1, keepdim=True) * u_flat
                for ep in dirs:
                    vp = vp - torch.sum(v * ep, dim=-1, keepdim=True) * ep
                vp_n = torch.norm(vp, dim=-1, keepdim=True) + 1e-6
                dirs.append(vp / vp_n)
        else:
            if self.teacher_forcing:
                dirs = [
                    v.unsqueeze(0).expand(u_flat.shape[0], -1)
                    for v in self.expert_vecs
                ]
            else:
                dirs = [
                    v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
                    for v in self.expert_vecs
                ]

        # Effective intensities with cosine decay
        eff_e = [e * self.tau for e in self.intensities]

        # Unit sphere constraint: w_base^2 + sum(w_i^2) = 1
        w_base = math.sqrt(max(0.0, 1.0 - sum(e ** 2 for e in eff_e)))
        H_t = w_base * u_flat
        for dv, ee in zip(dirs, eff_e):
            H_t += ee * dv
        H_t *= norm_ref

        # Envelope interpolation
        H_i = (1.0 - self.E) * dH_ref + self.E * H_t

        # Renormalize to original norm
        H_o = H_i / (torch.norm(H_i, dim=-1, keepdim=True) + 1e-8) * norm_ref

        # Reshape back (teacher-forcing)
        if self.teacher_forcing:
            H_o = H_o.reshape(B, S, D)

        # NaN guard on output
        if torch.isnan(H_o).any() or torch.isinf(H_o).any():
            H_o = torch.nan_to_num(H_o, nan=0.0, posinf=0.0, neginf=0.0)

        if self.teacher_forcing:
            dH[:, :, :].copy_(H_o.to(dtype=orig_dtype))
            return output
        else:
            return (H_o.to(dtype=orig_dtype),) + output[1:] \
                if isinstance(output, tuple) else H_o.to(dtype=orig_dtype)


class ActAddHook:
    """Pure ActAdd hook: raw vector addition without any constraints.

    Used as the baseline for comparison with GEMS.

    Args:
        raw_vectors: List of raw (un-normalized) diff vectors.
        intensities: Per-vector intervention intensities.
        layer_idx: Current layer index (unused, kept for API consistency).
        teacher_forcing: If True, operate on all token positions.
    """

    def __init__(self, raw_vectors, intensities, layer_idx,
                 teacher_forcing=False):
        self.raw_vecs = [
            v.cuda() if v.device.type != "cuda" else v for v in raw_vectors
        ]
        self.intensities = intensities
        self.N = len(intensities)
        self.teacher_forcing = teacher_forcing

    def __call__(self, module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        orig_dtype = h.dtype

        if not self.teacher_forcing and (h.shape[1] > 1 or self.N == 0):
            return output

        h_f = h.float()
        delta = torch.zeros_like(h_f)
        for v, alpha in zip(self.raw_vecs, self.intensities):
            delta += alpha * v
        h_f = h_f + delta

        if torch.isnan(h_f).any() or torch.isinf(h_f).any():
            h_f = torch.nan_to_num(h_f, nan=0.0, posinf=0.0, neginf=0.0)

        if self.teacher_forcing:
            h[:, :, :].copy_(h_f.to(dtype=orig_dtype))
            return output
        else:
            return (h_f.to(dtype=orig_dtype),) + output[1:] \
                if isinstance(output, tuple) else h_f.to(dtype=orig_dtype)
