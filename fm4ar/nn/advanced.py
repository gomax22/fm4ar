import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# -----------------------------
# RoPE utility
# -----------------------------
class RotaryPositionalEmbeddings(nn.Module):

  def __init__(self, head_dim: int, base: int = 10_000):

    super().__init__()
    self.base = base
    self.head_dim = head_dim
    self.cos_cached = None
    self.sin_cached = None

  def _build_cache(self, x: torch.Tensor):

    if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
      return

    seq_len = x.shape[0]
    theta = 1. / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)).to(x.device) # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)
    seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device) #Position Index -> [0,1,2...seq-1]
    idx_theta = torch.einsum('n,d->nd', seq_idx, theta)  #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1) # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]
    self.cos_cached = idx_theta2.cos()[:, None, None, :] #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d]
    self.sin_cached = idx_theta2.sin()[:, None, None, :] #cache [sinTHETA_1, sinTHETA_2...sinTHETA_d]

  def _neg_half(self, x: torch.Tensor):
    d_2 = self.head_dim // 2 #
    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1) # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]

  def forward(self, x: torch.Tensor):
    self._build_cache(x)
    neg_half_x = self._neg_half(x)
    x_rope = (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]]) # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]
    return x_rope
     

class WavelengthAwareRoPE(nn.Module):
    """
    Rotary Positional Embeddings that operate on *log-scaled physical wavelengths*.

    Expected input:
        positions = log(wavelength / min_wavelength)
    """

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE dimension must be even."
        self.head_dim = head_dim
        self.base = base
        self.cos_cached = None
        self.sin_cached = None

    def _compute_frequencies(self, device):
        """Standard RoPE angular frequencies."""
        return 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, device=device).float() / self.head_dim))

    def _build_cache(self, positions: torch.Tensor):
        """
        Cache cos/sin values for given (log-scaled) wavelength positions.

        Args:
            positions: (B, N) or (N,) tensor of log(wavelength / min_wavelength)
        """
        device = positions.device
        pos_flat = positions.view(-1)
        theta = self._compute_frequencies(device)  # (d/2,)
        idx_theta = torch.einsum('n,d->nd', pos_flat, theta)  # (N, d/2)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)  # (N, d)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]  # (N, 1, 1, d)
        self.sin_cached = idx_theta2.sin()[:, None, None, :]  # (N, 1, 1, d)

    def _neg_half(self, x: torch.Tensor):
        """Rotate feature halves: (x1, x2) → (-x2, x1)"""
        d_2 = self.head_dim // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        """
        Args:
            x: (B, H, N, D)
            positions: (B, N) or (N,) log-scaled wavelength positions

        Returns:
            x_rope: same shape as x, rotated according to wavelength positions
        """
        # Build or refresh cache
        self._build_cache(positions)
        neg_half_x = self._neg_half(x)
        cos = self.cos_cached[:x.shape[0]].to(x.device)
        sin = self.sin_cached[:x.shape[0]].to(x.device)
        return (x * cos) + (neg_half_x * sin)


# -----------------------------
# Dilated MHSA
# -----------------------------
class DilatedMHSABlock(nn.Module): 
    def __init__(self, d_model, n_heads, k=8, dilation=1, 
                 rope=None, use_qk_norm=True, learn_qk_scale=False
    ): 
        super().__init__() 
        self.d_model = d_model 
        self.n_heads = n_heads 
        self.d_head = d_model // n_heads 
        self.k = k 
        self.dilation = dilation 
        self.qkv = nn.Linear(d_model, 3 * d_model) 
        self.out = nn.Linear(d_model, d_model) 
        self.rope = rope 
        self.use_qk_norm = use_qk_norm 
        self.learn_qk_scale = learn_qk_scale 
        
        # Learnable scale for QK-norm 
        if use_qk_norm and learn_qk_scale: 
            self.scale = nn.Parameter(torch.ones(1)) 
        else: 
            self.register_buffer("scale", torch.tensor(1.0 if use_qk_norm else 1.0/self.d_head)) 
            
    def forward(self, x, positions=None): 
        B, N, D = x.shape 
        qkv = self.qkv(x) 
        q, k, v = qkv.chunk(3, dim=-1) 
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2) 
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2) 
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2) 
        
        if self.rope is not None:
            if isinstance(self.rope, RotaryPositionalEmbeddings):
                q = self.rope(q) 
                k = self.rope(k) 
            if isinstance(self.rope, WavelengthAwareRoPE):
                assert positions is not None, "Positions required for WavelengthAwareRoPE"
                pos_rep = positions.repeat_interleave(self.n_heads, dim=0) 
                q = self.rope(q, pos_rep) 
                k = self.rope(k, pos_rep) 
            q = q.view(B, self.n_heads, N, self.d_head) 
            k = k.view(B, self.n_heads, N, self.d_head) 
            
        if self.use_qk_norm: 
            q = q / (q.norm(dim=-1, keepdim=True)+1e-6) 
            k = k / (k.norm(dim=-1, keepdim=True)+1e-6) 
            scale = self.scale # learnable or fixed 
        else: 
            scale = 1.0 / (self.d_head ** 0.5) 
            
        mask = torch.ones((N, N), dtype=torch.bool, device=x.device) 
        idxs = torch.arange(N, device=x.device) 
        for i in range(N): 
            jmin = max(0, i - self.k * self.dilation) 
            jmax = min(N - 1, i + self.k * self.dilation) 
            allowed = idxs[jmin:jmax + 1] 
            allowed = allowed[((allowed - i) % self.dilation) == 0] 
            mask[i, allowed] = False 
                
        attn_scores = (q @ k.transpose(-2, -1)) * scale 
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf")) 
        attn = attn_scores.softmax(dim=-1) 
        out = attn @ v 
        out = out.transpose(1, 2).reshape(B, N, D) 
        return self.out(out)



# -----------------------------
# Dilated MHCA
# -----------------------------
class DilatedMHCABlock(nn.Module):
    def __init__(self, d_model, n_heads, k=8, dilation=1, rope_module=None,
                 use_qk_norm=False, learnable_qk_scale=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k = k
        self.dilation = dilation
        self.rope = rope_module
        self.use_qk_norm = use_qk_norm

        # Separate projections for clarity (Q from x, KV from context)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        if use_qk_norm and learnable_qk_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.tensor(1.0 if use_qk_norm else 1.0 / (self.d_head ** 0.5)))


    def forward(self, x, positions=None, context=None, context_positions=None):
        """
        x: (B, N_q, D)
        context: (B, N_k, D) or None (self-attention)
        positions: (B, N_q, dim_pos)
        context_positions: (B, N_k, dim_pos), optional (if using RoPE)
        """
        B, N_q, D = x.shape
        if context is None:
            context = x
            context_positions = positions
        Bc, N_k, Dc = context.shape

        # projections
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # reshape for multihead
        q = q.view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE (different lengths allowed)
        # Apply RoPE (different lengths allowed)
        if self.rope is not None:
            if isinstance(self.rope, RotaryPositionalEmbeddings):
                q = self.rope(q)
                k = self.rope(k)
            if isinstance(self.rope, WavelengthAwareRoPE):
                assert positions is not None, "Positions required for WavelengthAwareRoPE"

                q = self.rope(q, positions)
                k = self.rope(k, context_positions if context_positions is not None else positions)

            q = q.view(B, self.n_heads, N_q, self.d_head)
            k = k.view(B, self.n_heads, N_k, self.d_head)

        if self.use_qk_norm:
            q = q / (q.norm(dim=-1, keepdim=True)+1e-6)
            k = k / (k.norm(dim=-1, keepdim=True)+1e-6)
            scale = self.scale  # learnable or fixed
        else:
            scale = 1.0 / (self.d_head ** 0.5)

        mask = torch.ones((N_q, N_k), dtype=torch.bool, device=x.device)
        idxs = torch.arange(N_q, device=x.device)
        for i in range(N_q):
            jmin = max(0, i - self.k * self.dilation)
            jmax = min(N_k - 1, i + self.k * self.dilation)
            allowed = idxs[jmin:jmax + 1]
            allowed = allowed[((allowed - i) % self.dilation) == 0]
            mask[i, allowed] = False

        attn_scores = (q @ k.transpose(-2, -1)) * scale
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = attn_scores.softmax(dim=-1)
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N_q, D)
        return self.out(out)
    
# -----------------------------
# FFN Variants
# -----------------------------
class BaseFFN(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model)
        )
    def forward(self, x): return self.net(x)


class GeGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model
        self.w1 = nn.Linear(d_model, hidden_dim * 2)
        self.w2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return F.linear(self.dropout(F.gelu(x1) * x2), self.w2.weight, self.w2.bias)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model
        self.w1 = nn.Linear(d_model, hidden_dim * 2)
        self.w2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return F.linear(self.dropout(F.silu(x1) * x2), self.w2.weight, self.w2.bias)


# -----------------------------
# RMSNorm
# -----------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    def forward(self, x):
        rms = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return x / (rms + self.eps) * self.scale

# -----------------------------
# Patch Embedding with overlap
# -----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, n_channels, d_model, stride=None):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.proj = nn.Linear(patch_size * n_channels, d_model)

    def forward(self, x):
        B, N, C = x.shape
        N_patches = 1 + (N - self.patch_size) // self.stride
        patches = []
        for i in range(N_patches):
            start = i * self.stride
            end = start + self.patch_size
            patch = x[:, start:end, :].reshape(B, -1)
            patches.append(patch)
        x_patched = torch.stack(patches, dim=1)
        return self.proj(x_patched)
    

# -----------------------------
# Patch Embedding with overlap
# -----------------------------
class PatchEmbeddingWithoutOverlap(nn.Module):
    """Converts 1D signal into patch embeddings for Transformer input."""
    def __init__(self, in_channels: int, patch_size: int, emb_dim: int, seq_length: int):
        super().__init__()
        assert seq_length % patch_size == 0, "Signal length must be divisible by patch size."
        self.num_patches = seq_length // patch_size
        self.proj = nn.Conv1d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [B, C, L]
        Returns:
            Tensor of shape [B, num_patches, emb_dim]
        """
        x = self.proj(x)  # [B, emb_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, emb_dim]
        return x


# -----------------------------
# Fusion Modules
# -----------------------------
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_fine, x_coarse):
        fused, _ = self.attn(
            query=x_fine.transpose(0,1),
            key=x_coarse.transpose(0,1),
            value=x_coarse.transpose(0,1)
        )
        fused = fused.transpose(0,1)
        return self.norm(x_fine + fused)


class PoolBroadcastFusion(nn.Module):
    def __init__(self, d_model, fusion_hidden=512):
        super().__init__()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model*2, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_fine, x_coarse):
        pooled = x_coarse.mean(dim=1, keepdim=True)
        broadcast = pooled.repeat(1, x_fine.size(1), 1)
        fused = torch.cat([x_fine, broadcast], dim=-1)
        fused = self.fusion_mlp(fused)
        return self.norm(fused)


class AttentionPoolingFusion(nn.Module):
    def __init__(self, d_model, n_heads=4, n_queries=16, fusion_hidden=512):
        super().__init__()
        self.n_queries = n_queries
        self.query_vectors = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model*2, fusion_hidden),
            nn.GELU(),
            nn.Linear(fusion_hidden, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_fine, x_coarse):
        B, N_fine, D = x_fine.shape
        # 1. Learnable queries
        queries = self.query_vectors.unsqueeze(1).repeat(1, B, 1)  # (n_queries, B, D)
        # 2. Cross-attention
        summary, _ = self.attn(
            query=queries,
            key=x_coarse.transpose(0,1),
            value=x_coarse.transpose(0,1)
        )  # (n_queries, B, D)
        # 3. Broadcast along sequence dimension
        summary_fine = summary.transpose(0,1).unsqueeze(1)           # (B, 1, n_queries, D)
        summary_fine = summary_fine.expand(-1, N_fine, -1, -1)      # (B, N_fine, n_queries, D)
        summary_fine = summary_fine.mean(dim=2)                      # (B, N_fine, D)
        # 4. Fuse
        fused = torch.cat([x_fine, summary_fine], dim=-1)            # (B, N_fine, 2*D)
        fused = self.fusion_mlp(fused)
        return self.norm(fused)

class MultiScalePositionalEmbedding(nn.Module):
    def __init__(self, d_model, patch_sizes, max_patches_per_scale=2048):
        super().__init__()
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.max_patches_per_scale = max_patches_per_scale

        # Learnable positional embedding per scale
        self.scale_embeddings = nn.Parameter(torch.randn(len(patch_sizes), d_model))

        # Optional: per-patch embeddings (up to max_patches)
        self.patch_embeddings = nn.Parameter(torch.randn(len(patch_sizes), max_patches_per_scale, d_model))

    def forward(self, features_per_scale):
        """
        features_per_scale: list of tensors [(B, N1, d_model), (B, N2, d_model), ...]
        returns: concatenated tensor with positional embeddings
        """
        B = features_per_scale[0].size(0)
        out = []
        for i, f in enumerate(features_per_scale):
            N = f.size(1)
            scale_emb = self.scale_embeddings[i].unsqueeze(0).unsqueeze(1)  # (1,1,d_model)
            patch_emb = self.patch_embeddings[i, :N, :].unsqueeze(0)        # (1,N,d_model)
            f = f + scale_emb + patch_emb  # broadcast over batch
            out.append(f)
        return torch.cat(out, dim=1)  # (B, total_patches, d_model)

def get_ffn(ffn_type, d_model, ffn_hidden_mult, ffn_dropout):
    if ffn_type=='base':
        return BaseFFN(d_model, hidden_dim=ffn_hidden_mult*d_model, dropout=ffn_dropout)
    elif ffn_type=='geglu':
        return GeGLUFFN(d_model, hidden_dim=ffn_hidden_mult*d_model, dropout=ffn_dropout)
    else:
        return SwiGLUFFN(d_model, hidden_dim=ffn_hidden_mult*d_model, dropout=ffn_dropout)


def build_rope(head_dim, rope_type="none", base=10_000):
    """
    Factory method to build RoPE module.
    rope_type ∈ {"none", "standard", "wavelength"}
    """
    if rope_type == "none":
        return None
    elif rope_type == "standard":
        return RotaryPositionalEmbeddings(head_dim, base=base)
    elif rope_type == "wavelength":
        return WavelengthAwareRoPE(head_dim, base=base)
    else:
        raise ValueError(f"Unknown rope_type: {rope_type}")

# -----------------------
# Modulation function for AdaLN-Zero
# -----------------------
def modulate(x, shift, scale, unsqueeze=True):
    scale = scale.unsqueeze(1) if unsqueeze else scale
    shift = shift.unsqueeze(1) if unsqueeze else shift
    return x * (1 + scale) + shift

# -----------------------
# Classic Multi-Head Self-Attention with optional RoPE + QK-norm
# -----------------------
class ClassicMHSABlock(nn.Module): 
    def __init__(self, d_model, n_heads, rope_module=None, 
                 use_qk_norm=False, learnable_qk_scale=False): 
        super().__init__() 
        self.d_model = d_model 
        self.n_heads = n_heads 
        self.d_head = d_model // n_heads 
        self.rope = rope_module 
        self.use_qk_norm = use_qk_norm 
        self.qkv = nn.Linear(d_model, 3*d_model) 
        self.out = nn.Linear(d_model, d_model) 
        if use_qk_norm and learnable_qk_scale: 
            self.scale = nn.Parameter(torch.ones(1)) 
        else: 
            self.register_buffer("scale", torch.tensor(1.0 if use_qk_norm else 1.0/self.d_head)) 
            
    def forward(self, x, positions=None): 
        B, N, D = x.shape 
        qkv = self.qkv(x) 
        q, k, v = qkv.chunk(3, dim=-1) 
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2) 
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2) 
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2) 
        if self.rope is not None:
            if isinstance(self.rope, RotaryPositionalEmbeddings):
                q = self.rope(q) 
                k = self.rope(k) 
            if isinstance(self.rope, WavelengthAwareRoPE):
                assert positions is not None, "Positions required for WavelengthAwareRoPE"
                q = self.rope(q, positions) 
                k = self.rope(k, positions) 

            q = q.view(B, self.n_heads, N, self.d_head) 
            k = k.view(B, self.n_heads, N, self.d_head) 
            
        if self.use_qk_norm: 
            q = q / (q.norm(dim=-1, keepdim=True)+1e-6) 
            k = k / (k.norm(dim=-1, keepdim=True)+1e-6) 
            scale = self.scale 
        else: 
            scale = 1.0 / (self.d_head ** 0.5) 
        
        attn = torch.matmul(q, k.transpose(-2,-1)) * scale 
        attn = F.softmax(attn, dim=-1) 
        out = torch.matmul(attn, v) 
        out = out.transpose(1,2).reshape(B,N,D) 
        return self.out(out)


# -----------------------
# Classic Multi-Head Cross-Attention with optional RoPE + QK-norm
# -----------------------
class ClassicMHCABlock(nn.Module):
    def __init__(self, d_model, n_heads, rope_module=None,
                 use_qk_norm=False, learnable_qk_scale=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope = rope_module
        self.use_qk_norm = use_qk_norm

        # Separate projections for clarity (Q from x, KV from context)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        if use_qk_norm and learnable_qk_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("scale", torch.tensor(1.0 if use_qk_norm else 1.0 / (self.d_head ** 0.5)))

    def forward(self, x, positions=None, context=None, context_positions=None):
        """
        x: (B, N_q, D)
        context: (B, N_k, D) or None (self-attention)
        positions: (B, N_q, dim_pos)
        context_positions: (B, N_k, dim_pos), optional (if using RoPE)
        """
        B, N_q, D = x.shape
        if context is None:
            context = x
            context_positions = positions
        Bc, N_k, Dc = context.shape

        # projections
        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # reshape for multihead
        q = q.view(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N_k, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE (different lengths allowed)
        if self.rope is not None:
            if isinstance(self.rope, RotaryPositionalEmbeddings):
                q = self.rope(q)
                k = self.rope(k)
            if isinstance(self.rope, WavelengthAwareRoPE):
                assert positions is not None, "Positions required for WavelengthAwareRoPE"
                q = self.rope(q, positions)
                k = self.rope(k, context_positions if context_positions is not None else positions)

            q = q.view(B, self.n_heads, N_q, self.d_head)
            k = k.view(B, self.n_heads, N_k, self.d_head)

        if self.use_qk_norm:
            q = q / (q.norm(dim=-1, keepdim=True)+1e-6)
            k = k / (k.norm(dim=-1, keepdim=True)+1e-6)
            scale = self.scale
        else:
            scale = 1.0 / (self.d_head ** 0.5)

        attn = torch.matmul(q, k.transpose(-2,-1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1,2).reshape(B,N_q,D)
        return self.out(out)

# -----------------------
# SiT Block with optional AdaLN-Zero
# -----------------------
class SiTBlock(nn.Module):
    def __init__(self, d_model, n_heads, rope_type='standard', 
                 use_qk_norm=False, learnable_qk_scale=False, 
                 ffn_type='swiglu', ffn_hidden_mult=4.0, ffn_dropout=0.0, 
                 use_adaln=True):
        super().__init__()
        self.use_adaln = use_adaln
        self.pre_mhsa_rms = RMSNorm(d_model)
        self.post_mhsa_rms = RMSNorm(d_model)

        self.pre_ffn_rms = RMSNorm(d_model)
        self.post_ffn_rms = RMSNorm(d_model)

        self.final_norm = RMSNorm(d_model)
        rope_module = build_rope(head_dim=d_model // n_heads, rope_type=rope_type)
        self.attn = ClassicMHSABlock(d_model, n_heads, rope_module, use_qk_norm, learnable_qk_scale)
        self.mlp = get_ffn(ffn_type, d_model, ffn_hidden_mult, ffn_dropout)

        if use_adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, 6 * d_model, bias=True)
            )

    def forward(self, x, c=None):
        if self.use_adaln:
            assert c is not None, "Conditioning c required when use_adaln=True"
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.post_mhsa_rms(
                self.attn(modulate(self.pre_mhsa_rms(x), shift_msa, scale_msa))
            )
            x = x + gate_mlp.unsqueeze(1) * self.post_ffn_rms(
                self.mlp(modulate(self.pre_ffn_rms(x), shift_mlp, scale_mlp))
            )
        else:
            x = x + self.post_mhsa_rms(self.attn(self.pre_mhsa_rms(x)))
            x = x + self.post_ffn_rms(self.mlp(self.pre_ffn_rms(x)))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """
    def __init__(self, d_model, output_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(d_model, output_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale, unsqueeze=False)
        x = self.linear(x) # 256 -> output_dim
        return x

# -----------------------
# Full SiT Sequence-to-Vector Model
# -----------------------
class SiT(nn.Module):
    def __init__(self, 
                 input_shape, output_dim, context_dim_first, context_dim_second,
                 d_model, n_heads, num_layers, n_channels,
                 ffn_type='swiglu', ffn_hidden_mult=4, ffn_dropout=0.0,
                 use_qk_norm=True, learnable_qk_scale=True,
                 rope_type='standard', use_adaln=True
    ):
        super().__init__()

        self.input_dim = input_shape[0]
        self.output_dim = output_dim
        self.context_dim_first = context_dim_first
        self.context_dim_second = context_dim_second

        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))
        self.use_adaln = use_adaln

        # AdaLN projections
        self.adaLN_first_proj = nn.Identity()
        self.adaLN_second_proj = nn.Identity()
        if use_adaln:
            if context_dim_first is not None and context_dim_first != d_model:
                self.adaLN_first_proj = nn.Linear(context_dim_first, d_model)
            else:
                self.adaLN_first_proj = nn.Identity()
            if context_dim_second is not None and context_dim_second != d_model:
                self.adaLN_second_proj = nn.Linear(context_dim_second, d_model) 
            else:
                self.adaLN_second_proj = nn.Identity()

        self.proj = nn.Linear(n_channels, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            SiTBlock(
                d_model=d_model,
                n_heads=n_heads,
                rope_type=rope_type,
                use_qk_norm=use_qk_norm,
                learnable_qk_scale=learnable_qk_scale,
                ffn_type=ffn_type,
                ffn_hidden_mult=ffn_hidden_mult,
                ffn_dropout=ffn_dropout,
                use_adaln=use_adaln
            ) for _ in range(num_layers)
        ])

        # Output projection
        self.output = FinalLayer(d_model, output_dim)
        self.initialize_weights()

        
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in SiT blocks:
        for layer in self.layers:
            nn.init.constant_(layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(layer.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.output.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.output.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.output.linear.weight, 0)
        nn.init.constant_(self.output.linear.bias, 0)


    def forward(self, 
        x: torch.Tensor,    # latents
        first_context: torch.Tensor | None = None, # aux_data
        second_context: torch.Tensor | None = None, # t_theta
    ) -> torch.Tensor:
        B = x.size(0)

        if x.dim() == 2:
            x = x.unsqueeze(-1)
        else:
            x = x.view(B, self.n_channels, -1) # 512 or previous d_model
        # Linear projection
            
        x = self.proj(x) # (B, N, d_model)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Compute conditioning vector for AdaLN
        if self.use_adaln:
            if second_context is not None and first_context is not None:
                c = self.adaLN_first_proj(first_context) + self.adaLN_second_proj(second_context)
            elif first_context is None:
                c = self.adaLN_second_proj(second_context)
            elif second_context is None:
                c = self.adaLN_first_proj(first_context)
        else:
            c = None

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, c=c)

        # Return CLS token output as vectorfield
        return self.output(x[:,0], c=c)


def get_mhsa_block(mhsa_block_type, d_model, n_heads, k=8, dilation=1, rope_module=None, use_qk_norm=False, learn_qk_scale=False):
    if mhsa_block_type == 'dilated':
        return DilatedMHSABlock(d_model, n_heads, k, dilation, rope_module, use_qk_norm, learn_qk_scale)
    elif mhsa_block_type == 'classic':
        return ClassicMHSABlock(d_model, n_heads, rope_module, use_qk_norm, learn_qk_scale)
    else:
        raise ValueError(f"Unknown mhsa_block_type: {mhsa_block_type}")


def get_mhca_block(mhca_block_type, d_model, n_heads, k=8, dilation=1, rope_module=None, use_qk_norm=False, learn_qk_scale=False):
    if mhca_block_type == 'dilated':
        return DilatedMHCABlock(d_model, n_heads, k, dilation, rope_module, use_qk_norm, learn_qk_scale)
    elif mhca_block_type == 'classic':
        return ClassicMHCABlock(d_model, n_heads, rope_module, use_qk_norm, learn_qk_scale)
    else:
        raise ValueError(f"Unknown mhca_block_type: {mhca_block_type}")

# -----------------------------
# Hierarchical Multi-Scale Encoder
# -----------------------------
class HierarchicalMultiScaleSpectralEncoder(nn.Module):
    def __init__(self, d_model=256, n_heads=4, k=4, n_layers_per_scale=2,
                 patch_sizes=[16,32,64], strides=[1, 2, 4], dilations=[2, 4, 8], 
                 mhsa_block_type='dilated', mhca_block_type='dilated',
                 incremental_cross_attention=False, 
                 use_ms_pos_embed=False, n_channels=1, rope_type='standard', 
                 use_qk_norm=True, learn_qk_scale=False, pre_rms=True, post_rms=True,
                 ffn_type='swiglu', ffn_hidden_mult=4, ffn_dropout=0.0,
                 fusion_type='cross_attention', fusion_hidden=512):
        super().__init__()
        self.scales = nn.ModuleList()  # each scale is a ModuleDict
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.dilations = dilations
        self.use_ms_pos_embed = use_ms_pos_embed
        self.ffn_type = ffn_type
        self.post_rms = post_rms
        self.mhsa_block_type = mhsa_block_type
        self.incremental_cross_attention = incremental_cross_attention

        for patch_size, stride, dilation in zip(self.patch_sizes, self.strides, self.dilations):
            layers = nn.ModuleList()
            rope_module = build_rope(head_dim=d_model // n_heads, rope_type=rope_type)
            for _ in range(n_layers_per_scale):
                mhsa = get_mhsa_block(
                    mhsa_block_type, d_model, n_heads, k, dilation, 
                    rope_module, use_qk_norm, learn_qk_scale
                )
                mhca = get_mhca_block(
                    mhca_block_type, d_model, n_heads, k, dilation, 
                    rope_module, use_qk_norm, learn_qk_scale
                )
                pre_mhsa_rms = RMSNorm(d_model) if pre_rms else nn.Identity()
                pre_mhca_rms = RMSNorm(d_model) if pre_rms else nn.Identity()
                pre_ffn_rms = RMSNorm(d_model) if pre_rms else nn.Identity()
                ff = get_ffn(self.ffn_type, d_model, ffn_hidden_mult, ffn_dropout)
                post_mhsa_rms = RMSNorm(d_model) if post_rms else nn.Identity()
                post_mhca_rms = RMSNorm(d_model) if post_rms else nn.Identity()
                post_ffn_rms = RMSNorm(d_model) if post_rms else nn.Identity()
                layers.append(nn.ModuleDict(
                    {'mhsa': mhsa, 
                     'mhca': mhca,
                     'ff': ff, 
                     'pre_mhsa_rms': pre_mhsa_rms, 
                     'pre_mhca_rms': pre_mhca_rms,
                     'post_mhca_rms': post_mhca_rms,
                     'post_mhsa_rms': post_mhsa_rms,
                     'pre_ffn_rms': pre_ffn_rms,
                     'post_ffn_rms': post_ffn_rms

                    }
                ))

            patch_embed = PatchEmbedding(
                patch_size=patch_size, 
                n_channels=n_channels, 
                d_model=d_model, 
                stride=stride
            )

            # Create a ModuleDict containing all modules for this scale
            scale_module = nn.ModuleDict({
                'patch_embed': patch_embed,
                'layers': layers
            })

            # Store both ModuleDict and patch_size as attribute
            self.scales.append(scale_module)
            setattr(self, f'patch_size_{len(self.scales)-1}', patch_size)

        # Choose fusion module
        if fusion_type == 'cross_attention':
            self.fusion_module = CrossAttentionFusion(d_model, n_heads)
        elif fusion_type == 'pool_broadcast':
            self.fusion_module = PoolBroadcastFusion(d_model, fusion_hidden)
        elif fusion_type == 'attention_pooling':
            self.fusion_module = AttentionPoolingFusion(d_model, n_heads, n_queries=16, fusion_hidden=fusion_hidden)
        elif fusion_type == 'concat':
            self.fusion_module = None  # handled separately
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.fusion_type = fusion_type

        #  Optional multi-scale positional embedding
        if self.use_ms_pos_embed:
            self.ms_pos_embed = MultiScalePositionalEmbedding(d_model, patch_sizes)
        
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        # x: (B, C, N) -> (B, N, C)
        x = x.permute(0, 2, 1)
        wavelengths = x[:, :, -1]

        features_per_scale = []
        positions_per_scale = []

        for i, scale in enumerate(self.scales):
            patch_size = self.patch_sizes[i]
            stride = self.strides[i]
            # N_patches = 1 + (N - patch_size) // stride

            x_scale = scale['patch_embed'](x)

            w_patch = wavelengths.unfold(1, patch_size, stride)  # (B, N_patches, patch_size)
            positions = w_patch.median(dim=-1).values # median wavelength per patch
            positions = torch.log(positions / positions.min(dim=1, keepdim=True).values)

            for layer in scale['layers']:
                x_scale = x_scale + layer['post_mhsa_rms'](
                    layer['mhsa'](
                        layer['pre_mhsa_rms'](x_scale), 
                        positions, 
                    )
                )
                x_scale = x_scale + layer['post_mhca_rms'](
                    layer['mhca'](
                        layer['pre_mhca_rms'](x_scale), 
                        positions, 
                        features_per_scale[-1] if (
                        self.incremental_cross_attention and i > 0
                        ) else None,
                        positions_per_scale[-1] if (
                            self.incremental_cross_attention and i > 0
                        ) else None
                    )
                )
                x_scale = x_scale + layer['post_ffn_rms'](
                    layer['ff'](
                        layer['pre_ffn_rms'](x_scale)
                    )
                )

            features_per_scale.append(x_scale)
            positions_per_scale.append(positions)

        if self.incremental_cross_attention:
            return self.norm(features_per_scale[-1]).flatten(start_dim=1)

        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat(
                features_per_scale, 
                dim=1
            ) if not self.use_ms_pos_embed else \
                self.ms_pos_embed(features_per_scale)
        else:
            fused = features_per_scale[0]
            for f_coarse in features_per_scale[1:]:
                fused = self.fusion_module(fused, f_coarse)
        fused = self.norm(fused)
        fused = fused.flatten(start_dim=1)
        
        return fused

if __name__ == "__main__":
    from torchinfo import summary
    # Example usage
    B, C, N = 8, 3, 102400  # multi-channel spectrum
    x = torch.randn(B, C, N)
    wavelengths = torch.linspace(0.3, 2.4, N).unsqueeze(0).repeat(B, 1)
    encoder = HierarchicalMultiScaleSpectralEncoder(
        d_model=256,
        n_heads=8,
        k=4,
        n_layers_per_scale=8,
        patch_sizes=[2048, 4096, 8192, 16384],
        strides=[256, 512, 1024, 2048],
        dilations=[2, 4, 8, 16],
        n_channels=C,
        mhsa_block_type='classic',
        mhca_block_type='classic',
        incremental_cross_attention=True,
        fusion_type='concat',
        fusion_hidden=512,
        rope_type='wavelength',
        use_ms_pos_embed=True,
        ffn_type='geglu',
        pre_rms=True,
        post_rms=True,
        use_qk_norm=True,
        learn_qk_scale=True
    )

    summary(encoder, input_data=x, depth=8)

    out = encoder(x)
    print(out.shape)

    # Example SiTSeqToVector
    # TODO: test when n_channels == 1, e.g., latents = output of a DenseResidualNet
    model = SiT(
        input_shape=(512,),
        output_dim=6,
        context_dim_first=512,
        context_dim_second=512,
        d_model=256,
        n_heads=4,
        num_layers=12,
        n_channels=1,
        ffn_type='geglu',
        ffn_hidden_mult=4,
        ffn_dropout=0.0,
        use_qk_norm=True,
        learnable_qk_scale=True,
        use_adaln=True,
        rope_type='standard'
    )
    summary(model, input_data=[torch.randn(B, 512), torch.randn(B,512), torch.randn(B,512)], depth=6)
    out = model(torch.randn(B, 512), None, torch.randn(B,512))
    print(out.shape)    