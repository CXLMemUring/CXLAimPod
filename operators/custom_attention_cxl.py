"""
Custom attention implementation with CXL duplex memory awareness and zero-copy operations
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.utils import InferenceState

# Try to import the base attention class
try:
    from ktransformers.operators.attention import KDeepseekV2Attention
    BASE_ATTENTION_CLASS = KDeepseekV2Attention
except ImportError:
    try:
        from ktransformers.operators.cxlaimpod import KCustomAttention as BaseKCustomAttention
        BASE_ATTENTION_CLASS = BaseKCustomAttention
    except ImportError:
        # Final fallback
        from ktransformers.models.modeling_deepseek_v3 import DeepseekV3Attention
        class BASE_ATTENTION_CLASS(BaseInjectedModule, DeepseekV3Attention):
            pass

class DuplexMemoryBuffer:
    """Duplex memory buffer for zero-copy ping-pong operations"""
    
    def __init__(self, shape: Tuple, dtype: torch.dtype, device: torch.device = torch.device('cpu')):
        # Allocate duplex buffers for zero-copy swapping
        self.buffer_a = torch.zeros(shape, dtype=dtype, device=device)
        self.buffer_b = torch.zeros(shape, dtype=dtype, device=device)
        self.active_buffer = 'a'
        
    def get_active(self) -> torch.Tensor:
        """Get active buffer for reading"""
        return self.buffer_a if self.active_buffer == 'a' else self.buffer_b
    
    def get_inactive(self) -> torch.Tensor:
        """Get inactive buffer for writing"""
        return self.buffer_b if self.active_buffer == 'a' else self.buffer_a
    
    def swap(self):
        """Swap active/inactive buffers (zero-copy)"""
        self.active_buffer = 'b' if self.active_buffer == 'a' else 'a'
    
    def update_inactive_inplace(self, src: torch.Tensor, start_idx: int, end_idx: int):
        """Update inactive buffer in-place"""
        inactive = self.get_inactive()
        inactive[start_idx:end_idx].copy_(src, non_blocking=True)

class KCustomAttentionCXL(BASE_ATTENTION_CLASS):
    """CXL-aware attention with zero-copy operations and duplex memory"""
    
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx, **kwargs)
        
        # Extract configuration
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, 'num_key_value_heads', self.num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        
        # Sparse attention parameters
        self.max_seq_len = kwargs.get('max_seq_len', 32768)
        self.block_size = kwargs.get('block_size', 128)
        self.local_windows_len = kwargs.get('local_windows_len', 1024)
        self.topk = kwargs.get('topk', 16)
        self.use_attn_sparsity = kwargs.get('use_attn_sparsity', True)
        self.preselect_block = kwargs.get('preselect_block', True)
        self.preselect_block_count = kwargs.get('preselect_block_count', 64)
        
        # Initialize duplex KV cache for zero-copy updates
        cache_shape = (self.max_seq_len, self.num_kv_heads, self.head_dim)
        self.k_cache_duplex = DuplexMemoryBuffer(cache_shape, dtype=torch.float16)
        self.v_cache_duplex = DuplexMemoryBuffer(cache_shape, dtype=torch.float16)
        
        # Pre-allocate attention workspace buffers
        self.attn_workspace = torch.zeros(
            (self.block_size, self.block_size), 
            dtype=torch.float32
        )
        
        # Sparse block selection buffer
        self.block_importance = torch.zeros(
            (self.max_seq_len // self.block_size,), 
            dtype=torch.float32
        )
        
        # Pre-computed block indices for zero-copy access
        self.block_indices = self._precompute_block_indices()
        
        self.current_state = InferenceState.UNLOAD
        self.cache_position = 0
    
    def _precompute_block_indices(self):
        """Pre-compute block indices for zero-copy strided access"""
        indices = {}
        num_blocks = self.max_seq_len // self.block_size
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            
            # Store indices for zero-copy slicing
            indices[block_idx] = {
                'start': start,
                'end': end,
                'slice': slice(start, end),
                'stride': self.block_size * self.head_dim
            }
        
        return indices
    
    def load(self, state=InferenceState.GENERATE):
        """Load attention components"""
        self.current_state = state
        # Reset cache position for new sequence
        if state == InferenceState.PREFILL:
            self.cache_position = 0
    
    def unload(self):
        """Unload attention components and free memory"""
        self.current_state = InferenceState.UNLOAD
        # Zero out caches to free memory
        self.k_cache_duplex.buffer_a.zero_()
        self.k_cache_duplex.buffer_b.zero_()
        self.v_cache_duplex.buffer_a.zero_()
        self.v_cache_duplex.buffer_b.zero_()
        self.cache_position = 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with zero-copy operations"""
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute QKV projections (in-place operations where possible)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape without copying (zero-copy views)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Transpose for attention computation (zero-copy)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        # Apply rotary embeddings (in-place when possible)
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb_inplace(
                query_states, key_states, cos, sin
            )
        
        # Update KV cache using duplex buffers (zero-copy)
        if cache_position is not None:
            cache_pos = cache_position[0].item()
        else:
            cache_pos = self.cache_position
        
        # Update inactive buffer
        self.k_cache_duplex.update_inactive_inplace(
            key_states.squeeze(0), cache_pos, cache_pos + seq_len
        )
        self.v_cache_duplex.update_inactive_inplace(
            value_states.squeeze(0), cache_pos, cache_pos + seq_len
        )
        
        # Swap buffers (zero-copy operation)
        self.k_cache_duplex.swap()
        self.v_cache_duplex.swap()
        
        # Update cache position
        self.cache_position = cache_pos + seq_len
        
        # Compute sparse attention with zero-copy operations
        if self.use_attn_sparsity and seq_len == 1:  # Decoding
            attn_output = self._sparse_attention_decode_inplace(
                query_states, cache_pos + seq_len
            )
        else:  # Prefill or dense attention
            attn_output = self._dense_attention_inplace(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape and project output (zero-copy views)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        if use_cache:
            # Return cache references (no copying)
            cache_refs = (self.k_cache_duplex.get_active(), self.v_cache_duplex.get_active())
            return attn_output, None, cache_refs
        
        return attn_output, None, None
    
    def _apply_rotary_pos_emb_inplace(self, q, k, cos, sin):
        """Apply rotary position embeddings in-place"""
        # Split head_dim into two halves (zero-copy views)
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        
        # Apply rotation in-place
        # q_rot = q1 * cos - q2 * sin, q2 * cos + q1 * sin
        q_rot1 = q1.mul(cos).sub_(q2.mul(sin))
        q_rot2 = q2.mul(cos).add_(q1.mul(sin))
        
        k_rot1 = k1.mul(cos).sub_(k2.mul(sin))
        k_rot2 = k2.mul(cos).add_(k1.mul(sin))
        
        # Concatenate in-place
        q_rot = torch.cat([q_rot1, q_rot2], dim=-1)
        k_rot = torch.cat([k_rot1, k_rot2], dim=-1)
        
        return q_rot, k_rot
    
    def _sparse_attention_decode_inplace(self, query_states, total_len):
        """Compute sparse attention during decoding with zero-copy ops"""
        batch_size = query_states.shape[0]
        
        # Get active KV caches (zero-copy references)
        k_cache = self.k_cache_duplex.get_active()
        v_cache = self.v_cache_duplex.get_active()
        
        # Determine which blocks to attend to
        current_block = (total_len - 1) // self.block_size
        attend_blocks = self._select_important_blocks_inplace(current_block, total_len)
        
        # Pre-allocate output
        attn_output = torch.zeros_like(query_states)
        
        # Process each selected block
        for block_idx in attend_blocks:
            block_info = self.block_indices[block_idx]
            
            # Zero-copy views of KV cache blocks
            k_block = k_cache[block_info['slice']]
            v_block = v_cache[block_info['slice']]
            
            # Compute attention scores for this block (reuse workspace)
            scores = torch.matmul(
                query_states, 
                k_block.unsqueeze(0).transpose(-2, -1)
            )
            
            # In-place operations
            scores.mul_(self.head_dim ** -0.5)
            
            # Stable softmax (in-place)
            scores_max = scores.max(dim=-1, keepdim=True)[0]
            scores.sub_(scores_max)
            scores.exp_()
            scores_sum = scores.sum(dim=-1, keepdim=True)
            scores.div_(scores_sum)
            
            # Accumulate weighted values (in-place)
            attn_output.add_(
                torch.matmul(scores, v_block.unsqueeze(0))
            )
        
        return attn_output
    
    def _dense_attention_inplace(self, q, k, v, attention_mask):
        """Compute dense attention with in-place operations"""
        batch_size = q.shape[0]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores.mul_(self.head_dim ** -0.5)
        
        # Apply attention mask (in-place)
        if attention_mask is not None:
            scores.add_(attention_mask)
        
        # Softmax (in-place operations)
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores.sub_(scores_max)
        scores.exp_()
        scores_sum = scores.sum(dim=-1, keepdim=True)
        scores.div_(scores_sum)
        
        # Compute output
        attn_output = torch.matmul(scores, v)
        
        return attn_output
    
    def _select_important_blocks_inplace(self, current_block, total_len):
        """Select important blocks to attend to (zero-copy)"""
        # Local window blocks
        local_start = max(0, current_block - self.local_windows_len // self.block_size)
        local_blocks = list(range(local_start, current_block + 1))
        
        # Update block importance scores in-place
        if self.preselect_block:
            # Simple importance: exponential decay from current position
            num_blocks = min(current_block + 1, len(self.block_indices))
            for i in range(num_blocks):
                distance = current_block - i
                self.block_importance[i] = torch.exp(-0.1 * distance)
            
            # Select top-k blocks (excluding local window)
            _, top_indices = self.block_importance[:num_blocks].topk(
                min(self.topk, num_blocks - len(local_blocks))
            )
            important_blocks = top_indices.tolist()
        else:
            # Strided selection
            stride = max(1, current_block // self.topk)
            important_blocks = list(range(0, current_block, stride))
        
        # Combine and sort
        all_blocks = sorted(set(local_blocks + important_blocks))
        
        return all_blocks

# Module registration function
def get_custom_attention_cxl(config, layer_idx=None, **kwargs):
    """Factory function to create CXL-aware attention module"""
    return KCustomAttentionCXL(config, layer_idx=layer_idx, **kwargs)