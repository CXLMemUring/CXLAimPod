"""
CXL-aware custom attention and MoE modules with zero-copy operations
and duplex memory allocation awareness
"""

import torch
from typing import Optional, List, Dict, Set, Tuple
from ktransformers.operators.experts import KTransformersExperts
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.utils import InferenceState

# Import CXL memory management with proper error handling
try:
    from ktransformers.operators.cxl_memory_manager import (
        CXLMemoryAllocator, MemoryTier, ZeroCopyTensorView, DuplexMemoryBuffer
    )
    CXL_AVAILABLE = True
except ImportError:
    CXL_AVAILABLE = False
    # Fallback implementations
    class MemoryTier:
        LOCAL_DRAM = 0
        CXL_NEAR = 1
        CXL_FAR = 2
    
    class DuplexMemoryBuffer:
        def __init__(self, shape, dtype, device=torch.device('cpu')):
            self.buffer_a = torch.zeros(shape, dtype=dtype, device=device)
            self.buffer_b = torch.zeros(shape, dtype=dtype, device=device)
            self.active_buffer = 'a'
        
        def get_active(self):
            return self.buffer_a if self.active_buffer == 'a' else self.buffer_b
        
        def get_inactive(self):
            return self.buffer_b if self.active_buffer == 'a' else self.buffer_a
        
        def swap(self):
            self.active_buffer = 'b' if self.active_buffer == 'a' else 'a'
        
        def update_inactive_inplace(self, src, start_idx, end_idx):
            inactive = self.get_inactive()
            inactive[start_idx:end_idx].copy_(src, non_blocking=True)

# Try to import base attention class
try:
    from ktransformers.operators.attention import KDeepseekV2Attention
except ImportError:
    from ktransformers.operators.base_operator import BaseInjectedModule
    from ktransformers.models.modeling_deepseek_v3 import DeepseekV3Attention
    class KDeepseekV2Attention(BaseInjectedModule, DeepseekV3Attention):
        pass

# Import dynamic attention with fallback
try:
    from ktransformers.operators.dynamic_attention import DynamicScaledDotProductAttention
    DYNAMIC_ATTENTION_AVAILABLE = True
except ImportError:
    DYNAMIC_ATTENTION_AVAILABLE = False


class KCustomAttention(KDeepseekV2Attention):
    """CXL-aware custom attention with zero-copy operations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extract configuration
        self.max_seq_len = kwargs.get('max_seq_len', 32768)
        self.block_size = kwargs.get('block_size', 128)
        self.local_windows_len = kwargs.get('local_windows_len', 1024)
        self.topk = kwargs.get('topk', 16)
        self.threads_num = kwargs.get('threads_num', 8)
        self.use_attn_sparsity = kwargs.get('use_attn_sparsity', True)
        self.preselect_block = kwargs.get('preselect_block', True)
        self.preselect_block_count = kwargs.get('preselect_block_count', 64)
        
        # Initialize CXL memory allocator if available
        self.cxl_enabled = CXL_AVAILABLE and kwargs.get('use_cxl', True)
        if self.cxl_enabled:
            self.allocator = CXLMemoryAllocator()
            self._init_cxl_buffers()
        
        # Initialize dynamic attention if available
        if DYNAMIC_ATTENTION_AVAILABLE:
            self.dynamic_attn = DynamicScaledDotProductAttention(
                max_seq_len=self.max_seq_len,
                block_size=self.block_size,
                config=self.config,
                device=torch.device('cpu'),
                local_windows_len=self.local_windows_len,
                topk=self.topk,
                threads_num=self.threads_num,
                use_attn_sparsity=self.use_attn_sparsity,
                preselect_block=self.preselect_block,
                preselect_block_count=self.preselect_block_count
            )
        else:
            # Initialize duplex KV cache for zero-copy updates
            cache_shape = (self.max_seq_len, self.num_heads, self.head_dim)
            self.k_cache_duplex = DuplexMemoryBuffer(cache_shape, torch.float16)
            self.v_cache_duplex = DuplexMemoryBuffer(cache_shape, torch.float16)
            
            # Pre-allocate workspace
            self.attn_workspace = torch.zeros(
                (self.block_size, self.block_size), dtype=torch.float32
            )
            
            # Block importance scores
            self.block_importance = torch.zeros(
                (self.max_seq_len // self.block_size,), dtype=torch.float32
            )
        
        self.current_state = InferenceState.UNLOAD
        self.cache_position = 0
    
    def _init_cxl_buffers(self):
        """Initialize CXL-aware buffers with proper memory tier allocation"""
        cache_shape = (self.max_seq_len, self.num_heads, self.head_dim)
        
        # Allocate KV cache in CXL near memory
        self.k_cache_cxl = self.allocator.allocate_tensor_cxl(
            cache_shape, torch.float16, MemoryTier.CXL_NEAR, zero_copy=True
        )
        self.v_cache_cxl = self.allocator.allocate_tensor_cxl(
            cache_shape, torch.float16, MemoryTier.CXL_NEAR, zero_copy=True
        )
        
        # Duplex buffers for zero-copy updates
        self.k_cache_duplex = DuplexMemoryBuffer(cache_shape, torch.float16)
        self.v_cache_duplex = DuplexMemoryBuffer(cache_shape, torch.float16)
        
        # Workspace in local DRAM for fast computation
        self.attn_workspace = self.allocator.allocate_tensor_cxl(
            (self.block_size, self.block_size), torch.float32,
            MemoryTier.LOCAL_DRAM, zero_copy=True
        )
        
        # Block importance in CXL far memory
        self.block_importance = self.allocator.allocate_tensor_cxl(
            (self.max_seq_len // self.block_size,), torch.float32,
            MemoryTier.CXL_FAR, zero_copy=True
        )
    
    def load(self, state=InferenceState.GENERATE):
        """Load attention components into memory"""
        self.current_state = state
        if state == InferenceState.PREFILL:
            self.cache_position = 0
    
    def unload(self):
        """Unload attention components from memory"""
        self.current_state = InferenceState.UNLOAD
        # Free memory by clearing cached states
        if DYNAMIC_ATTENTION_AVAILABLE and hasattr(self, 'dynamic_attn'):
            self.dynamic_attn.cache_key_states.zero_()
            self.dynamic_attn.cache_value_states.zero_()
            self.dynamic_attn.cache_importance.zero_()
        else:
            self.k_cache_duplex.buffer_a.zero_()
            self.k_cache_duplex.buffer_b.zero_()
            self.v_cache_duplex.buffer_a.zero_()
            self.v_cache_duplex.buffer_b.zero_()
        self.cache_position = 0
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False,
                cache_position=None, **kwargs):
        """Forward pass with zero-copy operations"""
        
        if DYNAMIC_ATTENTION_AVAILABLE and hasattr(self, 'dynamic_attn'):
            # Use existing dynamic attention implementation
            return super().forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache,
                cache_position, **kwargs
            )
        
        # Custom zero-copy implementation
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute QKV projections
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
        
        # Apply rotary embeddings if available
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        
        # Update cache position
        if cache_position is not None:
            cache_pos = cache_position[0].item() if torch.is_tensor(cache_position[0]) else cache_position[0]
        else:
            cache_pos = self.cache_position
        
        # Update KV cache using duplex buffers (zero-copy)
        self.k_cache_duplex.update_inactive_inplace(
            key_states.squeeze(0), cache_pos, cache_pos + seq_len
        )
        self.v_cache_duplex.update_inactive_inplace(
            value_states.squeeze(0), cache_pos, cache_pos + seq_len
        )
        
        # Swap buffers (zero-copy operation)
        self.k_cache_duplex.swap()
        self.v_cache_duplex.swap()
        
        self.cache_position = cache_pos + seq_len
        
        # Compute attention
        if self.use_attn_sparsity and seq_len == 1:
            # Sparse attention for decoding
            attn_output = self._sparse_attention_decode_inplace(
                query_states, self.cache_position
            )
        else:
            # Dense attention for prefill
            attn_output = self._dense_attention_inplace(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, None
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embeddings"""
        # Implementation depends on the specific rotary embedding used
        # This is a placeholder - use the actual implementation from your model
        return q, k
    
    def _sparse_attention_decode_inplace(self, query_states, total_len):
        """Compute sparse attention during decoding with zero-copy ops"""
        batch_size = query_states.shape[0]
        
        # Get active KV caches (zero-copy references)
        k_cache = self.k_cache_duplex.get_active()
        v_cache = self.v_cache_duplex.get_active()
        
        # Pre-allocate output
        attn_output = torch.zeros_like(query_states)
        
        # Determine blocks to attend to
        current_block = (total_len - 1) // self.block_size
        attend_blocks = self._select_important_blocks(current_block, total_len)
        
        # Process each selected block
        for block_idx in attend_blocks:
            start = block_idx * self.block_size
            end = min(start + self.block_size, total_len)
            
            # Zero-copy views of KV cache blocks
            k_block = k_cache[start:end]
            v_block = v_cache[start:end]
            
            # Compute attention scores
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
    
    def _select_important_blocks(self, current_block, total_len):
        """Select important blocks to attend to"""
        # Local window blocks
        local_start = max(0, current_block - self.local_windows_len // self.block_size)
        local_blocks = list(range(local_start, current_block + 1))
        
        if self.preselect_block:
            # Update block importance scores in-place
            num_blocks = min(current_block + 1, self.max_seq_len // self.block_size)
            for i in range(num_blocks):
                distance = current_block - i
                self.block_importance[i] = torch.exp(-0.1 * distance)
            
            # Select top-k blocks
            _, top_indices = self.block_importance[:num_blocks].topk(
                min(self.topk, num_blocks - len(local_blocks))
            )
            important_blocks = top_indices.tolist()
        else:
            # Strided selection
            stride = max(1, current_block // self.topk)
            important_blocks = list(range(0, current_block, stride))
        
        return sorted(set(local_blocks + important_blocks))


class CustomKTransformersExperts(KTransformersExperts):
    """CXL-aware MoE implementation with zero-copy expert loading"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded_experts = set()  # Track which experts are currently loaded
        
        # Initialize CXL support if available
        self.cxl_enabled = CXL_AVAILABLE and kwargs.get('use_cxl', True)
        if self.cxl_enabled:
            self.allocator = CXLMemoryAllocator()
            self._init_cxl_expert_storage()
        
        # Expert memory tier tracking
        self.expert_memory_tier = {}
        self.expert_weights_cxl = {}
    
    def _init_cxl_expert_storage(self):
        """Initialize CXL storage for experts"""
        num_experts = len(self.orig_module)
        
        for expert_idx in range(num_experts):
            # Assign memory tier based on expert importance
            if expert_idx < 2:
                tier = MemoryTier.LOCAL_DRAM
            elif expert_idx < 8:
                tier = MemoryTier.CXL_NEAR
            else:
                tier = MemoryTier.CXL_FAR
            
            self.expert_memory_tier[expert_idx] = tier
            
            # Pre-allocate weight storage
            expert = self.orig_module[expert_idx]
            self.expert_weights_cxl[expert_idx] = {}
            
            for name, param in expert.named_parameters():
                # Allocate in CXL with zero-copy
                cxl_weight = self.allocator.allocate_tensor_cxl(
                    param.shape, param.dtype, tier, zero_copy=True
                )
                # Copy weights to CXL memory
                cxl_weight.copy_(param.data, non_blocking=True)
                self.expert_weights_cxl[expert_idx][name] = cxl_weight
    
    def load_specific_experts(self, expert_indices):
        """Load only specific experts into memory"""
        for idx in expert_indices:
            if idx not in self.loaded_experts and idx < len(self.orig_module):
                self.loaded_experts.add(idx)
                
                # If using CXL, potentially migrate to faster tier
                if self.cxl_enabled and self.expert_memory_tier.get(idx) != MemoryTier.LOCAL_DRAM:
                    self._migrate_expert_tier(idx, MemoryTier.LOCAL_DRAM)
    
    def unload_specific_experts(self, expert_indices):
        """Unload specific experts from memory"""
        for idx in expert_indices:
            if idx in self.loaded_experts:
                self.loaded_experts.remove(idx)
                
                # If using CXL, migrate back to slower tier
                if self.cxl_enabled:
                    if idx >= 8:
                        target_tier = MemoryTier.CXL_FAR
                    elif idx >= 2:
                        target_tier = MemoryTier.CXL_NEAR
                    else:
                        continue
                    
                    self._migrate_expert_tier(idx, target_tier)
    
    def _migrate_expert_tier(self, expert_idx, new_tier):
        """Migrate expert between memory tiers"""
        if not self.cxl_enabled or self.expert_memory_tier[expert_idx] == new_tier:
            return
        
        old_weights = self.expert_weights_cxl[expert_idx]
        new_weights = {}
        
        for name, old_tensor in old_weights.items():
            # Allocate in new tier
            new_tensor = self.allocator.allocate_tensor_cxl(
                old_tensor.shape, old_tensor.dtype, new_tier, zero_copy=True
            )
            # Copy data
            new_tensor.copy_(old_tensor, non_blocking=True)
            new_weights[name] = new_tensor
        
        # Update references
        self.expert_weights_cxl[expert_idx] = new_weights
        self.expert_memory_tier[expert_idx] = new_tier
    
    def forward(self, hidden_states, router_logits, **kwargs):
        """Forward pass with CXL-aware expert computation"""
        if self.cxl_enabled:
            return self._forward_cxl(hidden_states, router_logits, **kwargs)
        else:
            return super().forward(hidden_states, router_logits, **kwargs)
    
    def _forward_cxl(self, hidden_states, router_logits, **kwargs):
        """CXL-aware forward pass with zero-copy operations"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Get expert assignments
        expert_weights, expert_indices = router_logits.topk(2, dim=-1)
        expert_weights = torch.nn.functional.softmax(expert_weights, dim=-1)
        
        # Flatten for processing
        flat_hidden = hidden_states.view(-1, hidden_dim)
        flat_expert_indices = expert_indices.view(-1, 2)
        flat_expert_weights = expert_weights.view(-1, 2)
        
        # Pre-allocate output
        output = torch.zeros_like(flat_hidden)
        
        # Process each expert
        for expert_idx in range(len(self.orig_module)):
            # Find tokens for this expert
            expert_mask = (flat_expert_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue
            
            # Get tokens and weights
            expert_tokens = flat_hidden[expert_mask]
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            # Compute expert output using CXL weights
            if expert_idx in self.expert_weights_cxl:
                expert_output = self._compute_expert_cxl(expert_idx, expert_tokens)
            else:
                expert_output = self.orig_module[expert_idx](expert_tokens)
            
            # Apply gating weights and accumulate
            for i, token_idx in enumerate(token_indices):
                for j in range(2):
                    if flat_expert_indices[token_idx, j] == expert_idx:
                        weight = flat_expert_weights[token_idx, j]
                        output[token_idx].add_(expert_output[i].mul(weight))
        
        # Reshape to original
        output = output.view(batch_size, seq_len, hidden_dim)
        return output
    
    def _compute_expert_cxl(self, expert_idx, tokens):
        """Compute expert output using CXL-stored weights"""
        weights = self.expert_weights_cxl[expert_idx]
        expert = self.orig_module[expert_idx]
        
        # Simple feedforward computation
        x = tokens
        
        # This is a placeholder - adapt to your specific expert architecture
        if hasattr(expert, 'gate_proj') and 'gate_proj.weight' in weights:
            gate = torch.matmul(x, weights['gate_proj.weight'].t())
            gate = torch.nn.functional.silu(gate, inplace=True)
        
        if hasattr(expert, 'up_proj') and 'up_proj.weight' in weights:
            up = torch.matmul(x, weights['up_proj.weight'].t())
        
        if hasattr(expert, 'down_proj') and 'down_proj.weight' in weights:
            x = gate * up if 'gate' in locals() else up
            x = torch.matmul(x, weights['down_proj.weight'].t())
        
        return x


class SparseMemoryManager:
    """Memory manager for sparse attention and MoE modules"""
    
    def __init__(self, model):
        self.model = model
        self.attention_modules = {}
        self.moe_modules = {}
        self._collect_modules()
    
    def _collect_modules(self):
        """Find all sparse attention and MoE modules in the model"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'dynamic_attn') or isinstance(module, KCustomAttention):
                self.attention_modules[name] = module
            if isinstance(module, (KTransformersExperts, CustomKTransformersExperts)):
                self.moe_modules[name] = module
    
    def load_layer_attention(self, layer_indices):
        """Load attention for specific layers"""
        for idx in layer_indices:
            key = f"model.layers.{idx}.self_attn"
            if key in self.attention_modules:
                self.attention_modules[key].load(InferenceState.GENERATE)
    
    def unload_layer_attention(self, layer_indices):
        """Unload attention for specific layers"""
        for idx in layer_indices:
            key = f"model.layers.{idx}.self_attn"
            if key in self.attention_modules:
                self.attention_modules[key].unload()
    
    def load_layer_moe(self, layer_indices, expert_indices=None):
        """Load MoE for specific layers and optionally specific experts"""
        for idx in layer_indices:
            key = f"model.layers.{idx}.mlp.experts"
            if key in self.moe_modules:
                if expert_indices:
                    self.moe_modules[key].load_specific_experts(expert_indices)
                else:
                    self.moe_modules[key].load(InferenceState.GENERATE)
    
    def unload_layer_moe(self, layer_indices, expert_indices=None):
        """Unload MoE for specific layers and optionally specific experts"""
        for idx in layer_indices:
            key = f"model.layers.{idx}.mlp.experts"
            if key in self.moe_modules:
                if expert_indices:
                    self.moe_modules[key].unload_specific_experts(expert_indices)
                else:
                    self.moe_modules[key].unload()
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        stats = {
            'loaded_attention_layers': [],
            'loaded_moe_layers': [],
            'loaded_experts': {}
        }
        
        # Check attention modules
        for name, module in self.attention_modules.items():
            if hasattr(module, 'current_state') and module.current_state != InferenceState.UNLOAD:
                layer_idx = int(name.split('.')[2]) if 'layers' in name else -1
                stats['loaded_attention_layers'].append(layer_idx)
        
        # Check MoE modules
        for name, module in self.moe_modules.items():
            if hasattr(module, 'loaded_experts') and len(module.loaded_experts) > 0:
                layer_idx = int(name.split('.')[2]) if 'layers' in name else -1
                stats['loaded_moe_layers'].append(layer_idx)
                stats['loaded_experts'][layer_idx] = list(module.loaded_experts)
        
        # Add CXL memory stats if available
        if CXL_AVAILABLE:
            try:
                allocator = CXLMemoryAllocator()
                cxl_stats = allocator.get_memory_stats()
                stats['cxl_memory'] = cxl_stats
            except:
                pass
        
        return stats