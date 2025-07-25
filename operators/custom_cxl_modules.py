"""
Unified CXL-aware custom attention and MoE modules with zero-copy operations
and duplex memory allocation awareness
"""

import torch
from typing import Optional, Dict, Any, List, Set

from ktransformers.operators.experts import KTransformersExperts
from ktransformers.operators.base_operator import BaseInjectedModule
from ktransformers.util.utils import InferenceState

# Import CXL memory management
from ktransformers.operators.cxl_memory_manager import (
    CXLMemoryAllocator, MemoryTier, 
    ZeroCopyTensorView, DuplexMemoryBuffer
)

# Try to import base attention class
try:
    from ktransformers.operators.attention import KDeepseekV2Attention
    BASE_ATTENTION_CLASS = KDeepseekV2Attention
except ImportError:
    from ktransformers.models.modeling_deepseek_v3 import DeepseekV3Attention
    class BASE_ATTENTION_CLASS(BaseInjectedModule, DeepseekV3Attention):
        pass

class CXLDynamicScaledDotProductAttention:
    """CXL-aware dynamic sparse attention with zero-copy operations"""
    
    def __init__(
        self,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 128,
        local_windows_len: int = 1024,
        topk: int = 16,
        use_attn_sparsity: bool = True,
        preselect_block: bool = True,
        preselect_block_count: int = 64,
        device: torch.device = torch.device('cpu'),
        allocator: Optional[CXLMemoryAllocator] = None
    ):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.local_windows_len = local_windows_len
        self.topk = topk
        self.use_attn_sparsity = use_attn_sparsity
        self.preselect_block = preselect_block
        self.preselect_block_count = preselect_block_count
        self.device = device
        
        # CXL memory allocator
        self.allocator = allocator or CXLMemoryAllocator()
        
        # Initialize duplex KV cache in CXL near memory
        cache_shape = (max_seq_len, num_heads, head_dim)
        self.cache_key_states = self.allocator.allocate_tensor_cxl(
            cache_shape, torch.float16, MemoryTier.CXL_NEAR, zero_copy=True
        )
        self.cache_value_states = self.allocator.allocate_tensor_cxl(
            cache_shape, torch.float16, MemoryTier.CXL_NEAR, zero_copy=True
        )
        
        # Duplex buffers for zero-copy updates
        self.key_duplex = DuplexMemoryBuffer(cache_shape, torch.float16, device)
        self.value_duplex = DuplexMemoryBuffer(cache_shape, torch.float16, device)
        
        # Importance cache in CXL far memory
        self.cache_importance = self.allocator.allocate_tensor_cxl(
            (max_seq_len // block_size,), torch.float32, 
            MemoryTier.CXL_FAR, zero_copy=True
        )
        
        # Pre-allocate workspace in local DRAM for fast computation
        self.attn_workspace = self.allocator.allocate_tensor_cxl(
            (block_size, block_size), torch.float32,
            MemoryTier.LOCAL_DRAM, zero_copy=True
        )
        
        # Pre-compute block indices for zero-copy access
        self.block_indices = self._precompute_block_indices()
    
    def _precompute_block_indices(self) -> Dict[int, Dict]:
        """Pre-compute block indices for zero-copy strided access"""
        indices = {}
        num_blocks = self.max_seq_len // self.block_size
        
        for block_idx in range(num_blocks):
            start = block_idx * self.block_size
            end = start + self.block_size
            indices[block_idx] = {
                'start': start,
                'end': end,
                'slice': slice(start, end)
            }
        
        return indices
    
    def update_kv_cache_inplace(self, k: torch.Tensor, v: torch.Tensor, 
                               cache_position: int, seq_len: int):
        """Update KV cache using duplex buffers (zero-copy)"""
        # Update inactive duplex buffers
        self.key_duplex.update_inactive_inplace(k, cache_position, cache_position + seq_len)
        self.value_duplex.update_inactive_inplace(v, cache_position, cache_position + seq_len)
        
        # Swap buffers (zero-copy pointer swap)
        self.key_duplex.swap()
        self.value_duplex.swap()
        
        # Copy to persistent cache asynchronously
        active_key = self.key_duplex.get_active()
        active_value = self.value_duplex.get_active()
        
        self.cache_key_states[cache_position:cache_position + seq_len].copy_(
            active_key[cache_position:cache_position + seq_len], non_blocking=True
        )
        self.cache_value_states[cache_position:cache_position + seq_len].copy_(
            active_value[cache_position:cache_position + seq_len], non_blocking=True
        )
    
    def compute_sparse_attention_inplace(self, q: torch.Tensor, total_len: int) -> torch.Tensor:
        """Compute sparse attention with zero-copy operations"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Pre-allocate output in local DRAM
        output = torch.zeros_like(q)
        
        # Get active cache views (zero-copy)
        k_cache = self.key_duplex.get_active()[:total_len]
        v_cache = self.value_duplex.get_active()[:total_len]
        
        # Determine blocks to attend to
        current_block = (total_len - 1) // self.block_size
        attend_blocks = self._select_important_blocks(current_block, total_len)
        
        # Process blocks with zero-copy views
        for block_idx in attend_blocks:
            block_info = self.block_indices[block_idx]
            
            # Zero-copy views of cache blocks
            k_block = ZeroCopyTensorView.create_strided_view(
                k_cache, block_info['start'] * num_heads * head_dim,
                (block_info['end'] - block_info['start'], num_heads, head_dim)
            )
            v_block = ZeroCopyTensorView.create_strided_view(
                v_cache, block_info['start'] * num_heads * head_dim,
                (block_info['end'] - block_info['start'], num_heads, head_dim)
            )
            
            # Reuse workspace for attention computation
            attn_scores = torch.matmul(
                q.view(batch_size * num_heads, seq_len, head_dim),
                k_block.transpose(0, 1).reshape(num_heads, head_dim, -1)
            ).view(batch_size, num_heads, seq_len, -1)
            
            # In-place scaling and softmax
            attn_scores.mul_(head_dim ** -0.5)
            attn_scores = self._inplace_softmax(attn_scores)
            
            # Accumulate output (in-place)
            block_output = torch.matmul(
                attn_scores.view(batch_size * num_heads, seq_len, -1),
                v_block.transpose(0, 1).reshape(num_heads, -1, head_dim)
            ).view(batch_size, num_heads, seq_len, head_dim)
            
            output.add_(block_output)
        
        return output
    
    def _inplace_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute softmax in-place to save memory"""
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores.sub_(scores_max)  # In-place subtract
        scores.exp_()  # In-place exp
        scores_sum = scores.sum(dim=-1, keepdim=True)
        scores.div_(scores_sum)  # In-place divide
        return scores
    
    def _select_important_blocks(self, current_block: int, total_len: int) -> List[int]:
        """Select important blocks based on importance scores"""
        # Local window
        local_start = max(0, current_block - self.local_windows_len // self.block_size)
        local_blocks = list(range(local_start, current_block + 1))
        
        if self.preselect_block and current_block > len(local_blocks):
            # Update importance scores in-place
            num_blocks = min(current_block + 1, len(self.block_indices))
            for i in range(num_blocks):
                distance = current_block - i
                self.cache_importance[i] = torch.exp(-0.1 * distance).item()
            
            # Select top-k important blocks
            _, top_indices = self.cache_importance[:num_blocks].topk(
                min(self.topk, num_blocks - len(local_blocks))
            )
            important_blocks = top_indices.tolist()
        else:
            # Strided selection
            stride = max(1, current_block // self.topk)
            important_blocks = list(range(0, current_block, stride))
        
        return sorted(set(local_blocks + important_blocks))


class KCustomAttentionCXL(BASE_ATTENTION_CLASS):
    """CXL-aware custom attention with zero-copy operations"""
    
    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__(config, layer_idx=layer_idx)
        
        # Initialize CXL allocator
        self.allocator = CXLMemoryAllocator()
        
        # Create CXL-aware dynamic attention
        self.dynamic_attn = CXLDynamicScaledDotProductAttention(
            max_seq_len=kwargs.get('max_seq_len', 32768),
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            block_size=kwargs.get('block_size', 128),
            local_windows_len=kwargs.get('local_windows_len', 1024),
            topk=kwargs.get('topk', 16),
            use_attn_sparsity=kwargs.get('use_attn_sparsity', True),
            preselect_block=kwargs.get('preselect_block', True),
            preselect_block_count=kwargs.get('preselect_block_count', 64),
            device=torch.device('cpu'),
            allocator=self.allocator
        )
        
        self.current_state = InferenceState.UNLOAD
        self.cache_position = 0
    
    def load(self, state=InferenceState.GENERATE):
        """Load attention components into appropriate memory tiers"""
        self.current_state = state
        if state == InferenceState.PREFILL:
            self.cache_position = 0
    
    def unload(self):
        """Unload attention components and free memory"""
        self.current_state = InferenceState.UNLOAD
        # Zero out caches to free memory
        self.dynamic_attn.cache_key_states.zero_()
        self.dynamic_attn.cache_value_states.zero_()
        self.dynamic_attn.cache_importance.zero_()
        self.cache_position = 0
    
    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                past_key_value=None, output_attentions=False, use_cache=False, 
                cache_position=None, **kwargs):
        """Forward pass with CXL-aware zero-copy operations"""
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project QKV (standard operations)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape without copying (zero-copy views)
        query_states = ZeroCopyTensorView.create_reshaped_view(
            query_states, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        key_states = ZeroCopyTensorView.create_reshaped_view(
            key_states, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        value_states = ZeroCopyTensorView.create_reshaped_view(
            value_states, (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        
        # Transpose for attention (zero-copy)
        query_states = ZeroCopyTensorView.create_transposed_view(query_states, 1, 2)
        key_states = ZeroCopyTensorView.create_transposed_view(key_states, 1, 2)
        value_states = ZeroCopyTensorView.create_transposed_view(value_states, 1, 2)
        
        # Apply rotary embeddings if available
        if hasattr(self, 'rotary_emb'):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = self._apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
        
        # Determine cache position
        if cache_position is not None:
            cache_pos = cache_position[0].item() if torch.is_tensor(cache_position[0]) else cache_position[0]
        else:
            cache_pos = self.cache_position
        
        # Update KV cache with zero-copy duplex buffers
        self.dynamic_attn.update_kv_cache_inplace(
            key_states.squeeze(0), value_states.squeeze(0), 
            cache_pos, seq_len
        )
        self.cache_position = cache_pos + seq_len
        
        # Compute attention
        if self.dynamic_attn.use_attn_sparsity and seq_len == 1:
            # Sparse attention for decoding
            attn_output = self.dynamic_attn.compute_sparse_attention_inplace(
                query_states, self.cache_position
            )
        else:
            # Dense attention for prefill
            attn_output = self._dense_attention_cxl(
                query_states, key_states, value_states, attention_mask
            )
        
        # Reshape and project output (zero-copy views)
        attn_output = ZeroCopyTensorView.create_transposed_view(attn_output, 1, 2)
        attn_output = ZeroCopyTensorView.create_reshaped_view(
            attn_output, (batch_size, seq_len, self.hidden_size)
        )
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, None
    
    def _dense_attention_cxl(self, q, k, v, mask):
        """Dense attention with CXL-aware memory access"""
        # Allocate workspace in local DRAM
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores.mul_(self.head_dim ** -0.5)
        
        if mask is not None:
            scores.add_(mask)
        
        # In-place softmax
        scores = self.dynamic_attn._inplace_softmax(scores)
        
        # Output computation
        output = torch.matmul(scores, v)
        return output


class CustomKTransformersExpertsCXL(KTransformersExperts):
    """CXL-aware MoE with zero-copy expert loading and duplex memory"""
    
    def __init__(self, config, expert_modules, **kwargs):
        super().__init__(config, expert_modules, **kwargs)
        
        # CXL memory allocator
        self.allocator = CXLMemoryAllocator()
        
        # Expert memory management
        self.loaded_experts: Set[int] = set()
        self.expert_memory_tier: Dict[int, MemoryTier] = {}
        self.expert_weights_cxl: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Duplex buffers for expert computation
        self.expert_workspace_duplex = {}
        
        # Initialize expert memory placement
        self._initialize_expert_memory()
    
    def _initialize_expert_memory(self):
        """Initialize expert weights in appropriate CXL memory tiers"""
        num_experts = len(self.orig_module)
        
        for expert_idx in range(num_experts):
            # Assign memory tier based on expert importance
            if expert_idx < 2:
                tier = MemoryTier.LOCAL_DRAM  # Hot experts in fast memory
            elif expert_idx < 8:
                tier = MemoryTier.CXL_NEAR    # Warm experts in CXL near
            else:
                tier = MemoryTier.CXL_FAR     # Cold experts in CXL far
            
            self.expert_memory_tier[expert_idx] = tier
            
            # Pre-allocate weight storage
            expert = self.orig_module[expert_idx]
            self.expert_weights_cxl[expert_idx] = {}
            
            for name, param in expert.named_parameters():
                # Allocate in CXL with zero-copy
                cxl_weight = self.allocator.allocate_tensor_cxl(
                    param.shape, param.dtype, tier, zero_copy=True
                )
                # Initial weight copy (one-time)
                cxl_weight.copy_(param.data, non_blocking=True)
                self.expert_weights_cxl[expert_idx][name] = cxl_weight
            
            # Create duplex workspace for this expert
            if hasattr(expert, 'intermediate_size'):
                workspace_shape = (1, expert.intermediate_size)
                self.expert_workspace_duplex[expert_idx] = DuplexMemoryBuffer(
                    workspace_shape, torch.float16, torch.device('cpu')
                )
    
    def load_specific_experts(self, expert_indices: List[int]):
        """Load specific experts into working memory (zero-copy)"""
        for idx in expert_indices:
            if idx not in self.loaded_experts and idx < len(self.orig_module):
                # Mark as loaded (weights already in CXL)
                self.loaded_experts.add(idx)
                
                # Optionally migrate to faster tier if frequently used
                if self.expert_memory_tier[idx] != MemoryTier.LOCAL_DRAM:
                    self._migrate_expert_tier_inplace(idx, MemoryTier.LOCAL_DRAM)
    
    def unload_specific_experts(self, expert_indices: List[int]):
        """Unload specific experts from working memory"""
        for idx in expert_indices:
            if idx in self.loaded_experts:
                self.loaded_experts.remove(idx)
                
                # Migrate back to slower tier to free fast memory
                if idx >= 8:
                    target_tier = MemoryTier.CXL_FAR
                elif idx >= 2:
                    target_tier = MemoryTier.CXL_NEAR
                else:
                    continue  # Keep top experts in fast memory
                
                self._migrate_expert_tier_inplace(idx, target_tier)
    
    def _migrate_expert_tier_inplace(self, expert_idx: int, new_tier: MemoryTier):
        """Migrate expert between memory tiers with minimal copying"""
        if self.expert_memory_tier[expert_idx] == new_tier:
            return
        
        old_weights = self.expert_weights_cxl[expert_idx]
        new_weights = {}
        
        for name, old_tensor in old_weights.items():
            # Allocate in new tier
            new_tensor = self.allocator.allocate_tensor_cxl(
                old_tensor.shape, old_tensor.dtype, new_tier, zero_copy=True
            )
            # Copy data (only copy operation needed)
            new_tensor.copy_(old_tensor, non_blocking=True)
            new_weights[name] = new_tensor
        
        # Update references
        self.expert_weights_cxl[expert_idx] = new_weights
        self.expert_memory_tier[expert_idx] = new_tier
    
    def forward(self, hidden_states, router_logits, **kwargs):
        """Forward pass with zero-copy expert computation"""
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
            
            # Get tokens and weights (zero-copy views)
            expert_tokens = flat_hidden[expert_mask]
            token_indices = expert_mask.nonzero(as_tuple=True)[0]
            
            # Compute expert output using CXL weights
            expert_output = self._compute_expert_cxl(expert_idx, expert_tokens)
            
            # Apply gating weights and accumulate (in-place)
            for i, token_idx in enumerate(token_indices):
                for j in range(2):
                    if flat_expert_indices[token_idx, j] == expert_idx:
                        weight = flat_expert_weights[token_idx, j]
                        output[token_idx].add_(expert_output[i].mul(weight))
        
        # Reshape to original
        output = output.view(batch_size, seq_len, hidden_dim)
        return output
    
    def _compute_expert_cxl(self, expert_idx: int, tokens: torch.Tensor) -> torch.Tensor:
        """Compute expert forward pass using CXL-stored weights"""
        weights = self.expert_weights_cxl[expert_idx]
        expert = self.orig_module[expert_idx]
        
        # Use duplex workspace if available
        if expert_idx in self.expert_workspace_duplex:
            workspace = self.expert_workspace_duplex[expert_idx]
            
            # Process with zero-copy workspace swapping
            # This is a simplified MLP - adapt to your expert architecture
            x = tokens
            
            # Gate projection
            if 'gate_proj.weight' in weights:
                gate = torch.matmul(x, weights['gate_proj.weight'].t())
                gate = torch.nn.functional.silu(gate, inplace=True)
            
            # Up projection
            if 'up_proj.weight' in weights:
                up = torch.matmul(x, weights['up_proj.weight'].t())
            
            # Combine and down project
            if 'down_proj.weight' in weights:
                x = gate * up if 'gate_proj.weight' in weights else up
                x = torch.matmul(x, weights['down_proj.weight'].t())
            
            return x
        else:
            # Fallback to standard computation
            return expert(tokens)


class SparseMemoryManagerCXL:
    """CXL-aware sparse memory manager with zero-copy operations"""
    
    def __init__(self, model):
        self.model = model
        self.allocator = CXLMemoryAllocator()
        self.attention_modules = {}
        self.moe_modules = {}
        
        # Replace modules with CXL-aware versions
        self._replace_with_cxl_modules()
    
    def _replace_with_cxl_modules(self):
        """Replace standard modules with CXL-aware implementations"""
        for name, module in self.model.named_modules():
            # Replace attention modules
            if 'self_attn' in name and hasattr(module, 'q_proj'):
                parent_name = '.'.join(name.split('.')[:-1])
                parent = self._get_parent_module(parent_name)
                
                # Create CXL-aware attention
                config = module.config if hasattr(module, 'config') else self.model.config
                layer_idx = int(name.split('.')[2]) if 'layers' in name else None
                
                cxl_attn = KCustomAttentionCXL(
                    config, 
                    layer_idx=layer_idx,
                    max_seq_len=32768,
                    block_size=128,
                    local_windows_len=1024,
                    topk=16,
                    use_attn_sparsity=True,
                    preselect_block=True,
                    preselect_block_count=64
                )
                
                # Copy weights
                cxl_attn.q_proj = module.q_proj
                cxl_attn.k_proj = module.k_proj
                cxl_attn.v_proj = module.v_proj
                cxl_attn.o_proj = module.o_proj
                if hasattr(module, 'rotary_emb'):
                    cxl_attn.rotary_emb = module.rotary_emb
                
                setattr(parent, name.split('.')[-1], cxl_attn)
                self.attention_modules[name] = cxl_attn
            
            # Replace MoE modules
            elif 'experts' in name and hasattr(module, 'orig_module'):
                parent_name = '.'.join(name.split('.')[:-1])
                parent = self._get_parent_module(parent_name)
                
                config = module.config if hasattr(module, 'config') else self.model.config
                
                cxl_experts = CustomKTransformersExpertsCXL(
                    config,
                    module.orig_module
                )
                
                setattr(parent, name.split('.')[-1], cxl_experts)
                self.moe_modules[name] = cxl_experts
    
    def _get_parent_module(self, parent_name: str):
        """Get parent module from name"""
        if not parent_name:
            return self.model
        
        parent = self.model
        for part in parent_name.split('.'):
            parent = getattr(parent, part)
        return parent
    
    def load_layer_attention(self, layer_indices: List[int]):
        """Load attention for specific layers"""
        for idx in layer_indices:
            key = f"model.layers.{idx}.self_attn"
            if key in self.attention_modules:
                self.attention_modules[key].load(InferenceState.GENERATE)
    
    def unload_layer_attention(self, layer_indices: List[int]):
        """Unload attention for specific layers"""
        for idx in layer_indices:
            key = f"model.layers.{idx}.self_attn"
            if key in self.attention_modules:
                self.attention_modules[key].unload()
    
    def load_layer_moe(self, layer_indices: List[int], expert_indices: Optional[List[int]] = None):
        """Load MoE for specific layers and optionally specific experts"""
        for idx in layer_indices:
            key = f"model.layers.{idx}.mlp.experts"
            if key in self.moe_modules:
                if expert_indices:
                    self.moe_modules[key].load_specific_experts(expert_indices)
                else:
                    self.moe_modules[key].load(InferenceState.GENERATE)
    
    def unload_layer_moe(self, layer_indices: List[int], expert_indices: Optional[List[int]] = None):
        """Unload MoE for specific layers and optionally specific experts"""
        for idx in layer_indices:
            key = f"model.layers.{idx}.mlp.experts"
            if key in self.moe_modules:
                if expert_indices:
                    self.moe_modules[key].unload_specific_experts(expert_indices)
                else:
                    self.moe_modules[key].unload()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        stats = self.allocator.get_memory_stats()
        
        # Add module-specific stats
        stats['loaded_attention_layers'] = []
        stats['loaded_moe_layers'] = []
        
        for name, module in self.attention_modules.items():
            if module.current_state != InferenceState.UNLOAD:
                layer_idx = int(name.split('.')[2])
                stats['loaded_attention_layers'].append(layer_idx)
        
        for name, module in self.moe_modules.items():
            if len(module.loaded_experts) > 0:
                layer_idx = int(name.split('.')[2])
                stats['loaded_moe_layers'].append({
                    'layer': layer_idx,
                    'loaded_experts': list(module.loaded_experts),
                    'expert_tiers': {
                        idx: tier.name 
                        for idx, tier in module.expert_memory_tier.items()
                    }
                })
        
        return stats


# Helper function to enable CXL optimization
def enable_cxl_sparse_optimization(model):
    """Enable CXL-aware sparse optimization for a model"""
    manager = SparseMemoryManagerCXL(model)
    
    # Optimize initial memory placement
    # Load first few layers in fast memory
    manager.load_layer_attention(list(range(min(4, len(model.model.layers)))))
    manager.load_layer_moe(list(range(min(4, len(model.model.layers)))), expert_indices=[0, 1])
    
    return manager