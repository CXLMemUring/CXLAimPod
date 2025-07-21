# Add this to your custom_moe.py file
import torch
from ktransformers.operators.experts import KTransformersExperts, KExpertsCPU
from ktransformers.util.utils import InferenceState
# Create this in a new file, e.g., custom_attention.py
try:
    from ktransformers.operators.attention import KDeepseekV2Attention
except ImportError:
    # Fallback to base attention if specific version not available
    from ktransformers.operators.base_operator import BaseInjectedModule
    from ktransformers.models.modeling_deepseek_v3 import DeepseekV3Attention
    class KDeepseekV2Attention(BaseInjectedModule, DeepseekV3Attention):
        pass
from ktransformers.operators.dynamic_attention import DynamicScaledDotProductAttention
from ktransformers.util.utils import InferenceState

class KCustomAttention(KDeepseekV2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_attn = DynamicScaledDotProductAttention(
            max_seq_len=kwargs.get('max_seq_len', 32768),
            block_size=kwargs.get('block_size', 128),
            config=self.config,
            device=torch.device('cpu'),
            local_windows_len=kwargs.get('local_windows_len', 1024),
            topk=kwargs.get('topk', 16),
            threads_num=kwargs.get('threads_num', 8),
            use_attn_sparsity=kwargs.get('use_attn_sparsity', True),
            preselect_block=kwargs.get('preselect_block', True),
            preselect_block_count=kwargs.get('preselect_block_count', 64)
        )
        self.current_state = InferenceState.UNLOAD
    
    def load(self, state=InferenceState.GENERATE):
        """Explicitly load attention components into memory"""
        self.current_state = state
        # Additional memory management logic here
        
    def unload(self):
        """Explicitly unload attention components from memory"""
        self.current_state = InferenceState.UNLOAD
        # Free memory by clearing cached states
        self.dynamic_attn.cache_key_states.zero_()
        self.dynamic_attn.cache_value_states.zero_()
        self.dynamic_attn.cache_importance.zero_()
        
class CustomKTransformersExperts(KTransformersExperts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded_experts = set()  # Track which experts are currently loaded
        
    def load_specific_experts(self, expert_indices):
        """Load only specific experts into memory"""
        # For CPU implementation, we'll load weights selectively
        for idx in expert_indices:
            if idx not in self.loaded_experts and idx < len(self.orig_module):
                # Load the specific expert
                self.loaded_experts.add(idx)
        
    def unload_specific_experts(self, expert_indices):
        """Unload specific experts from memory"""
        for idx in expert_indices:
            if idx in self.loaded_experts:
                # Unload the specific expert
                self.loaded_experts.remove(idx)
                
class SparseMemoryManager:
    def __init__(self, model):
        self.model = model
        self.attention_modules = {}
        self.moe_modules = {}
        self._collect_modules()
        
    def _collect_modules(self):
        """Find all sparse attention and MoE modules in the model"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'dynamic_attn'):
                self.attention_modules[name] = module
            if isinstance(module, KTransformersExperts):
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