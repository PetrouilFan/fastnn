"""Weight mapping utilities for loading model weights.

This module provides a declarative DSL for mapping HuggingFace
weight names to fastnn model parameter names.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple
import re
import numpy as np
import fastnn._core as _core


MappingRule = Tuple[str, str, Optional[Callable]]


class WeightMapper:
    """Declarative weight mapping from HuggingFace to fastnn.
    
    Usage:
        mapper = WeightMapper()
        
        # Simple mapping with auto-transpose detection
        mapper.map("model.layers.{i}.mlp.gate_proj.weight", 
                   "layers.{i}.mlp.gate_proj.weight")
        
        # Mapping with custom transform
        mapper.map("model.layers.{i}.mlp.up_proj.weight",
                   "layers.{i}.mlp.up_proj.weight",
                   transform=lambda w: w.T)
        
        # Apply to model
        mapper.apply( hf_state_dict, fastnn_model)
    """
    
    def __init__(self):
        self.mappings: List[MappingRule] = []
        self._layer_idx_pattern = re.compile(r'\{i\}')
    
    def map(
        self, 
        hf_pattern: str, 
        fnn_pattern: str, 
        transform: Optional[Callable] = None,
        transpose_check: bool = False
    ) -> 'WeightMapper':
        """Register a mapping rule.
        
        Args:
            hf_pattern: HuggingFace weight name pattern (e.g., "model.layers.{i}.mlp.gate_proj.weight")
            fnn_pattern: fastnn parameter name pattern (e.g., "layers.{i}.mlp.gate_proj.weight")
            transform: Optional transform function to apply to weights
            transpose_check: If True, auto-detect if transpose is needed based on shape
            
        Returns:
            Self for chaining
        """
        self.mappings.append((hf_pattern, fnn_pattern, transform, transpose_check))
        return self
    
    def apply(
        self, 
        hf_state: Dict[str, Any], 
        fnn_model: Any
    ) -> None:
        """Apply all mapping rules to load weights into model.
        
        Args:
            hf_state: HuggingFace state dictionary
            fnn_model: fastnn model instance
        """
        # Build parameter map from model
        param_map = {}
        for name, param in fnn_model.named_parameters():
            param_map[name] = param
        
        # Get number of layers (try to detect from config or weights)
        num_layers = self._detect_num_layers(hf_state)
        
        # Apply each mapping rule
        for hf_pattern, fnn_pattern, transform, transpose_check in self.mappings:
            self._apply_mapping(
                hf_pattern, fnn_pattern, transform, transpose_check,
                hf_state, param_map, num_layers
            )
    
    def _detect_num_layers(self, state: Dict[str, Any]) -> int:
        """Detect number of layers from state dict."""
        max_layer = 0
        for key in state.keys():
            match = re.search(r'layers\.(\d+)\.', key)
            if match:
                layer_idx = int(match.group(1))
                max_layer = max(max_layer, layer_idx + 1)
        return max_layer if max_layer > 0 else 32
    
    def _apply_mapping(
        self,
        hf_pattern: str,
        fnn_pattern: str,
        transform: Optional[Callable],
        transpose_check: bool,
        hf_state: Dict[str, Any],
        param_map: Dict,
        num_layers: int
    ) -> None:
        """Apply a single mapping rule across all layers."""
        
        # Handle layer iteration
        if '{i}' in hf_pattern:
            for layer_idx in range(num_layers):
                hf_name = hf_pattern.format(i=layer_idx)
                fnn_name = fnn_pattern.format(i=layer_idx)
                
                if hf_name in hf_state and fnn_name in param_map:
                    self._load_weight(
                        hf_state[hf_name], 
                        param_map[fnn_name], 
                        transform, 
                        transpose_check
                    )
        else:
            # Direct mapping (no layer iteration)
            if hf_pattern in hf_state and fnn_pattern in param_map:
                self._load_weight(
                    hf_state[hf_pattern],
                    param_map[fnn_pattern],
                    transform,
                    transpose_check
                )
    
    def _load_weight(
        self,
        hf_tensor: Any,
        fnn_param: Any,
        transform: Optional[Callable],
        transpose_check: bool
    ) -> None:
        """Load a single weight tensor into a parameter."""
        
        # Convert to numpy
        if hasattr(hf_tensor, 'numpy'):
            np_data = hf_tensor.numpy()
        else:
            np_data = np.array(hf_tensor)
        
        # Apply transform if provided
        if transform is not None:
            np_data = transform(np_data)
        
        # Auto transpose check
        if transpose_check:
            target_shape = fnn_param.tensor.shape
            if len(np_data.shape) == 2 and len(target_shape) == 2:
                if np_data.shape[0] != target_shape[0]:
                    np_data = np_data.T
        
        # Create fastnn tensor and copy
        fastnn_data = tensor_from_array(np_data.astype(np.float32))
        fnn_param.tensor.copy_(fastnn_data)


class ModelWeightMapper(WeightMapper):
    """Pre-configured weight mapper with common mappings.
    
    Provides ready-to-use mappings for common architectures.
    """
    
    def __init__(self, model_type: str):
        super().__init__()
        self.model_type = model_type
        
        if model_type in ("llama", "llama2", "llama3"):
            self._setup_llama_mappings()
        elif model_type in ("lfm2", "lfm2.5"):
            self._setup_lfm_mappings()
    
    def _setup_llama_mappings(self):
        """Setup mappings for Llama models."""
        
        for i in range(32):  # Support up to 32 layers
            # Attention
            self.map(
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"layers.{i}.attention.q_proj.weight"
            )
            self.map(
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"layers.{i}.attention.k_proj.weight"
            )
            self.map(
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"layers.{i}.attention.v_proj.weight"
            )
            self.map(
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"layers.{i}.attention.o_proj.weight"
            )
            
            # MLP - transposed in HF format
            self.map(
                f"model.layers.{i}.mlp.gate_proj.weight",
                f"layers.{i}.ffn.gate_proj.weight",
                transform=lambda w: w.T
            )
            self.map(
                f"model.layers.{i}.mlp.up_proj.weight",
                f"layers.{i}.ffn.up_proj.weight",
                transform=lambda w: w.T
            )
            self.map(
                f"model.layers.{i}.mlp.down_proj.weight",
                f"layers.{i}.ffn.down_proj.weight",
                transform=lambda w: w.T
            )
            
            # Layer norms
            self.map(
                f"model.layers.{i}.input_layernorm.weight",
                f"layers.{i}.input_norm.weight"
            )
            self.map(
                f"model.layers.{i}.post_attention_layernorm.weight",
                f"layers.{i}.output_norm.weight"
            )
        
        # Shared weights
        self.map("model.embed_tokens.weight", "embedding.weight")
        self.map("model.norm.weight", "output_norm.weight")
        self.map("lm_head.weight", "lm_head.weight")
    
    def _setup_lfm_mappings(self):
        """Setup mappings for LFM models."""
        
        for i in range(16):  # LFM2.5 has 16 layers
            # LIV Conv blocks (layer_types[i] == "conv")
            # FFN w1, w2, w3
            self.map(
                f"model.layers.{i}.feed_forward.w1.weight",
                f"layers.{i}.ffn.w1.weight",
                transpose_check=True
            )
            self.map(
                f"model.layers.{i}.feed_forward.w3.weight",
                f"layers.{i}.ffn.w3.weight",
                transpose_check=True
            )
            self.map(
                f"model.layers.{i}.feed_forward.w2.weight",
                f"layers.{i}.ffn.w2.weight",
                transpose_check=True
            )
            
            # Attention blocks (layer_types[i] == "attention" or "full_attention")
            self.map(
                f"model.layers.{i}.self_attn.q_proj.weight",
                f"layers.{i}.attention.q_proj.weight"
            )
            self.map(
                f"model.layers.{i}.self_attn.k_proj.weight",
                f"layers.{i}.attention.k_proj.weight",
                transpose_check=True
            )
            self.map(
                f"model.layers.{i}.self_attn.v_proj.weight",
                f"layers.{i}.attention.v_proj.weight",
                transpose_check=True
            )
            self.map(
                f"model.layers.{i}.self_attn.o_proj.weight",
                f"layers.{i}.attention.o_proj.weight"
            )
            
            # Layer norms (if present)
            # LFM uses ffn_norm and operator_norm
            self.map(
                f"model.layers.{i}.ffn_norm.weight",
                f"layers.{i}.ffn_norm.weight"
            )
            self.map(
                f"model.layers.{i}.operator_norm.weight",
                f"layers.{i}.operator_norm.weight"
            )
        
        # Shared weights
        self.map("model.embed_tokens.weight", "embedding.weight")
        self.map("model.embedding_norm.weight", "embedding_norm.weight")
        # LFM ties embedding and lm_head
        self.map("model.embed_tokens.weight", "lm_head.weight")