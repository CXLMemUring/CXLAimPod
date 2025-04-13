'''
Description  :  
Author       : Boxin Zhang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved. 
'''
import torch
from typing import Dict, Optional

class GraphRunner:
    def __init__(self):
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}
        self.model = None
        self.device = None

    def capture(
        self,
        model,
        cur_token,
        position_ids,
        cache_position,
        past_key_values,
        main_device,
        **kwargs,
    ) -> None:
        # Store model and device
        self.model = model
        self.device = main_device

        # Create initial embeddings
        inputs_embeds = model.model.embed_tokens(cur_token.to("cpu"))
        if main_device != "cpu":
            inputs_embeds = inputs_embeds.to(main_device)

        # Run forward pass to initialize buffers
        logits = model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=past_key_values,
            **kwargs
        )[0]

        if past_key_values is not None:
            past_key_values.change_seq_length(-1)

        # Save the input and output buffers
        self.input_buffers = {
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
        self.output_buffers = {"logits": logits}

    def forward(
        self,
        cur_token,
        position_ids,
        cache_position,
    ) -> torch.Tensor:
        # Create embeddings
        inputs_embeds = self.model.model.embed_tokens(cur_token.to("cpu"))
        if self.device != "cpu":
            inputs_embeds = inputs_embeds.to(self.device)

        # Update input buffers
        self.input_buffers["inputs_embeds"].copy_(inputs_embeds)
        self.input_buffers["position_ids"].copy_(position_ids)
        self.input_buffers["cache_position"].copy_(cache_position)

        # Run forward pass
        with torch.no_grad():
            logits = self.model(
                inputs_embeds=self.input_buffers["inputs_embeds"],
                position_ids=self.input_buffers["position_ids"],
                cache_position=self.input_buffers["cache_position"],
            )[0]
            self.output_buffers["logits"].copy_(logits)

        return self.output_buffers["logits"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
