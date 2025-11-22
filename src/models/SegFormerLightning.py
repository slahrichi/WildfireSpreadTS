from typing import Any

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from .BaseModel import BaseModel


class SegFormerLightning(BaseModel):
    """SegFormer model for semantic segmentation using Hugging Face transformers.
    
    SegFormer is a simple, efficient yet powerful semantic segmentation framework
    which unifies Transformers with lightweight multilayer perceptron (MLP) decoders.
    """
    
    def __init__(
        self,
        model_name: str,
        n_channels: int,
        flatten_temporal_dimension: bool,
        pos_class_weight: float,
        encoder_weights=None,
        required_img_size: tuple = None,  # Add option to specify required image size
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(
            n_channels=n_channels,
            flatten_temporal_dimension=flatten_temporal_dimension,
            pos_class_weight=pos_class_weight,
            required_img_size=required_img_size,  # Pass through to BaseModel
            *args,
            **kwargs
        )
        self.save_hyperparameters()
        
        # Available SegFormer model variants
        # Update the model_variants to include ImageNet-only options
        model_variants = {
            # ADE20k fine-tuned models
            "segformer-b0-ade": "nvidia/segformer-b0-finetuned-ade-512-512",
            "segformer-b1-ade": "nvidia/segformer-b1-finetuned-ade-512-512", 
            "segformer-b2-ade": "nvidia/segformer-b2-finetuned-ade-512-512",
            "segformer-b3-ade": "nvidia/segformer-b3-finetuned-ade-512-512",
            "segformer-b4-ade": "nvidia/segformer-b4-finetuned-ade-512-512",
            "segformer-b5-ade": "nvidia/segformer-b5-finetuned-ade-512-512",
            
            # ImageNet-only pretrained models (encoder only)
            "segformer-b0": "nvidia/mit-b0",
            "segformer-b1": "nvidia/mit-b1", 
            "segformer-b2": "nvidia/mit-b2",
            "segformer-b3": "nvidia/mit-b3",
            "segformer-b4": "nvidia/mit-b4",
            "segformer-b5": "nvidia/mit-b5"
        }

        # Use the provided model name or default to b0
        if model_name in model_variants:
            pretrained_model_name = model_variants[model_name]
        else:
            pretrained_model_name = model_name  # Allow custom model names
        
        # Determine number of labels based on loss function
        # BCE/Focal with sigmoid: 1 class (binary)
        # CrossEntropy: 2 classes (background + fire)
        if kwargs.get('loss_function') in ['BCE', 'Focal', 'Dice', 'Jaccard', 'Lovasz']:
            num_labels = 1  # Binary segmentation with sigmoid
        else:
            num_labels = 2  # Multi-class with softmax (background + fire)
        
        # Load configuration and modify for our use case
        if encoder_weights == "imagenet":
            # Use ImageNet-only pretrained encoder
            pretrained_model_name = model_variants[model_name]  # e.g., "nvidia/mit-b2"
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            print(f"Loaded ImageNet pretrained SegFormer model: {pretrained_model_name}")
            
        elif encoder_weights == "ade20k":
            # Use ADE20k fine-tuned model
            pretrained_model_name = model_variants[model_name + "-ade"]  # e.g., "nvidia/segformer-b2-finetuned-ade-512-512"
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            print(f"Loaded ADE20k pretrained SegFormer model: {pretrained_model_name}")
            
        elif encoder_weights != "none" and encoder_weights is not None:
            # Custom pretrained model
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                encoder_weights,
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
            print(f"Loaded custom pretrained SegFormer model: {encoder_weights}")
            
        else:
            # Train from scratch - use config from any model variant to get architecture
            config_model_name = model_variants[model_name]  # e.g., "nvidia/mit-b2"
            config = SegformerConfig.from_pretrained(config_model_name)
            config.num_labels = num_labels
            self.model = SegformerForSemanticSegmentation(config)
            print(f"Created SegFormer from scratch using architecture: {model_name} (num_labels={num_labels})")
            
        # If using pretrained weights with custom channels, we follow SwinUnet pattern:
        # Drop the first layer weights and reinitialize (similar to deleting mismatched keys)
        if n_channels != 3:
            print(f"Input channels ({n_channels}) != 3, dropping pretrained weights for first layer")
            self._modify_input_channels(n_channels)
            
    def _modify_input_channels(self, n_channels: int):
        """Modify the first convolutional layer to accept n_channels input channels.
        
        Following the SwinUnet pattern: when input channels don't match pretrained weights,
        we drop the pretrained weights for that layer and initialize from scratch.
        """
        # The first layer is in the patch embedding
        first_conv = self.model.segformer.encoder.patch_embeddings[0].proj
        
        # Create new convolutional layer with the desired number of input channels
        new_conv = nn.Conv2d(
            in_channels=n_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        # Initialize from scratch (similar to SwinUnet approach of dropping mismatched layers)
        # This avoids issues with pretrained weights that expect 3 channels
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        if new_conv.bias is not None:
            nn.init.constant_(new_conv.bias, 0)
        
        # Replace the layer
        self.model.segformer.encoder.patch_embeddings[0].proj = new_conv
        print(f"Modified input layer to accept {n_channels} channels (initialized from scratch)")

    def forward(self, x: torch.Tensor, doys: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through SegFormer model.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W) or (B, C, H, W)
            doys: Day of year tensor (not used by SegFormer but included for compatibility)
            
        Returns:
            Output tensor of shape (B, H, W) for binary or (B, num_classes, H, W) for multi-class
        """
        # Handle temporal dimension flattening if needed
        if self.hparams.flatten_temporal_dimension and len(x.shape) == 5:
            x = x.flatten(start_dim=1, end_dim=2)
        
        # SegFormer expects input of shape (B, C, H, W)
        outputs = self.model(pixel_values=x)
        
        # Get logits and resize to input resolution
        logits = outputs.logits  # Shape: (B, num_labels, H/4, W/4)
        
        # Upsample to original resolution
        logits = nn.functional.interpolate(
            logits,
            size=x.shape[-2:],  # (H, W)
            mode='bilinear',
            align_corners=False
        )
        
        # Handle output shape based on loss function
        if self.hparams.loss_function in ['BCE', 'Focal', 'Dice', 'Jaccard', 'Lovasz']:
            # Binary segmentation - remove channel dimension
            return logits.squeeze(1)  # Shape: (B, H, W)
        else:
            # Multi-class segmentation - keep channel dimension
            return logits  # Shape: (B, num_classes, H, W)
