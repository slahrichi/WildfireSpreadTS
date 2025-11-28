# code taken from https://github.com/HuCaoFighting/Swin-Unet/tree/main

# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from collections import OrderedDict

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from src.models.utae_paps_models.ltae import LTAE2d
from src.models.utae_paps_models.utae import Temporal_Aggregator

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=self.num_classes,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        
        self.ltae = LTAE2d(
            in_channels=self.swin_unet.num_features,
            n_head=16,
            d_k=4,
            mlp=[256, self.swin_unet.num_features],
            dropout=0.2,
            d_model=256,
            T=1000,
            return_att=True,
            positional_encoding=True,
        )

        self.temporal_aggregator = Temporal_Aggregator(mode='att_group')


    def forward(self, x, batch_positions):
        B, T, C, H, W = x.shape
        features_list = []
        x_downsample_list = []

        pad_value = 0  
        pad_mask = None

        for t in range(T):
            x_t = x[:, t, :, :, :]  # shape (B, C, H, W)
            features_t, x_downsample_t = self.swin_unet.forward_features(x_t)
            features_list.append(features_t)
            x_downsample_list.append(x_downsample_t)

        H_prime = self.swin_unet.patches_resolution[0] // (2 ** (self.swin_unet.num_layers - 1))
        W_prime = self.swin_unet.patches_resolution[1] // (2 ** (self.swin_unet.num_layers - 1))
        features_list = [features_t.view(B, H_prime, W_prime, -1) for features_t in features_list]


        # Stack along time dimension
        features = torch.stack(features_list, dim=1)  # shape (B, T, H', W', C)
        features = features.permute(0, 1, 4, 2, 3)    # Permute to (B, T, C, H', W')
                                                                               
        features, attn = self.ltae(features, batch_positions=batch_positions)   # features: (B, C', H', W'), attn: (n_head, B, T, H', W')
        features = features.permute(0, 2, 3, 1).view(B, -1, features.shape[1])  # Shape (B, L, C')

        x_downsample_agg = []
        for l in range(self.swin_unet.num_layers):
            features_l_t = [x_downsample_list[t][l] for t in range(T)]  # List of T tensors
            H_l = self.swin_unet.layers[l].input_resolution[0]
            W_l = self.swin_unet.layers[l].input_resolution[1]
            features_l_t = [f_t.view(B, H_l, W_l, -1) for f_t in features_l_t]
            features_l = torch.stack(features_l_t, dim=1)  # Shape (B, T, H_l, W_l, C_l)
            features_l = features_l.permute(0, 1, 4, 2, 3) # Permute to (B, T, C_l, H_l, W_l)


            features_l_agg = self.temporal_aggregator(features_l, pad_mask=pad_mask, attn_mask=attn)  # features_l_agg: (B, C_l, H_l, W_l)
            features_l_agg = features_l_agg.permute(0, 2, 3, 1).view(B, -1, features_l_agg.shape[1])  # Shape (B, L_l, C_l)
            x_downsample_agg.append(features_l_agg)

        # Decoder
        x = self.swin_unet.forward_up_features(features, x_downsample_agg)
        x = self.swin_unet.up_x4(x)
        return x
    


    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")
            
            model_dict = self.swin_unet.state_dict()
            # new_pretrained_dict = OrderedDict()
            # for k in pretrained_dict.keys():
            #     new_key =  + k  # rename key
            #     new_pretrained_dict[new_key] = pretrained_dict[k]

            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            #print(msg)
            print(f"Number of keys loaded: {len(model_dict) - len(msg.missing_keys)}")
            print(f"Number of keys missing: {len(msg.missing_keys)}")
            print(f"Missing keys: {msg.missing_keys}")
            print(f"Number of unexpected keys: {len(msg.unexpected_keys)}")
            print(f"Unexpected keys: {msg.unexpected_keys}")
        else:
            print("none pretrain")
 