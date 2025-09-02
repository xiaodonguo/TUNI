import warnings
from collections import OrderedDict
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
# from mmcv.cnn.utils.weight_init import (constant_init, trunc_normal_,
#                                         trunc_normal_init)
# from mmcv.runner import (BaseModule, CheckpointLoader, ModuleList,
#                          load_state_dict)
from mmengine.model.base_module import BaseModule
from mmengine.runner.checkpoint import load_state_dict
# from mmcv.utils import to_2tuple
import math

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x

class LocalAttentionRGBT(nn.Module):
    def __init__(self, dim, ratio=8):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim // 2)
        self.linear2 = nn.Linear(dim // 2, dim // 2)
        self.conv1 = nn.Conv2d(dim//2, dim//2, 7, padding=3, groups=dim//2)
        self.conv2 = nn.Conv2d(dim//2, dim//2, 7, padding=3, groups=dim//2)
        self.fc_channel = nn.Sequential(nn.Linear(dim, dim // ratio, bias=False),
                                        nn.GELU(),
                                        nn.Linear(dim // ratio, dim, bias=False),)
        self.linear3 = nn.Linear(dim, dim//2)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, rgb, t):
        rgb = self.linear1(rgb).permute(0, 3, 1, 2)
        t = self.linear2(t).permute(0, 3, 1, 2)
        co  = self.conv1(rgb * t)
        di = self.conv2(torch.abs(rgb-t))
        co_di = torch.cat([co, di], 1).permute(0, 2, 3, 1)
        b, h, w, c = co_di.size()

        attention_map = torch.mean(co_di.permute(0, 3, 1, 2), dim=1, keepdim=True).permute(0, 2, 3, 1)
        dot_product = torch.sum(attention_map * co_di, dim=[1, 2], keepdim=True) # B,1,1,C
        norm_1 = torch.norm(attention_map, p=2, dim=[1, 2], keepdim=True)
        norm_2 = torch.norm(co_di, p=2, dim=[1, 2], keepdim=True)
        cos_sim = dot_product / (norm_1 * norm_2 + 1e-6)
        attention_c = self.fc_channel(cos_sim).sigmoid()
        out = self.linear3(co_di * attention_c)
        return out

class Attention(nn.Module):
    def __init__(self, dim, num_head=8, window=7, norm_cfg=dict(type='BN', requires_grad=True), drop_depth=False):
        super().__init__()
        self.num_head = num_head
        self.window = window

        self.local_rr1 = nn.Linear(dim, dim)
        self.local_rr2 = nn.Linear(dim, dim)
        self.local_rr3 = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.local_rx = LocalAttentionRGBT(dim)

        self.proj = nn.Linear(dim//2*3, dim)
        if not drop_depth:
            self.proj_e = nn.Linear(dim//2*3, dim//2)
        if window != 0:
            # self.kv_rr = nn.Linear(dim, dim * 2)
            # self.q_rr = nn.Linear(dim, dim)
            self.kv_rx = nn.Linear(dim, dim)
            self.q_rx = nn.Linear(dim // 2 * 3, dim // 2)
            self.pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
            self.proj = nn.Linear(dim * 2, dim)
            if not drop_depth:
                self.proj_e = nn.Linear(dim*2, dim//2)

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim//2, eps=1e-6, data_format="channels_last")
        self.drop_depth = drop_depth

    def forward(self, x, x_e):
        B, H, W, C = x.size()
        x = self.norm(x)
        x_e = self.norm_e(x_e)

        # local RGB_RGB attention
        local_rr1 = self.local_rr1(x)
        local_rr2 = self.local_rr2(x).permute(0, 3, 1, 2)
        local_rr2 = self.local_rr3(self.conv(local_rr2).permute(0, 2, 3, 1))
        local_rr = local_rr1 * local_rr2

        # local RGB_X attention
        local_rx = self.local_rx(x, x_e)
        # attention can be added here

        if self.window != 0:
            # global rr
            # kv_rr = self.kv_rr(x)
            # kv_rr = kv_rr.reshape(B, H * W, 2, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
            # k_rr, v_rr = kv_rr.unbind(0)
            # x_pool = self.pool(x.permute(0,3,1,2)).permute(0, 2, 3, 1)
            # q_rr = self.q_rr(x_pool)
            # q_rr = q_rr.reshape(B, -1, self.num_head, C // self.num_head).permute(0, 2, 1, 3)
            # global_rr = (q_rr * (C // self.num_head) ** -0.5) @ k_rr.transpose(-2, -1)
            # global_rr = global_rr.softmax(dim=-1)
            # global_rr = (global_rr @ v_rr).reshape(B, self.num_head, self.window, self.window, C // self.num_head).permute(0, 1,4,2,3).reshape(B, C, self.window, self.window)
            # global_rr = F.interpolate(global_rr, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            # global rx
            kv_rx = self.kv_rx(x)
            kv_rx = kv_rx.reshape(B, H * W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
            k_rx, v_rx = kv_rx.unbind(0)
            rx = torch.cat([x, x_e], dim=3)
            rx_pool = self.pool(rx.permute(0,3,1,2)).permute(0, 2, 3, 1)
            q_rx = self.q_rx(rx_pool)
            q_rx = q_rx.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
            global_rx = (q_rx * (C // self.num_head // 2) ** -0.5) @ k_rx.transpose(-2, -1)
            global_rx = global_rx.softmax(dim=-1)
            global_rx = (global_rx @ v_rx).reshape(B, self.num_head, self.window, self.window,C // self.num_head // 2).permute(0, 1, 4, 2, 3).reshape(B, C // 2,self.window,self.window)
            global_rx = F.interpolate(global_rx, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
            x = torch.cat([local_rr, local_rx, global_rx], dim=3)
        else:
            x = torch.cat([local_rr, local_rx], dim=3)

        if not self.drop_depth:
            x_e = self.proj_e(x)
        else:
            x_e = local_rx

        x = self.proj(x)

        return x, x_e

class Block(nn.Module):
    def __init__(self, index, dim, num_head, norm_cfg=dict(type='BN', requires_grad=True), mlp_ratio=4.,block_index=0, last_block_index=50, window=7, dropout_layer=None,drop_depth=False):
        super().__init__()
        
        self.index = index
        layer_scale_init_value = 1e-6  
        if block_index>last_block_index:
            window=0 
        self.attn = Attention(dim, num_head, window=window, norm_cfg=norm_cfg, drop_depth=drop_depth)
        self.mlp = MLP(dim, mlp_ratio, norm_cfg=norm_cfg)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
                 
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
        if not drop_depth:
            self.layer_scale_1_e = nn.Parameter(layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
            self.layer_scale_2_e = nn.Parameter(layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
            self.mlp_e2 = MLP(dim//2, mlp_ratio)

        self.drop_depth=drop_depth

    def forward(self, x, x_e):
        res_x, res_e = x, x_e
        x, x_e = self.attn(x, x_e)

        
        x = res_x + self.dropout_layer(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x )

        
        x = x + self.dropout_layer(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))

        if not self.drop_depth:
            x_e = res_e + self.dropout_layer(self.layer_scale_1_e.unsqueeze(0).unsqueeze(0) * x_e)
            x_e = x_e + self.dropout_layer(self.layer_scale_2_e.unsqueeze(0).unsqueeze(0) * self.mlp_e2(x_e))

        return x, x_e


class Model1_Backbone(BaseModule):
    def __init__(self, in_channels=4, depths=(2, 2, 8, 2), dims=(32, 64, 128, 256), out_indices=(0, 1, 2, 3), windows=[7, 7, 7, 7], norm_cfg=dict(type='BN', requires_grad=True),
                 mlp_ratios=[8, 8, 4, 4], num_heads=(2, 4, 10, 16), last_block=[50,50,50,50], drop_path_rate=0.1, init_cfg=None):
        super().__init__()
        print(drop_path_rate)
        self.depths = depths
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
                nn.Conv2d(3, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]),
        )

        self.downsample_layers_e = nn.ModuleList() 
        stem_e = nn.Sequential(
                nn.Conv2d(1, dims[0] // 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 4),
                nn.GELU(),
                nn.Conv2d(dims[0] // 4, dims[0]//2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]//2),
        )

        self.downsample_layers.append(stem)
        self.downsample_layers_e.append(stem_e)

        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i])[1],
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

            downsample_layer_e = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i]//2)[1],
                    nn.Conv2d(dims[i]//2, dims[i+1]//2, kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer_e)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur+j, 
                        dim=dims[i], 
                        window=windows[i],
                        dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]), 
                        num_head=num_heads[i], 
                        norm_cfg=norm_cfg,
                        block_index=depths[i]-j,
                        last_block_index=last_block[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_depth=((i==3)&(j==depths[i]-1))) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
       
        # for i in out_indices:
        #     layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
        #     layer_name = f'norm{i}'
        #     self.add_module(layer_name, layer)


    def init_weights(self, pretrained):
       
        _state_dict = torch.load(pretrained)
        if 'state_dict_ema' in _state_dict.keys():
            _state_dict = _state_dict['state_dict_ema']
        else:
            _state_dict = _state_dict['state_dict']

        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith('Backbone.'):
                state_dict[k[9:]] = v
            else:
                state_dict[k] = v

        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        
        load_state_dict(self, state_dict, strict=False)
        print('pretrained weights have been loaded')

    def forward(self, x, x_e=None):
        if x_e is None:
            x_e = x
        if len(x.shape)==3:
            x=x.unsqueeze(0)
        if len(x_e.shape)==3:
            x_e=x_e.unsqueeze(0)

        x_e=x_e[:,0,:,:].unsqueeze(1)
        
        outs = []
        for i in range(4):
          
            x = self.downsample_layers[i](x)
            x_e = self.downsample_layers_e[i](x_e)
           
            x = x.permute(0, 2, 3, 1)
            x_e = x_e.permute(0, 2, 3, 1)

            for blk in self.stages[i]:
                x, x_e = blk(x, x_e)
            x = x.permute(0, 3, 1, 2)
            x_e = x_e.permute(0, 3, 1, 2)
            outs.append(x)
        return outs

def Tiny(pretrained=False, **kwargs):   # 81.5
    model = Model1_Backbone(dims=[32, 64, 128, 256], mlp_ratios=[8, 8, 4, 4], depths=[3, 4, 4, 6], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)
    return model

def backbone_384_2242(pretrained=False, **kwargs):   # 81.0
    model = Model1_Backbone(dims=[48, 96, 192, 384], mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)

    return model

def backbone_512_2242(pretrained=False, **kwargs):   # 81.0
    model = Model1_Backbone(dims=[64, 128, 256, 512], mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 4, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)

    return model

def backbone_320_2262(pretrained=False, **kwargs):   # 81.0
    model = Model1_Backbone(dims=[40, 80, 160, 320], mlp_ratios=[8, 8, 4, 4], depths=[2, 2, 6, 2], num_heads=[1, 2, 4, 8], windows=[0, 7, 7, 7], **kwargs)

    return model



if __name__ == '__main__':
    model = backbone_320_2262(pretrained=False)
    # model.init_weights(pretrained='/home/ubuntu/code/pretrain_weight/model3_384_2242/model_best.pth.tar')
    rgb = torch.randn(1, 3, 480, 640)
    t = torch.randn(1, 1, 480, 640)
    out = model(rgb, t)
    for i in out:
        print(i.shape)

    from fvcore.nn import flop_count_table, FlopCountAnalysis

    with torch.no_grad():
        print(flop_count_table(FlopCountAnalysis(model, (rgb, t))))