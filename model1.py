import torch
import torch.nn as nn
from proposed.backbone_model.TUNI import *
from proposed.backbone_model.ablation_wo_localrgbt import wo_localrgbt
from proposed.backbone_model.ablation_wo_globalrgbt import wo_globalrgbt
from proposed.backbone_model.ablation_wo_localrgbrgb import wo_localrgbrgb
from proposed.decoder.MLP import Decoder_MLP
from proposed.decoder.Hamburger import Decoder_Ham
import torch.nn.functional as F


class Encoder_RGBX(nn.Module):
    def __init__(self, mode, input):
        super(Encoder_RGBX, self).__init__()

        if mode == 'TUNI':
            self.enc = backbone_384_2242()
            if input=='RGBD':
                self.enc.init_weights(
                pretrained="/home/ubuntu/code/pretrain_weight/model3_384_2242/TUNI/RGB_Depth/model_best.pth.tar"
            )
                print('load from RGBD')
            if input=='RGBT':
                self.enc.init_weights(
                pretrained="/home/ubuntu/code/pretrain_weight/model3_384_2242/TUNI/RGB_T/model_best.pth.tar"
            )
                print('load from RGBT')
            if input=='RGBRGB':
                self.enc.init_weights(
                pretrained="/home/ubuntu/code/pretrain_weight/model3_384_2242/TUNI/RGB_RGB/model_best.pth.tar"
            )
                print('load from RGBRGB')

        if mode == 'wo_localrgbt':
            self.enc = wo_localrgbt(pretrained=True)
            print('ablation_wo_localrgbt')

        if mode == 'wo_globalrgbt':
            self.enc = wo_globalrgbt(pretrained=True)
            print('ablation_wo_globalrgbt')

        if mode == 'wo_localrgbrgb':
            self.enc = wo_localrgbrgb(pretrained=True)
            print('ablation_wo_localrgbrgb')





    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        outs = self.enc(rgb, t)
        return outs

class Model(nn.Module):
    def __init__(self, mode, input='RGBT', n_class=9):
        super(Model, self).__init__()

        channels = [48, 96, 192, 384]
        emb_c = 256

        self.encoder = Encoder_RGBX(mode=mode, input=input)
        self.decoder = Decoder_MLP(in_channels=channels, embed_dim=emb_c, num_classes=n_class)
        # self.decoder = Decoder_Ham(channels, emb_c, emb_c, n_class)
    def forward(self, rgb, t=None):
        if t == None:
            t = rgb
        f_rgb = self.encoder(rgb, t)
        sem = self.decoder(f_rgb)
        sem = F.interpolate(sem, scale_factor=4, mode='bilinear', align_corners=False)
        return sem


if __name__ == '__main__':
    rgb = torch.rand(1, 3, 480, 640).cuda()
    t = torch.rand(1, 3, 480, 640).cuda()
    model = Model(mode='TUNI', input='RGBT', n_class=12).eval().cuda()
    out = model(rgb, t)
    print(out.shape)

    # from ptflops import get_model_complexity_info
    #
    # flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    # print('Flops ' + flops)
    # print('Params ' + params)

    from fvcore.nn import flop_count_table, FlopCountAnalysis

    print(flop_count_table(FlopCountAnalysis(model, rgb)))
    from thop import profile
    flops, params = profile(model, inputs=(rgb, t))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    print(f"Parameters: {params / 1e6:.2f} M")
