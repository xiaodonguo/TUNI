
from .log import get_logger
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay

from .metrics_CART import averageMeter, runningScore
from .log import get_logger
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', "glassrgbt", "mirrorrgbd", 'glassrgbt_merged', 'SUS', 'CART',
                              'CART_Terrain', 'KP', 'FMB']

    if cfg['dataset'] == 'irseg':
        from .datasets.MFNet import IRSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

    if cfg['dataset'] == 'SUS':
        from .datasets.SUS import SUS
        return SUS(cfg, mode='train'), SUS(cfg, mode='val'), SUS(cfg, mode='test')
        # return SUS(cfg, mode='train_night'), SUS(cfg, mode='val'), SUS(cfg, mode='test_night')

    if cfg['dataset'] == 'CART':
        from .datasets.CART import CART
        # return SUS(cfg, mode='trainval'), SUS(cfg, mode='test')
        return CART(cfg, mode='train'), CART(cfg, mode='val'), CART(cfg, mode='test')

    if cfg['dataset'] == 'CART_Terrain':
        from .datasets.CART_Terrain import Terrain
        return Terrain(cfg, mode='train'), Terrain(cfg, mode='val'), Terrain(cfg, mode='test')

    if cfg['dataset'] == 'pst900':
        from .datasets.pst900 import PST900

        return PST900(cfg, mode='train'), PST900(cfg, mode='test')

    if cfg['dataset'] == 'KP':
        from .datasets.KP import KP

        return KP(cfg, mode='train'), KP(cfg, mode='val'), KP(cfg, mode='test')

    if cfg['dataset'] == 'FMB':
        from .datasets.FMB import FMB

        return FMB(cfg, mode='train'), FMB(cfg, mode='test')



def get_model(cfg):

    ############# model_others ################
#  RGB_T
    if cfg['model_name'] == 'MFNet':
        from model_others.RGB_T.MFNet import MFNet
        return MFNet(6)

    if cfg['model_name'] == 'MMSMCNet':
        from model_others.RGB_T.MMSMCNet import nation
        return nation()

    if cfg['model_name'] == 'SGFNet':
        from model_others.RGB_T.SGFNet.SGFNet import SGFNet
        return SGFNet(12)

    if cfg['model_name'] == 'CMX':
        from model_others.RGB_T.CMX.models.builder import EncoderDecoder
        return EncoderDecoder(cfg['n_classes'])

    if cfg['model_name'] == 'TSFANet-T':
        from model_others.RGB_T.MAINet import MAINet
        return MAINet(False)

    if cfg['model_name'] == 'GMNet':
        from model_others.RGB_T.GMNet import GMNet
        return GMNet(12)

    if cfg['model_name'] == 'CAINet':
        from model_others.RGB_T.CAINet import mobilenetGloRe3_CRRM_dule_arm_bou_att
        return mobilenetGloRe3_CRRM_dule_arm_bou_att(cfg['n_classes'])

    if cfg['model_name'] == 'EGFNet':
        from model_others.RGB_T.EGFNet import EGFNet
        return EGFNet(12)

    if cfg['model_name'] == 'RTFNet':
        from model_others.RGB_T.RTFNet import RTFNet
        return RTFNet(6)

    if cfg['model_name'] == 'SFAFMA':
        from model_others.RGB_T.SFAFMA import SFAFMA
        return SFAFMA(12)

    if cfg['model_name'] == 'EAEFNet':
        from model_others.RGB_T.EAEFNet import EAEFNet
        return EAEFNet(6)

    if cfg['model_name'] == 'CENet':
        from model_others.RGB_T.TSmodel import Teacher_model
        return Teacher_model(6)

    if cfg['model_name'] == 'MSIRNet':
        from model_others.RGB_T.MS_IRTNet_main.Convnextv2.builder import Convnextv2
        return Convnextv2()

    if cfg['model_name'] == 'CLNet_T':
        from model_others.RGB_T.CLNet_T import Teacher
        return Teacher(cfg['n_classes'])

    if cfg['model_name'] == 'ECM':
        from model_others.RGB_T.ECM import ECM
        return ECM(12)

    if cfg['model_name'] == 'CMNext':
        from model_others.RGB_T.CMNeXt.models.cmnext import CMNeXt
        modals = ['img', 'depth']
        return CMNeXt('CMNeXt-B2', cfg['n_classes'], modals)

    if cfg['model_name'] == 'DPLNet':
        from model_others.RGB_T.DPLNet.DPLNet import DPLNet
        return DPLNet()

    if cfg['model_name'] == 'sigma_tiny':
        from model_others.RGB_T.sigma.builder import EncoderDecoder
        return EncoderDecoder(backbone='sigma_tiny', decoder='MambaDecoder', num_classes=9)

    if cfg['model_name'] == 'sigma_small':
        from model_others.RGB_T.sigma.builder import EncoderDecoder
        return EncoderDecoder(backbone='sigma_small', decoder='MambaDecoder', num_classes=9)

    if cfg['model_name'] == 'MCNet_S':
        from proposed.KD.KD_mem import Model
        return Model(name='nano', num_classes=6)


    if cfg['model_name'] == 'MCNet_T':
        from model_others.RGB_T.MCNet.teacher.teacher import Model
        return Model(name='base', num_classes=cfg['n_classes'])

    if cfg['model_name'] == 'MDNet':
        from model_others.RGB_T.MDNet.model import MDNet
        return MDNet(n_class=cfg['n_classes'])

# RGB
    if cfg['model_name'] == 'FCN_8s':
        from model_others.RGB.FCN import FCN
        return FCN(6)

    if cfg['model_name'] == 'PSPNet':
        from model_others.RGB.PSPNet import PSPNet
        return PSPNet(6)

    if cfg['model_name'] == 'PSANet':
        from model_others.RGB.PSANet import PSANet
        return PSANet(6)

    if cfg['model_name'] == 'DeeplabV3+':
        from model_others.RGB.DeeplabV3.modeling import deeplabv3plus_resnet101
        return deeplabv3plus_resnet101(6)



    # DFormer

    if cfg['model_name'] == 'DFormer_rgbt_tiny':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='tiny', input='rgbt', n_class=12)

    if cfg['model_name'] == 'DFormer_rgbt_small':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='small', input='rgbt', n_class=12)

    if cfg['model_name'] == 'DFormer_rgbt_large':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='large', input='rgbt', n_class=12)

    if cfg['model_name'] == 'DFormer_base':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='base', input='rgbt', n_class=cfg['n_classes'])


# model

    if cfg['model_name'] == 'model1_b1_CM-SSM':
        from proposed.model1 import Model
        return Model(mode='b1', n_class=cfg['n_classes'], inputs='rgbt', fusion_mode='CMSSM')

    if cfg['model_name'] == 'model1_b3_CM-SSM':
        from proposed.model1 import Model
        return Model(mode='b3', n_class=cfg['n_classes'], inputs='rgbt', fusion_mode='CMSSM')

    if cfg['model_name'] == 'model2_atto_CM-SSM':
        from proposed.model2 import Model
        return Model(mode='atto', n_class=cfg['n_classes'], inputs='rgbt', fusion_mode='CMSSM')

    if cfg['model_name'] == 'model2_tiny_CM-SSM':
        from proposed.model2 import Model
        return Model(mode='tiny', n_class=cfg['n_classes'], inputs='rgbt', fusion_mode='CMSSM')

    if cfg['model_name'] == 'model3_b0_CM-SSM':
        from proposed.model3 import Model
        return Model(mode='b0', n_class=cfg['n_classes'], inputs='rgbt', fusion_mode='CMSSM')

# ablation
    if cfg['model_name'] == 'model1_b1_sigma':
        from proposed.model1 import Model
        return Model(mode='b1', n_class=cfg['n_classes'], inputs='rgbt', fusion_mode='sigma')

    if cfg['model_name'] == 'model1_b1_CMX':
        from proposed.model1 import Model
        return Model(mode='b1', n_class=cfg['n_classes'], inputs='rgbt', fusion_mode='CMX')


# DFormer
    if cfg['model_name'] == 'DFormer_tiny_RGBD':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='tiny', input='RGBD', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'DFormer_tiny_RGBT':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='tiny', input='RGBT', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'DFormer_large':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='large', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'DFormer_small_RGBD':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='small', input='RGBD', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'DFormer_small_RGBT':
        from model_others.RGB_T.DFormer import Model
        return Model(mode='small', input='RGBT', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'DFormerV2_small':
        from model_others.RGB_T.DFormerV2 import Model
        return Model(mode='s', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'MILNet':
        from model_others.RGB_T.MILNet.M0 import Model
        return Model(cfg['n_classes'])


# modify

    if cfg['model_name'] == 'TUNI':
        from proposed.model1 import Model
        return Model(mode='TUNI', input='RGBT', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'TUNI_RR':
        from proposed.model1 import Model
        return Model(mode='TUNI', input='RGBRGB', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'TUNI_RD':
        from proposed.model1 import Model
        return Model(mode='TUNI', input='RGBD', n_class=cfg['n_classes'])

    # ablation

    if cfg['model_name'] == 'wo_localrgbt':
        from proposed.model1 import Model
        return Model(mode='wo_localrgbt', input='RGBT', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'wo_globalrgbt':
        from proposed.model1 import Model
        return Model(mode='wo_globalrgbt', input='RGBT', n_class=cfg['n_classes'])

    if cfg['model_name'] == 'wo_localrgbrgb':
        from proposed.model1 import Model
        return Model(mode='wo_localrgbrgb', input='RGBT', n_class=cfg['n_classes'])













