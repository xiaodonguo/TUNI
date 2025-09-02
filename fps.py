# TODO - 计算模型的推理时间
def calcTime():

    import numpy as np
    import torch
    from torch.backends import cudnn
    import tqdm
    '''  导入你的模型
    '''
    # modals = ['img', 'depth']
    # from model_others.RGB_T.CAINet import mobilenetGloRe3_CRRM_dule_arm_bou_att
    # from model_others.RGB_T.MDNet.model import MDNet
    # from model_others.RGB_T.DFormer import Model
    from proposed.model1 import Model


    cudnn.benchmark = True

    device = 'cuda:0'

    repetitions = 1000
    model = Model(mode='TFormer', input='RGBT', n_class=12).eval().to(device)
    # model = Model(mode='atto', inputs='rgbt', n_class=12, fusion_mode='CMSSM-KD').eval().to(device)
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    print('Flops ' + flops)
    print('Params ' + params)
    dummy_input1 = torch.rand(1, 3, 480, 640).to(device)
    dummy_input2 = torch.rand(1, 3, 480, 640).to(device)
    # 预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    with torch.no_grad():
        for _ in range(100):
            # _ = model(dummy_input1)
            _ = model(dummy_input1, dummy_input2)

    # synchronize 等待所有 GPU 任务处理完才返回 CPU 主线程
    torch.cuda.synchronize()

    # 设置用于测量时间的 cuda Event, 这是PyTorch 官方推荐的接口,理论上应该最靠谱
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # 初始化一个时间容器
    timings = np.zeros((repetitions, 1))

    print('testing ...\n')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            # _ = model(dummy_input1)
            _ = model(dummy_input1, dummy_input2)
            ender.record()
            torch.cuda.synchronize()  # 等待GPU任务完成
            curr_time = starter.elapsed_time(ender)  # 从 starter 到 ender 之间用时,单位为毫秒
            timings[rep] = curr_time

    avg = 1 / (timings.sum() / repetitions) * 1000
    print('\navg={}\n'.format(avg))

if __name__ == '__main__':
    calcTime()