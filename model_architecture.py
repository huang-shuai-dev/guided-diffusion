import torch
from torchviz import make_dot
from guided_diffusion.unet import UNetModel

def draw_unet_architecture():
    # 实例化模型
    model = UNetModel(
        image_size=64,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[8, 16, 32],
        dropout=0.1,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=1000,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )

    # 创建输入张量
    x = torch.randn(1, 3, 64, 64)
    t = torch.randint(0, 1000, (1,))
    y = torch.randint(0, 1000, (1,))

    # 前向传播
    output = model(x, t, y)

    # 使用 torchviz 生成架构图
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render('unet_architecture', format='png', cleanup=True)

if __name__ == '__main__':
    draw_unet_architecture() 