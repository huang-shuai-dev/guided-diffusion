"""
评估生成的信道矩阵性能。
计算 FID、sFID、Precision 和 Recall 等指标。
支持.mat格式的数据输入。
"""

import argparse
import os
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm.auto import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt

from evaluator import Evaluator
from guided_diffusion.channel_datasets import ChannelDataset


def load_data(data_path, debug_mode=False):
    """
    加载数据，支持.mat和.npz格式
    
    Args:
        data_path: 数据文件路径
        debug_mode: 是否开启调试模式，开启时只加载100个样本
    
    Returns:
        numpy.ndarray: 处理后的数据
    """
    if data_path.endswith('.mat'):
        # 使用ChannelDataset加载.mat数据
        dataset = ChannelDataset(data_path=data_path)
        # 获取所有样本的H数据
        data = []
        max_samples = 100 if debug_mode else len(dataset)
        for i in range(max_samples):
            sample = dataset[i]
            # 确保数据维度正确 [2, Tx, Rx]
            h_data = sample["H"].numpy()
            data.append(h_data)
        data = np.stack(data, axis=0)  # [N, 2, Tx, Rx]
    elif data_path.endswith('.npz'):
        data = np.load(data_path)['arr_0']
        print("data.shape", data.shape)
        if debug_mode:
            data = data[:100]  # 只取前100个样本
        # 调整维度顺序从 [N, Tx, Rx, C] 到 [N, C, Tx, Rx]
        if len(data.shape) == 4 and data.shape[-1] == 2:  # 如果是 [N, Tx, Rx, C] 格式
            data = np.transpose(data, (0, 3, 1, 2))
        # 确保数据维度正确
        if len(data.shape) == 3:  # [N, C, L]
            data = data.reshape(data.shape[0], data.shape[1], 1, data.shape[2])
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")
    print("data.shape", data.shape)
    return data


def preprocess_data(data):
    """
    确保数据格式正确，并转换为 [N, H, W, 3] 格式
    输入数据格式: [N, C, Tx, Rx] 或 [N, Tx, Rx, C]
    """
    print("预处理前数据维度:", data.shape)
    
    # 确保数据维度正确
    if len(data.shape) == 3:  # [N, Tx, Rx]
        data = np.expand_dims(data, axis=1)  # 添加通道维度 [N, 1, Tx, Rx]
    
    # 如果数据是 [N, Tx, Rx, C] 格式，转换为 [N, C, Tx, Rx]
    if len(data.shape) == 4 and data.shape[-1] == 2:
        data = np.transpose(data, (0, 3, 1, 2))
    
    print("转换后数据维度:", data.shape)
    
    # 计算每个样本的实部和虚部的最大值
    max_vals = np.max(np.abs(data), axis=(2, 3), keepdims=True)  # [N, C, 1, 1]
    # 避免除以0
    max_vals = np.maximum(max_vals, 1e-6)
    
    # 对每个样本进行归一化
    data = data / max_vals
    
    # 将归一化后的数据映射到[0, 255]范围
    data = ((data + 1) * 127.5).astype(np.uint8)
    
    # 直接转换为 [N, H, W, 3] 格式
    N, C, H, W = data.shape
    rgb_data = np.zeros((N, H, W, 3), dtype=np.uint8)
    
    # 调整维度顺序并赋值
    rgb_data[..., 0] = data[:, 0, :, :]  # R 通道 = 实部
    rgb_data[..., 1] = data[:, 1, :, :]  # G 通道 = 虚部
    rgb_data[..., 2] = 0  # B 通道 = 0
    
    print("最终RGB数据维度:", rgb_data.shape)
    return rgb_data


def evaluate_channel_matrices(ref_batch, sample_batch, batch_size=100):
    """
    评估生成的信道矩阵性能
    
    Args:
        ref_batch: 参考数据集（训练数据）
        sample_batch: 生成的样本数据
        batch_size: 批处理大小，用于控制内存使用
    
    Returns:
        dict: 包含各项评估指标的字典
    """
    print("\n=== 评估开始 ===")
    print(f"参考数据集形状: {ref_batch.shape}")
    print(f"生成样本形状: {sample_batch.shape}")
    
    # 1. 设置 TensorFlow 配置
    config = tf.ConfigProto(
        allow_soft_placement=True
    )
    config.gpu_options.allow_growth = True
    
    # 2. 创建评估器
    evaluator = Evaluator(tf.Session(config=config))
    
    # 3. 预热 TensorFlow
    print("\n=== TensorFlow 预热 ===")
    evaluator.warmup()
    
    # 4. 分批计算激活值
    print("\n=== 计算参考数据集激活值 ===")
    ref_acts_pooled = []
    ref_acts_spatial = []
    for i in range(0, len(ref_batch), batch_size):
        batch = ref_batch[i:i + batch_size]
        print(f"处理参考数据集批次 {i//batch_size + 1}, 形状: {batch.shape}")
        acts = evaluator.compute_activations([batch])
        ref_acts_pooled.append(acts[0])  # pooled features
        ref_acts_spatial.append(acts[1])  # spatial features
    
    print("\n=== 计算生成样本激活值 ===")
    sample_acts_pooled = []
    sample_acts_spatial = []
    for i in range(0, len(sample_batch), batch_size):
        batch = sample_batch[i:i + batch_size]
        print(f"处理生成样本批次 {i//batch_size + 1}, 形状: {batch.shape}")
        acts = evaluator.compute_activations([batch])
        sample_acts_pooled.append(acts[0])  # pooled features
        sample_acts_spatial.append(acts[1])  # spatial features
    
    # 5. 合并激活值
    print("\n=== 合并激活值 ===")
    ref_acts_combined = [
        np.concatenate(ref_acts_pooled, axis=0),
        np.concatenate(ref_acts_spatial, axis=0)
    ]
    sample_acts_combined = [
        np.concatenate(sample_acts_pooled, axis=0),
        np.concatenate(sample_acts_spatial, axis=0)
    ]
    print(f"参考数据集合并后形状: {[x.shape for x in ref_acts_combined]}")
    print(f"生成样本合并后形状: {[x.shape for x in sample_acts_combined]}")
    
    # 6. 计算统计信息
    print("\n=== 计算统计信息 ===")
    ref_stats = evaluator.compute_statistics(ref_acts_combined[0])
    ref_stats_spatial = evaluator.compute_statistics(ref_acts_combined[1])
    sample_stats = evaluator.compute_statistics(sample_acts_combined[0])
    sample_stats_spatial = evaluator.compute_statistics(sample_acts_combined[1])
    
    # 7. 计算评估指标
    print("\n=== 计算评估指标 ===")
    inception_score = evaluator.compute_inception_score(sample_acts_combined[0])
    fid = sample_stats.frechet_distance(ref_stats)
    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    prec, recall = evaluator.compute_prec_recall(ref_acts_combined[0], sample_acts_combined[0])
    
    return {
        'inception_score': inception_score,
        'fid': fid,
        'sfid': sfid,
        'precision': prec,
        'recall': recall
    }


def visualize_matrices(ref_batch, sample_batch, output_dir, num_samples=5):
    """
    可视化对比实际矩阵和生成矩阵
    
    Args:
        ref_batch: 参考数据集 [N, H, W, 3]
        sample_batch: 生成的样本数据 [N, H, W, 3]
        output_dir: 输出目录
        num_samples: 要可视化的样本数量
    """
    print("\n=== 开始可视化 ===")
    print(f"参考数据集形状: {ref_batch.shape}")
    print(f"生成样本形状: {sample_batch.shape}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保样本数量不超过实际数据量
    num_samples = min(num_samples, len(ref_batch), len(sample_batch))
    print(f"\n=== 选择样本 ===")
    print(f"将可视化 {num_samples} 个样本")
    
    # 随机选择样本
    indices = np.random.choice(min(len(ref_batch), len(sample_batch)), num_samples, replace=False)
    print(f"选择的样本索引: {indices}")
    
    # 创建图像
    print("\n=== 生成实部虚部对比图 ===")
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    for i, idx in enumerate(indices):
        # 获取实际矩阵和生成矩阵
        ref_real = ref_batch[idx, :, :, 0]  # R通道 = 实部
        ref_imag = ref_batch[idx, :, :, 1]  # G通道 = 虚部
        sample_real = sample_batch[idx, :, :, 0]
        sample_imag = sample_batch[idx, :, :, 1]
        
        print(f"处理样本 {idx}:")
        print(f"  参考矩阵实部形状: {ref_real.shape}")
        print(f"  参考矩阵虚部形状: {ref_imag.shape}")
        print(f"  生成矩阵实部形状: {sample_real.shape}")
        print(f"  生成矩阵虚部形状: {sample_imag.shape}")
        
        # 计算幅度
        ref_mag = np.sqrt(ref_real**2 + ref_imag**2)
        sample_mag = np.sqrt(sample_real**2 + sample_imag**2)
        
        # 绘制实际矩阵
        im1 = axes[i, 0].imshow(ref_real, cmap='viridis')
        axes[i, 0].set_title(f'Real Matrix - Real Part (Sample {idx})')
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(ref_imag, cmap='viridis')
        axes[i, 1].set_title(f'Real Matrix - Imaginary Part (Sample {idx})')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # 绘制生成矩阵
        im3 = axes[i, 2].imshow(sample_real, cmap='viridis')
        axes[i, 2].set_title(f'Generated Matrix - Real Part (Sample {idx})')
        plt.colorbar(im3, ax=axes[i, 2])
        
        im4 = axes[i, 3].imshow(sample_imag, cmap='viridis')
        axes[i, 3].set_title(f'Generated Matrix - Imaginary Part (Sample {idx})')
        plt.colorbar(im4, ax=axes[i, 3])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'matrix_comparison.png'))
    plt.close()
    
    # 绘制幅度对比图
    print("\n=== 生成幅度对比图 ===")
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
    
    for i, idx in enumerate(indices):
        ref_mag = np.sqrt(ref_batch[idx, :, :, 0]**2 + ref_batch[idx, :, :, 1]**2)
        sample_mag = np.sqrt(sample_batch[idx, :, :, 0]**2 + sample_batch[idx, :, :, 1]**2)
        
        print(f"处理样本 {idx} 的幅度图:")
        print(f"  参考矩阵幅度形状: {ref_mag.shape}")
        print(f"  生成矩阵幅度形状: {sample_mag.shape}")
        
        im1 = axes[i, 0].imshow(ref_mag, cmap='viridis')
        axes[i, 0].set_title(f'Real Matrix - Magnitude (Sample {idx})')
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(sample_mag, cmap='viridis')
        axes[i, 1].set_title(f'Generated Matrix - Magnitude (Sample {idx})')
        plt.colorbar(im2, ax=axes[i, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'magnitude_comparison.png'))
    plt.close()
    print("\n=== 可视化完成 ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", type=str, required=True,
                      help="参考数据集路径 (.mat 或 .npz 文件)")
    parser.add_argument("--sample_batch", type=str, required=True,
                      help="生成样本路径 (.mat 或 .npz 文件)")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                      help="评估结果输出目录")
    parser.add_argument("--debug", action="store_true", 
                      help="开启调试模式，只加载100个样本")
    args = parser.parse_args()

    print("加载参考数据集:", args.ref_batch)
    ref_batch = load_data(args.ref_batch, debug_mode=args.debug)
    print("[Dataset Ref] Samples:", ref_batch.shape[0], "Tx:", ref_batch.shape[2], "Rx:", ref_batch.shape[3])

    print("加载生成样本:", args.sample_batch)
    sample_batch = load_data(args.sample_batch, debug_mode=args.debug)
    print("[Dataset Sample] Samples:", sample_batch.shape[0], "Tx:", sample_batch.shape[2], "Rx:", sample_batch.shape[3])
    
    print("预处理数据...")
    ref_batch = preprocess_data(ref_batch)
    sample_batch = preprocess_data(sample_batch)
    
    # 可视化矩阵对比
    print("生成可视化对比图...")
    visualize_matrices(ref_batch, sample_batch, args.output_dir)
    
    print("开始评估...")
    metrics = evaluate_channel_matrices(ref_batch, sample_batch)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存结果
    output_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(output_file, "w") as f:
        f.write("评估结果：\n")
        f.write(f"Inception Score: {metrics['inception_score']:.4f}\n")
        f.write(f"FID: {metrics['fid']:.4f}\n")
        f.write(f"sFID: {metrics['sfid']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
    
    # 打印结果
    print("\n评估结果：")
    print(f"Inception Score: {metrics['inception_score']:.4f}")
    print(f"FID: {metrics['fid']:.4f}")
    print(f"sFID: {metrics['sfid']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"\n结果已保存到: {output_file}")
    print(f"可视化结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main() 