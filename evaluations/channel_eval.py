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

from evaluator import Evaluator
from guided_diffusion.channel_datasets import ChannelDataset


def load_data(data_path):
    """
    加载数据，支持.mat和.npz格式
    
    Args:
        data_path: 数据文件路径
    
    Returns:
        numpy.ndarray: 处理后的数据
    """
    if data_path.endswith('.mat'):
        # 使用ChannelDataset加载.mat数据
        dataset = ChannelDataset(data_path=data_path)
        # 获取所有样本的H数据
        data = []
        for i in range(len(dataset)):
            sample = dataset[i]
            # 确保数据维度正确 [2, Tx, Rx]
            h_data = sample["H"].numpy()
            data.append(h_data)
        data = np.stack(data, axis=0)  # [N, 2, Tx, Rx]
    elif data_path.endswith('.npz'):
        data = np.load(data_path)['arr_0']
        # 确保数据维度正确
        if len(data.shape) == 3:  # [N, C, L]
            data = data.reshape(data.shape[0], data.shape[1], 1, data.shape[2])
    else:
        raise ValueError(f"不支持的文件格式: {data_path}")
    
    return data


def preprocess_data(data):
    """
    确保数据格式正确，并转换为 RGB 图像格式
    """
    # 确保数据在正确的范围内
    data = np.clip(data, -1, 1)
    
    # 确保数据维度正确
    if len(data.shape) == 3:  # [N, C, L]
        data = data.reshape(data.shape[0], data.shape[1], 1, data.shape[2])
    
    # 将信道矩阵转换为 RGB 图像格式
    # 1. 将实部和虚部映射到 [0, 255] 范围
    data = ((data + 1) * 127.5).astype(np.uint8)
    
    # 2. 创建 RGB 图像
    # 使用实部作为 R 通道，虚部作为 G 通道，B 通道设为 0
    N, C, H, W = data.shape
    rgb_data = np.zeros((N, H, W, 3), dtype=np.uint8)
    
    # 调整维度顺序并赋值
    for i in range(N):
        # 直接使用原始维度，不需要reshape
        rgb_data[i, :, :, 0] = data[i, 0, :, :]  # R 通道 = 实部
        rgb_data[i, :, :, 1] = data[i, 1, :, :]  # G 通道 = 虚部
        rgb_data[i, :, :, 2] = 0  # B 通道 = 0
    
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
    # 1. 设置 TensorFlow 配置
    config = tf.ConfigProto(
        allow_soft_placement=True
    )
    config.gpu_options.allow_growth = True
    
    # 2. 创建评估器
    evaluator = Evaluator(tf.Session(config=config))
    
    # 3. 预热 TensorFlow
    print("预热 TensorFlow...")
    evaluator.warmup()
    
    # 4. 分批计算激活值
    print("计算参考数据集的激活值...")
    ref_acts = []
    for i in range(0, len(ref_batch), batch_size):
        batch = ref_batch[i:i + batch_size]
        acts = evaluator.compute_activations([batch])
        ref_acts.append(acts)
    
    print("计算生成样本的激活值...")
    sample_acts = []
    for i in range(0, len(sample_batch), batch_size):
        batch = sample_batch[i:i + batch_size]
        acts = evaluator.compute_activations([batch])
        sample_acts.append(acts)
    
    # 5. 合并激活值
    ref_acts_combined = [np.concatenate([acts[0] for acts in ref_acts], axis=0),
                        np.concatenate([acts[1] for acts in ref_acts], axis=0)]
    sample_acts_combined = [np.concatenate([acts[0] for acts in sample_acts], axis=0),
                          np.concatenate([acts[1] for acts in sample_acts], axis=0)]
    
    # 6. 计算统计信息
    print("计算参考数据集的统计信息...")
    ref_stats = evaluator.compute_statistics(ref_acts_combined[0])
    ref_stats_spatial = evaluator.compute_statistics(ref_acts_combined[1])
    print("计算生成样本的统计信息...")
    sample_stats = evaluator.compute_statistics(sample_acts_combined[0])
    sample_stats_spatial = evaluator.compute_statistics(sample_acts_combined[1])
    
    # 7. 计算评估指标
    print("计算评估指标...")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_batch", type=str, required=True,
                      help="参考数据集路径 (.mat 或 .npz 文件)")
    parser.add_argument("--sample_batch", type=str, required=True,
                      help="生成样本路径 (.mat 或 .npz 文件)")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                      help="评估结果输出目录")
    args = parser.parse_args()

    # 1. 加载数据
    print(f"加载参考数据集: {args.ref_batch}")
    ref_data = load_data(args.ref_batch)
    print(f"加载生成样本: {args.sample_batch}")
    sample_data = load_data(args.sample_batch)
    
    # 2. 预处理数据
    print("预处理数据...")
    ref_batch = preprocess_data(ref_data)
    sample_batch = preprocess_data(sample_data)
    
    # 3. 评估
    print("开始评估...")
    metrics = evaluate_channel_matrices(ref_batch, sample_batch)
    
    # 4. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. 保存结果
    output_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(output_file, "w") as f:
        f.write("评估结果：\n")
        f.write(f"Inception Score: {metrics['inception_score']:.4f}\n")
        f.write(f"FID: {metrics['fid']:.4f}\n")
        f.write(f"sFID: {metrics['sfid']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
    
    # 6. 打印结果
    print("\n评估结果：")
    print(f"Inception Score: {metrics['inception_score']:.4f}")
    print(f"FID: {metrics['fid']:.4f}")
    print(f"sFID: {metrics['sfid']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main() 