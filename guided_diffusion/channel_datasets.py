import os
import numpy as np
import torch
from torch.utils import data
import hdf5storage

class ChannelDataset(data.Dataset):
    def __init__(self, data_path, image_size=None, pilot_length=1, snr_db=20.0,
                 quantize_y=True, n_bits=4, normalize=True,
                 subcarrier_index=0, seed=42):
        self.data_path = data_path
        self.image_size = image_size
        self.pilot_length = pilot_length
        self.snr_db = snr_db
        self.quantize_y = quantize_y
        self.n_bits = n_bits
        self.normalize = normalize
        self.subcarrier_index = subcarrier_index
        self.seed = seed
        
        # 加载数据
        mat = hdf5storage.loadmat(self.data_path)
        H_full = mat['output_h']  # [N, S, Tx, Rx]
        assert H_full.ndim == 4
        self.H_all = H_full[:, self.subcarrier_index, :, :]
        self.num_samples, self.tx, self.rx = self.H_all.shape
        
        # 生成导频
        self.pilot = self._generate_qpsk_pilot(self.rx, self.pilot_length)
        self.p_concat = np.stack([np.real(self.pilot), np.imag(self.pilot)], axis=0)  # [2, Rx, L]
        
        # 计算H的最大值用于归一化
        self.h_max = np.max(np.abs(self.H_all))
        if self.normalize:
            self.H_all_norm = self.H_all / self.h_max
            
        self.scale_y = self._compute_y_quant_scale()
        self.rng = np.random.default_rng(seed)
        
        # 如果指定了image_size，检查数据尺寸是否匹配
        if self.image_size is not None:
            if isinstance(self.image_size, tuple):
                height, width = self.image_size
                if height != self.tx or width != self.rx:
                    raise ValueError(f"Specified image size ({height}, {width}) does not match data dimensions ({self.tx}, {self.rx})")
            else:
                size = int(self.image_size)
                if size != self.tx or size != self.rx:
                    raise ValueError(f"Specified image size {size} does not match data dimensions ({self.tx}, {self.rx})")
        
        print(f"[Dataset] Samples: {self.num_samples}, Tx: {self.tx}, Rx: {self.rx}")
        print(f"[Pilot] Shape: {self.pilot.shape}, SNR: {self.snr_db} dB")
        print(f"[Global scale] h_max: {self.h_max:.4f}, scale_y (0.99) {self.scale_y:.4f}")
        
    def _generate_qpsk_pilot(self, rx_dim, L):
        rng = np.random.default_rng(self.seed)
        phases = rng.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=(rx_dim, L))
        return np.exp(1j * phases)
        
    def _compute_y_quant_scale(self):
        y_vals = []
        for i in range(min(self.num_samples, 500)):
            if self.normalize:
                H = self.H_all_norm[i]
            else:
                H = self.H_all[i]
            y = H @ self.pilot
            y_vals.append(np.abs(np.real(y)).ravel())
            y_vals.append(np.abs(np.imag(y)).ravel())
        return np.quantile(np.concatenate(y_vals), 0.99)
        
    def _uniform_quantize(self, x, scale, n_bits):
        levels = 2 ** n_bits
        x_clipped = np.clip(x / scale, -1.0, 1.0)
        x_scaled = (x_clipped + 1) * (levels - 1) / 2
        x_rounded = np.round(x_scaled)
        x_dequant = x_rounded / (levels - 1) * 2 - 1
        return x_dequant * scale
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.normalize:
            H = self.H_all_norm[idx]  # [Tx, Rx]
        else:
            H = self.H_all[idx]  # [Tx, Rx]
            
        y = H @ self.pilot
        
        # 添加噪声
        signal_power = np.mean(np.abs(y) ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        noise = self.rng.normal(0, noise_std, y.shape) + 1j * self.rng.normal(0, noise_std, y.shape)
        y = y + noise
        
        H_concat = np.stack([np.real(H), np.imag(H)], axis=0)  # [2, Tx, Rx]
        y_concat = np.stack([np.real(y), np.imag(y)], axis=0)  # [2, Tx, L]
        
        # y 量化（基于全局 scale_y）
        y_quant = self._uniform_quantize(y_concat, self.scale_y, self.n_bits) if self.quantize_y else y_concat
        
        return {
            "H": torch.from_numpy(H_concat.astype(np.float32)),
            "y": torch.from_numpy(y_concat.astype(np.float32)),
            "y_quant": torch.from_numpy(y_quant.astype(np.float32)),
            "p": torch.from_numpy(self.p_concat.astype(np.float32)),
            "h_max": torch.tensor(self.h_max, dtype=torch.float32),
            "scale_y": torch.tensor(self.scale_y, dtype=torch.float32),
        } 