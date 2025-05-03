import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

class DenoisingSpectrogramDataset(Dataset):

    def __init__(
            self, 
            root, 
            split='train', 
            target_frames=1024,
            target_bins=257,
            ):
        self.target_frames = target_frames
        self.target_bins = target_bins
        self.clean_files = []
        self.noisy_map = {}

        # Support multiple folders if split is 'train+validation'
        splits = split.split('+')
        for s in splits:
            clean_dir = os.path.join(root, s.strip(), 'clean')
            noisy_dir = os.path.join(root, s.strip(), 'noisy')

            clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.npy')))
            for clean_file in clean_files:
                base = os.path.splitext(os.path.basename(clean_file))[0]
                pattern = os.path.join(noisy_dir, f"*_{base}.npy")
                noisy_files = glob.glob(pattern)
                if noisy_files:
                    self.clean_files.append(clean_file)
                    self.noisy_map[clean_file] = noisy_files

    def __len__(self):
        return sum(len(v) for v in self.noisy_map.values())

    def __getitem__(self, idx):
        flat_pairs = []
        for clean_path, noisy_list in self.noisy_map.items():
            for noisy_path in noisy_list:
                flat_pairs.append((clean_path, noisy_path))

        clean_path, noisy_path = flat_pairs[idx]

        clean = np.load(clean_path).astype(np.float32)
        noisy = np.load(noisy_path).astype(np.float32)

        clean = self._pad_or_crop(clean, self.target_frames, self.target_bins)
        noisy = self._pad_or_crop(noisy, self.target_frames, self.target_bins)

        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)

        return noisy, clean

    def _pad_or_crop(self, spec, target_frames, target_bins):
        freq_bins, time_frames = spec.shape

        if freq_bins < target_bins:
            pad_bins = target_bins - freq_bins
            spec = np.pad(spec, ((0, pad_bins), (0, 0)), mode='constant')
        elif freq_bins > target_bins:
            spec = spec[:target_bins, :]

        if time_frames < target_frames:
            pad_width = target_frames - time_frames
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        elif time_frames > target_frames:
            start = np.random.randint(0, time_frames - target_frames)
            spec = spec[:, start:start + target_frames]

        if spec.shape != (target_bins, target_frames):
            print(f"❗️BAD SHAPE: {spec.shape}, expected ({target_bins}, {target_frames})")

        return spec
