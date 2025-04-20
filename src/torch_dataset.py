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
        
        self.clean_dir = os.path.join(root, split, 'clean')
        self.noisy_dir = os.path.join(root, split, 'noisy')
        self.target_frames = target_frames
        self.target_bins = target_bins

        self.clean_files = sorted(glob.glob(os.path.join(self.clean_dir, '*.npy')))
        self.noisy_map = self._map_noisy_versions()

    def _map_noisy_versions(self):
        mapping = {}
        for clean_file in self.clean_files:
            base = os.path.splitext(os.path.basename(clean_file))[0]
            pattern = os.path.join(self.noisy_dir, f"{base}__*.npy")
            noisy_files = glob.glob(pattern)
            if noisy_files:
                mapping[clean_file] = noisy_files
        return mapping

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

        # Ensure the shapes are correct
        if noisy.shape[1] > self.target_frames:
            noisy = noisy[:, :self.target_frames]
        
        if clean.shape[1] > self.target_frames:
            clean = clean[:, :self.target_frames]

        return noisy, clean

    def _pad_or_crop(self, spec, target_frames, target_bins):
        freq_bins, time_frames = spec.shape
        
        # Pad or crop frequency bins
        if freq_bins < target_bins:
            pad_bins = target_bins - freq_bins
            spec = np.pad(spec, ((0, pad_bins), (0, 0)), mode='constant')
        elif freq_bins > target_bins:
            spec = spec[:target_bins, :]

        # Pad or crop time frames
        if time_frames < target_frames:
            pad_width = target_frames - time_frames
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        elif time_frames > target_frames:
            start = np.random.randint(0, time_frames - target_frames)
            spec = spec[:, start:start + target_frames]

        if spec.shape != (target_bins, target_frames):
            print(f"❗️BAD SHAPE: {spec.shape}, expected ({target_bins}, {target_frames})")
        
        return spec
