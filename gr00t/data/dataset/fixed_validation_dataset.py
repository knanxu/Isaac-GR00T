"""Fixed, complete action windows from independent LeRobot validation episodes."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.types import EmbodimentTag, MessageType


class FixedValidationDataset(Dataset):
    def __init__(self, specifications, modality_configs, processor):
        # Reuse training-only normalization without changing training augmentation
        # or computing any statistics from the held-out data.
        self.processor = deepcopy(processor)
        self.processor.eval()
        self.loaders = []
        self.samples = []
        self._cached_episode = None
        self._cached_table = None
        for path, embodiment in specifications:
            tag = EmbodimentTag(embodiment)
            modalities = modality_configs[embodiment]
            offsets = [i for config in modalities.values() for i in config.delta_indices]
            if min(offsets) != 0:
                raise ValueError("Fixed validation requires nonnegative offsets including zero")
            if "mask" in modalities:
                raise ValueError("Fixed validation does not support mask modalities")
            horizon = max(offsets) + 1
            loader = LeRobotEpisodeLoader(
                Path(path), modalities, decoder_kwargs={"num_ffmpeg_threads": 2}
            )
            index = len(self.loaders)
            self.loaders.append((loader, modalities, tag, horizon))
            for episode, length in enumerate(loader.episode_lengths):
                last_start = length - horizon
                if last_start < 0:
                    raise ValueError(f"Validation episode {episode} is shorter than its horizon")
                # Two fixed interior windows per episode, never crossing its end.
                for step in sorted({last_start // 3, 2 * last_start // 3}):
                    self.samples.append((index, episode, step))
        if not self.samples:
            raise ValueError("Validation dataset is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        loader_index, episode, step = self.samples[index]
        loader, modalities, tag, horizon = self.loaders[loader_index]
        cache_key = (loader_index, episode)
        if self._cached_episode != cache_key:
            self._cached_table = loader._load_parquet_data(episode)
            self._cached_episode = cache_key
        window = self._cached_table.iloc[step : step + horizon].copy()
        video_offsets = np.asarray(modalities["video"].delta_indices)
        videos = loader._load_video_data(episode, step + video_offsets)
        for key, images in videos.items():
            column = np.empty(horizon, dtype=object)
            column[:] = None
            for offset, frame in zip(video_offsets, images, strict=True):
                column[offset] = frame
            window[f"video.{key}"] = column
        content = extract_step_data(window, 0, modalities, tag)
        return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": content}])

    def close(self):
        self._cached_table = None
