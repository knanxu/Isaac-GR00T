from types import SimpleNamespace

from gr00t.data.dataset import fixed_validation_dataset as module
from gr00t.data.dataset.factory import DatasetFactory
from gr00t.data.types import EmbodimentTag, ModalityConfig
import numpy as np
import pandas as pd
import pytest


class Processor:
    def __init__(self):
        self.training = True
        self.statistics = {"training_mean": 12.0}

    def eval(self):
        self.training = False

    def __call__(self, messages):
        return messages[0]["content"]


def test_validation_preserves_windows_and_training_statistics(monkeypatch, tmp_path):
    class Loader:
        episode_lengths = [50, 40]

        def __init__(self, *args, **kwargs):
            pass

        def _load_parquet_data(self, episode):
            values = np.arange(self.episode_lengths[episode]) + episode * 100
            return pd.DataFrame(
                {
                    "state.pose": [np.array([x], dtype=np.float32) for x in values],
                    "action.delta": [np.array([x], dtype=np.float32) for x in values],
                    "language.task": ["Sort the parcels."] * len(values),
                }
            )

        def _load_video_data(self, episode, indices):
            return {"head": np.stack([np.full((2, 2, 3), x, dtype=np.uint8) for x in indices])}

    monkeypatch.setattr(module, "LeRobotEpisodeLoader", Loader)
    tag = EmbodimentTag.NEW_EMBODIMENT.value
    modalities = {
        "state": ModalityConfig(delta_indices=[0], modality_keys=["pose"]),
        "action": ModalityConfig(delta_indices=list(range(40)), modality_keys=["delta"]),
        "video": ModalityConfig(delta_indices=[0], modality_keys=["head"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }
    processor = Processor()
    dataset = module.FixedValidationDataset([(tmp_path, tag)], {tag: modalities}, processor)
    assert len(dataset) == 3
    assert processor.training and not dataset.processor.training
    assert dataset.processor.statistics == processor.statistics
    dataset.processor.statistics["training_mean"] = 99
    assert processor.statistics["training_mean"] == 12.0
    first, second, last = [dataset[i] for i in range(3)]
    np.testing.assert_array_equal(first.actions["delta"].ravel(), np.arange(3, 43))
    np.testing.assert_array_equal(second.actions["delta"].ravel(), np.arange(6, 46))
    np.testing.assert_array_equal(last.actions["delta"].ravel(), np.arange(100, 140))
    assert first.states["pose"][0, 0] == first.images["head"][0][0, 0, 0] == 3
    assert first.text == "Sort the parcels."


def test_validation_rejects_training_path_alias(tmp_path):
    training = tmp_path / "train"
    training.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(training, target_is_directory=True)
    config = SimpleNamespace(
        training=SimpleNamespace(eval_strategy="steps"),
        data=SimpleNamespace(
            datasets=[SimpleNamespace(dataset_paths=[str(training)], val_dataset_path=str(alias))]
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        DatasetFactory(config).build(None)
