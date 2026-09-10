from types import SimpleNamespace
from unittest.mock import Mock

from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as module
import numpy as np
import pytest
from transformers import Qwen2VLImageProcessorFast


@pytest.fixture
def local_vlm_processor(monkeypatch):
    def create(*args, **kwargs):
        return SimpleNamespace(
            tokenizer=SimpleNamespace(padding_side="right"),
            image_processor=Qwen2VLImageProcessorFast(
                patch_size=16,
                size={"shortest_edge": 65536, "longest_edge": 16777216},
            ),
        )

    monkeypatch.setattr(module, "Qwen3VLProcessor", SimpleNamespace(from_pretrained=create))
    monkeypatch.setattr(module, "is_offline_mode", lambda: False)


def grid(processor):
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"].tolist()


def test_pixel_bound_preserves_224_in_actual_image_processor(local_vlm_processor):
    assert grid(module.build_processor("local", {})) == [[1, 16, 16]]
    assert grid(module.build_processor("local", {}, 224 * 224)) == [[1, 14, 14]]


def test_training_override_and_checkpoint_reload_preserve_image_size(tmp_path, local_vlm_processor):
    old = module.Gr00tN1d7Processor(
        modality_configs={},
        statistics={},
        image_crop_size=[230, 230],
        image_target_size=[256, 256],
        shortest_image_edge=None,
        crop_fraction=None,
        use_albumentations=True,
    )
    old.save_pretrained(tmp_path / "base")
    trained = module.Gr00tN1d7Processor.from_pretrained(
        tmp_path / "base",
        shortest_image_edge=224,
        crop_fraction=1.0,
        vlm_min_pixels=224 * 224,
    )
    assert trained.image_crop_size is None and trained.image_target_size is None
    assert trained.shortest_image_edge == 224 and trained.crop_fraction == 1.0
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    assert trained.eval_image_transform(image=image)["image"].shape == (224, 224, 3)
    assert grid(trained.processor) == grid(trained.collator.processor) == [[1, 14, 14]]
    trained.save_pretrained(tmp_path / "trained")
    reloaded = module.Gr00tN1d7Processor.from_pretrained(tmp_path / "trained")
    assert reloaded.vlm_min_pixels == 50176
    assert grid(reloaded.processor) == grid(reloaded.collator.processor) == [[1, 14, 14]]
    assert reloaded.shortest_image_edge == 224 and reloaded.crop_fraction == 1.0


def test_invalid_pixel_bound_fails_before_hub_access(monkeypatch):
    loader = Mock()
    monkeypatch.setattr(module, "Qwen3VLProcessor", loader)
    with pytest.raises(ValueError, match="positive"):
        module.build_processor("local", {}, 0)
    loader.from_pretrained.assert_not_called()
