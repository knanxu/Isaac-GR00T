from unittest.mock import Mock

from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as module
import pytest


@pytest.mark.parametrize("offline,local_only", [(True, False), (False, True)])
def test_offline_repo_id_resolves_cached_snapshot(monkeypatch, offline, local_only):
    processor = Mock()
    snapshot = Mock(return_value="/cache/snapshots/pinned")
    monkeypatch.setattr(module, "Qwen3VLProcessor", processor)
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    monkeypatch.setattr(module, "is_offline_mode", lambda: offline)
    kwargs = {"local_files_only": local_only, "revision": "pinned", "cache_dir": "/cache"}
    module.build_processor("nvidia/Cosmos-Reason2-2B", kwargs)
    snapshot.assert_called_once_with(
        "nvidia/Cosmos-Reason2-2B", local_files_only=True, revision="pinned", cache_dir="/cache"
    )
    processor.from_pretrained.assert_called_once_with("/cache/snapshots/pinned", **kwargs)


def test_existing_local_directory_does_not_resolve_repo(monkeypatch, tmp_path):
    processor, snapshot = Mock(), Mock()
    monkeypatch.setattr(module, "Qwen3VLProcessor", processor)
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    monkeypatch.setattr(module, "is_offline_mode", lambda: True)
    module.build_processor(str(tmp_path), {"local_files_only": True})
    snapshot.assert_not_called()
    processor.from_pretrained.assert_called_once_with(str(tmp_path), local_files_only=True)


def test_online_repo_loading_keeps_original_behavior(monkeypatch):
    processor, snapshot = Mock(), Mock()
    monkeypatch.setattr(module, "Qwen3VLProcessor", processor)
    monkeypatch.setattr(module, "snapshot_download", snapshot)
    monkeypatch.setattr(module, "is_offline_mode", lambda: False)
    module.build_processor("nvidia/Cosmos-Reason2-2B", {})
    snapshot.assert_not_called()
    processor.from_pretrained.assert_called_once_with("nvidia/Cosmos-Reason2-2B")
