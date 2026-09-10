"""Training modalities matching the parcel_4f_v5.9 checkpoint and WA1 client."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


register_modality_config(
    {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["head", "left", "right"]),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["left_tcp", "left_wrist_force", "right_tcp", "right_wrist_force"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(40)),
            modality_keys=["left_delta_tcp", "left_pinch", "right_delta_tcp", "right_pinch"],
            # Deltas are computed by the converter. The processor must leave them as-is.
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
                for _ in range(4)
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    },
    embodiment_tag=EmbodimentTag.NAVIAI_WA1_HEAD_LR_WF,
)
