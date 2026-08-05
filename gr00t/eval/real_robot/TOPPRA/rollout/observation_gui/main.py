import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="Gym has been unmaintained")

import rospy

from agents.lerobuffer import LeroBuffer
from agents.rollout_passive import PassiveRolloutAgent
from gui import ObservationGUI
from gui.rollout_control import RolloutControlPanel, configured_namespace
from observations import (
    LeftDeltaTCP,
    LeftFingerPressure,
    LeftTCP,
    LeftWristForce,
    ObservationController,
    ParcelCamera,
    ParcelLeftPinch,
    ParcelRightPinch,
    RightDeltaTCP,
    RightFingerPressure,
    RightTCP,
    RightWristForce,
)


def main() -> None:
    rospy.init_node("learning_framework_parcel_rollout_view")
    controller = ObservationController(
        [
            ParcelCamera(
                "head",
                "/zj_humanoid/sensor/realsense_head/color/image_raw/compressed",
            ),
            ParcelCamera(
                "chest",
                "/zj_humanoid/sensor/realsense_up/color/image_raw/compressed",
            ),
            ParcelCamera(
                "left",
                "/zj_humanoid/sensor/left_wrist/image_raw/compressed",
            ),
            ParcelCamera(
                "right",
                "/zj_humanoid/sensor/right_wrist/image_raw/compressed",
            ),
            LeftTCP(),
            RightTCP(),
            LeftDeltaTCP(),
            RightDeltaTCP(),
            LeftFingerPressure(),
            RightFingerPressure(),
            ParcelLeftPinch(visible=False),
            ParcelRightPinch(visible=False),
            LeftWristForce(deadzone=10),
            RightWristForce(deadzone=10),
        ]
    )
    buffer = LeroBuffer(
        "parcel_rollout_view",
        "datasets/parcel_rollout_view",
        controller.generate_spec(),
        fps=30,
    )
    agent = PassiveRolloutAgent()
    rollout_panel = RolloutControlPanel(configured_namespace())
    gui = ObservationGUI(
        controller,
        buffer,
        agent,
        fps=30,
        default_task="move parcel onto conveyor belt one by one",
        modules=[rollout_panel],
    )
    gui.run()


if __name__ == "__main__":
    main()
