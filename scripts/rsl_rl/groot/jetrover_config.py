from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

jetrover_config = {
    # Video: single wrist camera only
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist"],  # Only wrist camera (matching real robot)
    ),
    # State: current joint positions
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["single_arm", "gripper", "goal_rel"],
    ),
    # Action: 16-step prediction horizon
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["single_arm", "gripper"],
        action_configs=[
            # Arm joints (0-4): RELATIVE delta from current state
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            # Gripper (5): ABSOLUTE target (binary open/close)
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(jetrover_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
