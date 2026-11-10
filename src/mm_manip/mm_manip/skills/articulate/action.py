from functools import partial

import numpy as np

from mm_interfaces.action import Articulate

from ...errors import PerceptionError
from ..action import SkillAction, motion_message
from . import skill
from .config import ArticulateConfig
from .door import DoorType, HandleType
from .gemini import GeminiClient, GeminiError
from .keypoints import GeminiDoorKeypoints
from .locator import DoorLocator

assert all(getattr(Articulate.Result, f"TYPE_{t.name}") == t for t in DoorType)
assert all(getattr(Articulate.Result, f"HANDLE_{h.name}") == h for h in HandleType)


class ArticulateAction(SkillAction):
    action = Articulate
    config = ArticulateConfig
    phases = skill.Phase

    def __init__(self, node, runtime, startup):
        super().__init__(node, runtime, startup)
        self._doors = DoorLocator(node, runtime.camera, ee_frame=startup.frames.ee)

    def goal(self, request) -> skill.ArticulateGoal:
        if request.mode == Articulate.Goal.MODE_POINT:
            to_base = self.runtime.scene.to_base
            return skill.ArticulateGoal(
                handle=to_base(request.handle.header, request.handle.point),
                hinge=to_base(request.hinge.header, request.hinge.point),
                open_angle=request.open_angle,
            )
        return skill.ArticulateGoal(
            prompt=request.prompt, open_angle=request.open_angle
        )

    def locator(self, cfg):
        return partial(self._locate, cfg)

    def _locate(self, cfg, prompt, cancel):
        try:
            client = GeminiClient(
                cfg.gemini.model, cfg.gemini.thinking_level, cfg.gemini.timeout
            )
        except GeminiError as e:
            raise PerceptionError(f"gemini: {e}") from e
        return self._doors.locate(
            prompt,
            GeminiDoorKeypoints(client, max_side=cfg.gemini.image_max_side),
            cancel,
            depth_band=cfg.door.depth_band,
            shrink=cfg.door.quad_shrink,
            debug_dir=cfg.door.debug_dir,
        )

    def run(self, goal, cfg, shared, ctx):
        return skill.run(goal, cfg, shared, ctx)

    def message(self, outcome, shared) -> str:
        opened, asked = np.rad2deg(outcome.opened), np.rad2deg(outcome.asked)
        return f"{motion_message(shared, 'opened')}: {opened:.0f} of {asked:.0f} deg"

    def fill_result(self, result, outcome, kept):
        door = outcome.door if outcome is not None else kept.get("door")
        if door is not None:
            result.type = int(door.type)
            result.handle_type = int(door.handle_type)
