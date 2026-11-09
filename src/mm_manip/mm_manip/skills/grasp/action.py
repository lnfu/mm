from geometry_msgs.msg import PointStamped

from mm_interfaces.action import Grasp

from ...config import GraspConfig
from ..action import SkillAction, motion_message
from . import skill
from .locator import Locator


class GraspAction(SkillAction):
    action = Grasp
    config = GraspConfig
    phases = skill.Phase

    def __init__(self, node, runtime, shared):
        super().__init__(node, runtime, shared)
        self._locator = Locator(
            node,
            runtime.camera,
            shared.detect_service,
            service_timeout=shared.service_timeout,
            depth_band=shared.depth_band,
            callback_group=runtime.callback_group,
        )

    def goal(self, request) -> skill.GraspGoal:
        if request.mode == Grasp.Goal.MODE_POINT:
            point = request.target_point
            return skill.GraspGoal(
                point=self.runtime.scene.to_base(point.header, point.point)
            )
        return skill.GraspGoal(prompt=request.prompt)

    def locator(self, cfg):
        return self._locator.locate

    def run(self, goal, cfg, shared, ctx):
        return skill.run(goal, cfg, shared, ctx)

    def message(self, outcome, shared) -> str:
        return motion_message(shared, "grasped")

    def fill_result(self, result, outcome, kept):
        if outcome is None:
            return
        msg = PointStamped()
        msg.header.frame_id = self.runtime.scene.base_frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = (float(v) for v in outcome)
        result.grasped_point = msg
