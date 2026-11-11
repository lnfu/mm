class Canceled(Exception):
    pass


class SkillError(RuntimeError):
    """An expected failure. The goal aborts with this message."""


class MotionError(SkillError):
    pass


class PerceptionError(SkillError):
    pass


class OutOfReach(SkillError):
    pass


class GraspFailed(SkillError):
    pass


class ArticulateFailed(SkillError):
    pass
