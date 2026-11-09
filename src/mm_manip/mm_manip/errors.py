class Canceled(Exception):
    pass


class SkillError(RuntimeError):
    """An expected failure. The goal aborts with this message."""
