"""Errors shared by the skills and the hardware interfaces. No ROS."""


class Canceled(Exception):
    """The action goal was canceled; whatever moves must stop."""
