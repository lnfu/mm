import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from ..config import StartupConfig
from ..errors import SkillError
from ..runtime import build


def run_tool(main, *configs) -> int:
    """Run main(node, runtime, StartupConfig, *configs) on a node named mm_manip.

    Returns the exit code.

    The node takes the server's name, so the server's params file applies.
    Stop manipulation_server first.
    """
    rclpy.init()
    node = rclpy.create_node("mm_manip")
    for config in (StartupConfig, *configs):
        config.declare(node)
    runtime = build(node, StartupConfig.from_node(node))
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        return main(
            node, runtime, *(c.from_node(node) for c in (StartupConfig, *configs))
        )
    except KeyboardInterrupt:
        runtime.robot.stop()
        print("interrupted: motion canceled")
        return 1
    except (SkillError, TimeoutError) as e:
        runtime.robot.stop()
        print(f"error: {e}")
        return 1
    finally:
        # Stop the executor before the node is destroyed.
        executor.shutdown()
        spin.join(timeout=2.0)
        node.destroy_node()
        rclpy.try_shutdown()


def wait_for(get, what: str, timeout: float = 5.0):
    """The first value get() returns without SkillError; TimeoutError after `timeout` s."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            value = get()
        except SkillError:
            value = None
        if value is not None:
            return value
        time.sleep(0.05)
    raise TimeoutError(f"no {what} after {timeout} s (is stretch_driver running?)")
