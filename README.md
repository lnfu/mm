# mm

Grasping a Pringles using Hello Robot Stretch 4.
The robot finds an object with the gripper camera and moves the gripper to it.

Machines:

- **Dev PC**: this repository.
- **GPU PC**: runs the detector.
- **Robot**: runs the driver, the camera and the grasp server.

## Install (one time)

On the dev PC:

1. Copy `.env.example` to `.env`. Set the hosts and `GEMINI_API_KEY` in `.env`.
   `scripts/deploy.sh robot` copies `.env` to the robot.
2. Install the Python packages on the GPU PC and the robot:
   ```
   scripts/setup_venv.sh all
   ```

## Deploy

On the dev PC, after each change to the code:

```
scripts/deploy.sh all --build
```

## Run a grasp

In each terminal on the GPU PC or the robot that uses `mm_ws`, do this command first:

```
cd ~/mm_ws && source /opt/ros/jazzy/setup.bash && source .venv/bin/activate && source install/setup.bash
```

1. On the robot, terminal 1, start the driver:
   ```
   ros2 launch stretch_core stretch_driver.launch.py action_timeout:=15.0
   ```
2. On the robot, terminal 2, start the gripper camera:
   ```
   ros2 launch stretch_core gripper_camera.launch.py
   ```
3. On the GPU PC, start the detector. Wait until it shows `ready`.
   ```
   ros2 run mm_perception detector
   ```
4. On the robot, terminal 3, start the server. Wait until it shows `READY`.
   The first command loads `GEMINI_API_KEY`, which only articulate needs.
   ```
   set -a && source .env && set +a
   ros2 run mm_manip manipulation_server --ros-args --params-file src/mm_manip/config/manipulation.yaml -p dry_run:=false -p grasp.base_dofs:=x,y,theta
   ```
5. On the robot, terminal 4, send a grasp goal:
   ```
   ros2 action send_goal --feedback /mm_manip/grasp mm_interfaces/action/Grasp '{prompt: pringles}'
   ```

## Open a door

Do steps 1 to 4 of [Run a grasp](#run-a-grasp) (the detector is not needed), then
send an articulate goal on the robot:

```
ros2 action send_goal --feedback /mm_manip/articulate mm_interfaces/action/Articulate '{prompt: the cabinet door}'
```

Without the camera, give the handle and a point on the hinge axis in `base_link`:

```
ros2 action send_goal --feedback /mm_manip/articulate mm_interfaces/action/Articulate \
  '{mode: 1, handle: {header: {frame_id: base_link}, point: {x: 0.82, y: -0.09, z: 0.75}}, hinge: {header: {frame_id: base_link}, point: {x: 0.82, y: 0.21, z: 0.75}}}'
```

`open_angle` (rad) sets how far to open; 0 opens a quarter turn. The result says
how far the plan got, e.g. `opened: 60 of 90 deg`.

## Change parameters

Do these commands on the robot before you send a goal. The values stay until the grasp server stops.

- Set the lift height of the start pose:
  ```
  ros2 param set /mm_manip grasp.start_pose.lift_joint 0.8
  ```
- Do not lift the object after the grasp:
  ```
  ros2 param set /mm_manip grasp.lift_height 0.0
  ```

## Retract the arm

On the robot, do these commands in sequence:

```
ros2 action send_goal /follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{trajectory: {joint_names: [arm_joint], points: [{positions: [0.05], time_from_start: {sec: 3}}]}}"
ros2 action send_goal /follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{trajectory: {joint_names: [wrist_yaw_joint], points: [{positions: [0.0], time_from_start: {sec: 3}}]}}"
```

If a command shows `Goal timed out`, the joint is already at the position.
