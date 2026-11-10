# SO-101 + OInK demo

```
so101_oink/                   MuJoCo scene (two cubes), gym env, action-space encodings,
                              scripted pick expert, and oink_controller.py: the OInK loadouts
plugins/so101_oink_robot/     LeRobot Robot plugin (--robot.type=so101_oink) + OInK processor steps
plugins/keyboard_pose_teleop/ LeRobot Teleoperator plugin (--teleop.type=keyboard_pose)
scripts/                      check_kinematics / teleoperate / record / collect_scripted / eval_policy / inspect_policy
```

The MuJoCo model is the SO-101 from [robot-learning-101](https://github.com/sea-bass/robot-learning-101); the kinematic model RoboPlan solves on is the SO-101 URDF shipped with RoboPlan.
