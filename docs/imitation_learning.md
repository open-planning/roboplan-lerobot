# Imitation learning with OInK in the loop

Code: [`so101_oink_demo/`](../so101_oink_demo).

Teleoperate a simulated SO-101 with the keyboard, record demonstrations as LeRobot datasets, train a diffusion policy, and roll it back out in the sim.
Every joint command along the way, from teleop to policy rollout, comes out of RoboPlan's OInK solver.

The scene holds a **red** and a **green** cube in random slots; the episode's *target color* decides which one must be lifted, and each episode stores the matching instruction ("Pick up the red cube and lift it.").
The sim is integrated like real hardware: a `Robot` plugin (`--robot.type=so101_oink`) and a `Teleoperator` plugin (`--teleop.type=keyboard_pose`) that LeRobot's CLIs auto-discover.

## What OInK does here

[OInK](https://roboplan.readthedocs.io/en/latest/concepts/inverse_kinematics.html) solves a small quadratic program at every control step for a joint displacement that pursues *tasks* under *constraints* and *barriers*.
It runs at 100 Hz (4 solves per 25 Hz policy tick) in one of two loadouts, defined in [`oink_controller.py`](../so101_oink_demo/so101_oink/oink_controller.py):

| | `joint` | `cartesian` |
|---|---|---|
| Dataset action | 6 joint positions, degrees | gripper-frame pose (xyz + two rotation columns) + gripper |
| Primary task | `ConfigurationTask` toward the target configuration | `FrameTask` on `gripper_frame_link` toward the target pose |
| Nullspace task | - | `ConfigurationTask` toward the home posture |
| Constraints | `PositionLimit`, `VelocityLimit` | same |
| Barriers | `SelfCollisionBarrier` (closest link pairs, 1 cm), `PositionBarrier` (gripper and jaw above the table) | same |

The same solver plays three roles:

1. **Teleop**: the keyboard moves a virtual end-effector target and a `FrameTask` follows it. For joint datasets this happens in the teleop processor pipeline, so the dataset stores joints; for Cartesian datasets the robot tracks the pose itself.
2. **Scripted data collection**: the expert commands poses, and the same pipeline records them in either space.
3. **Policy execution**: each policy action becomes the task target for one tick. A network output below the table or through the arm turns into the closest safe motion.

The barriers are [control barrier functions](https://roboplan.readthedocs.io/en/latest/concepts/inverse_kinematics.html#control-barrier-functions-barriers): inequality constraints that limit how fast the arm may approach a boundary as a function of its distance to it, so motion slows smoothly into the boundary instead of stopping dead.

Every `OinkConfig` field is a command-line flag: `--oink.table_z=0.02`, `--oink.self_collision=false`, `--oink.max_joint_velocity=1.5`.

## Setup

One pixi environment ([`pixi.toml`](../pixi.toml)) holds RoboPlan's Python bindings from conda-forge, LeRobot from PyPI, MuJoCo, and this package with both plugins installed editable.
The `so101-*` tasks run inside `so101_oink_demo/`, so the `data/` and `outputs/` paths below land there.
Extra arguments are passed through.

```bash
pixi run so101-check
```

checks that RoboPlan's forward kinematics of the RoboPlan SO-101 URDF matches the MuJoCo gripper frame, that both loadouts converge, and that the table barrier holds.

**Remote machines**: add `--viewer viser` to any task to get a browser viewer instead of the MuJoCo window; the script prints the URL and keyboard teleop works from the browser tab.

## 1. Practice teleoperating

```bash
pixi run so101-teleoperate
pixi run so101-teleoperate --action-space cartesian
```

| key | action |
|---|---|
| ↑ / ↓, ← / → | end-effector forward / back, left / right |
| w / s | end-effector up / down |
| q / e | pitch gripper down / up |
| a / d | roll jaws |
| space | toggle gripper |
| x | reset scene (next target color) |
| ESC | quit |

Type in the terminal that launched the script; the viewer window keeps its own shortcuts.

Push the gripper into the table or fold the wrist back onto the forearm: the arm slows and stops, and the barrier values printed in the terminal go to zero.

**Grasping**: pitch down with `q` until the gripper is near-vertical, descend with the open jaws *behind* the cube, nudge forward so the cube sits between the jaw tips, close, lift.

## 2. Record datasets

Pick the action space when recording (`--action-space joint`, the default, or `cartesian`); training and evaluation read it from the data.
The observation always holds the joint positions and the gripper-frame pose.

```bash
pixi run so101-record --repo-id you/so101_pick_joint --root data/pick_joint --episodes 100 --display-data
pixi run so101-record --repo-id you/so101_pick_cart --root data/pick_cart --episodes 100 --action-space cartesian
```

**n** saves an episode, **r** discards and re-records it, **ESC** stops and finalizes the dataset.

A scripted expert produces the same datasets headlessly (about two minutes per 100 episodes); `--show-viewer` watches it:

```bash
pixi run so101-collect --repo-id you/so101_pick_joint --root data/pick_joint --episodes 100 --perturb 0.5
pixi run so101-collect --repo-id you/so101_pick_cart --root data/pick_cart --episodes 100 --perturb 0.5 --action-space cartesian
```

The expert never misses, so a dataset of clean picks teaches no recovery: a policy that fumbles a grasp hovers instead of trying again.
`--perturb 0.5` slides the cube away mid-descent in half the episodes, and the expert's retry gets recorded.

Replay an episode with rerun:

```bash
pixi run so101-viz --repo-id you/so101_pick_cart --root data/pick_cart --episode-index 0
```

## 3. Train a policy

`lerobot-train` does not care which action space the dataset uses; the policy's output dimension follows the data.

```bash
pixi run so101-train \
    --dataset.repo_id=you/so101_pick_joint --dataset.root=data/pick_joint \
    --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
    --output_dir=outputs/train/so101_diffusion_joint --job_name=so101_diffusion_joint \
    --batch_size=16 --steps=10000 --save_freq=2500 \
    --trackio.enable=true --trackio.project=so101_oink
```

Swap `pick_joint` for `pick_cart` (and the job name) for the Cartesian policy.
Trackio keeps the loss curves in a local database (no account needed); `pixi run trackio show --project so101_oink` opens the dashboard.
Diffusion uses about 6.3 GB of VRAM at batch 16 and runs at 4 steps/s on a laptop RTX 5070, so 10k steps take about 40 minutes.

ACT trains on the same datasets (2.3 GB, 9 steps/s) with two changes to LeRobot's defaults.
`--policy.n_action_steps=10` replans every 0.4 s instead of executing the whole 100-step chunk open loop (4 s at 25 Hz), so the policy can react to a fumbled grasp.
The preset optimizer is a constant 1e-5 learning rate sized for 100k-step runs; this gives ACT the same warmup-and-cosine schedule diffusion uses, which converges in 10k.

```bash
pixi run so101-train \
    --dataset.repo_id=you/so101_pick_joint --dataset.root=data/pick_joint \
    --policy.type=act --policy.n_action_steps=10 --policy.device=cuda --policy.push_to_hub=false \
    --use_policy_training_preset=false --optimizer.type=adamw --optimizer.lr=1e-4 --optimizer.weight_decay=1e-4 \
    --scheduler.type=diffuser --scheduler.name=cosine --scheduler.num_warmup_steps=500 \
    --output_dir=outputs/train/so101_act_joint --job_name=so101_act_joint \
    --batch_size=16 --steps=10000 --save_freq=2500 \
    --trackio.enable=true --trackio.project=so101_oink
```

## 4. Roll it out

Checkpoints live in `outputs/train/<job>/checkpoints/<step>/pretrained_model/`, with `last` pointing at the newest.

```bash
pixi run so101-eval --policy-path outputs/train/so101_diffusion_joint/checkpoints/last/pretrained_model --episodes 20
pixi run so101-eval --policy-path outputs/train/so101_diffusion_cart/checkpoints/last/pretrained_model --episodes 20
```

The action space is read from the checkpoint (6 outputs = joint, 10 = Cartesian) and the robot gets the matching loadout.
The seeds are fixed, so checkpoints are compared on identical scenes.
Add `--show` to watch, and change the solver at rollout time to see what the barriers do:

```bash
pixi run so101-eval --policy-path ... --episodes 10 --show --oink.table_z=0.02
pixi run so101-eval --policy-path ... --episodes 10 --show --oink.table_barrier=false --oink.self_collision=false
```

Diffusion samples each chunk with the 100 DDPM steps it was trained on, about 2 s on a laptop GPU.
`--num-inference-steps 10` samples with DDIM in 10 steps instead (0.2 s) from the same weights, no retraining; on the 50 evaluation episodes it scores 48 / 50 in both action spaces.

`pixi run so101-inspect --policy-path ...` prints a checkpoint's module tree and parameter counts.

### Results

Both policies trained on 100 perturbed demonstrations for 10k steps and evaluated on the same 50 held-out seeded episodes.
The gripper frame never crossed the table barrier, and the solver failed on 1 of 50,000 ticks and held the previous command.

| policy | action space | success |
|---|---|---|
| diffusion | joint | 48 / 50 (96%) |
| diffusion | cartesian | 50 / 50 (100%) |
| ACT | joint | 39 / 50 (78%) |
| ACT | cartesian | 42 / 50 (84%) |

Diffusion success climbs between 5k and 7.5k steps and more demonstrations or steps stay within evaluation noise; ACT is still improving at 10k (72% / 66% at 7.5k).
Without the perturbed demonstrations diffusion loses a few more episodes to fumbled grasps it never retries.
`--resume=true --config_path=outputs/train/<job>/checkpoints/last/pretrained_model/train_config.json` continues a run.

## Notes

- The SO-101 has five joints, so teleop and expert targets are *yaw-coupled* ([`ee_target.py`](../so101_oink_demo/so101_oink/ee_target.py)): the orientation target follows the direction of the target position plus the pitch and roll offsets, and the `FrameTask` weighs position 10x more than orientation.
- The self-collision barrier needs link distances every step, and mesh geometry makes that cost ~100 ms. [`collision_model.py`](../so101_oink_demo/so101_oink/collision_model.py) rewrites the URDF's collision geometry as boxes on load (~30 µs per solve).
- RoboPlan's barriers default to a safe-displacement damping term that is far larger than the task weights for an arm this small; the controller sets it to 0.
- The task gain is 0.25 per 10 ms solver step. The default of 1 is dead-beat: each 25 Hz policy action is reached in the first solver step and then held, and the arm visibly shakes during rollouts; 0.25 spreads the motion over the tick at no cost in success.
- Scripts that leave background threads running until exit (dataset writers, torch) end with a harmless `nanobind: leaked N instances!` report from the bindings; the eval script skips interpreter teardown (`os._exit`) to avoid it.
- The MuJoCo model (jaw collision pads, elliptic friction cone, gripper close speed) comes from [robot-learning-101](https://github.com/sea-bass/robot-learning-101), where those choices are explained.
