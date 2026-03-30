# Supported Robots

All 52 built-in robot configurations, organized by manufacturer.

## Universal Robots

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `ur3e` | UR3e | 6 | OPW | Compact tabletop arm |
| `ur5e` | UR5e | 6 | OPW | Most popular UR model |
| `ur10e` | UR10e | 6 | OPW | Long reach (1300mm) |
| `ur16e` | UR16e | 6 | OPW | Heavy payload (16kg) |
| `ur20` | UR20 | 6 | OPW | Next-gen 20kg payload |
| `ur30` | UR30 | 6 | OPW | Next-gen 30kg payload |

## Franka Emika

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `franka_panda` | Panda | 7 | DLS | Research-grade, torque sensing |

## KUKA

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `kuka_iiwa7` | LBR iiwa 7 | 7 | DLS | 7-DOF collaborative |
| `kuka_iiwa14` | LBR iiwa 14 | 7 | DLS | 14kg payload variant |
| `kuka_kr6` | KR 6 | 6 | OPW | Industrial 6kg |

## ABB

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `abb_irb1200` | IRB 1200 | 6 | OPW | Compact industrial |
| `abb_irb4600` | IRB 4600 | 6 | OPW | High payload industrial |
| `abb_yumi_left` | YuMi (left) | 7 | DLS | Dual-arm collaborative |
| `abb_yumi_right` | YuMi (right) | 7 | DLS | Dual-arm collaborative |

## Fanuc

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `fanuc_crx10ia` | CRX-10iA | 6 | OPW | Collaborative |
| `fanuc_lr_mate_200id` | LR Mate 200iD | 6 | OPW | Compact industrial |

## Yaskawa/Motoman

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `yaskawa_gp7` | GP7 | 6 | OPW | General purpose |
| `yaskawa_hc10` | HC10 | 6 | OPW | Collaborative 10kg |

## UFactory (xArm)

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `xarm5` | xArm 5 | 5 | DLS | 5-DOF variant |
| `xarm6` | xArm 6 | 6 | OPW | 6-DOF variant |
| `xarm7` | xArm 7 | 7 | DLS | 7-DOF redundant |

## Kinova

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `kinova_gen3` | Gen3 | 7 | DLS | 7-DOF research arm |
| `kinova_gen3_lite` | Gen3 Lite | 6 | DLS | 6-DOF lightweight |
| `jaco2_6dof` | Jaco2 | 6 | DLS | Assistive robotics |

## Trossen Robotics

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `trossen_px100` | PincherX 100 | 4 | DLS | Budget tabletop |
| `trossen_rx150` | ReactorX 150 | 5 | DLS | Mid-range |
| `trossen_wx250s` | WidowX 250s | 6 | OPW | Research platform |
| `viperx_300` | ViperX 300 | 6 | OPW | 6-DOF research |
| `widowx_250` | WidowX 250 | 6 | OPW | Research platform |

## Research / Education

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `aloha_left` | ALOHA (left) | 6 | OPW | Bimanual teleop |
| `aloha_right` | ALOHA (right) | 6 | OPW | Bimanual teleop |
| `koch_v1` | Koch v1 | 6 | OPW | Open-source arm |
| `lerobot_so100` | LeRobot SO-100 | 6 | DLS | AI training arm |
| `so_arm100` | SO-ARM100 | 5 | DLS | Compact research |
| `open_manipulator_x` | OpenMANIPULATOR-X | 4 | DLS | ROBOTIS, 4-DOF |
| `robotis_open_manipulator_p` | OpenMANIPULATOR-P | 6 | OPW | ROBOTIS, 6-DOF |
| `mycobot_280` | myCobot 280 | 6 | OPW | Elephant Robotics |

## Rethink Robotics

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `baxter_left` | Baxter (left) | 7 | DLS | Dual-arm research |
| `baxter_right` | Baxter (right) | 7 | DLS | Dual-arm research |
| `sawyer` | Sawyer | 7 | DLS | Single-arm research |

## Other Manufacturers

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `denso_vs068` | Denso VS068 | 6 | OPW | Compact industrial |
| `dobot_cr5` | Dobot CR5 | 6 | OPW | Collaborative |
| `elite_ec66` | Elite EC66 | 6 | OPW | Collaborative |
| `flexiv_rizon4` | Flexiv Rizon 4 | 7 | DLS | Force-controlled |
| `meca500` | Mecademic Meca500 | 6 | OPW | Micro industrial |
| `niryo_ned2` | Niryo Ned2 | 6 | OPW | Education |
| `staubli_tx260` | Staubli TX2-60 | 6 | OPW | High-speed |
| `techman_tm5_700` | Techman TM5-700 | 6 | OPW | Collaborative with vision |

## Mobile Manipulators

| Config Name | Model | DOF | IK Solver | Notes |
|-------------|-------|-----|-----------|-------|
| `fetch` | Fetch | 8 | DLS | Mobile base + 7-DOF arm |
| `pr2` | PR2 | 8 | DLS | Willow Garage dual-arm |
| `stretch_re2` | Stretch RE2 | 5 | DLS | Hello Robot, mobile |
| `tiago` | TIAGo | 8 | DLS | PAL Robotics mobile |

## Usage

```rust
use kinetic::prelude::*;

// Load any robot by config name
let robot = Robot::from_name("ur5e")?;

// Check available named poses
if let Some(joints) = robot.named_pose("home") {
    println!("Home: {:?}", joints);
}
```

## Adding Your Own

See the Custom Robots guide for step-by-step instructions on adding a new
robot configuration. Contributions are welcome via pull request.
