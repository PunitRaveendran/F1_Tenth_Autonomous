"""
SLAM Mapping Launch File for F1TENTH

Launches the simulator bridge (single agent), SLAM Toolbox in online async
mapping mode, and RViz for visualization.  Drive the car with teleop to
build a map, then save it with:

    ros2 run nav2_map_server map_saver_cli -f <map_name>
"""

import os
import pathlib
import tempfile
import yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _patch_numba_coverage_compat():
    """Prevent coverage/numba crash on import."""
    try:
        import coverage.types as coverage_types
    except Exception:
        return
    for name in ('Tracer', 'TShouldTraceFn', 'TShouldStartContextFn'):
        if not hasattr(coverage_types, name):
            setattr(coverage_types, name, object)


def _resolve_yaml_path(base_path: pathlib.Path) -> pathlib.Path:
    if base_path.suffix in ('.yaml', '.yml'):
        return base_path
    candidates = (
        base_path.with_suffix('.yaml'),
        base_path.with_suffix('.yml'),
        base_path.parent / f"{base_path.stem}_map.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_path.with_suffix('.yaml')


def _resolve_map_yaml_path(map_path: str, package_share: str) -> pathlib.Path:
    if map_path.startswith(('/', '\\')):
        return _resolve_yaml_path(pathlib.Path(map_path))
    if '/' in map_path or '\\' in map_path:
        return _resolve_yaml_path(pathlib.Path(package_share) / map_path)
    try:
        _patch_numba_coverage_compat()
        from f1tenth_gym.envs.track.utils import find_track_dir
        track_dir = find_track_dir(map_path)
        yaml_path = track_dir / f"{track_dir.stem}.yaml"
        if not yaml_path.exists():
            yaml_path = track_dir / f"{track_dir.stem}_map.yaml"
        return yaml_path
    except Exception:
        return _resolve_yaml_path(pathlib.Path(package_share) / 'maps' / map_path)


def generate_launch_description():
    ld = LaunchDescription()
    package_share = get_package_share_directory('f1tenth_gym_ros')

    # ── Read sim.yaml for map + vehicle config ────────────────
    config = os.path.join(package_share, 'config', 'sim.yaml')
    with open(config, 'r') as f:
        config_dict = yaml.safe_load(f)

    bridge_params = config_dict['bridge']['ros__parameters']
    use_sim_time = bridge_params.get('use_sim_time', False)

    # ── Simulator bridge (single agent for mapping) ───────────
    # Override: force 1 agent and enable teleop for SLAM mapping
    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[config, {
            'use_sim_time': False,
            'use_sim_time_bridge': use_sim_time,
            'num_agent': 1,
            'kb_teleop': True,
        }],
    )

    # ── Map server (for the base map in sim) ──────────────────
    map_path = bridge_params['map_path']
    map_yaml_path = _resolve_map_yaml_path(map_path, package_share)
    with open(map_yaml_path, 'r') as f:
        map_yaml = yaml.safe_load(f)

    scale = bridge_params.get('scale', 1.0)
    map_yaml['resolution'] *= scale
    origin = map_yaml['origin']
    map_yaml['origin'] = (origin[0] * scale, origin[1] * scale, origin[2])

    image_field = map_yaml.get('image')
    image_path = pathlib.Path(image_field)
    if not image_path.is_absolute():
        image_path = map_yaml_path.parent / image_path
    map_img_ext = image_path.suffix
    map_yaml['image'] = 'scaled_map' + map_img_ext

    temp_dir = tempfile.mkdtemp(prefix='f1tenth_gym_ros_slam_map_')
    temp_yaml_path = os.path.join(temp_dir, 'scaled_map.yaml')
    temp_img_path = os.path.join(temp_dir, 'scaled_map' + map_img_ext)

    with open(temp_yaml_path, 'w') as f:
        yaml.dump(map_yaml, f)
    with open(temp_img_path, 'wb') as f:
        with open(image_path, 'rb') as img:
            f.write(img.read())

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[
            {'yaml_filename': temp_yaml_path},
            {'topic': 'map'},
            {'frame_id': 'map'},
            {'output': 'screen'},
            {'use_sim_time': use_sim_time},
        ],
    )
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['map_server']},
        ],
    )

    # ── Robot state publisher ─────────────────────────────────
    vehicle = bridge_params.get('vehicle_params', 'f1tenth')
    xacro_map = {
        'f1tenth': 'ego_racecar.xacro',
        'fullscale': 'ego_racecar_fullscale.xacro',
        'f1fifth': 'ego_racecar_f1fifth.xacro',
    }
    ego_xacro = xacro_map.get(vehicle, 'ego_racecar.xacro')

    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[
            {'robot_description': Command([
                'xacro ',
                os.path.join(package_share, 'urdf', ego_xacro),
            ])},
            {'use_sim_time': use_sim_time},
        ],
        remappings=[('/robot_description', 'ego_robot_description')],
    )

    # ── SLAM Toolbox (online async) ───────────────────────────
    slam_config = os.path.join(package_share, 'config', 'slam_toolbox_mapping.yaml')
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_config, {'use_sim_time': use_sim_time}],
        # The gym bridge publishes LiDAR on /ego_racecar/scan,
        # but SLAM Toolbox expects /scan by default.
        remappings=[('/scan', '/ego_racecar/scan')],
    )

    # ── RViz ──────────────────────────────────────────────────
    rviz_config = os.path.join(package_share, 'config', 'rviz', 'gym_bridge.rviz')
    launch_rviz_arg = LaunchConfiguration('launch_rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', rviz_config],
        condition=IfCondition(launch_rviz_arg),
    )

    # ── Declare launch arguments ──────────────────────────────
    ld.add_action(DeclareLaunchArgument(
        'launch_rviz', default_value='true',
        description='Launch RViz for SLAM visualization (default: true).',
    ))

    # ── Add nodes ─────────────────────────────────────────────
    ld.add_action(LogInfo(msg='\033[32m[SLAM] Starting F1TENTH SLAM mapping pipeline...\033[0m'))
    ld.add_action(bridge_node)
    ld.add_action(map_server_node)
    ld.add_action(nav_lifecycle_node)
    ld.add_action(ego_robot_publisher)
    ld.add_action(slam_node)
    ld.add_action(rviz_node)

    return ld
