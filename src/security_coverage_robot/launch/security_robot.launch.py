import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('security_coverage_robot')
    turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    launch_file_dir = os.path.join(turtlebot3_gazebo, 'launch')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    world_file = LaunchConfiguration('world_file')

    declare_world_file = DeclareLaunchArgument(
        'world_file',
        default_value='world1_simple.world',
        description='World file name (in the package worlds folder)'
    )

    # Resolve the world file path from the launch arg
    world_path = PathJoinSubstitution([pkg_share, 'worlds', world_file])

    # Gazebo (new gz-sim) server
    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': ['-r -s -v2 ', world_path],
            'on_exit_shutdown': 'true'
        }.items()
    )

    # Gazebo client (GUI)
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': '-g -v2 ',
            'on_exit_shutdown': 'true'
        }.items()
    )

    # Robot state publisher
    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()
    )

    # Spawn TurtleBot3
    spawn_turtlebot_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': x_pose,
            'y_pose': y_pose
        }.items()
    )

    # Make TB3 models visible to gz-sim
    set_env_vars_resources = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(turtlebot3_gazebo, 'models')
    )

    # ros_gz bridge — without this our nodes get NO scan / odom / image, and
    # /cmd_vel never reaches the simulated robot. Topic names below are the
    # defaults exposed by the standard TB3 SDF; if your SDF publishes under
    # /model/<robot>/... you'll need to remap these accordingly.
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Our nodes
    common_params = [{'use_sim_time': use_sim_time}]

    perception = Node(
        package='security_coverage_robot',
        executable='perception_node',
        name='perception_node',
        output='screen',
        parameters=common_params,
    )

    planner = Node(
        package='security_coverage_robot',
        executable='planner_node',
        name='planner_node',
        output='screen',
        parameters=common_params,
    )

    control = Node(
        package='security_coverage_robot',
        executable='control_node',
        name='control_node',
        output='screen',
        parameters=common_params,
    )

    metrics = Node(
        package='security_coverage_robot',
        executable='metrics_node',
        name='metrics_node',
        output='screen',
        parameters=common_params,
    )

    ld = LaunchDescription()
    ld.add_action(declare_world_file)
    ld.add_action(set_env_vars_resources)
    ld.add_action(gzserver_cmd)
    ld.add_action(gzclient_cmd)
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(spawn_turtlebot_cmd)
    ld.add_action(gz_bridge)
    ld.add_action(perception)
    ld.add_action(planner)
    ld.add_action(control)
    ld.add_action(metrics)
    return ld