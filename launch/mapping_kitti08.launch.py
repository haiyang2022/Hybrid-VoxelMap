from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("hybrid_voxel_map")
    config_file = PathJoinSubstitution([pkg_share, "config", "kitti08.yaml"])
    rviz_config = PathJoinSubstitution([pkg_share, "rviz_cfg", "voxel_mapping.rviz"])
    default_result_dir = PathJoinSubstitution([EnvironmentVariable("HOME", default_value="."), "hvm_results"])
    default_sequence = PathJoinSubstitution(
        [EnvironmentVariable("HOME", default_value="."), "datasets", "KITTI", "dataset", "sequences", "08"]
    )
    default_video_dump_dir = PathJoinSubstitution([EnvironmentVariable("HOME", default_value="."), "hvm_video_dump"])

    sequence = LaunchConfiguration("sequence")
    lid_topic = LaunchConfiguration("lid_topic")
    publisher_topic = LaunchConfiguration("publisher_topic")
    rate = LaunchConfiguration("rate")
    start_index = LaunchConfiguration("start_index")
    end_index = LaunchConfiguration("end_index")
    gt_semantics = LaunchConfiguration("gt_semantics")
    label_dir = LaunchConfiguration("label_dir")
    plane_labels = LaunchConfiguration("plane_labels")
    down_sample_size = LaunchConfiguration("down_sample_size")
    max_iteration = LaunchConfiguration("max_iteration")
    max_distance = LaunchConfiguration("max_distance")
    voxel_size = LaunchConfiguration("voxel_size")
    plannar_threshold = LaunchConfiguration("plannar_threshold")
    update_size_threshold = LaunchConfiguration("update_size_threshold")
    init_plane_threshold = LaunchConfiguration("init_plane_threshold")
    sigma_num = LaunchConfiguration("sigma_num")
    abs_residual_gating = LaunchConfiguration("abs_residual_gating")
    use_step_convergence = LaunchConfiguration("use_step_convergence")
    min_iteration = LaunchConfiguration("min_iteration")
    converge_translation = LaunchConfiguration("converge_translation")
    converge_rotation = LaunchConfiguration("converge_rotation")
    iekf_step_scale = LaunchConfiguration("iekf_step_scale")
    min_effective_features = LaunchConfiguration("min_effective_features")
    max_solver_condition = LaunchConfiguration("max_solver_condition")
    skip_map_update_on_unstable = LaunchConfiguration("skip_map_update_on_unstable")
    ranging_cov = LaunchConfiguration("ranging_cov")
    angle_cov = LaunchConfiguration("angle_cov")
    calib_laser = LaunchConfiguration("calib_laser")
    calib_vertical_angle_deg = LaunchConfiguration("calib_vertical_angle_deg")
    kitti_vertical_filter = LaunchConfiguration("kitti_vertical_filter")
    blind = LaunchConfiguration("blind")
    point_filter_num = LaunchConfiguration("point_filter_num")
    publish_point_time = LaunchConfiguration("publish_point_time")
    use_lidar_constant_velocity = LaunchConfiguration("use_lidar_constant_velocity")
    use_lidar_constant_velocity_deskew = LaunchConfiguration("use_lidar_constant_velocity_deskew")
    publish_iteration_debug = LaunchConfiguration("publish_iteration_debug")
    semantic_debug = LaunchConfiguration("semantic_debug")
    semantic_debug_interval = LaunchConfiguration("semantic_debug_interval")
    debug_csv_enable = LaunchConfiguration("debug_csv_enable")
    debug_csv_dir = LaunchConfiguration("debug_csv_dir")
    debug_csv_prefix = LaunchConfiguration("debug_csv_prefix")
    tum_result_path = LaunchConfiguration("tum_result_path")
    efficiency_path = LaunchConfiguration("efficiency_path")
    per_frame_efficiency_enable = LaunchConfiguration("per_frame_efficiency_enable")
    per_frame_efficiency_path = LaunchConfiguration("per_frame_efficiency_path")
    video_dump_enable = LaunchConfiguration("video_dump_enable")
    video_dump_dir = LaunchConfiguration("video_dump_dir")
    video_dump_map_interval = LaunchConfiguration("video_dump_map_interval")
    video_dump_points_enable = LaunchConfiguration("video_dump_points_enable")
    video_dump_map_enable = LaunchConfiguration("video_dump_map_enable")
    video_dump_updates_enable = LaunchConfiguration("video_dump_updates_enable")

    return LaunchDescription([
        DeclareLaunchArgument(
            "sequence",
            default_value=default_sequence,
        ),
        DeclareLaunchArgument("lid_topic", default_value="/velodyne_points"),
        DeclareLaunchArgument("publisher_topic", default_value="/velodyne_points"),
        DeclareLaunchArgument("rate", default_value="10.0"),
        DeclareLaunchArgument("start_index", default_value="0"),
        DeclareLaunchArgument("end_index", default_value="-1"),
        DeclareLaunchArgument("gt_semantics", default_value="true"),
        DeclareLaunchArgument("label_dir", default_value=""),
        DeclareLaunchArgument("plane_labels", default_value="40,44,48,49,50,51,52,60,72"),
        DeclareLaunchArgument("down_sample_size", default_value="0.5"),
        DeclareLaunchArgument("max_iteration", default_value="20"),
        DeclareLaunchArgument("max_distance", default_value="120.0"),
        DeclareLaunchArgument("voxel_size", default_value="0.8"),
        DeclareLaunchArgument("plannar_threshold", default_value="0.002"),
        DeclareLaunchArgument("update_size_threshold", default_value="5"),
        DeclareLaunchArgument("init_plane_threshold", default_value="5"),
        DeclareLaunchArgument("sigma_num", default_value="3"),
        DeclareLaunchArgument("abs_residual_gating", default_value="true"),
        DeclareLaunchArgument("use_step_convergence", default_value="true"),
        DeclareLaunchArgument("min_iteration", default_value="3"),
        DeclareLaunchArgument("converge_translation", default_value="0.001"),
        DeclareLaunchArgument("converge_rotation", default_value="0.0001"),
        DeclareLaunchArgument("iekf_step_scale", default_value="0.5"),
        DeclareLaunchArgument("min_effective_features", default_value="0"),
        DeclareLaunchArgument("max_solver_condition", default_value="0.0"),
        DeclareLaunchArgument("skip_map_update_on_unstable", default_value="false"),
        DeclareLaunchArgument("ranging_cov", default_value="0.05"),
        DeclareLaunchArgument("angle_cov", default_value="0.1"),
        DeclareLaunchArgument("calib_laser", default_value="true"),
        DeclareLaunchArgument("calib_vertical_angle_deg", default_value="0.2"),
        DeclareLaunchArgument("kitti_vertical_filter", default_value="false"),
        DeclareLaunchArgument("blind", default_value="0.5"),
        DeclareLaunchArgument("point_filter_num", default_value="1"),
        # KITTI Odometry scans are already motion-compensated; do not synthesize point times.
        DeclareLaunchArgument("publish_point_time", default_value="false"),
        DeclareLaunchArgument("use_lidar_constant_velocity", default_value="true"),
        DeclareLaunchArgument("use_lidar_constant_velocity_deskew", default_value="false"),
        DeclareLaunchArgument("publish_iteration_debug", default_value="false"),
        DeclareLaunchArgument("semantic_debug", default_value="false"),
        DeclareLaunchArgument("semantic_debug_interval", default_value="100"),
        DeclareLaunchArgument("debug_csv_enable", default_value="false"),
        DeclareLaunchArgument("debug_csv_dir", default_value=default_result_dir),
        DeclareLaunchArgument("debug_csv_prefix", default_value="kitti08_hybrid_voxel_map_debug"),
        DeclareLaunchArgument("tum_result_path", default_value=""),
        DeclareLaunchArgument("efficiency_path", default_value=""),
        DeclareLaunchArgument("per_frame_efficiency_enable", default_value="false"),
        DeclareLaunchArgument("per_frame_efficiency_path", default_value=""),
        DeclareLaunchArgument("video_dump_enable", default_value="false"),
        DeclareLaunchArgument(
            "video_dump_dir",
            default_value=default_video_dump_dir,
        ),
        DeclareLaunchArgument("video_dump_map_interval", default_value="1"),
        DeclareLaunchArgument("video_dump_points_enable", default_value="true"),
        DeclareLaunchArgument("video_dump_map_enable", default_value="true"),
        DeclareLaunchArgument("video_dump_updates_enable", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("loop", default_value="false"),
        Node(
            package="hybrid_voxel_map",
            executable="voxel_mapping_odom",
            name="voxel_mapping_odom",
            output="screen",
            parameters=[
                config_file,
                {
                    "common.lid_topic": ParameterValue(lid_topic, value_type=str),
                    "mapping.down_sample_size": ParameterValue(down_sample_size, value_type=float),
                    "mapping.max_iteration": ParameterValue(max_iteration, value_type=int),
                    "mapping.max_distance": ParameterValue(max_distance, value_type=float),
                    "mapping.voxel_size": ParameterValue(voxel_size, value_type=float),
                    "mapping.plannar_threshold": ParameterValue(plannar_threshold, value_type=float),
                    "mapping.update_size_threshold": ParameterValue(update_size_threshold, value_type=int),
                    "mapping.init_plane_threshold": ParameterValue(init_plane_threshold, value_type=int),
                    "mapping.sigma_num": ParameterValue(sigma_num, value_type=int),
                    "mapping.abs_residual_gating": ParameterValue(abs_residual_gating, value_type=bool),
                    "mapping.use_step_convergence": ParameterValue(use_step_convergence, value_type=bool),
                    "mapping.min_iteration": ParameterValue(min_iteration, value_type=int),
                    "mapping.converge_translation": ParameterValue(converge_translation, value_type=float),
                    "mapping.converge_rotation": ParameterValue(converge_rotation, value_type=float),
                    "mapping.iekf_step_scale": ParameterValue(iekf_step_scale, value_type=float),
                    "mapping.min_effective_features": ParameterValue(min_effective_features, value_type=int),
                    "mapping.max_solver_condition": ParameterValue(max_solver_condition, value_type=float),
                    "mapping.skip_map_update_on_unstable": ParameterValue(skip_map_update_on_unstable, value_type=bool),
                    "noise_model.ranging_cov": ParameterValue(ranging_cov, value_type=float),
                    "noise_model.angle_cov": ParameterValue(angle_cov, value_type=float),
                    "preprocess.calib_laser": ParameterValue(calib_laser, value_type=bool),
                    "preprocess.calib_vertical_angle_deg": ParameterValue(calib_vertical_angle_deg, value_type=float),
                    "preprocess.kitti_vertical_filter": ParameterValue(kitti_vertical_filter, value_type=bool),
                    "preprocess.blind": ParameterValue(blind, value_type=float),
                    "preprocess.point_filter_num": ParameterValue(point_filter_num, value_type=int),
                    "imu.use_lidar_constant_velocity": ParameterValue(use_lidar_constant_velocity, value_type=bool),
                    "imu.use_lidar_constant_velocity_deskew": ParameterValue(use_lidar_constant_velocity_deskew, value_type=bool),
                    "mapping.publish_iteration_debug": ParameterValue(publish_iteration_debug, value_type=bool),
                    "mapping.semantic_debug": ParameterValue(semantic_debug, value_type=bool),
                    "mapping.semantic_debug_interval": ParameterValue(semantic_debug_interval, value_type=int),
                    "mapping.debug_csv_enable": ParameterValue(debug_csv_enable, value_type=bool),
                    "mapping.debug_csv_dir": ParameterValue(debug_csv_dir, value_type=str),
                    "mapping.debug_csv_prefix": ParameterValue(debug_csv_prefix, value_type=str),
                    "Result.tum_result_path": ParameterValue(tum_result_path, value_type=str),
                    "Result.efficiency_path": ParameterValue(efficiency_path, value_type=str),
                    "Result.per_frame_efficiency_enable": ParameterValue(per_frame_efficiency_enable, value_type=bool),
                    "Result.per_frame_efficiency_path": ParameterValue(per_frame_efficiency_path, value_type=str),
                    "Result.video_dump_enable": ParameterValue(video_dump_enable, value_type=bool),
                    "Result.video_dump_dir": ParameterValue(video_dump_dir, value_type=str),
                    "Result.video_dump_map_interval": ParameterValue(video_dump_map_interval, value_type=int),
                    "Result.video_dump_points_enable": ParameterValue(video_dump_points_enable, value_type=bool),
                    "Result.video_dump_map_enable": ParameterValue(video_dump_map_enable, value_type=bool),
                    "Result.video_dump_updates_enable": ParameterValue(video_dump_updates_enable, value_type=bool),
                },
            ],
        ),
        Node(
            package="hybrid_voxel_map",
            executable="kitti_sequence_publisher_cpp",
            name="kitti_sequence_publisher",
            output="screen",
            parameters=[{
                "sequence_path": sequence,
                "topic": publisher_topic,
                "frame_id": "camera_init",
                "rate": ParameterValue(rate, value_type=float),
                "loop": ParameterValue(LaunchConfiguration("loop"), value_type=bool),
                "start_index": ParameterValue(start_index, value_type=int),
                "end_index": ParameterValue(end_index, value_type=int),
                "qos_depth": 512,
                "wait_for_subscriber": True,
                "publish_point_time": ParameterValue(publish_point_time, value_type=bool),
                "publish_gt_semantics": ParameterValue(gt_semantics, value_type=bool),
                "label_dir": label_dir,
                "semantic_plane_labels": ParameterValue(plane_labels, value_type=str),
                "log_semantic_stats": True,
            }],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
