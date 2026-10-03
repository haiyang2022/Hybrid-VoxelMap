#include "IMU_Processing.hpp"
#include "preprocess.h"
#include "voxelmapplus_util.hpp"
#include <stdexcept>
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <common_lib.h>
#include <csignal>
#include <ctime>
#include <fstream>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <iomanip>
#include <limits>
#include <math.h>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <so3_math.h>
#include <sstream>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <thread>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <hybrid_voxel_map/msg/states.hpp>

#ifdef HAVE_LIVOX_ROS_DRIVER2
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif

#define INIT_TIME (0.0)
#define CALIB_ANGLE_COV (0.01)

using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
using ImuMsg = sensor_msgs::msg::Imu;
using PathMsg = nav_msgs::msg::Path;
using OdometryMsg = nav_msgs::msg::Odometry;
using PoseStampedMsg = geometry_msgs::msg::PoseStamped;
using QuaternionMsg = geometry_msgs::msg::Quaternion;
using MarkerArrayMsg = visualization_msgs::msg::MarkerArray;
using PointCloud2Pub = rclcpp::Publisher<PointCloud2Msg>::SharedPtr;
using OdometryPub = rclcpp::Publisher<OdometryMsg>::SharedPtr;
using PathPub = rclcpp::Publisher<PathMsg>::SharedPtr;
#ifdef HAVE_LIVOX_ROS_DRIVER2
using LivoxCustomMsg = livox_ros_driver2::msg::CustomMsg;
#endif

std::shared_ptr<rclcpp::Node> g_node;
std::unique_ptr<tf2_ros::TransformBroadcaster> g_tf_broadcaster;

inline double stampSec(const builtin_interfaces::msg::Time &stamp)
{
    return rclcpp::Time(stamp).seconds();
}

inline builtin_interfaces::msg::Time stampFromSec(double sec)
{
    const int64_t nanoseconds = static_cast<int64_t>(sec * 1e9);
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(nanoseconds / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(nanoseconds % 1000000000LL);
    return stamp;
}

inline builtin_interfaces::msg::Time rosNow()
{
    const int64_t nanoseconds = g_node ? g_node->now().nanoseconds() : rclcpp::Clock().now().nanoseconds();
    builtin_interfaces::msg::Time stamp;
    stamp.sec = static_cast<int32_t>(nanoseconds / 1000000000LL);
    stamp.nanosec = static_cast<uint32_t>(nanoseconds % 1000000000LL);
    return stamp;
}

template <typename T>
void declareAndGet(const std::string &name, T &value, const T &default_value)
{
    g_node->declare_parameter<T>(name, default_value);
    g_node->get_parameter(name, value);
}

bool ensureDirectory(const std::string &dir)
{
    if (dir.empty())
    {
        return false;
    }
    struct stat st;
    if (stat(dir.c_str(), &st) == 0)
    {
        return S_ISDIR(st.st_mode);
    }
    return mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST;
}

bool ensureDirectoryRecursive(const std::string &dir)
{
    if (dir.empty())
    {
        return false;
    }
    std::string current = (dir[0] == '/') ? "/" : "";
    size_t start = (dir[0] == '/') ? 1 : 0;
    while (start < dir.size())
    {
        const size_t slash = dir.find('/', start);
        const std::string part = dir.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (!part.empty())
        {
            if (!current.empty() && current.back() != '/')
            {
                current.push_back('/');
            }
            current += part;
            if (!ensureDirectory(current))
            {
                return false;
            }
        }
        if (slash == std::string::npos)
        {
            break;
        }
        start = slash + 1;
    }
    return true;
}

std::string joinPath(const std::string &dir, const std::string &name)
{
    if (dir.empty() || dir.back() == '/')
    {
        return dir + name;
    }
    return dir + "/" + name;
}

std::string localTimeString()
{
    std::time_t now = std::time(nullptr);
    std::tm local_tm;
    localtime_r(&now, &local_tm);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &local_tm);
    return std::string(buffer);
}

std::string parentPath(const std::string &path)
{
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos)
    {
        return "";
    }
    if (slash == 0)
    {
        return "/";
    }
    return path.substr(0, slash);
}

std::string frameStem(const int frame_idx, const double stamp)
{
    std::ostringstream oss;
    oss << std::setw(6) << std::setfill('0') << frame_idx << "_";
    oss << std::fixed << std::setprecision(6) << stamp;
    std::string text = oss.str();
    std::replace(text.begin(), text.end(), '.', '_');
    std::replace(text.begin(), text.end(), '-', 'm');
    return text;
}

extern std::string tum_result_dir;
extern std::string run_timestamp;
extern std::string efficiency_path;
extern bool per_frame_efficiency_enable;
extern std::string per_frame_efficiency_path;
extern std::ofstream per_frame_efficiency_file;
extern bool video_dump_enable;
extern std::string video_dump_dir;
extern std::string video_dump_points_dir;
extern std::string video_dump_map_dir;
extern std::string video_dump_updates_dir;
extern int video_dump_map_interval;
extern bool video_dump_points_enable;
extern bool video_dump_map_enable;
extern bool video_dump_updates_enable;
extern std::ofstream tum_result_file;
extern bool debug_csv_enable;
extern std::string debug_csv_dir;
extern std::string debug_csv_prefix;
extern std::ofstream debug_frame_csv;
extern std::ofstream debug_iter_csv;

double percentileCopy(std::vector<double> values, const double ratio)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double clamped = std::max(0.0, std::min(1.0, ratio));
    const size_t index = static_cast<size_t>(
        std::round(clamped * static_cast<double>(values.size() - 1)));
    return values[index];
}

std::string makeRunFilePath(const std::string &dir, const std::string &prefix,
                            const std::string &suffix)
{
    return joinPath(dir, prefix + "_" + run_timestamp + suffix);
}

std::string makeEfficiencyPathFromTum(const std::string &tum_path)
{
    const std::string tum_suffix = "_tum.txt";
    if (tum_path.size() >= tum_suffix.size() &&
        tum_path.compare(tum_path.size() - tum_suffix.size(), tum_suffix.size(), tum_suffix) == 0)
    {
        return tum_path.substr(0, tum_path.size() - tum_suffix.size()) + "_efficiency.txt";
    }
    const std::string txt_suffix = ".txt";
    if (tum_path.size() >= txt_suffix.size() &&
        tum_path.compare(tum_path.size() - txt_suffix.size(), txt_suffix.size(), txt_suffix) == 0)
    {
        return tum_path.substr(0, tum_path.size() - txt_suffix.size()) + "_efficiency.txt";
    }
    return tum_path + "_efficiency.txt";
}

void initEfficiencyFile()
{
    if (efficiency_path.empty())
    {
        return;
    }
    std::ofstream outfile(efficiency_path, std::ios::out | std::ios::trunc);
    if (!outfile.is_open())
    {
        RCLCPP_ERROR(g_node->get_logger(), "Can't open efficiency report: %s",
                     efficiency_path.c_str());
        efficiency_path.clear();
        return;
    }
    outfile << "# Hybrid-VoxelMap efficiency report\n";
    outfile << "# Generated when nodes shut down. Times are milliseconds unless noted.\n\n";
    outfile.close();
    RCLCPP_INFO(g_node->get_logger(), "Efficiency report will be written to: %s",
                efficiency_path.c_str());
}

void closeTumResultFile()
{
    if (tum_result_file.is_open())
    {
        tum_result_file.flush();
        tum_result_file.close();
    }
}

double currentRssMb()
{
    std::ifstream statm("/proc/self/statm");
    long pages_total = 0;
    long pages_resident = 0;
    if (!(statm >> pages_total >> pages_resident))
    {
        return 0.0;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
    {
        return 0.0;
    }
    return static_cast<double>(pages_resident) * static_cast<double>(page_size) /
           (1024.0 * 1024.0);
}

double peakRssMb()
{
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0)
    {
        return 0.0;
    }
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
}

void initDebugCsv()
{
    if (!debug_csv_enable)
    {
        return;
    }
    if (debug_csv_dir.empty())
    {
        debug_csv_dir = tum_result_dir.empty() ? "result" : tum_result_dir;
    }
    if (!ensureDirectory(debug_csv_dir))
    {
        RCLCPP_ERROR(g_node->get_logger(), "Can't create debug CSV directory: %s",
                     debug_csv_dir.c_str());
        debug_csv_enable = false;
        return;
    }

    const std::string frame_path = makeRunFilePath(debug_csv_dir, debug_csv_prefix, "_frame.csv");
    const std::string iter_path = makeRunFilePath(debug_csv_dir, debug_csv_prefix, "_iter.csv");
    debug_frame_csv.open(frame_path, std::ios::out | std::ios::trunc);
    debug_iter_csv.open(iter_path, std::ios::out | std::ios::trunc);
    if (!debug_frame_csv.is_open() || !debug_iter_csv.is_open())
    {
        RCLCPP_ERROR(g_node->get_logger(), "Can't open debug CSV files under: %s",
                     debug_csv_dir.c_str());
        debug_csv_enable = false;
        return;
    }

    debug_frame_csv << std::fixed << std::setprecision(9);
    debug_iter_csv << std::fixed << std::setprecision(9);
    debug_frame_csv
        << "scan_idx,lidar_beg,lidar_end,scan_dt,predicted,deskewed,"
        << "pred_pos_norm,pred_rot_norm,opt_pos_norm,opt_rot_norm,"
        << "raw_points,down_points,effect_points,plane_residuals,ndt_residuals,plane_ratio,"
        << "iter_used,last_res_abs,last_res_signed,last_res_p50,last_res_p95,last_res_p99,"
        << "last_plane_abs,last_ndt_abs,last_pos_step,last_rot_step,last_cond,"
        << "update_rejected,"
        << "semantic_plane,semantic_gaussian,semantic_object,"
        << "map_total,map_unknown,map_sem1,map_sem2,map_sem3,map_plane,map_ndt,"
        << "undistort_ms,downsample_ms,cov_ms,scan_match_ms,solve_ms,map_update_ms,total_ms,"
        << "velocity_norm,gyro_bias_norm,acc_bias_norm,gravity_norm,cov_min_eigenvalue\n";
    debug_iter_csv
        << "scan_idx,iter,lidar_end,pv_points,effect_points,plane_residuals,ndt_residuals,"
        << "res_abs,res_signed,res_p50,res_p95,res_p99,plane_abs,ndt_abs,"
        << "raw_pos_step,raw_rot_step,applied_pos_step,applied_rot_step,cond\n";
    debug_frame_csv.flush();
    debug_iter_csv.flush();
    RCLCPP_INFO(g_node->get_logger(), "Frame debug CSV: %s", frame_path.c_str());
    RCLCPP_INFO(g_node->get_logger(), "Iter debug CSV: %s", iter_path.c_str());
}

bool calib_laser = false;
double calib_vertical_angle_deg = 0.15;
bool write_kitti_log = false;
std::string result_path = "";
std::string tum_result_path = "";
std::string tum_result_dir = "";
std::string tum_result_prefix = "hybrid_voxel_map";
std::string run_timestamp = "";
std::string efficiency_path = "";
bool per_frame_efficiency_enable = false;
std::string per_frame_efficiency_path = "";
std::ofstream per_frame_efficiency_file;
bool video_dump_enable = false;
std::string video_dump_dir = "";
std::string video_dump_points_dir = "";
std::string video_dump_map_dir = "";
std::string video_dump_updates_dir = "";
int video_dump_map_interval = 1;
bool video_dump_points_enable = true;
bool video_dump_map_enable = true;
bool video_dump_updates_enable = true;
std::ofstream tum_result_file;
// params for imu
bool imu_en = true;
bool use_lidar_constant_velocity = true;
bool use_lidar_constant_velocity_deskew = true;
std::vector<double> extrinT;
std::vector<double> extrinR;

// params for publish function
bool publish_point_cloud = false;
int pub_point_cloud_skip = 1;

// record point usage
double mean_effect_points = 0;
double mean_ds_points = 0;
double mean_raw_points = 0;

// record time
double undistort_time_mean = 0;
double down_sample_time_mean = 0;
double calc_cov_time_mean = 0;
double scan_match_time_mean = 0;
double ekf_solve_time_mean = 0;
double map_update_time_mean = 0;
double iter_num_mean = 0;
double plane_residual_ratio_mean = 0;
double slam_algorithm_total_time_mean = 0;
int slam_efficiency_frames = 0;
double slam_e2e_time_mean = 0;
std::chrono::steady_clock::time_point slam_efficiency_start;
std::chrono::steady_clock::time_point slam_efficiency_end;
bool slam_efficiency_started = false;

// params for IEKF convergence
bool use_step_convergence = true;
int min_iteration = 3;
double converge_translation = 0.001;
double converge_rotation = 0.0001;
double iekf_step_scale = 1.0;
int min_effective_features = 0;
double max_solver_condition = 0.0;
bool skip_map_update_on_unstable = false;
bool publish_iteration_debug = false;
bool semantic_debug = false;
int semantic_debug_interval = 100;
bool debug_csv_enable = false;
std::string debug_csv_dir = "";
std::string debug_csv_prefix = "hybrid_voxel_map_debug";
std::ofstream debug_frame_csv;
std::ofstream debug_iter_csv;
int runtime_log_interval = 20;

mutex mtx_buffer;
condition_variable sig_buffer;
Eigen::Matrix3d last_rot = Eigen::Matrix3d::Zero();

string lid_topic, imu_topic;
bool reliable_lidar_qos = false;
int lidar_qos_depth = 512;
int imu_qos_depth = 2048;
int scanIdx = 0;

int iterCount, NUM_MAX_ITERATIONS, effct_feat_num, time_log_counter, publish_count = 0;

double NUM_MAX_DISTANCE = 0;
double first_lidar_time = 0;
double lidar_end_time = 0;
size_t received_lidar_messages = 0;
double res_mean_last = 0.05;
double total_distance = 0;
double gyr_cov_scale, acc_cov_scale;
double last_timestamp_lidar, last_timestamp_imu = -1.0;
double filter_size_surf_min;
double map_incremental_time, total_time, scan_match_time, solve_time;
bool lidar_pushed, flg_reset, flg_exit = false;
bool dense_map_en = true;
double time_diff_lidar_to_imu = 0.0;

deque<PointCloudXYZI::Ptr> lidar_surf_buffer;
deque<double> time_buffer;
deque<ImuMsg::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr surf_feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudNoeffect(new PointCloudXYZI(100000, 1));
pcl::VoxelGrid<PointType> downSizeFilterSurf;

V3D euler_cur;
V3D position_last(Zero3d);

// estimator inputs and output;
MeasureGroup Measures;
StatesGroup state;
StatesGroup lidar_only_prev_state;
StatesGroup lidar_only_last_state;
double lidar_only_prev_time = 0.0;
double lidar_only_last_time = 0.0;
bool lidar_only_has_prev_state = false;
bool lidar_only_has_last_state = false;

PathMsg path;
OdometryMsg odomAftMapped;
QuaternionMsg geoQuat;
PoseStampedMsg msg_body_pose;

int NUM_FEAT = 0;
int NUM_NDT = 0;
shared_ptr<Preprocess>
    p_pre(new Preprocess());

void update_lidar_only_state_history(const StatesGroup &optimized_state, const double lidar_time)
{
    if (lidar_only_has_last_state)
    {
        lidar_only_prev_state = lidar_only_last_state;
        lidar_only_prev_time = lidar_only_last_time;
        lidar_only_has_prev_state = true;
    }
    lidar_only_last_state = optimized_state;
    lidar_only_last_time = lidar_time;
    lidar_only_has_last_state = true;
}

bool predict_lidar_only_constant_velocity(StatesGroup &state_inout, const double lidar_time)
{
    if (!use_lidar_constant_velocity || imu_en || !lidar_only_has_prev_state || !lidar_only_has_last_state)
    {
        return false;
    }

    const double dt_prev = lidar_only_last_time - lidar_only_prev_time;
    const double dt_curr = lidar_time - lidar_only_last_time;
    if (dt_prev <= 1e-6 || dt_curr < 0.0)
    {
        return false;
    }

    const double scale = std::max(0.2, std::min(5.0, dt_curr / dt_prev));
    const M3D delta_rot = lidar_only_prev_state.rot_end.transpose() * lidar_only_last_state.rot_end;
    const V3D delta_rot_vec = Log(delta_rot) * scale;
    const M3D scaled_delta_rot = Exp(delta_rot_vec(0), delta_rot_vec(1), delta_rot_vec(2));
    const V3D delta_pos_body =
        lidar_only_prev_state.rot_end.transpose() *
        (lidar_only_last_state.pos_end - lidar_only_prev_state.pos_end);

    state_inout.rot_end = lidar_only_last_state.rot_end * scaled_delta_rot;
    state_inout.pos_end = lidar_only_last_state.pos_end +
                          lidar_only_last_state.rot_end * (delta_pos_body * scale);
    state_inout.vel_end = (state_inout.pos_end - lidar_only_last_state.pos_end) /
                          std::max(dt_curr, 1e-6);
    state_inout.bias_g = lidar_only_last_state.bias_g;
    state_inout.bias_a = lidar_only_last_state.bias_a;
    state_inout.gravity = lidar_only_last_state.gravity;
    state_inout.cov = lidar_only_last_state.cov;
    for (int i = 0; i < 3; ++i)
    {
        state_inout.cov(i, i) = std::max(state_inout.cov(i, i), 0.01);
        state_inout.cov(i + 3, i + 3) = std::max(state_inout.cov(i + 3, i + 3), 0.25);
        state_inout.cov(i + 6, i + 6) = std::max(state_inout.cov(i + 6, i + 6), 1.0);
    }
    return true;
}

bool deskew_lidar_only_constant_velocity(const PointCloudXYZI::Ptr &cloud, const double scan_duration)
{
    if (!use_lidar_constant_velocity_deskew || imu_en || !cloud ||
        !lidar_only_has_prev_state || !lidar_only_has_last_state)
    {
        return false;
    }

    const double dt_prev = lidar_only_last_time - lidar_only_prev_time;
    if (dt_prev <= 1e-6)
    {
        return false;
    }

    const double scan_dt = scan_duration > 1e-6 ? scan_duration : dt_prev;
    const double motion_scale = std::max(0.2, std::min(5.0, scan_dt / dt_prev));
    const M3D delta_rot = lidar_only_prev_state.rot_end.transpose() * lidar_only_last_state.rot_end;
    const V3D delta_rot_vec = Log(delta_rot) * motion_scale;
    const V3D delta_pos_body =
        lidar_only_prev_state.rot_end.transpose() *
        (lidar_only_last_state.pos_end - lidar_only_prev_state.pos_end) *
        motion_scale;
    const M3D end_rot = Exp(delta_rot_vec(0), delta_rot_vec(1), delta_rot_vec(2));

    bool deskewed = false;
    for (auto &point : cloud->points)
    {
        const double point_time = std::max(0.0, static_cast<double>(point.curvature) / 1000.0);
        if (point_time <= 0.0)
        {
            continue;
        }
        const double scale = std::max(0.0, std::min(1.0, point_time / scan_dt));
        const V3D scaled_rot_vec = delta_rot_vec * scale;
        const M3D point_rot = Exp(scaled_rot_vec(0), scaled_rot_vec(1), scaled_rot_vec(2));
        const V3D point_trans = delta_pos_body * scale;
        const V3D raw_point(point.x, point.y, point.z);
        const V3D deskewed_point =
            end_rot.transpose() * (point_rot * raw_point + point_trans - delta_pos_body);
        point.x = static_cast<float>(deskewed_point.x());
        point.y = static_cast<float>(deskewed_point.y());
        point.z = static_cast<float>(deskewed_point.z());
        deskewed = true;
    }
    return deskewed;
}

void SigHandle(int sig)
{
    flg_exit = true;
    RCLCPP_WARN(rclcpp::get_logger("hybrid_voxel_map"), "catch sig %d", sig);
    sig_buffer.notify_all();
}

const bool var_contrast(const pointWithCov &x, const pointWithCov &y)
{
    return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
};

inline int pointHybridSemantic(const PointType &point)
{
    if (!p_pre || p_pre->lidar_type != KITTI)
    {
        return -1;
    }
    const int semantic_id = static_cast<int>(std::lround(point.normal_x));
    return (semantic_id >= 1 && semantic_id <= 3) ? semantic_id : -1;
}

inline VOXEL_LOC semanticDownsampleKey(const PointType &point, const double leaf_size)
{
    return VOXEL_LOC(
        static_cast<int64_t>(std::floor(point.x / leaf_size)),
        static_cast<int64_t>(std::floor(point.y / leaf_size)),
        static_cast<int64_t>(std::floor(point.z / leaf_size)));
}

void transferDownsampledSemantics(const PointCloudXYZI::Ptr &source,
                                  PointCloudXYZI::Ptr &target,
                                  const double leaf_size)
{
    if (!p_pre || p_pre->lidar_type != KITTI || !source || !target || source->empty() ||
        target->empty() || leaf_size <= 0.0)
    {
        return;
    }

    std::unordered_map<VOXEL_LOC, std::array<int, 4>> semantic_counts;
    semantic_counts.reserve(source->size());
    for (const auto &point : source->points)
    {
        const int semantic_id = pointHybridSemantic(point);
        if (semantic_id < 1 || semantic_id > 3)
        {
            continue;
        }
        auto &counts = semantic_counts[semanticDownsampleKey(point, leaf_size)];
        counts[semantic_id]++;
    }

    for (auto &point : target->points)
    {
        const auto iter = semantic_counts.find(semanticDownsampleKey(point, leaf_size));
        if (iter == semantic_counts.end())
        {
            point.normal_x = 0.0f;
            point.normal_y = 0.0f;
            point.normal_z = 0.0f;
            continue;
        }

        int best_semantic = -1;
        int best_count = 0;
        for (int semantic_id = 1; semantic_id <= 3; semantic_id++)
        {
            if (iter->second[semantic_id] > best_count)
            {
                best_count = iter->second[semantic_id];
                best_semantic = semantic_id;
            }
        }

        point.normal_x = static_cast<float>(best_semantic);
        point.normal_y = 0.0f;
        point.normal_z = 0.0f;
    }
}

struct VoxelMapSemanticStats
{
    size_t total = 0;
    size_t unknown = 0;
    size_t semantic1 = 0;
    size_t semantic2 = 0;
    size_t semantic3 = 0;
    size_t plane = 0;
    size_t ndt = 0;
};

VoxelMapSemanticStats collectVoxelMapSemanticStats(
    const std::unordered_map<VOXEL_LOC, UnionFindNode *> &voxel_map)
{
    VoxelMapSemanticStats stats;
    stats.total = voxel_map.size();
    for (const auto &item : voxel_map)
    {
        const UnionFindNode *node = item.second;
        if (node == nullptr)
        {
            continue;
        }
        if (node->semantic_categary == 1)
        {
            ++stats.semantic1;
        }
        else if (node->semantic_categary == 2)
        {
            ++stats.semantic2;
        }
        else if (node->semantic_categary == 3)
        {
            ++stats.semantic3;
        }
        else
        {
            ++stats.unknown;
        }
        if (node->is_plane)
        {
            ++stats.plane;
        }
        if (node->is_NDT)
        {
            ++stats.ndt;
        }
    }
    return stats;
}

inline void kitti_log(FILE *fp)
{
    Eigen::Matrix4d T_lidar_to_cam;
    T_lidar_to_cam << 0.00042768, -0.999967, -0.0080845, -0.01198, -0.00721062, 0.0080811998,
        -0.99994131, -0.0540398, 0.999973864, 0.00048594, -0.0072069, -0.292196, 0, 0, 0, 1.0;
    V3D rot_ang(Log(state.rot_end));
    MD(4, 4)
    T;
    T.block<3, 3>(0, 0) = state.rot_end;
    T.block<3, 1>(0, 3) = state.pos_end;
    T(3, 0) = 0;
    T(3, 1) = 0;
    T(3, 2) = 0;
    T(3, 3) = 1;
    T = T_lidar_to_cam * T * T_lidar_to_cam.inverse();
    for (int i = 0; i < 3; i++)
    {
        if (i == 2)
            fprintf(fp, "%lf %lf %lf %lf", T(i, 0), T(i, 1), T(i, 2), T(i, 3));
        else
            fprintf(fp, "%lf %lf %lf %lf ", T(i, 0), T(i, 1), T(i, 2), T(i, 3));
    }
    fprintf(fp, "\n");
    fflush(fp);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state.rot_end * (p_body) + state.pos_end);
    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
    po->normal_y = pi->normal_y;
    po->normal_z = pi->normal_z;
    float intensity = pi->intensity;
    intensity = intensity - floor(intensity);

    int reflection_map = intensity * 10000;
}

void standard_pcl_cbk(const PointCloud2Msg::ConstSharedPtr &msg)
{
    ++received_lidar_messages;
    mtx_buffer.lock();
    const double msg_time = stampSec(msg->header.stamp);
    if (msg_time < last_timestamp_lidar)
    {
        RCLCPP_ERROR(g_node->get_logger(), "lidar loop back, clear buffer");
        lidar_surf_buffer.clear();
    }
    PointCloudXYZI::Ptr surf_ptr(new PointCloudXYZI());
    p_pre->process(msg, surf_ptr);
    lidar_surf_buffer.push_back(surf_ptr);
    time_buffer.push_back(msg_time);
    last_timestamp_lidar = msg_time;

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

#ifdef HAVE_LIVOX_ROS_DRIVER2
void livox_pcl_cbk(const LivoxCustomMsg::ConstSharedPtr &msg)
{
    ++received_lidar_messages;
    mtx_buffer.lock();
    const double msg_time = stampSec(msg->header.stamp);
    if (msg_time < last_timestamp_lidar)
    {
        RCLCPP_ERROR(g_node->get_logger(), "lidar loop back, clear buffer");
        lidar_surf_buffer.clear();
    }
    PointCloudXYZI::Ptr surf_ptr(new PointCloudXYZI());
    p_pre->process(msg, surf_ptr);
    lidar_surf_buffer.push_back(surf_ptr);
    time_buffer.push_back(msg_time);
    last_timestamp_lidar = msg_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}
#endif

void imu_cbk(const ImuMsg::ConstSharedPtr &msg_in)
{
    publish_count++;
    ImuMsg::SharedPtr msg(new ImuMsg(*msg_in));
    msg->header.stamp = stampFromSec(stampSec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    
    double timestamp = stampSec(msg->header.stamp);
    mtx_buffer.lock();
    if (timestamp < last_timestamp_imu)
    {
        RCLCPP_ERROR(g_node->get_logger(), "imu loop back, clear buffer");
        imu_buffer.clear();
        flg_reset = true;
    }
    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup &meas)
{
    if (!imu_en)
    {
        if (!lidar_surf_buffer.empty())
        {
            meas.surf_lidar = lidar_surf_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            lidar_end_time = meas.lidar_beg_time;
            if (meas.surf_lidar->points.size() > 1)
            {
                lidar_end_time += meas.surf_lidar->points.back().curvature / double(1000);
            }
            time_buffer.pop_front();
            lidar_surf_buffer.pop_front();
            return true;
        }

        return false;
    }

    if (lidar_surf_buffer.empty() || imu_buffer.empty())
    {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed)
    {
        meas.surf_lidar = lidar_surf_buffer.front();
        if (meas.surf_lidar->points.size() <= 1)
        {
            lidar_surf_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = time_buffer.front();
        lidar_end_time =
            meas.lidar_beg_time + meas.surf_lidar->points.back().curvature / double(1000);
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    meas.imu.clear();
    // Retain the first IMU sample after scan end.
    while (!imu_buffer.empty() && stampSec(imu_buffer.front()->header.stamp) <= lidar_end_time)
    {
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_surf_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

void publish_surf_frame_world(const PointCloud2Pub &pubLaserCloud, const int point_skip)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_map_en ? surf_feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    }
    PointCloudXYZI::Ptr laserCloudWorldPub(new PointCloudXYZI);
    for (int i = 0; i < size; i += point_skip)
    {
        laserCloudWorldPub->points.push_back(laserCloudWorld->points[i]);
    }
    PointCloud2Msg laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorldPub, laserCloudmsg);
    laserCloudmsg.header.stamp = rosNow();
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloud->publish(laserCloudmsg);
}

void publish_effect(const PointCloud2Pub &pubLaserCloudEffect)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr effect_cloud_world(
        new pcl::PointCloud<pcl::PointXYZRGB>);
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
        pcl::PointXYZRGB pi;
        pi.x = laserCloudWorld->points[i].x;
        pi.y = laserCloudWorld->points[i].y;
        pi.z = laserCloudWorld->points[i].z;
        float v = laserCloudWorld->points[i].intensity / 100;
        v = 1.0 - v;
        uint8_t r, g, b;
        MapJet(v, 0, 1, r, g, b);
        pi.r = r;
        pi.g = g;
        pi.b = b;
        effect_cloud_world->points.push_back(pi);
    }

    PointCloud2Msg laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = rosNow();
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

template <typename T>
void set_posestamp(T &out)
{
    out.position.x = state.pos_end(0);
    out.position.y = state.pos_end(1);
    out.position.z = state.pos_end(2);
    out.orientation.x = geoQuat.x;
    out.orientation.y = geoQuat.y;
    out.orientation.z = geoQuat.z;
    out.orientation.w = geoQuat.w;
}

void publish_odometry(const OdometryPub &pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "aft_mapped";
    odomAftMapped.header.stamp = rosNow();
    set_posestamp(odomAftMapped.pose.pose);
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = odomAftMapped.header.stamp;
    transform.header.frame_id = "camera_init";
    transform.child_frame_id = "aft_mapped";
    transform.transform.translation.x = state.pos_end(0);
    transform.transform.translation.y = state.pos_end(1);
    transform.transform.translation.z = state.pos_end(2);
    transform.transform.rotation = geoQuat;
    g_tf_broadcaster->sendTransform(transform);
    if (pubOdomAftMapped->get_subscription_count() > 0)
    {
        pubOdomAftMapped->publish(odomAftMapped);
    }

    if (!tum_result_file.is_open())
    {
        return;
    }

    tum_result_file
        << lidar_end_time << " "
        << odomAftMapped.pose.pose.position.x << " "
        << odomAftMapped.pose.pose.position.y << " "
        << odomAftMapped.pose.pose.position.z << " "
        << odomAftMapped.pose.pose.orientation.x << " "
        << odomAftMapped.pose.pose.orientation.y << " "
        << odomAftMapped.pose.pose.orientation.z << " "
        << odomAftMapped.pose.pose.orientation.w
        << "\n";
}

void publish_path(const PathPub &pubPath)
{
    if (pubPath->get_subscription_count() == 0)
    {
        return;
    }
    set_posestamp(msg_body_pose.pose);
    msg_body_pose.header.stamp = rosNow();
    msg_body_pose.header.frame_id = "camera_init";
    path.poses.push_back(msg_body_pose);
    pubPath->publish(path);
}

void recordSlamEfficiencyFrame(const std::chrono::steady_clock::time_point &frame_start,
                               const std::chrono::steady_clock::time_point &frame_end)
{
    if (!slam_efficiency_started)
    {
        slam_efficiency_start = frame_start;
        slam_efficiency_started = true;
    }
    slam_efficiency_end = frame_end;
    const double frame_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                                frame_end - frame_start)
                                .count();
    slam_efficiency_frames++;
    slam_e2e_time_mean =
        slam_e2e_time_mean * (slam_efficiency_frames - 1) / slam_efficiency_frames +
        frame_ms / slam_efficiency_frames;
}

void writeSlamEfficiencyReport()
{
    if (efficiency_path.empty())
    {
        return;
    }
    std::ofstream outfile(efficiency_path, std::ios::out | std::ios::app);
    if (!outfile.is_open())
    {
        std::cerr << "Can't open " << efficiency_path << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(6);
    double overall_frame_ms = 0.0;
    double overall_fps = 0.0;
    if (slam_efficiency_started && slam_efficiency_frames > 0)
    {
        const double wall_time_s = std::chrono::duration_cast<std::chrono::duration<double>>(
                                       slam_efficiency_end - slam_efficiency_start)
                                       .count();
        overall_frame_ms = wall_time_s * 1000.0 / static_cast<double>(slam_efficiency_frames);
        overall_fps = wall_time_s > 1e-9 ? static_cast<double>(slam_efficiency_frames) / wall_time_s : 0.0;
    }
    outfile << "[Overall]\n";
    outfile << "frames=" << slam_efficiency_frames << "\n";
    outfile << "avg_frame_ms=" << overall_frame_ms << "\n";
    outfile << "fps=" << overall_fps << "\n\n";
    outfile << "[SLAM]\n";
    outfile << "frames=" << slam_efficiency_frames << "\n";
    outfile << "avg_processing_ms=" << slam_algorithm_total_time_mean << "\n";
    outfile << "fps=" << (slam_algorithm_total_time_mean > 1e-9 ? 1000.0 / slam_algorithm_total_time_mean : 0.0) << "\n";
    outfile << "process_peak_rss_mb=" << peakRssMb() << "\n";
    outfile << "gpu_peak_mb=0.000000\n\n";
    outfile.close();
}

void initPerFrameEfficiencyFile()
{
    if (!per_frame_efficiency_enable)
    {
        return;
    }
    if (per_frame_efficiency_path.empty())
    {
        const std::string root = video_dump_dir.empty() ? "video_dump" : video_dump_dir;
        per_frame_efficiency_path = joinPath(root, "efficiency/slam_per_frame_efficiency.txt");
    }
    const std::string parent = parentPath(per_frame_efficiency_path);
    if (!parent.empty() && !ensureDirectoryRecursive(parent))
    {
        RCLCPP_ERROR(g_node->get_logger(), "Can't create per-frame efficiency directory: %s",
                     parent.c_str());
        per_frame_efficiency_enable = false;
        return;
    }
    per_frame_efficiency_file.open(per_frame_efficiency_path, std::ios::out | std::ios::trunc);
    if (!per_frame_efficiency_file.is_open())
    {
        RCLCPP_ERROR(g_node->get_logger(), "Can't open per-frame efficiency file: %s",
                     per_frame_efficiency_path.c_str());
        per_frame_efficiency_enable = false;
        return;
    }
    per_frame_efficiency_file << std::fixed << std::setprecision(6);
    per_frame_efficiency_file << "frame_idx timestamp slam_ms cpu_rss_mb gpu_mb\n";
    per_frame_efficiency_file.flush();
    RCLCPP_INFO(g_node->get_logger(), "SLAM per-frame efficiency: %s",
                per_frame_efficiency_path.c_str());
}

void recordSlamPerFrameEfficiency(const int frame_idx,
                                  const double stamp,
                                  const double slam_ms)
{
    if (!per_frame_efficiency_enable || !per_frame_efficiency_file.is_open())
    {
        return;
    }
    per_frame_efficiency_file << frame_idx << " " << stamp << " " << slam_ms << " "
                              << currentRssMb() << " 0.000000\n";
}

void closePerFrameEfficiencyFile()
{
    if (per_frame_efficiency_file.is_open())
    {
        per_frame_efficiency_file.flush();
        per_frame_efficiency_file.close();
    }
}

V3D planeNormalRaw(const Plane &plane, double &d)
{
    d = plane.n_vec[2];
    if (plane.main_direction == 0)
    {
        return V3D(plane.n_vec[0], plane.n_vec[1], 1.0);
    }
    if (plane.main_direction == 1)
    {
        return V3D(plane.n_vec[0], 1.0, plane.n_vec[1]);
    }
    return V3D(1.0, plane.n_vec[0], plane.n_vec[1]);
}

void writeVoxelRow(std::ofstream &out,
                   const VOXEL_LOC &loc,
                   const UnionFindNode *node)
{
    if (node == nullptr)
    {
        return;
    }
    const UnionFindNode *root = FindRootNoCompress(const_cast<UnionFindNode *>(node));
    const UnionFindNode *ndt = node->original_ptr_ != nullptr ? node->original_ptr_ : node;
    const bool use_plane = root != nullptr && root->is_plane && root->plane_ptr_ != nullptr &&
                           root->plane_ptr_->is_plane;
    const bool use_ndt = ndt != nullptr && ndt->is_NDT && ndt->plane_ptr_ != nullptr;
    const UnionFindNode *model_node = use_plane ? root : (use_ndt ? ndt : node);
    const PlanePtr &plane_ptr = model_node->plane_ptr_;
    const Plane &plane = *plane_ptr;

    int model_type = 0;
    if (use_plane)
    {
        model_type = 1;
    }
    else if (use_ndt)
    {
        model_type = 2;
    }

    double plane_d = 0.0;
    V3D normal = V3D::Zero();
    if (use_plane)
    {
        normal = planeNormalRaw(plane, plane_d);
        const double norm = normal.norm();
        if (norm > 1e-12)
        {
            normal /= norm;
            plane_d /= norm;
        }
    }

    const M3D &cov = plane.covariance;
    const V3D &mean = plane.center;
    out << loc.x << " " << loc.y << " " << loc.z << " "
        << node->voxel_center_[0] << " " << node->voxel_center_[1] << " " << node->voxel_center_[2] << " "
        << node->semantic_categary << " " << model_type << " " << plane.points_size << " "
        << normal[0] << " " << normal[1] << " " << normal[2] << " " << plane_d << " "
        << mean[0] << " " << mean[1] << " " << mean[2] << " "
        << cov(0, 0) << " " << cov(0, 1) << " " << cov(0, 2) << " "
        << cov(1, 1) << " " << cov(1, 2) << " " << cov(2, 2) << "\n";
}

void initVideoDump()
{
    if (!video_dump_enable)
    {
        return;
    }
    if (video_dump_dir.empty())
    {
        video_dump_dir = joinPath("video_dump",
                                  "kitti08_harp_" + run_timestamp);
    }
    video_dump_points_dir = joinPath(video_dump_dir, "frames");
    video_dump_map_dir = joinPath(video_dump_dir, "map_snapshots");
    video_dump_updates_dir = joinPath(video_dump_dir, "map_updates");
    if ((video_dump_points_enable && !ensureDirectoryRecursive(video_dump_points_dir)) ||
        (video_dump_map_enable && !ensureDirectoryRecursive(video_dump_map_dir)) ||
        (video_dump_updates_enable && !ensureDirectoryRecursive(video_dump_updates_dir)))
    {
        RCLCPP_ERROR(g_node->get_logger(), "Can't create video dump directories under: %s",
                     video_dump_dir.c_str());
        video_dump_enable = false;
        return;
    }
    ensureDirectoryRecursive(video_dump_dir);
    std::ofstream meta(joinPath(video_dump_dir, "metadata.txt"), std::ios::out | std::ios::trunc);
    if (meta.is_open())
    {
        meta << "run_timestamp " << run_timestamp << "\n";
        meta << "points_format x y z intensity semantic_id is_plane_point\n";
        meta << "voxel_format voxel_x voxel_y voxel_z center_x center_y center_z semantic_id model_type point_count "
             << "normal_x normal_y normal_z plane_d mean_x mean_y mean_z cov_xx cov_xy cov_xz cov_yy cov_yz cov_zz\n";
        meta << "model_type 0=uninitialized 1=plane 2=NDT\n";
        meta << "map_snapshot_policy interval>0 saves full map every interval frames; interval=0 saves frame 0 full map only and reconstructs later frames from map_updates\n";
    }
    RCLCPP_INFO(g_node->get_logger(), "Video dump directory: %s", video_dump_dir.c_str());
}

void dumpFramePoints(const int frame_idx,
                     const double stamp,
                     const StatesGroup &state,
                     const PointCloudXYZI::Ptr &cloud_body)
{
    if (!video_dump_enable || !video_dump_points_enable || cloud_body == nullptr)
    {
        return;
    }
    const std::string path = joinPath(video_dump_points_dir, frameStem(frame_idx, stamp) + "_points.txt");
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        return;
    }
    out << std::fixed << std::setprecision(6);
    for (const auto &point : cloud_body->points)
    {
        V3D p(point.x, point.y, point.z);
        p = state.rot_end * p + state.pos_end;
        int semantic_id = pointHybridSemantic(point);
        if (semantic_id < 0)
        {
            semantic_id = 0;
        }
        out << p[0] << " " << p[1] << " " << p[2] << " "
            << point.intensity << " " << semantic_id << " "
            << (semantic_id == 1 ? 1 : 0) << "\n";
    }
}

void dumpMapSnapshot(const int frame_idx,
                     const double stamp,
                     const std::unordered_map<VOXEL_LOC, UnionFindNode *> &voxel_map)
{
    if (!video_dump_enable || !video_dump_map_enable)
    {
        return;
    }
    if (video_dump_map_interval == 0 && frame_idx != 0)
    {
        return;
    }
    if (video_dump_map_interval < 0 ||
        (video_dump_map_interval > 0 && frame_idx % video_dump_map_interval != 0))
    {
        return;
    }
    const std::string path = joinPath(video_dump_map_dir, frameStem(frame_idx, stamp) + "_voxels.txt");
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        return;
    }
    out << std::fixed << std::setprecision(6);
    for (const auto &item : voxel_map)
    {
        writeVoxelRow(out, item.first, item.second);
    }
}

void dumpMapUpdates(const int frame_idx,
                    const double stamp,
                    const std::unordered_map<VOXEL_LOC, UnionFindNode *> &voxel_map,
                    const std::vector<VOXEL_LOC> &updated_voxels)
{
    if (!video_dump_enable || !video_dump_updates_enable)
    {
        return;
    }
    const std::string path = joinPath(video_dump_updates_dir, frameStem(frame_idx, stamp) + "_updates.txt");
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        return;
    }
    out << std::fixed << std::setprecision(6);
    for (const auto &loc : updated_voxels)
    {
        const auto iter = voxel_map.find(loc);
        if (iter != voxel_map.end())
        {
            writeVoxelRow(out, loc, iter->second);
        }
    }
}

void dumpVideoFrame(const int frame_idx,
                    const double stamp,
                    const StatesGroup &state,
                    const PointCloudXYZI::Ptr &cloud_body,
                    const std::unordered_map<VOXEL_LOC, UnionFindNode *> &voxel_map,
                    const std::vector<VOXEL_LOC> &updated_voxels)
{
    if (!video_dump_enable)
    {
        return;
    }
    dumpFramePoints(frame_idx, stamp, state, cloud_body);
    dumpMapSnapshot(frame_idx, stamp, voxel_map);
    dumpMapUpdates(frame_idx, stamp, voxel_map, updated_voxels);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    g_node = std::make_shared<rclcpp::Node>("voxel_mapping_odom");
    g_tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*g_node);

    double ranging_cov = 0.0;
    double angle_cov = 0.0;
    std::vector<double> layer_point_size;

    // cummon params
    declareAndGet<string>("common.lid_topic", lid_topic, "/livox/lidar");
    declareAndGet<string>("common.imu_topic", imu_topic, "/livox/imu");
    declareAndGet<double>("common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    declareAndGet<bool>("common.reliable_lidar_qos", reliable_lidar_qos, false);
    declareAndGet<int>("common.lidar_qos_depth", lidar_qos_depth, 512);
    lidar_qos_depth = std::max(1, lidar_qos_depth);
    declareAndGet<int>("common.imu_qos_depth", imu_qos_depth, 2048);
    imu_qos_depth = std::max(1, imu_qos_depth);

    // noise model params
    declareAndGet<double>("noise_model.ranging_cov", ranging_cov, 0.02);
    declareAndGet<double>("noise_model.angle_cov", angle_cov, 0.05);
    declareAndGet<double>("noise_model.gyr_cov_scale", gyr_cov_scale, 0.1);
    declareAndGet<double>("noise_model.acc_cov_scale", acc_cov_scale, 0.1);
    bool use_calibrated_imu_noise = true;
    double calibrated_gyr_cov_scale = gyr_cov_scale;
    double calibrated_acc_cov_scale = acc_cov_scale;
    declareAndGet<bool>("noise_model.use_calibrated_imu_noise", use_calibrated_imu_noise, true);
    declareAndGet<double>("noise_model.calibrated_gyr_cov_scale", calibrated_gyr_cov_scale, gyr_cov_scale);
    declareAndGet<double>("noise_model.calibrated_acc_cov_scale", calibrated_acc_cov_scale, acc_cov_scale);
    if (use_calibrated_imu_noise)
    {
        if (!std::isfinite(calibrated_gyr_cov_scale) || calibrated_gyr_cov_scale <= 0.0 ||
            !std::isfinite(calibrated_acc_cov_scale) || calibrated_acc_cov_scale <= 0.0)
            throw std::invalid_argument("Calibrated IMU noise variances must be positive and finite");
        gyr_cov_scale = calibrated_gyr_cov_scale;
        acc_cov_scale = calibrated_acc_cov_scale;
    }

    // imu params
    declareAndGet<bool>("imu.imu_en", imu_en, false);
    declareAndGet<bool>("imu.use_lidar_constant_velocity", use_lidar_constant_velocity, true);
    declareAndGet<bool>("imu.use_lidar_constant_velocity_deskew", use_lidar_constant_velocity_deskew, true);
    declareAndGet<vector<double>>("imu.extrinsic_T", extrinT, vector<double>{0.0, 0.0, 0.0});
    declareAndGet<vector<double>>("imu.extrinsic_R", extrinR, vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});

    // mapping algorithm params
    declareAndGet<bool>("mapping.gaussian_neighbor_search", gaussian_neighbor_search, true);
    declareAndGet<bool>("mapping.gaussian_uncertainty_gating", gaussian_uncertainty_gating, true);
    declareAndGet<int>("mapping.max_iteration", NUM_MAX_ITERATIONS, 4);
    declareAndGet<double>("mapping.max_distance", NUM_MAX_DISTANCE, 80.0);
    declareAndGet<int>("mapping.update_size_threshold", update_size_threshold, 5);
    declareAndGet<int>("mapping.init_plane_threshold", init_plane_threshold, 100);
    declareAndGet<int>("mapping.sigma_num", sigma_num, 3);
    declareAndGet<double>("mapping.voxel_size", voxel_size, 1.0);
    quater_length = voxel_size / 4;
    declareAndGet<double>("mapping.down_sample_size", filter_size_surf_min, 0.5);
    declareAndGet<double>("mapping.plannar_threshold", planer_threshold, 0.01);
    declareAndGet<bool>("mapping.abs_residual_gating", use_abs_residual_gating, true);
    declareAndGet<bool>("mapping.use_step_convergence", use_step_convergence, true);
    declareAndGet<int>("mapping.min_iteration", min_iteration, 3);
    declareAndGet<double>("mapping.converge_translation", converge_translation, 0.001);
    declareAndGet<double>("mapping.converge_rotation", converge_rotation, 0.0001);
    declareAndGet<double>("mapping.iekf_step_scale", iekf_step_scale, 1.0);
    declareAndGet<int>("mapping.min_effective_features", min_effective_features, 0);
    declareAndGet<double>("mapping.max_solver_condition", max_solver_condition, 0.0);
    declareAndGet<bool>("mapping.skip_map_update_on_unstable", skip_map_update_on_unstable, false);
    declareAndGet<bool>("mapping.publish_iteration_debug", publish_iteration_debug, false);
    declareAndGet<bool>("mapping.semantic_debug", semantic_debug, false);
    declareAndGet<int>("mapping.semantic_debug_interval", semantic_debug_interval, 100);
    declareAndGet<bool>("mapping.debug_csv_enable", debug_csv_enable, false);
    declareAndGet<string>("mapping.debug_csv_dir", debug_csv_dir, "");
    declareAndGet<string>("mapping.debug_csv_prefix", debug_csv_prefix, "hybrid_voxel_map_debug");
    declareAndGet<int>("runtime.log_interval", runtime_log_interval, 20);
    iekf_step_scale = std::max(0.05, std::min(1.0, iekf_step_scale));
    min_effective_features = std::max(0, min_effective_features);
    runtime_log_interval = std::max(0, runtime_log_interval);
    if (max_solver_condition < 0.0)
    {
        max_solver_condition = 0.0;
    }
    semantic_debug_interval = std::max(1, semantic_debug_interval);

    // preprocess params
    declareAndGet<double>("preprocess.blind", p_pre->blind, 0.01);
    declareAndGet<bool>("preprocess.calib_laser", calib_laser, false);
    declareAndGet<double>("preprocess.calib_vertical_angle_deg", calib_vertical_angle_deg, 0.15);
    declareAndGet<int>("preprocess.lidar_type", p_pre->lidar_type, AVIA);
    declareAndGet<int>("preprocess.scan_line", p_pre->N_SCANS, 6);
    declareAndGet<int>("preprocess.point_filter_num", p_pre->point_filter_num, 1);
    declareAndGet<bool>("preprocess.kitti_vertical_filter", p_pre->kitti_vertical_filter, false);

    // visualization params
    declareAndGet<bool>("visualization.pub_point_cloud", publish_point_cloud, false);
    declareAndGet<int>("visualization.pub_point_cloud_skip", pub_point_cloud_skip, 1);
    declareAndGet<bool>("visualization.dense_map_enable", dense_map_en, false);

    // result params
    declareAndGet<bool>("Result.write_kitti_log", write_kitti_log, false);
    declareAndGet<string>("Result.result_path", result_path,
                     "kitt_log.txt");
    declareAndGet<string>("Result.tum_result_dir", tum_result_dir, "");
    declareAndGet<string>("Result.tum_result_prefix", tum_result_prefix, "hybrid_voxel_map");
    declareAndGet<string>("Result.tum_result_path", tum_result_path, "");
    declareAndGet<string>("Result.efficiency_path", efficiency_path, "");
    declareAndGet<bool>("Result.per_frame_efficiency_enable", per_frame_efficiency_enable, false);
    declareAndGet<string>("Result.per_frame_efficiency_path", per_frame_efficiency_path, "");
    declareAndGet<bool>("Result.video_dump_enable", video_dump_enable, false);
    declareAndGet<string>("Result.video_dump_dir", video_dump_dir, "");
    declareAndGet<int>("Result.video_dump_map_interval", video_dump_map_interval, 1);
    declareAndGet<bool>("Result.video_dump_points_enable", video_dump_points_enable, true);
    declareAndGet<bool>("Result.video_dump_map_enable", video_dump_map_enable, true);
    declareAndGet<bool>("Result.video_dump_updates_enable", video_dump_updates_enable, true);
    const bool tum_result_disabled =
        (tum_result_path == "none" || tum_result_path == "NONE" || tum_result_path == "off");
    const bool efficiency_disabled =
        (efficiency_path == "none" || efficiency_path == "NONE" || efficiency_path == "off");
    if (tum_result_disabled)
    {
        tum_result_path.clear();
        tum_result_dir.clear();
    }
    if (efficiency_disabled)
    {
        efficiency_path.clear();
    }
    run_timestamp = localTimeString();
    if (!tum_result_dir.empty())
    {
        if (!ensureDirectory(tum_result_dir))
        {
            RCLCPP_ERROR(g_node->get_logger(), "Can't create TUM result directory: %s", tum_result_dir.c_str());
            rclcpp::shutdown();
            return 1;
        }
        if (tum_result_path.empty())
        {
            tum_result_path = makeRunFilePath(tum_result_dir, tum_result_prefix, "_tum.txt");
        }
    }
    if (!tum_result_path.empty())
    {
        tum_result_file.open(tum_result_path, std::ios::out | std::ios::trunc);
        if (!tum_result_file.is_open())
        {
            RCLCPP_ERROR(g_node->get_logger(), "Can't open TUM trajectory file: %s", tum_result_path.c_str());
            tum_result_path.clear();
        }
        else
        {
            tum_result_file << std::fixed << std::setprecision(9);
            RCLCPP_INFO(g_node->get_logger(), "TUM trajectory will be written to: %s", tum_result_path.c_str());
        }
    }
    if (!efficiency_disabled && efficiency_path.empty() && !tum_result_path.empty())
    {
        efficiency_path = makeEfficiencyPathFromTum(tum_result_path);
    }
    initEfficiencyFile();
    initPerFrameEfficiencyFile();
    initVideoDump();
    initDebugCsv();
    cout << "p_pre->lidar_type " << p_pre->lidar_type << endl;

    rclcpp::Subscription<PointCloud2Msg>::SharedPtr sub_pcl;
#ifdef HAVE_LIVOX_ROS_DRIVER2
    rclcpp::Subscription<LivoxCustomMsg>::SharedPtr sub_livox;
#endif
    rclcpp::QoS lidar_qos = rclcpp::SensorDataQoS();
    if (reliable_lidar_qos)
    {
        lidar_qos = rclcpp::QoS(rclcpp::KeepLast(lidar_qos_depth)).reliable();
    }
    if (p_pre->lidar_type == AVIA)
    {
#ifdef HAVE_LIVOX_ROS_DRIVER2
        sub_livox = g_node->create_subscription<LivoxCustomMsg>(lid_topic, lidar_qos, livox_pcl_cbk);
#else
        RCLCPP_ERROR(g_node->get_logger(), "lidar_type=1 requires livox_ros_driver2, but it was not found at build time");
        rclcpp::shutdown();
        return 1;
#endif
    }
    else
    {
        sub_pcl = g_node->create_subscription<PointCloud2Msg>(lid_topic, lidar_qos, standard_pcl_cbk);
    }

    rclcpp::Subscription<ImuMsg>::SharedPtr sub_imu;
    if (imu_en)
    {
        bool reliable_imu_qos = false;
        declareAndGet<bool>("common.reliable_imu_qos", reliable_imu_qos, false);
        rclcpp::QoS imu_qos = rclcpp::SensorDataQoS().keep_last(imu_qos_depth);
        if (reliable_imu_qos) imu_qos.reliable();
        sub_imu = g_node->create_subscription<ImuMsg>(imu_topic, imu_qos, imu_cbk);
    }

    PointCloud2Pub pubLaserCloudSurfFull =
        g_node->create_publisher<PointCloud2Msg>("/cloud_registered_surf", 100);
    PointCloud2Pub pubLaserCloudEffect =
        g_node->create_publisher<PointCloud2Msg>("/cloud_effected", 100);
    OdometryPub pubOdomAftMapped = g_node->create_publisher<OdometryMsg>("/aft_mapped_to_init", 10);
    PathPub pubPath = g_node->create_publisher<PathMsg>("/path", 10);
    auto voxel_map_pub = g_node->create_publisher<MarkerArrayMsg>("/planes", 10000);

    path.header.stamp = rosNow();
    path.header.frame_id = "camera_init";

    /*** variables definition ***/
    VD(DIM_STATE)
    solution;
    MD(DIM_STATE, DIM_STATE)
    G, H_T_H, I_STATE;
    V3D rot_add, t_add;
    StatesGroup state_propagat;
    PointType pointOri, pointSel, coeff;
    int frame_num = 0;
    double aver_time_consu = 0;
    bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0, is_first_frame = true;
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                   filter_size_surf_min);

    shared_ptr<ImuProcess> p_imu(new ImuProcess());
    p_imu->imu_en = imu_en;
    Eigen::Vector3d extT = V3D::Zero();
    Eigen::Matrix3d extR = M3D::Identity();
    extT << extrinT[0], extrinT[1], extrinT[2];
    extR << extrinR[0], extrinR[1], extrinR[2], extrinR[3], extrinR[4], extrinR[5], extrinR[6],
        extrinR[7], extrinR[8];
    p_imu->set_extrinsic(extT, extR);

    p_imu->set_gyr_cov_scale(V3D(gyr_cov_scale, gyr_cov_scale, gyr_cov_scale));
    p_imu->set_acc_cov_scale(V3D(acc_cov_scale, acc_cov_scale, acc_cov_scale));
    p_imu->set_gyr_bias_cov(V3D(0.00001, 0.00001, 0.00001));
    p_imu->set_acc_bias_cov(V3D(0.00001, 0.00001, 0.00001));

    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    /*** debug record ***/
    FILE *fp_kitti;

    if (write_kitti_log)
    {
        fp_kitti = fopen(result_path.c_str(), "w");
    }

    signal(SIGINT, SigHandle);
    bool status = rclcpp::ok();

    // for Plane Map
    bool init_map = false;
    std::unordered_map<VOXEL_LOC, UnionFindNode *> voxel_map;
    last_rot << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    bool logged_cv_prediction = false;
    bool logged_cv_deskew = false;

    while (status)
    {
        bool processed_measurement = false;
        if (flg_exit)
        {
            break;
        }
        if (!rclcpp::ok())
        {
            break;
        }
        try
        {
            rclcpp::spin_some(g_node);
        }
        catch (const rclcpp::exceptions::RCLError &)
        {
            break;
        }

        /*** 1.Sync Package ***/
        if (sync_packages(Measures))
        {
            processed_measurement = true;
            const auto slam_frame_start = std::chrono::steady_clock::now();
            if (flg_reset)
            {
                RCLCPP_WARN(g_node->get_logger(), "reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                continue;
            }

            const bool should_log_frame =
                runtime_log_interval > 0 && (scanIdx % runtime_log_interval == 0);
            if (should_log_frame)
            {
                std::cout << "scanIdx:" << scanIdx << std::endl;
            }
            const StatesGroup state_before_prediction = state;
            bool cv_predicted = false;
            bool cv_deskewed = false;
            double pred_pos_norm = 0.0;
            double pred_rot_norm = 0.0;
            double opt_pos_norm = 0.0;
            double opt_rot_norm = 0.0;
            double t0, t1, t2, t3, t4, t5, match_start, match_time, solve_start, svd_time;
            match_time = 0;
            solve_time = 0;
            svd_time = 0;

            /*** 2.IMU Predict ***/
            /*** 3.PointCloud Undistort ***/
            auto undistort_start = std::chrono::high_resolution_clock::now();
            p_imu->Process(Measures, state, surf_feats_undistort);
            auto undistort_end = std::chrono::high_resolution_clock::now();
            auto undistort_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                      undistort_end - undistort_start)
                                      .count() *
                                  1000;
            const double lidar_state_time = imu_en ? Measures.lidar_beg_time : lidar_end_time;
            cv_predicted = predict_lidar_only_constant_velocity(state, lidar_state_time);
            if (cv_predicted && !logged_cv_prediction)
            {
                RCLCPP_INFO(g_node->get_logger(),
                            "No IMU: using constant-velocity LiDAR-only prediction");
                logged_cv_prediction = true;
            }
            cv_deskewed = deskew_lidar_only_constant_velocity(surf_feats_undistort,
                                                              lidar_end_time - Measures.lidar_beg_time);
            if (cv_deskewed && !logged_cv_deskew)
            {
                RCLCPP_INFO(g_node->get_logger(),
                            "No IMU: deskewing LiDAR scan with constant-velocity prediction");
                logged_cv_deskew = true;
            }
            // only for kitti
            if (calib_laser)
            {
                // calib the vertical angle for kitti dataset
                for (size_t i = 0; i < surf_feats_undistort->size(); i++)
                {
                    PointType pi = surf_feats_undistort->points[i];
                    double range = sqrt(pi.x * pi.x + pi.y * pi.y + pi.z * pi.z);
                    double calib_vertical_angle = deg2rad(calib_vertical_angle_deg);
                    double vertical_angle = asin(pi.z / range) + calib_vertical_angle;
                    double horizon_angle = atan2(pi.y, pi.x);
                    pi.z = range * sin(vertical_angle);
                    double project_len = range * cos(vertical_angle);
                    pi.x = project_len * cos(horizon_angle);
                    pi.y = project_len * sin(horizon_angle);
                    surf_feats_undistort->points[i] = pi;
                }
            }
            state_propagat = state;
            {
                const auto pred_delta = state_propagat - state_before_prediction;
                pred_rot_norm = pred_delta.block<3, 1>(0, 0).norm();
                pred_pos_norm = pred_delta.block<3, 1>(3, 0).norm();
            }

            if (is_first_frame)
            {
                first_lidar_time = Measures.lidar_beg_time;
                is_first_frame = false;
            }

            if (surf_feats_undistort->empty() || (surf_feats_undistort == NULL))
            {
                p_imu->first_lidar_time = first_lidar_time;
                cout << "LIO not ready!" << endl;
                continue;
            }

            // flg_EKF_inited = !((Measures.lidar_beg_time - first_lidar_time) < INIT_TIME);
            flg_EKF_inited =
                (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            /*** Build Voxel Map ***/
            if (flg_EKF_inited && !init_map)
            {
                pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(
                    new pcl::PointCloud<pcl::PointXYZI>);
                TransformLidar(state, p_imu, surf_feats_undistort, world_lidar);
                std::vector<pointWithCov> pv_list;
                pv_list.reserve(world_lidar->size());
                int semantic_counts[4] = {0, 0, 0, 0};
                for (size_t i = 0; i < world_lidar->size(); i++)
                {
                    const PointType &body_point = surf_feats_undistort->points[i];
                    pointWithCov pv;
                    pv.point << world_lidar->points[i].x, world_lidar->points[i].y,
                        world_lidar->points[i].z;
                    pv.point_world = pv.point;
                    pv.point_index = static_cast<int>(i);
                    pv.Semantic_ID = pointHybridSemantic(body_point);
                    if (pv.Semantic_ID >= 1 && pv.Semantic_ID <= 3)
                    {
                        semantic_counts[pv.Semantic_ID]++;
                    }
                    V3D point_this(body_point.x, body_point.y, body_point.z);
                    if (point_this[2] == 0)
                    {
                        point_this[2] = 0.001;
                    }
                    M3D cov;
                    calcBodyCov(point_this, ranging_cov, angle_cov, cov);

                    point_this += Lidar_offset_to_IMU;
                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);
                    const M3D rotation_jacobian = -state.rot_end * point_crossmat;
                    cov = state.rot_end * cov * state.rot_end.transpose() +
                          rotation_jacobian * state.cov.block<3, 3>(0, 0) *
                              rotation_jacobian.transpose() +
                          state.cov.block<3, 3>(3, 3);
                    pv.cov = cov;
                    pv_list.push_back(pv);
                    Eigen::Vector3d sigma_pv = pv.cov.diagonal();
                    sigma_pv[0] = sqrt(sigma_pv[0]);
                    sigma_pv[1] = sqrt(sigma_pv[1]);
                    sigma_pv[2] = sqrt(sigma_pv[2]);
                }

                std::vector<VOXEL_LOC> updated_voxels;
                UpdateVoxelMap(pv_list, voxel_map,
                               video_dump_enable ? &updated_voxels : nullptr);
                dumpVideoFrame(scanIdx, lidar_end_time, state, surf_feats_undistort,
                               voxel_map, updated_voxels);
                if (semantic_debug)
                {
                    const auto map_stats = collectVoxelMapSemanticStats(voxel_map);
                    RCLCPP_INFO(g_node->get_logger(),
                                "Semantic debug init scan %d input plane=%d gaussian=%d object=%d | map voxels total=%zu unknown=%zu sem1=%zu sem2=%zu sem3=%zu plane=%zu ndt=%zu",
                                scanIdx, semantic_counts[1], semantic_counts[2], semantic_counts[3],
                                map_stats.total, map_stats.unknown, map_stats.semantic1,
                                map_stats.semantic2, map_stats.semantic3, map_stats.plane, map_stats.ndt);
                }
                std::cout << "build voxel map" << std::endl;
                if (write_kitti_log)
                {
                    kitti_log(fp_kitti);
                }

                update_lidar_only_state_history(state, lidar_state_time);
                scanIdx++;

                init_map = true;
                cout << "Finish First Frame" << endl;
                continue;
            }

            /*** 4. VoxelGrid Downsample ***/
            auto t_downsample_start = std::chrono::high_resolution_clock::now();
            downSizeFilterSurf.setInputCloud(surf_feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            transferDownsampledSemantics(surf_feats_undistort, feats_down_body,
                                         filter_size_surf_min);
            auto t_downsample_end = std::chrono::high_resolution_clock::now();
            if (should_log_frame)
            {
                std::cout << "feats size:" << surf_feats_undistort->size()
                          << ", down size:" << feats_down_body->size() << std::endl;
            }
            auto t_downsample = std::chrono::duration_cast<std::chrono::duration<double>>(
                                    t_downsample_end - t_downsample_start)
                                    .count() *
                                1000;

            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);

            double total_residual;

            scan_match_time = 0.0;

            /*** 5. Calculate Body Cov ***/
            std::vector<M3D> body_var;
            std::vector<M3D> crossmat_list;
            body_var.reserve(feats_down_body->size());
            crossmat_list.reserve(feats_down_body->size());
            auto calc_point_cov_start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < feats_down_body->size(); i++)
            {
                V3D point_this(feats_down_body->points[i].x, feats_down_body->points[i].y,
                               feats_down_body->points[i].z);
                if (point_this[2] == 0)
                {
                    point_this[2] = 0.001;
                }
                M3D cov;
                if (calib_laser)
                {
                    calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV, cov);
                }
                else
                {
                    calcBodyCov(point_this, ranging_cov, angle_cov, cov);
                }
                M3D point_crossmat;
                point_crossmat << SKEW_SYM_MATRX(point_this);
                crossmat_list.push_back(point_crossmat);
                body_var.push_back(cov);
            }

            auto calc_point_cov_end = std::chrono::high_resolution_clock::now();
            double calc_point_cov_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                             calc_point_cov_end - calc_point_cov_start)
                                             .count() *
                                         1000;
            int iter_used = NUM_MAX_ITERATIONS;
            double last_res_abs = 0.0;
            double last_res_signed = 0.0;
            double last_res_p50 = 0.0;
            double last_res_p95 = 0.0;
            double last_res_p99 = 0.0;
            double last_plane_abs = 0.0;
            double last_ndt_abs = 0.0;
            double last_pos_step = 0.0;
            double last_rot_step = 0.0;
            double last_cond = -1.0;
            int last_plane_residual_count = 0;
            int last_ndt_residual_count = 0;
            bool measurement_update_valid = false;
            bool update_rejected = false;
            bool skip_this_map_update = false;
            StatesGroup best_valid_state = state;
            /*** 6. IESEKF Update ***/
            for (iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++)
            {
                laserCloudOri->clear();
                laserCloudNoeffect->clear();
                total_residual = 0.0;
                NUM_FEAT = 0;
                NUM_NDT = 0;

                std::vector<ptpl> ptpl_list;

                vector<pointWithCov> pv_list;
                pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(
                    new pcl::PointCloud<pcl::PointXYZI>);
                TransformLidar(state, p_imu, feats_down_body, world_lidar);
                pv_list.reserve(feats_down_body->size());
                ptpl_list.reserve(feats_down_body->size());
                const M3D rot_var = state.cov.block<3, 3>(0, 0);
                const M3D t_var = state.cov.block<3, 3>(3, 3);
                for (size_t i = 0; i < feats_down_body->size(); i++)
                {
                    const PointType &body_point = feats_down_body->points[i];
                    pointWithCov pv;
                    pv.point << body_point.x, body_point.y, body_point.z;
                    pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y,
                        world_lidar->points[i].z;
                    pv.Semantic_ID = pointHybridSemantic(body_point);
                    pv.point_index = static_cast<int>(i);
                    double dis_out = body_point.x * body_point.x + body_point.y * body_point.y +
                                     body_point.z * body_point.z;
                    if (dis_out > NUM_MAX_DISTANCE * NUM_MAX_DISTANCE)
                    {
                        continue;
                    }
                    M3D cov = body_var[i];
                    M3D point_crossmat = crossmat_list[i];
                    const M3D rotation_jacobian = -state.rot_end * point_crossmat;
                    cov = state.rot_end * cov * state.rot_end.transpose() +
                          rotation_jacobian * rot_var * rotation_jacobian.transpose() + t_var;
                    pv.cov = cov;
                    pv_list.push_back(pv);
                }

                auto scan_match_time_start = std::chrono::high_resolution_clock::now();
                BuildResidualListOMP(voxel_map, pv_list, ptpl_list);
                auto scan_match_time_end = std::chrono::high_resolution_clock::now();
                effct_feat_num = ptpl_list.size();
                double signed_residual = 0.0;
                double plane_abs_residual = 0.0;
                double ndt_abs_residual = 0.0;
                int plane_residual_count = 0;
                int ndt_residual_count = 0;
                std::vector<double> abs_residual_values;
                if (debug_csv_enable)
                {
                    abs_residual_values.reserve(static_cast<size_t>(effct_feat_num));
                }
                for (int i = 0; i < effct_feat_num; i++)
                {
                    const double abs_dist = std::abs(ptpl_list[i].dist);
                    total_residual += abs_dist;
                    signed_residual += ptpl_list[i].dist;
                    if (debug_csv_enable)
                    {
                        abs_residual_values.push_back(abs_dist);
                    }
                    if (ptpl_list[i].pp_res)
                    {
                        plane_abs_residual += abs_dist;
                        plane_residual_count++;
                    }
                    else
                    {
                        ndt_abs_residual += abs_dist;
                        ndt_residual_count++;
                    }
                }
                res_mean_last = effct_feat_num > 0 ? total_residual / effct_feat_num : 0.0;
                const double signed_res_mean = effct_feat_num > 0 ? signed_residual / effct_feat_num : 0.0;
                const double plane_res_mean =
                    plane_residual_count > 0 ? plane_abs_residual / plane_residual_count : 0.0;
                const double ndt_res_mean =
                    ndt_residual_count > 0 ? ndt_abs_residual / ndt_residual_count : 0.0;
                const double res_p50 = debug_csv_enable ? percentileCopy(abs_residual_values, 0.50) : 0.0;
                const double res_p95 = debug_csv_enable ? percentileCopy(abs_residual_values, 0.95) : 0.0;
                const double res_p99 = debug_csv_enable ? percentileCopy(abs_residual_values, 0.99) : 0.0;
                last_res_abs = res_mean_last;
                last_res_signed = signed_res_mean;
                last_res_p50 = res_p50;
                last_res_p95 = res_p95;
                last_res_p99 = res_p99;
                last_plane_abs = plane_res_mean;
                last_ndt_abs = ndt_res_mean;
                last_plane_residual_count = plane_residual_count;
                last_ndt_residual_count = ndt_residual_count;
                scan_match_time += std::chrono::duration_cast<std::chrono::duration<double>>(
                                       scan_match_time_end - scan_match_time_start)
                                       .count() *
                                   1000;

                auto t_solve_start = std::chrono::high_resolution_clock::now();
                MatrixXd Hsub(effct_feat_num, 6);
                MatrixXd Hsub_T_R_inv(6, effct_feat_num);
                VectorXd R_inv(effct_feat_num);
                VectorXd meas_vec(effct_feat_num);

                for (int i = 0; i < effct_feat_num; i++)
                {
                    V3D laser_p = ptpl_list[i].point;
                    V3D point_this(laser_p(0), laser_p(1), laser_p(2));
                    M3D cov; 
                    if (calib_laser)
                    {
                        calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV, cov);
                    }
                    else
                    {
                        calcBodyCov(point_this, ranging_cov, angle_cov, cov);
                    }
                    M3D point_crossmat;
                    
                    point_crossmat << SKEW_SYM_MATRX(point_this);
                    if (ptpl_list[i].pp_res == true)
                    {
                        V3D norm_p = ptpl_list[i].omega; 
                        V3D norm_vec(norm_p(0), norm_p(1), norm_p(2));
                        V3D point_world = ptpl_list[i].point_world;
                        Eigen::Matrix<double, 1, 3> J_abd; 
                        Eigen::Matrix<double, 1, 3> J_pw;  
                        V3D Omega = ptpl_list[i].omega;
                        double Omega_norm = ptpl_list[i].omega_norm;
                        double dist = ptpl_list[i].dist;
                        if (ptpl_list[i].main_direction == 0)
                        { // ax+by+z+d = 0;
                            J_abd << point_world(0) - Omega(0) * dist / Omega_norm,
                                point_world(1) - Omega(1) * dist / Omega_norm, 1;
                        }
                        else if (ptpl_list[i].main_direction == 1)
                        { // ax+y+bz+d = 0;
                            J_abd << point_world(0) - Omega(0) * dist / Omega_norm,
                                point_world(2) - Omega(2) * dist / Omega_norm, 1;
                        }
                        else
                        { // x+ay+bz+d = 0;
                            J_abd << point_world(1) - Omega(1) * dist / Omega_norm,
                                point_world(2) - Omega(2) * dist / Omega_norm, 1;
                        }
                        J_abd /= Omega_norm;
                        J_pw = Omega.transpose() * state.rot_end / Omega_norm;
                        double sigma_l = J_abd * ptpl_list[i].plane_cov * J_abd.transpose();
                        double huber_delta = 0.3; 
                        double huber_weight = 1.0;
                        if (std::abs(ptpl_list[i].dist) > huber_delta)
                            huber_weight = huber_delta / std::abs(ptpl_list[i].dist);
                        R_inv(i) = huber_weight / (sigma_l + J_pw * cov * J_pw.transpose());
                        
                        V3D n = Omega / Omega_norm; 
                        V3D A(point_crossmat * state.rot_end.transpose() * n);
                        
                        Hsub.row(i) << VEC_FROM_ARRAY(A), n[0], n[1], n[2];
                        Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i),
                            n[0] * R_inv(i), n[1] * R_inv(i), n[2] * R_inv(i);

                        double lam_tx = 1.0, lam_ty = 1.0, lam_tz = 1.0;
                        switch (ptpl_list[i].main_direction)
                        {
                        case 0:
                            lam_tx = 0.0001;
                            lam_ty = 0.0001;
                            break;
                        case 1: 
                            break;
                        case 2: 
                            break;
                        }
                        Hsub(i, 3) *= lam_tx;
                        Hsub_T_R_inv(3, i) *= lam_tx;
                        Hsub(i, 4) *= lam_ty;
                        Hsub_T_R_inv(4, i) *= lam_ty;
                        Hsub(i, 5) *= lam_tz;
                        Hsub_T_R_inv(5, i) *= lam_tz;

                        meas_vec(i) = -dist;
                        NUM_FEAT++;
                    }
                    else
                    {
                        Eigen::Matrix<double, 1, 3> J_p = ptpl_list[i].J_T_NDT.transpose(); 
                        Eigen::Matrix<double, 1, 3> J_mu = (-1) * J_p;                      
                        const M3D point_cov_world = state.rot_end * cov * state.rot_end.transpose();
                        double sigma_p = J_p * point_cov_world * J_p.transpose();
                        double sigma_mu = J_mu * ptpl_list[i].cov_mu * J_mu.transpose();
                        double huber_delta = 0.3; 
                        double huber_weight = 1.0;
                        if (std::abs(ptpl_list[i].dist) > huber_delta)
                            huber_weight = huber_delta / std::abs(ptpl_list[i].dist);

                        R_inv(i) = huber_weight / (sigma_p + sigma_mu);

                        V3D n = ptpl_list[i].J_T_NDT;
                        V3D A(point_crossmat * state.rot_end.transpose() * n);

                        Hsub.row(i) << VEC_FROM_ARRAY(A), n[0], n[1], n[2];
                        Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i),
                            n[0] * R_inv(i), n[1] * R_inv(i), n[2] * R_inv(i);

                        meas_vec(i) = -ptpl_list[i].dist;
                        NUM_NDT++;
                    }
                }

                if ((NUM_NDT + NUM_FEAT) == 0)
                    continue;

                MatrixXd K(DIM_STATE, effct_feat_num);

                EKF_stop_flg = false;
                flg_EKF_converged = false;
                if (!flg_EKF_inited)
                {
                    cout << "||||||||||Initiallizing LiDar||||||||||" << endl;
                    /*** only run in initialization period ***/
                    MatrixXd H_init(MD(9, DIM_STATE)::Zero());
                    MatrixXd z_init(VD(9)::Zero());
                    H_init.block<3, 3>(0, 0) = M3D::Identity();
                    H_init.block<3, 3>(3, 3) = M3D::Identity();
                    H_init.block<3, 3>(6, 15) = M3D::Identity();
                    z_init.block<3, 1>(0, 0) = -Log(state.rot_end); 
                    z_init.block<3, 1>(3, 0) = -state.pos_end;      

                    auto H_init_T = H_init.transpose();
                    auto &&K_init =
                        state.cov * H_init_T *
                        (H_init * state.cov * H_init_T + 0.0001 * MD(9, 9)::Identity()).inverse();
                    solution = K_init * z_init;

                    state.resetpose();
                    EKF_stop_flg = true;
                }
                else
                {
                    H_T_H.setZero();
                    H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
                    double normal_cond = -1.0;
                    if (publish_iteration_debug || debug_csv_enable || max_solver_condition > 0.0)
                    {
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eig_solver(
                            H_T_H.block<6, 6>(0, 0));
                        if (eig_solver.info() == Eigen::Success)
                        {
                            const auto evals = eig_solver.eigenvalues();
                            const double max_eval = evals.cwiseAbs().maxCoeff();
                            double min_eval = std::numeric_limits<double>::infinity();
                            for (int eig_i = 0; eig_i < evals.size(); ++eig_i)
                            {
                                const double abs_eval = std::abs(evals(eig_i));
                                if (abs_eval > 1e-12)
                                {
                                    min_eval = std::min(min_eval, abs_eval);
                                }
                            }
                            if (std::isfinite(min_eval))
                            {
                                normal_cond = max_eval / min_eval;
                            }
                        }
                    }
                    const bool too_few_features =
                        min_effective_features > 0 && effct_feat_num < min_effective_features;
                    const bool ill_conditioned =
                        max_solver_condition > 0.0 &&
                        (!std::isfinite(normal_cond) || normal_cond > max_solver_condition);
                    if (too_few_features || ill_conditioned)
                    {
                        update_rejected = true;
                        if (measurement_update_valid)
                        {
                            state = best_valid_state;
                        }
                        else
                        {
                            state = state_propagat;
                            skip_this_map_update = true;
                        }
                        iter_used = iterCount + 1;
                        last_pos_step = 0.0;
                        last_rot_step = 0.0;
                        last_cond = normal_cond;
                        if (publish_iteration_debug)
                        {
                            RCLCPP_WARN(g_node->get_logger(),
                                        "Reject unstable IEKF update at frame %d iter %d: effect=%d cond=%.3e",
                                        scanIdx, iterCount + 1, effct_feat_num, normal_cond);
                        }
                        break;
                    }
                    MD(DIM_STATE, DIM_STATE)
                    K1 = (H_T_H + state.cov.inverse()).inverse();
                    K = K1.block<DIM_STATE, 6>(0, 0) * Hsub_T_R_inv;

                    auto vec = state_propagat - state;
                    Eigen::Matrix<double, DIM_STATE, 1> step =
                        (K * meas_vec + vec - K * Hsub * vec.block<6, 1>(0, 0)).eval();

                    const Eigen::Matrix<double, DIM_STATE, 1> applied_step =
                        (iekf_step_scale < 1.0) ? (step * iekf_step_scale).eval() : step;
                    state += applied_step;
                    solution = applied_step;

                    const double raw_rot_step_norm = step.block<3, 1>(0, 0).norm();
                    const double raw_pos_step_norm = step.block<3, 1>(3, 0).norm();
                    const double rot_step_norm = applied_step.block<3, 1>(0, 0).norm();
                    const double pos_step_norm = applied_step.block<3, 1>(3, 0).norm();
                    last_pos_step = pos_step_norm;
                    last_rot_step = rot_step_norm;
                    last_cond = normal_cond;
                    measurement_update_valid = true;
                    best_valid_state = state;
                    if (debug_csv_enable && debug_iter_csv.is_open())
                    {
                        debug_iter_csv
                            << scanIdx << "," << (iterCount + 1) << "," << lidar_end_time << ","
                            << pv_list.size() << "," << effct_feat_num << ","
                            << plane_residual_count << "," << ndt_residual_count << ","
                            << res_mean_last << "," << signed_res_mean << ","
                            << res_p50 << "," << res_p95 << "," << res_p99 << ","
                            << plane_res_mean << "," << ndt_res_mean << ","
                            << raw_pos_step_norm << "," << raw_rot_step_norm << ","
                            << pos_step_norm << "," << rot_step_norm << ","
                            << normal_cond << "\n";
                    }
                    if (publish_iteration_debug)
                    {
                        RCLCPP_INFO(g_node->get_logger(),
                                    "IEKF frame %d iter %d abs %.6f signed %.6f plane %.6f/%d ndt %.6f/%d pos_step %.6e rot_step %.6e cond %.3e scale %.2f effect %d",
                                    scanIdx, iterCount + 1, res_mean_last, signed_res_mean,
                                    plane_res_mean, plane_residual_count,
                                    ndt_res_mean, ndt_residual_count,
                                    pos_step_norm, rot_step_norm, normal_cond,
                                    iekf_step_scale, effct_feat_num);
                    }
                    if (use_step_convergence &&
                        iterCount + 1 >= std::max(1, min_iteration) &&
                        pos_step_norm < converge_translation &&
                        rot_step_norm < converge_rotation)
                    {
                        flg_EKF_converged = true;
                    }
                }
                if (!EKF_stop_flg &&
                    (flg_EKF_converged || (iterCount == NUM_MAX_ITERATIONS - 1)))
                {
                    if (flg_EKF_inited)
                    {
                        G.setZero();
                        G.block<DIM_STATE, 6>(0, 0) = K * Hsub;
                        state.cov = (I_STATE - G) * state.cov;
                        total_distance += (state.pos_end - position_last).norm();
                        position_last = state.pos_end;

                        euler_cur = RotMtoEuler(state.rot_end);
                        tf2::Quaternion q;
                        q.setRPY(euler_cur(0), euler_cur(1), euler_cur(2));
                        geoQuat = tf2::toMsg(q);
                    }
                    EKF_stop_flg = true;
                    iter_used = iterCount + 1;
                    for (int i = 0; i < effct_feat_num; i++)
                    {
                        PointType pi_body;
                        pi_body.x = ptpl_list[i].point(0);
                        pi_body.y = ptpl_list[i].point(1);
                        pi_body.z = ptpl_list[i].point(2);
                        laserCloudOri->push_back(pi_body);
                    }
                }
                auto t_solve_end = std::chrono::high_resolution_clock::now();
                solve_time += std::chrono::duration_cast<std::chrono::duration<double>>(
                                  t_solve_end - t_solve_start)
                                  .count() *
                              1000;
                if (EKF_stop_flg)
                {
                    break;
                }
            }
            if (!measurement_update_valid)
            {
                update_rejected = true;
                skip_this_map_update = true;
                state = state_propagat;
                if (iter_used <= 0)
                {
                    iter_used = NUM_MAX_ITERATIONS;
                }
            }
            {
                const auto opt_delta = state - state_propagat;
                opt_rot_norm = opt_delta.block<3, 1>(0, 0).norm();
                opt_pos_norm = opt_delta.block<3, 1>(3, 0).norm();
            }
            
            /*** 7. Update Voxel Map ***/
            auto map_incremental_start = std::chrono::high_resolution_clock::now();
            pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
            std::vector<pointWithCov> pv_list;
            pv_list.reserve(feats_down_body->size());
            TransformLidar(state, p_imu, feats_down_body, world_lidar);

            int semantic_counts[4] = {0, 0, 0, 0};
            for (size_t i = 0; i < world_lidar->size(); i++)
            {
                const PointType &body_point = feats_down_body->points[i];
                pointWithCov pv;
                pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
                pv.point_world = pv.point;
                pv.point_index = static_cast<int>(i);
                pv.Semantic_ID = pointHybridSemantic(body_point);
                if (pv.Semantic_ID >= 1 && pv.Semantic_ID <= 3)
                {
                    semantic_counts[pv.Semantic_ID]++;
                }
                M3D point_crossmat = crossmat_list[i];
                M3D cov = body_var[i];
                const M3D rotation_jacobian = -state.rot_end * point_crossmat;
                cov = state.rot_end * cov * state.rot_end.transpose() +
                      rotation_jacobian * state.cov.block<3, 3>(0, 0) *
                          rotation_jacobian.transpose() +
                      state.cov.block<3, 3>(3, 3);
                pv.cov = cov;
                pv_list.push_back(pv);
            }

            std::sort(pv_list.begin(), pv_list.end(), var_contrast);
            const bool do_map_update = !(skip_map_update_on_unstable && skip_this_map_update);
            std::vector<VOXEL_LOC> updated_voxels;
            if (do_map_update)
            {
                UpdateVoxelMap(pv_list, voxel_map,
                               video_dump_enable ? &updated_voxels : nullptr);
            }
            else
            {
                RCLCPP_WARN_THROTTLE(g_node->get_logger(), *g_node->get_clock(), 2000,
                                     "Skipping voxel map update after unstable/no-match IEKF frame");
            }
            if (semantic_debug && (scanIdx < 5 || scanIdx % semantic_debug_interval == 0))
            {
                const auto map_stats = collectVoxelMapSemanticStats(voxel_map);
                RCLCPP_INFO(g_node->get_logger(),
                            "Semantic debug scan %d input plane=%d gaussian=%d object=%d | map voxels total=%zu unknown=%zu sem1=%zu sem2=%zu sem3=%zu plane=%zu ndt=%zu",
                            scanIdx, semantic_counts[1], semantic_counts[2], semantic_counts[3],
                            map_stats.total, map_stats.unknown, map_stats.semantic1,
                            map_stats.semantic2, map_stats.semantic3, map_stats.plane, map_stats.ndt);
            }
            auto map_incremental_end = std::chrono::high_resolution_clock::now();
            map_incremental_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                       map_incremental_end - map_incremental_start)
                                       .count() *
                                   1000;

            total_time = t_downsample + scan_match_time + solve_time + map_incremental_time +
                         undistort_time + calc_point_cov_time;
            dumpVideoFrame(scanIdx, lidar_end_time, state, surf_feats_undistort,
                           voxel_map, updated_voxels);
            const int final_effect = NUM_FEAT + NUM_NDT;
            const double plane_residual_ratio =
                final_effect > 0 ? static_cast<double>(NUM_FEAT) / static_cast<double>(final_effect) : 0.0;
            if (debug_csv_enable && debug_frame_csv.is_open())
            {
                const auto map_stats = collectVoxelMapSemanticStats(voxel_map);
                debug_frame_csv
                    << scanIdx << "," << Measures.lidar_beg_time << "," << lidar_end_time << ","
                    << (lidar_end_time - Measures.lidar_beg_time) << ","
                    << (cv_predicted ? 1 : 0) << "," << (cv_deskewed ? 1 : 0) << ","
                    << pred_pos_norm << "," << pred_rot_norm << ","
                    << opt_pos_norm << "," << opt_rot_norm << ","
                    << surf_feats_undistort->size() << "," << feats_down_body->size() << ","
                    << final_effect << "," << NUM_FEAT << "," << NUM_NDT << ","
                    << plane_residual_ratio << "," << iter_used << ","
                    << last_res_abs << "," << last_res_signed << ","
                    << last_res_p50 << "," << last_res_p95 << "," << last_res_p99 << ","
                    << last_plane_abs << "," << last_ndt_abs << ","
                    << last_pos_step << "," << last_rot_step << "," << last_cond << ","
                    << (update_rejected ? 1 : 0) << ","
                    << semantic_counts[1] << "," << semantic_counts[2] << "," << semantic_counts[3] << ","
                    << map_stats.total << "," << map_stats.unknown << ","
                    << map_stats.semantic1 << "," << map_stats.semantic2 << ","
                    << map_stats.semantic3 << "," << map_stats.plane << "," << map_stats.ndt << ","
                    << undistort_time << "," << t_downsample << "," << calc_point_cov_time << ","
                    << scan_match_time << "," << solve_time << "," << map_incremental_time << ","
                    << total_time << "," << state.vel_end.norm() << ","
                    << state.bias_g.norm() << "," << state.bias_a.norm() << ","
                    << state.gravity.norm() << ","
                    << Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, DIM_STATE, DIM_STATE>>(
                        (0.5 * (state.cov + state.cov.transpose())).eval()).eigenvalues().minCoeff()
                    << "\n";
                debug_frame_csv.flush();
            }

            /*** 8. Publish functions:  ***/
            publish_odometry(pubOdomAftMapped);
            publish_path(pubPath);
            if (publish_point_cloud)
            {
                publish_surf_frame_world(pubLaserCloudSurfFull, pub_point_cloud_skip);
                publish_effect(pubLaserCloudEffect);
            }

            frame_num++;
            mean_raw_points = mean_raw_points * (frame_num - 1) / frame_num +
                              (double)(surf_feats_undistort->size()) / frame_num;
            mean_ds_points = mean_ds_points * (frame_num - 1) / frame_num +
                             (double)(feats_down_body->size()) / frame_num;
            mean_effect_points = mean_effect_points * (frame_num - 1) / frame_num +
                                 (double)effct_feat_num / frame_num;

            undistort_time_mean =
                undistort_time_mean * (frame_num - 1) / frame_num + (undistort_time) / frame_num;
            down_sample_time_mean =
                down_sample_time_mean * (frame_num - 1) / frame_num + (t_downsample) / frame_num;
            calc_cov_time_mean = calc_cov_time_mean * (frame_num - 1) / frame_num +
                                 (calc_point_cov_time) / frame_num;
            scan_match_time_mean =
                scan_match_time_mean * (frame_num - 1) / frame_num + (scan_match_time) / frame_num;
            ekf_solve_time_mean =
                ekf_solve_time_mean * (frame_num - 1) / frame_num + (solve_time) / frame_num;
            map_update_time_mean = map_update_time_mean * (frame_num - 1) / frame_num +
                                   (map_incremental_time) / frame_num;
            iter_num_mean =
                iter_num_mean * (frame_num - 1) / frame_num + (double)(iter_used) / frame_num;
            plane_residual_ratio_mean =
                plane_residual_ratio_mean * (frame_num - 1) / frame_num + plane_residual_ratio / frame_num;

            aver_time_consu =
                aver_time_consu * (frame_num - 1) / frame_num + (total_time) / frame_num;
            slam_algorithm_total_time_mean = aver_time_consu;

            time_log_counter++;
            if (should_log_frame)
            {
                cout << "pos:" << state.pos_end.transpose() << endl;
                cout << "[ Time ]: "
                     << "average undistort: " << undistort_time_mean << std::endl;
                cout << "[ Time ]: "
                     << "average down sample: " << down_sample_time_mean << std::endl;
                cout << "[ Time ]: "
                     << "average calc cov: " << calc_cov_time_mean << std::endl;
                cout << "[ Time ]: "
                     << "average scan match: " << scan_match_time_mean << std::endl;
                cout << "[ Time ]: "
                     << "average solve: " << ekf_solve_time_mean << std::endl;
                cout << "[ Time ]: "
                     << "average map incremental: " << map_update_time_mean << std::endl;
                cout << "[ IEKF ]: "
                     << "average iterations: " << iter_num_mean << std::endl;
                cout << "[ Effect ]: final plane residuals: " << NUM_FEAT
                     << ", final NDT residuals: " << NUM_NDT
                     << ", plane ratio: " << plane_residual_ratio
                     << ", average plane ratio: " << plane_residual_ratio_mean << std::endl;
                cout << "[ Time ]: "
                     << " average total " << aver_time_consu << endl;
                cout << "--------------------------------------------" << endl;
            }
            if (write_kitti_log)
            {
                kitti_log(fp_kitti);
            }

            update_lidar_only_state_history(state, lidar_state_time);
            const auto slam_frame_end = std::chrono::steady_clock::now();
            recordSlamPerFrameEfficiency(scanIdx, lidar_end_time, total_time);
            recordSlamEfficiencyFrame(slam_frame_start, slam_frame_end);
            scanIdx++;
        }
        status = rclcpp::ok();
        if (!processed_measurement)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }
    closeTumResultFile();
    writeSlamEfficiencyReport();
    closePerFrameEfficiencyFile();
    std::cout << "INPUT_COUNTS lidar=" << received_lidar_messages
              << " imu=" << publish_count << std::endl;
    // Release ROS entities before shutting down the context.
    g_tf_broadcaster.reset();
    sub_pcl.reset();
#ifdef HAVE_LIVOX_ROS_DRIVER2
    sub_livox.reset();
#endif
    sub_imu.reset();
    pubLaserCloudSurfFull.reset();
    pubLaserCloudEffect.reset();
    pubOdomAftMapped.reset();
    pubPath.reset();
    voxel_map_pub.reset();
    g_node.reset();
    if (rclcpp::ok())
    {
        rclcpp::shutdown();
    }
    return 0;
}
