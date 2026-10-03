#include <Eigen/Eigen>
#include <cassert>
#include <cmath>
#include <common_lib.h>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <geometry_msgs/msg/vector3.hpp>
#include <math.h>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <so3_math.h>
#include <thread>
#include <hybrid_voxel_map/msg/states.hpp>

/// *************Preconfiguration

#define MAX_INI_COUNT (200)

/*TEST3*/
static int TEST3 = 0;

const bool time_list(PointType &x, PointType &y)
{
    return (x.curvature < y.curvature);
};

/// *************IMU Process and undistortion
class ImuProcess
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuProcess();

    ~ImuProcess();

    void Reset();

    void Reset(double start_timestamp, const sensor_msgs::msg::Imu::ConstSharedPtr &lastimu);

    void set_extrinsic(const V3D &transl, const M3D &rot);

    void set_extrinsic(const V3D &transl);

    void set_extrinsic(const MD(4, 4) & T);

    void set_gyr_cov_scale(const V3D &scaler);

    void set_acc_cov_scale(const V3D &scaler);

    void set_gyr_bias_cov(const V3D &b_g);

    void set_acc_bias_cov(const V3D &b_a);

    // void Process(const MeasureGroup &meas, StatesGroup &state,
    //              PointCloudXYZI::Ptr &surf_pcl_un_,  PointCloudXYZI::Ptr &corn_pcl_un_);
    void Process(const MeasureGroup &meas, StatesGroup &stat,
                 PointCloudXYZI::Ptr &cur_surf_pcl_un_);

    ofstream fout_imu;
    V3D cov_acc;
    V3D cov_gyr;
    V3D cov_acc_scale;
    V3D cov_gyr_scale;
    V3D cov_bias_gyr;
    V3D cov_bias_acc;
    M3D Lid_rot_to_IMU;
    V3D Lid_offset_to_IMU;
    double first_lidar_time;
    bool imu_en;
    bool is_first_frame = true;

private:
    void IMU_init(const MeasureGroup &meas, StatesGroup &state, int &N);
    void UndistortPcl(const MeasureGroup &meas, StatesGroup &state_inout, PointCloudXYZI &surf_pcl_out);

    PointCloudXYZI::Ptr cur_pcl_un_;
    sensor_msgs::msg::Imu::ConstSharedPtr last_imu_;
    deque<sensor_msgs::msg::Imu::ConstSharedPtr> v_imu_;
    vector<Pose6D> IMUpose;
    vector<M3D> v_rot_pcl_;

    V3D mean_acc;
    V3D mean_gyr;
    V3D angvel_last;
    V3D acc_s_last;
    double start_timestamp_;
    double time_last_scan_;
    int init_iter_num = 1;
    bool b_first_frame_ = true;
    bool imu_need_init_ = true;
};

ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
    imu_en = true;
    init_iter_num = 1;
    time_last_scan_ = -1.0;
    cov_acc = V3D(0.1, 0.1, 0.1);
    cov_gyr = V3D(0.1, 0.1, 0.1);
    // old
    cov_acc_scale = V3D(1, 1, 1);
    cov_gyr_scale = V3D(1, 1, 1);

    cov_bias_gyr = V3D(0.1, 0.1, 0.1);
    cov_bias_acc = V3D(0.1, 0.1, 0.1);
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    angvel_last = Zero3d;
    Lid_offset_to_IMU = Zero3d;
    Lid_rot_to_IMU = Eye3d;
    last_imu_.reset(new sensor_msgs::msg::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset()
{
    RCLCPP_WARN(rclcpp::get_logger("hybrid_voxel_map"), "Reset ImuProcess");
    mean_acc = V3D(0, 0, -1.0);
    mean_gyr = V3D(0, 0, 0);
    acc_s_last = Zero3d;
    angvel_last = Zero3d;
    imu_need_init_ = true;
    start_timestamp_ = -1;
    init_iter_num = 1;
    v_imu_.clear();
    time_last_scan_ = -1.0;
    IMUpose.clear();
    last_imu_.reset(new sensor_msgs::msg::Imu());
    cur_pcl_un_.reset(new PointCloudXYZI());
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
    Lid_offset_to_IMU = transl;
    Lid_rot_to_IMU = rot;
}

void ImuProcess::set_gyr_cov_scale(const V3D &scaler)
{
    cov_gyr_scale = scaler;
}

void ImuProcess::set_acc_cov_scale(const V3D &scaler)
{
    cov_acc_scale = scaler;
}

void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }

void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }

void ImuProcess::IMU_init(const MeasureGroup &meas, StatesGroup &state_inout,
                          int &N)
{
    /** 1. initializing the gravity, gyro bias
     ** 2. normalize the acceleration measurenments to unit gravity **/
    RCLCPP_INFO(rclcpp::get_logger("hybrid_voxel_map"), "IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
    V3D cur_acc, cur_gyr;

    if (b_first_frame_)
    {
        Reset();
        N = 1;
        b_first_frame_ = false;
        const auto &imu_acc = meas.imu.front()->linear_acceleration;
        const auto &gyr_acc = meas.imu.front()->angular_velocity;
        mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        first_lidar_time = meas.lidar_beg_time;
    }

    for (const auto &imu : meas.imu)
    {
        const auto &imu_acc = imu->linear_acceleration;
        const auto &gyr_acc = imu->angular_velocity;
        cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
        cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

        mean_acc += (cur_acc - mean_acc) / N;
        mean_gyr += (cur_gyr - mean_gyr) / N;
        N++;
    }

    state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;
    state_inout.rot_end = Eye3d;
    // state_inout.rot_end = Lid_rot_to_IMU;
    state_inout.bias_g = mean_gyr;
    last_imu_ = meas.imu.back();
    double max_offset_ms = 0.0;
    for (const auto &point : meas.surf_lidar->points)
        max_offset_ms = std::max(max_offset_ms, static_cast<double>(point.curvature));
    time_last_scan_ = meas.lidar_beg_time + max_offset_ms / 1000.0;
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas,
                              StatesGroup &state_inout,
                              PointCloudXYZI &surf_pcl_out)
{
    /*** add the imu of the last frame-tail to the of current frame-head ***/
    auto v_imu = meas.imu;
    v_imu.push_front(last_imu_);
    const double imu_beg_time = rclcpp::Time(v_imu.front()->header.stamp).seconds();
    const double imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
    const double &pcl_beg_time = meas.lidar_beg_time;
    // The input state belongs to the previous scan end.
    const double propagation_start = time_last_scan_ >= 0.0
        ? time_last_scan_ : pcl_beg_time;
    /*** sort point clouds by offset time ***/
    surf_pcl_out = *(meas.surf_lidar);
    sort(surf_pcl_out.points.begin(), surf_pcl_out.points.end(), time_list);
    const double &pcl_end_time =
        pcl_beg_time + surf_pcl_out.points.back().curvature / double(1000);
    /*** Initialize IMU pose ***/
    IMUpose.clear();
    IMUpose.push_back(set_pose6d(propagation_start - pcl_beg_time, acc_s_last, angvel_last,
                                 state_inout.vel_end, state_inout.pos_end,
                                 state_inout.rot_end));
    /*** forward propagation at each imu point ***/
    V3D acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end),
        pos_imu(state_inout.pos_end);
    M3D R_imu(state_inout.rot_end);
    MD(DIM_STATE, DIM_STATE)
    F_x, cov_w;
    double dt = 0;
    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
    {
        auto &&head = *(it_imu);
        auto &&tail = *(it_imu + 1);

        if (rclcpp::Time(tail->header.stamp).seconds() <= propagation_start)
        {
            continue;
        }

        angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
            0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
            0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
        acc_avr << 0.5 *
                       (head->linear_acceleration.x + tail->linear_acceleration.x),
            0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
            0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

        angvel_avr -= state_inout.bias_g;
        acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

        if (rclcpp::Time(head->header.stamp).seconds() < propagation_start)
        {
            dt = rclcpp::Time(tail->header.stamp).seconds() - propagation_start;
        }
        else
        {
            dt = rclcpp::Time(tail->header.stamp).seconds() - rclcpp::Time(head->header.stamp).seconds();
        }

        /*** covariance propagation of error state ***/
        M3D acc_avr_skew;
        M3D Exp_f = Exp(angvel_avr, dt);
        acc_avr_skew << SKEW_SYM_MATRX(acc_avr);
        // F_x
        F_x.setIdentity();
        F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
        F_x.block<3, 3>(0, 9) = -Eye3d * dt;
        F_x.block<3, 3>(3, 6) = Eye3d * dt;
        F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
        F_x.block<3, 3>(6, 12) = -R_imu * dt;
        F_x.block<3, 3>(6, 15) = Eye3d * dt;
        // Q
        cov_w.setZero();
        cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr * dt * dt;
        cov_w.block<3, 3>(6, 6) =
            R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
        cov_w.block<3, 3>(9, 9).diagonal() =
            cov_bias_gyr * dt * dt; // bias gyro covariance
        cov_w.block<3, 3>(12, 12).diagonal() =
            cov_bias_acc * dt * dt; // bias acc covariance
        // P = FPF^T + Q
        state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

        /*** propogation of nonmial state ***/
        R_imu = R_imu * Exp_f;
        acc_imu = R_imu * acc_avr + state_inout.gravity;
        pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
        vel_imu = vel_imu + acc_imu * dt;
        /* save the poses at each IMU measurements */
        angvel_last = angvel_avr;
        acc_s_last = acc_imu;
        double offs_t = rclcpp::Time(tail->header.stamp).seconds() - pcl_beg_time;
        IMUpose.push_back(
            set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    }
    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt = note * (pcl_end_time - imu_end_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end =
        pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
    time_last_scan_ = pcl_end_time;

    auto pos_liD_e =
        state_inout.pos_end + state_inout.rot_end * Lid_offset_to_IMU;
    /*** undistort each surf lidar point (backward propagation) ***/
    auto it_surf_pcl = surf_pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
    {
        auto head = it_kp - 1;
        auto tail = it_kp;
        R_imu << MAT_FROM_ARRAY(head->rot);
        acc_imu << VEC_FROM_ARRAY(head->acc);
        vel_imu << VEC_FROM_ARRAY(head->vel);
        pos_imu << VEC_FROM_ARRAY(head->pos);
        angvel_avr << VEC_FROM_ARRAY(head->gyr);

        for (; it_surf_pcl->curvature / double(1000) > head->offset_time; it_surf_pcl--)
        {
            dt = it_surf_pcl->curvature / double(1000) - head->offset_time;
            /* Transform to the 'end' frame, using only the rotation
             * Note: Compensation direction is INVERSE of Frame's moving direction
             * So if we want to compensate a point at timestamp-i to the frame-e
             * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is
             * represented in global frame */
            M3D R_i(R_imu * Exp(angvel_avr, dt));
            V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt +
                     R_i * Lid_offset_to_IMU - pos_liD_e);

            V3D P_i(it_surf_pcl->x, it_surf_pcl->y, it_surf_pcl->z);
            V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

            /// save Undistorted points and their rotation
            it_surf_pcl->x = P_compensate(0);
            it_surf_pcl->y = P_compensate(1);
            it_surf_pcl->z = P_compensate(2);

            if (it_surf_pcl == surf_pcl_out.points.begin())
                break;
        }
    }
}

void ImuProcess::Process(const MeasureGroup &meas, StatesGroup &stat,
                         PointCloudXYZI::Ptr &cur_surf_pcl_un_)
{
    if (meas.imu.empty() && imu_en)
    {
        return;
    }
    assert(meas.surf_lidar != nullptr);
    if (imu_need_init_ && imu_en)
    {
        /// The very first lidar frame
        IMU_init(meas, stat, init_iter_num);
        imu_need_init_ = true;
        last_imu_ = meas.imu.back();
        if (init_iter_num > MAX_INI_COUNT)
        {
            imu_need_init_ = false;
            cov_acc = Eye3d * cov_acc_scale;
            cov_gyr = Eye3d * cov_gyr_scale;
            RCLCPP_INFO(rclcpp::get_logger("hybrid_voxel_map"), "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f "
                     "%.4f %.4f; acc covarience: %.8f %.8f %.8f; gyr covarience: "
                     "%.8f %.8f %.8f",
                     stat.gravity[0], stat.gravity[1], stat.gravity[2],
                     stat.gravity.norm(), stat.bias_g[0], stat.bias_g[1],
                     stat.bias_g[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0],
                     cov_gyr[1], cov_gyr[2]);
        }
        return;
    }

    /// Undistort points： the first point is assummed as the base frame
    if (imu_en)
    {
        UndistortPcl(meas, stat, *cur_surf_pcl_un_);
        last_imu_ = meas.imu.back();
    }
    else
    {
        if (b_first_frame_)
        {
            RCLCPP_WARN(rclcpp::get_logger("hybrid_voxel_map"),
                        "No IMU input: using raw LiDAR scan without undistortion");
            b_first_frame_ = false;
        }
        cov_acc = Eye3d * cov_acc_scale;
        cov_gyr = Eye3d * cov_gyr_scale;
        *cur_surf_pcl_un_ = *(meas.surf_lidar);
    }
}
