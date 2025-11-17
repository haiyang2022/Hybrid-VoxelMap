#include <thread>
#include "preprocess.h"
#include <cmath>

Preprocess::Preprocess()
    : lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
    N_SCANS = 6;
}

Preprocess::~Preprocess() {}

// Avia input
void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg,
                         PointCloudXYZI::Ptr &pcl_surf_out)
{
    avia_handler(msg);
    *pcl_surf_out = pl_surf;
}

// Other input
void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg,
                         PointCloudXYZI::Ptr &pcl_out)
{
    switch (lidar_type)
    {
    case L515:
        l515_handler(msg);
        break;
    case VELO16:
        velodyne_handler(msg);
        break;
    case OUST64:
        oust64_handler(msg);
        break;
    case XT32:
        xt32_handler(msg);
        break;
    default:
        printf("Error LiDAR Type");
        break;
    }
    *pcl_out = pl_surf;
}


void Preprocess::avia_handler(
    const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    auto t_feature_start = std::chrono::high_resolution_clock::now();
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();
    int plsize = msg->point_num;
    std::vector<bool> is_valid_pt(plsize, false);

    pl_corn.reserve(plsize);
    pl_surf.reserve(plsize);
    pl_full.resize(plsize);

    for (int i = 0; i < N_SCANS; i++)
    {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
    }
    uint valid_num = 0;

    for (uint i = 1; i < plsize; i++)
    {
        if ((msg->points[i].line < N_SCANS) &&
            ((msg->points[i].tag & 0x30) == 0x10 ||
             (msg->points[i].tag & 0x30) == 0x00))
        {
            if (i % point_filter_num == 0)
            {
                pl_full[i].x = msg->points[i].x;
                pl_full[i].y = msg->points[i].y;
                pl_full[i].z = msg->points[i].z;
                pl_full[i].intensity = msg->points[i].reflectivity;
                pl_full[i].curvature =
                    msg->points[i].offset_time /
                    float(1000000); // use curvature as time of each laser points

                if (isinf(pl_full[i].x) || isinf(pl_full[i].y) || isinf(pl_full[i].z))
                {
                    continue;
                }

                if ((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) ||
                    (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
                    (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7))
                {
                    is_valid_pt[i] = true;
                    valid_num++;
                }
            }
        }
    }

    for (uint i = 1; i < plsize; i++)
    {
        if (is_valid_pt[i])
        {
            pl_surf.points.push_back(pl_full[i]);
        }
    }
}

void Preprocess::l515_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;

        if (i % point_filter_num == 0)
        {
            pl_surf.push_back(added_pt);
        }
    }
}

#define MAX_LINE_NUM 64

void Preprocess::velodyne_handler(
    const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();

    float startOri = -atan2(pl_orig.points[0].y, pl_orig.points[0].x);
    float endOri = -atan2(pl_orig.points[plsize - 1].y,
                          pl_orig.points[plsize - 1].x) +
                   2 * M_PI;

    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }

    bool halfPassed = false;
    for (int i = 0; i < pl_orig.size(); i++)
    {
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;

        float angle = atan(added_pt.z / sqrt(added_pt.x * added_pt.x +
                                             added_pt.y * added_pt.y)) *
                      180 / M_PI;
        int scanID = 0;
        if (angle >= -8.83)
            scanID = int((2 - angle) * 3.0 + 0.5);
        else
            scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

        if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
        {
            continue;
        }
        float ori = -atan2(added_pt.y, added_pt.x);
        if (!halfPassed)
        {
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        added_pt.curvature = (ori - startOri) / (endOri - startOri) * 100.00;
        pl_surf.push_back(added_pt);
    }
}

void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();
    pcl::PointCloud<ouster_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.size();
    pl_corn.reserve(plsize);
    pl_surf.reserve(plsize);

    double time_stamp = msg->header.stamp.toSec();
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
        Eigen::Vector3d pt_vec;
        PointType added_pt;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.curvature = pl_orig.points[i].t * 1e-6; // curvature unit: ms

        pl_surf.points.push_back(added_pt);
    }
}

void Preprocess::xt32_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<xt32_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    pl_surf.reserve(plsize);

    bool is_first[MAX_LINE_NUM];
    double yaw_fp[MAX_LINE_NUM] = {0};     // yaw of first scan point
    double omega_l = 3.61;                 // scan angular velocity
    float yaw_last[MAX_LINE_NUM] = {0.0};  // yaw of last scan point
    float time_last[MAX_LINE_NUM] = {0.0}; // last offset time

    memset(is_first, true, sizeof(is_first));
    double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
    double yaw_end = yaw_first;
    int layer_first = pl_orig.points[0].ring;
    for (uint i = plsize - 1; i > 0; i--)
    {
        if (pl_orig.points[i].ring == layer_first)
        {
            yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
            break;
        }
    }

    double time_head = pl_orig.points[0].timestamp;

    for (int i = 0; i < plsize; i++)
    {
        PointType added_pt;

        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = (pl_orig.points[i].timestamp - time_head) * 1000.f;
        pl_surf.points.push_back(added_pt);
    }
}