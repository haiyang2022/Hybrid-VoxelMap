#include <livox_ros_driver/CustomMsg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

using namespace std;

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

enum LID_TYPE
{
    AVIA = 1,
    VELO16,
    L515,
    OUST64,
    XT32
}; //{1, 2, 3}

namespace velodyne_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        float time;
        uint16_t ring;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
                                  (float, x, x)(float, y, y)(float, z, z)(
                                      float, intensity,
                                      intensity)(float, time, time)(uint16_t,
                                                                    ring, ring))

namespace ouster_ros
{
    struct EIGEN_ALIGN16 Point
    {
        PCL_ADD_POINT4D;
        float intensity;
        uint32_t t;
        uint16_t reflectivity;
        uint8_t ring;
        uint16_t ambient;
        uint32_t range;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
} // namespace ouster_ros

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

/*** Hesai_XT32 ***/
namespace xt32_ros
{
struct EIGEN_ALIGN16 Point
{
    PCL_ADD_POINT4D;
    float    intensity;
    double   timestamp;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace xt32_ros
POINT_CLOUD_REGISTER_POINT_STRUCT( xt32_ros::Point,
                                   ( float, x, x )( float, y, y )( float, z, z )( float, intensity, intensity )( double, timestamp,
                                                                                                                 timestamp )( uint16_t, ring, ring ) )
/*****************/
                                                                        
class Preprocess {
public:
    Preprocess();

    ~Preprocess();

    void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_surf_out);

    void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);

    PointCloudXYZI pl_full, pl_corn, pl_surf;
    PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
    int lidar_type, point_filter_num, N_SCANS;;
    double blind;
    ros::Publisher pub_full, pub_surf, pub_corn;

private:
    void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);

    void l515_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);

    void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);

    void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);

    void xt32_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
};
