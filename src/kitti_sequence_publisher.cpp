#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

using PointCloud2Msg = sensor_msgs::msg::PointCloud2;
using PointFieldMsg = sensor_msgs::msg::PointField;

namespace
{
std::vector<std::string> listBinScans(const std::string &velodyne_dir)
{
    std::vector<std::string> scans;
    DIR *dir = opendir(velodyne_dir.c_str());
    if (dir == nullptr)
    {
        throw std::runtime_error("Cannot open velodyne directory: " + velodyne_dir);
    }

    while (dirent *entry = readdir(dir))
    {
        const std::string name(entry->d_name);
        if (name.size() >= 4 && name.substr(name.size() - 4) == ".bin")
        {
            scans.push_back(velodyne_dir + "/" + name);
        }
    }
    closedir(dir);
    std::sort(scans.begin(), scans.end());
    return scans;
}

std::vector<double> loadTimes(const std::string &sequence_path)
{
    std::vector<double> times;
    std::ifstream file(sequence_path + "/times.txt");
    if (!file.is_open())
    {
        return times;
    }
    double value = 0.0;
    while (file >> value)
    {
        times.push_back(value);
    }
    return times;
}

bool readScan(const std::string &path, std::vector<float> &raw)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(4 * sizeof(float)) != 0)
    {
        return false;
    }

    raw.resize(static_cast<size_t>(size / sizeof(float)));
    file.seekg(0, std::ios::beg);
    return static_cast<bool>(file.read(reinterpret_cast<char *>(raw.data()), size));
}

std::string fileStem(const std::string &path)
{
    const size_t slash_pos = path.find_last_of('/');
    const size_t name_pos = slash_pos == std::string::npos ? 0 : slash_pos + 1;
    const size_t dot_pos = path.find_last_of('.');
    if (dot_pos == std::string::npos || dot_pos < name_pos)
    {
        return path.substr(name_pos);
    }
    return path.substr(name_pos, dot_pos - name_pos);
}

bool readLabels(const std::string &path, std::vector<uint32_t> &labels)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        return false;
    }

    const std::streamsize size = file.tellg();
    if (size <= 0 || size % static_cast<std::streamsize>(sizeof(uint32_t)) != 0)
    {
        return false;
    }

    labels.resize(static_cast<size_t>(size / sizeof(uint32_t)));
    file.seekg(0, std::ios::beg);
    return static_cast<bool>(file.read(reinterpret_cast<char *>(labels.data()), size));
}

std::unordered_set<uint32_t> parseSemanticLabelSet(const std::string &labels)
{
    std::unordered_set<uint32_t> label_set;
    std::stringstream stream(labels);
    std::string token;
    while (std::getline(stream, token, ','))
    {
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (token.empty())
        {
            continue;
        }
        label_set.insert(static_cast<uint32_t>(std::stoul(token)));
    }
    return label_set;
}

uint32_t semanticKittiToHybridCategory(const uint32_t raw_label,
                                       const std::unordered_set<uint32_t> &plane_labels)
{
    const uint32_t semantic_label = raw_label & 0xFFFFu;
    if (plane_labels.find(semantic_label) != plane_labels.end())
    {
        return 1; // plane model
    }

    switch (semantic_label)
    {
    case 40: // road
    case 44: // parking
    case 48: // sidewalk
    case 49: // other-ground
    case 50: // building
    case 51: // fence
    case 52: // other-structure
    case 60: // lane-marking
    case 70: // vegetation
    case 71: // trunk
    case 72: // terrain
    case 80: // pole
    case 81: // traffic-sign
    case 99: // other-object
        return 2; // Gaussian/NDT model

    case 10:  // car
    case 11:  // bicycle
    case 13:  // bus
    case 15:  // motorcycle
    case 16:  // on-rails
    case 18:  // truck
    case 20:  // other-vehicle
    case 30:  // person
    case 31:  // bicyclist
    case 32:  // motorcyclist
    case 252: // moving-car
    case 253: // moving-bicyclist
    case 254: // moving-person
    case 255: // moving-motorcyclist
    case 256: // moving-on-rails
    case 257: // moving-bus
    case 258: // moving-truck
    case 259: // moving-other-vehicle
        return 3; // object/dynamic, kept as non-plane Gaussian/NDT

    default:
        return 0; // unknown or unlabeled
    }
}

void writeFloat(std::vector<uint8_t> &data, const size_t offset, const float value)
{
    std::memcpy(data.data() + offset, &value, sizeof(float));
}

void writeUint32(std::vector<uint8_t> &data, const size_t offset, const uint32_t value)
{
    std::memcpy(data.data() + offset, &value, sizeof(uint32_t));
}
} // namespace

class KittiSequencePublisher : public rclcpp::Node
{
public:
    KittiSequencePublisher()
        : Node("kitti_sequence_publisher_cpp")
    {
        sequence_path_ = this->declare_parameter<std::string>(
            "sequence_path", "data/KITTI/dataset/sequences/08");
        topic_ = this->declare_parameter<std::string>("topic", "/velodyne_points");
        frame_id_ = this->declare_parameter<std::string>("frame_id", "camera_init");
        rate_ = this->declare_parameter<double>("rate", 10.0);
        loop_ = this->declare_parameter<bool>("loop", false);
        start_index_ = this->declare_parameter<int>("start_index", 0);
        end_index_ = this->declare_parameter<int>("end_index", -1);
        qos_depth_ = static_cast<int>(this->declare_parameter<int>("qos_depth", 512));
        qos_depth_ = std::max(1, qos_depth_);
        wait_for_subscriber_ = this->declare_parameter<bool>("wait_for_subscriber", true);
        // KITTI Odometry scans are already motion-compensated and have no raw point timestamps.
        publish_point_time_ = this->declare_parameter<bool>("publish_point_time", false);
        publish_gt_semantics_ = this->declare_parameter<bool>("publish_gt_semantics", false);
        label_dir_ = this->declare_parameter<std::string>("label_dir", "");
        log_semantic_stats_ = this->declare_parameter<bool>("log_semantic_stats", true);
        semantic_plane_labels_ = this->declare_parameter<std::string>(
            "semantic_plane_labels", "40,44,48,49,50,51,52,60,72");
        plane_labels_ = parseSemanticLabelSet(semantic_plane_labels_);
        if (label_dir_.empty())
        {
            label_dir_ = sequence_path_ + "/labels";
        }

        const std::string velodyne_dir = sequence_path_ + "/velodyne";
        scans_ = listBinScans(velodyne_dir);
        if (end_index_ >= 0 && end_index_ + 1 < static_cast<int>(scans_.size()))
        {
            scans_.resize(static_cast<size_t>(end_index_ + 1));
        }
        if (start_index_ > 0 && start_index_ < static_cast<int>(scans_.size()))
        {
            scans_.erase(scans_.begin(), scans_.begin() + start_index_);
        }
        if (scans_.empty())
        {
            throw std::runtime_error("No KITTI .bin scans found under " + velodyne_dir);
        }

        times_ = loadTimes(sequence_path_);
        publisher_ = this->create_publisher<PointCloud2Msg>(
            topic_, rclcpp::QoS(rclcpp::KeepLast(qos_depth_)).reliable());
        fields_ = makeFields();
        start_wall_ = std::chrono::steady_clock::now();
        next_pub_time_ = start_wall_;

        RCLCPP_INFO(this->get_logger(), "Publishing %zu KITTI scans from %s to %s at %.3f Hz",
                    scans_.size(), sequence_path_.c_str(), topic_.c_str(), rate_);
        if (publish_gt_semantics_)
        {
            RCLCPP_INFO(this->get_logger(), "SemanticKITTI GT labels enabled from %s", label_dir_.c_str());
            RCLCPP_INFO(this->get_logger(), "SemanticKITTI plane labels: %s",
                        semantic_plane_labels_.c_str());
        }
        timer_ = this->create_wall_timer(std::chrono::milliseconds(1),
                                         std::bind(&KittiSequencePublisher::publishDueScans, this));
    }

private:
    std::vector<PointFieldMsg> makeFields() const
    {
        std::vector<PointFieldMsg> fields;
        fields.reserve(publish_gt_semantics_ ? 7 : 6);
        fields.push_back(makeField("x", 0, PointFieldMsg::FLOAT32));
        fields.push_back(makeField("y", 4, PointFieldMsg::FLOAT32));
        fields.push_back(makeField("z", 8, PointFieldMsg::FLOAT32));
        fields.push_back(makeField("intensity", 16, PointFieldMsg::FLOAT32));
        fields.push_back(makeField("time", 20, PointFieldMsg::FLOAT32));
        fields.push_back(makeField("ring", 24, PointFieldMsg::UINT16));
        if (publish_gt_semantics_)
        {
            fields.push_back(makeField("semantic", 28, PointFieldMsg::UINT32));
        }
        return fields;
    }

    PointFieldMsg makeField(const std::string &name, const uint32_t offset, const uint8_t datatype) const
    {
        PointFieldMsg field;
        field.name = name;
        field.offset = offset;
        field.datatype = datatype;
        field.count = 1;
        return field;
    }

    void publishDueScans()
    {
        if (!rclcpp::ok())
        {
            return;
        }
        if (!started_)
        {
            if (wait_for_subscriber_ && publisher_->get_subscription_count() == 0)
            {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "Waiting for a subscriber on %s", topic_.c_str());
                return;
            }
            started_ = true;
            start_wall_ = std::chrono::steady_clock::now();
            next_pub_time_ = start_wall_;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now < next_pub_time_)
        {
            return;
        }
        next_pub_time_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / std::max(rate_, 1e-6)));

        if (index_ >= scans_.size())
        {
            if (loop_)
            {
                index_ = 0;
                next_pub_time_ = now;
            }
            else
            {
                const double elapsed = std::chrono::duration<double>(now - start_wall_).count();
                RCLCPP_INFO(this->get_logger(), "KITTI sequence finished, published %zu frames in %.3f s (%.3f Hz)",
                            index_, elapsed, static_cast<double>(index_) / std::max(elapsed, 1e-6));
                rclcpp::shutdown();
                return;
            }
        }

        std::vector<float> raw;
        if (!readScan(scans_[index_], raw))
        {
            RCLCPP_WARN(this->get_logger(), "Skipping malformed scan: %s", scans_[index_].c_str());
            ++index_;
            return;
        }

        const size_t point_num = raw.size() / 4;
        std::vector<uint32_t> labels;
        bool labels_ok = false;
        if (publish_gt_semantics_)
        {
            const std::string label_path = label_dir_ + "/" + fileStem(scans_[index_]) + ".label";
            labels_ok = readLabels(label_path, labels) && labels.size() == point_num;
            if (!labels_ok)
            {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                     "GT semantic labels unavailable or size-mismatched for %s",
                                     scans_[index_].c_str());
            }
        }

        PointCloud2Msg msg;
        if (!times_.empty() && start_index_ + static_cast<int>(index_) < static_cast<int>(times_.size()))
        {
            const double stamp = times_[static_cast<size_t>(start_index_) + index_];
            msg.header.stamp.sec = static_cast<int32_t>(stamp);
            msg.header.stamp.nanosec = static_cast<uint32_t>((stamp - static_cast<double>(msg.header.stamp.sec)) * 1e9);
        }
        else
        {
            msg.header.stamp = this->now();
        }
        msg.header.frame_id = frame_id_;
        msg.height = 1;
        msg.width = static_cast<uint32_t>(point_num);
        msg.fields = fields_;
        msg.is_bigendian = false;
        msg.point_step = 32;
        msg.row_step = msg.point_step * msg.width;
        msg.is_dense = true;
        msg.data.assign(point_num * msg.point_step, 0);

        size_t semantic_counts[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < point_num; ++i)
        {
            const size_t src = i * 4;
            const size_t dst = i * msg.point_step;
            writeFloat(msg.data, dst + 0, raw[src + 0]);
            writeFloat(msg.data, dst + 4, raw[src + 1]);
            writeFloat(msg.data, dst + 8, raw[src + 2]);
            writeFloat(msg.data, dst + 16, raw[src + 3]);
            const float rel_time = (publish_point_time_ && point_num > 1)
                                       ? static_cast<float>(0.1 * static_cast<double>(i) / point_num)
                                       : 0.0f;
            writeFloat(msg.data, dst + 20, rel_time);
            if (publish_gt_semantics_)
            {
                const uint32_t semantic_category =
                    labels_ok ? semanticKittiToHybridCategory(labels[i], plane_labels_) : 0;
                writeUint32(msg.data, dst + 28, semantic_category);
                if (semantic_category <= 3)
                {
                    ++semantic_counts[semantic_category];
                }
            }
        }

        publisher_->publish(std::move(msg));

        if (index_ == 0 || (index_ + 1) % 100 == 0)
        {
            const double elapsed = std::chrono::duration<double>(now - start_wall_).count();
            RCLCPP_INFO(this->get_logger(), "Published KITTI frame %zu/%zu, elapsed %.3f s, average %.3f Hz",
                        index_ + 1, scans_.size(), elapsed, static_cast<double>(index_ + 1) / std::max(elapsed, 1e-6));
            if (publish_gt_semantics_ && log_semantic_stats_)
            {
                RCLCPP_INFO(this->get_logger(),
                            "GT semantic categories frame %zu: unknown=%zu plane=%zu gaussian=%zu object=%zu",
                            start_index_ + index_, semantic_counts[0], semantic_counts[1],
                            semantic_counts[2], semantic_counts[3]);
            }
        }
        ++index_;
    }

    std::string sequence_path_;
    std::string topic_;
    std::string frame_id_;
    double rate_ = 10.0;
    bool loop_ = false;
    bool wait_for_subscriber_ = true;
    bool publish_point_time_ = false;
    bool publish_gt_semantics_ = false;
    bool log_semantic_stats_ = true;
    bool started_ = false;
    int qos_depth_ = 512;
    int start_index_ = 0;
    int end_index_ = -1;
    std::string label_dir_;
    std::string semantic_plane_labels_;
    std::unordered_set<uint32_t> plane_labels_;
    size_t index_ = 0;
    std::vector<std::string> scans_;
    std::vector<double> times_;
    std::vector<PointFieldMsg> fields_;
    rclcpp::Publisher<PointCloud2Msg>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::chrono::steady_clock::time_point start_wall_;
    std::chrono::steady_clock::time_point next_pub_time_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    try
    {
        rclcpp::spin(std::make_shared<KittiSequencePublisher>());
    }
    catch (const std::exception &exc)
    {
        RCLCPP_FATAL(rclcpp::get_logger("kitti_sequence_publisher_cpp"), "%s", exc.what());
        rclcpp::shutdown();
        return 1;
    }
    if (rclcpp::ok())
    {
        rclcpp::shutdown();
    }
    return 0;
}
