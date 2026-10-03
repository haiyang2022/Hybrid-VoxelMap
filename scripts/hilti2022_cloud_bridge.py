#!/usr/bin/env python3
"""Transform Hilti-Oxford 2022 Hesai clouds to the IMU frame."""
import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField


class HiltiCloudBridge(Node):
    def __init__(self):
        super().__init__('hilti2022_cloud_bridge')
        self.declare_parameter('input_topic', '/hesai/pandar')
        self.declare_parameter('output_topic', '/hybrid/points_imu')
        self.declare_parameter('counts_path', '')
        self.declare_parameter('blind', 0.5)
        self.declare_parameter('extrinsic_R', [0., -1., 0., -1., 0., 0., 0., 0., -1.])
        self.declare_parameter('extrinsic_T', [-0.001, -0.00855, 0.055])
        self.rotation = np.array(self.get_parameter('extrinsic_R').value).reshape(3, 3)
        self.translation = np.array(self.get_parameter('extrinsic_T').value)
        self.blind = float(self.get_parameter('blind').value)
        self.counts_path = self.get_parameter('counts_path').value
        self.clouds = self.published = self.points = self.removed = 0
        qos = QoSProfile(depth=512, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = self.create_publisher(PointCloud2, self.get_parameter('output_topic').value, qos)
        self.subscription = self.create_subscription(PointCloud2, self.get_parameter('input_topic').value, self.callback, qos)

    def callback(self, msg):
        self.clouds += 1
        fields = {f.name: f for f in msg.fields}
        if msg.is_bigendian or any(n not in fields for n in ['x', 'y', 'z', 'timestamp']):
            raise ValueError('Expected little-endian Hesai XYZ/timestamp fields')
        if any(fields[n].datatype != PointField.FLOAT32 for n in ['x', 'y', 'z']) or fields['timestamp'].datatype != PointField.FLOAT64:
            raise ValueError('Expected FLOAT32 XYZ and FLOAT64 timestamp')
        dtype = np.dtype({'names': ['x', 'y', 'z', 'timestamp'],
                         'formats': ['<f4', '<f4', '<f4', '<f8'],
                         'offsets': [fields[n].offset for n in ['x', 'y', 'z', 'timestamp']],
                         'itemsize': msg.point_step})
        points = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                            strides=(msg.row_step, msg.point_step)).reshape(-1)
        xyz = np.column_stack([points[n] for n in ['x', 'y', 'z']])
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(points['timestamp']) & (np.sum(xyz * xyz, axis=1) > self.blind ** 2)
        raw = np.ndarray((msg.height, msg.width, msg.point_step), dtype=np.uint8, buffer=msg.data,
                         strides=(msg.row_step, msg.point_step, 1)).reshape(-1, msg.point_step)[valid].copy()
        self.removed += int(len(valid) - len(raw))
        if len(raw) == 0:
            self.get_logger().warning('Skipping empty valid scan')
            return
        output = np.ndarray(len(raw), dtype=dtype, buffer=raw)
        transformed = xyz[valid] @ self.rotation.T + self.translation
        for axis, name in enumerate(['x', 'y', 'z']):
            output[name] = transformed[:, axis]
        first = float(output['timestamp'][0])
        seconds = int(first)
        carry, nanoseconds = divmod(int(round((first - seconds) * 1e9)), 1000000000)
        msg.header.stamp.sec, msg.header.stamp.nanosec = seconds + carry, nanoseconds
        msg.header.frame_id = 'alphasense_imu'
        msg.height, msg.width = 1, len(output)
        msg.row_step = len(output) * msg.point_step
        msg.data = raw.tobytes()
        msg.is_dense = True
        self.publisher.publish(msg)
        self.published += 1
        self.points += len(output)

    def close(self):
        if self.counts_path:
            Path(self.counts_path).write_text(json.dumps({'input_clouds': self.clouds,
                'published_clouds': self.published, 'published_points': self.points,
                'removed_points': self.removed}, indent=2) + '\n')


def main():
    rclpy.init()
    node = HiltiCloudBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
