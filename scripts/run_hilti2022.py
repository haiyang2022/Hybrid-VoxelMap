#!/usr/bin/env python3
"""Run Hybrid-VoxelMap on an already converted Hilti-Oxford 2022 ROS 2 bag."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from ament_index_python.packages import get_package_prefix, get_package_share_directory
import numpy as np
import yaml


def estimate_imu_noise(bag, storage_id, topic):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage_id),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    first, samples = None, []
    while reader.has_next():
        _, data, stamp = reader.read_next()
        if first is None:
            first = stamp
        if stamp >= first + 2000000000:
            break
        msg = deserialize_message(data, Imu)
        samples.append([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    values = np.asarray(samples)
    if len(values) < 400 or not np.isfinite(values).all():
        raise ValueError('Insufficient finite IMU samples in the first two seconds')
    if np.quantile(np.linalg.norm(values[:, :3], axis=1), .99) > .03 or np.max(np.std(values[:, 3:], axis=0)) > .1:
        raise ValueError('The first two seconds must be stationary for IMU noise calibration')
    variance = np.var(values, axis=0, ddof=1)
    return {'samples': len(values), 'gyro_axis_variance': variance[:3].tolist(),
            'acc_axis_variance': variance[3:].tolist(),
            'calibrated_gyr_cov_scale': float(max(variance[:3])),
            'calibrated_acc_cov_scale': float(max(variance[3:]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bag', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    share = Path(get_package_share_directory('hybrid_voxel_map'))
    executables = Path(get_package_prefix('hybrid_voxel_map')) / 'lib/hybrid_voxel_map'
    bag = args.bag.resolve()
    metadata = yaml.safe_load((bag / 'metadata.yaml').read_text())['rosbag2_bagfile_information']
    params = yaml.safe_load((args.config or share / 'config/hilti2022.yaml').read_text())
    mapper = params['voxel_mapping_odom']['ros__parameters']
    bridge_params = params['hilti2022_cloud_bridge']['ros__parameters']
    raw_topic, imu_topic = bridge_params['input_topic'], mapper['common']['imu_topic']
    expected = {t['topic_metadata']['name']: t['message_count'] for t in metadata['topics_with_message_count']}
    if raw_topic not in expected or imu_topic not in expected:
        raise ValueError('The bag must contain the configured Hesai and IMU topics')
    out = (args.output or Path('hilti2022_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    out.mkdir(parents=True, exist_ok=False)
    noise = mapper['noise_model']
    if noise.get('use_calibrated_imu_noise', True):
        calibration = estimate_imu_noise(bag, metadata['storage_identifier'], imu_topic)
        noise.update({k: calibration[k] for k in ['calibrated_gyr_cov_scale', 'calibrated_acc_cov_scale']})
        (out / 'imu_noise_calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    mapper['Result'].update(tum_result_path=str(out / 'trajectory_imu.tum'),
                            efficiency_path=str(out / 'efficiency.txt'), write_kitti_log=False)
    mapper['mapping'].update(debug_csv_enable=True, debug_csv_dir=str(out), debug_csv_prefix='mapping')
    bridge_params['counts_path'] = str(out / 'bridge_counts.json')
    mapper['use_sim_time'] = True
    config = out / 'config.yaml'
    config.write_text(yaml.safe_dump(params, sort_keys=False))
    qos = {t: {'reliability': 'reliable', 'durability': 'volatile', 'history': 'keep_last',
               'depth': 4096 if t == imu_topic else 512} for t in [raw_topic, imu_topic]}
    qos_file = out / 'qos.yaml'
    qos_file.write_text(yaml.safe_dump(qos))
    env = os.environ.copy()
    env.update(ROS_DOMAIN_ID=str(170 + os.getpid() % 40), ROS_LOCALHOST_ONLY='1')
    env.setdefault('OMP_NUM_THREADS', '8')
    children, logs = [], []
    summary = {'bag': str(bag), 'rate': 1.0, 'ros_domain_id': int(env['ROS_DOMAIN_ID']), 'status': 'running'}

    def launch(command, name):
        handle = (out / name).open('w')
        logs.append(handle)
        child = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child

    def stop(child):
        if child.poll() is not None:
            return
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)

    try:
        bridge = launch([sys.executable, str(executables / 'hilti2022_cloud_bridge.py'),
                         '--ros-args', '--params-file', str(config)], 'bridge.log')
        node = launch([str(executables / 'voxel_mapping_odom'), '--ros-args', '--params-file', str(config)], 'mapping.log')
        time.sleep(3)
        if any(c.poll() is not None for c in [bridge, node]):
            raise RuntimeError('A node exited before playback')
        effective = subprocess.check_output(['ros2', 'param', 'dump', '/voxel_mapping_odom', '--no-daemon', '--timeout', '10'],
                                            env=env, text=True, timeout=30)
        (out / 'effective_params.yaml').write_text(effective)
        player = launch(['ros2', 'bag', 'play', str(bag), '--rate', '1.0', '--clock', '100',
                         '--start-paused', '--wait-for-all-acked', '10000', '--read-ahead-queue-size', '10000',
                         '--qos-profile-overrides-path', str(qos_file), '--disable-keyboard-controls'], 'playback.log')
        subprocess.run([sys.executable, str(executables / 'hilti2022_playback_ready.py'),
                        str(out / 'playback_readiness.json'), raw_topic, imu_topic, bridge_params['output_topic']],
                       env=env, check=True, timeout=55)
        while player.poll() is None:
            if any(c.poll() is not None for c in [bridge, node]):
                raise RuntimeError('A node exited during playback')
            time.sleep(1)
        if player.returncode:
            raise RuntimeError('Bag playback failed')
        previous, stable, deadline = None, 0, time.monotonic() + 180
        while time.monotonic() < deadline:
            time.sleep(2)
            frames = list(out.glob('mapping_*_frame.csv'))
            size = frames[0].stat().st_size if frames else 0
            stable = stable + 1 if size == previous else 0
            previous = size
            if stable >= 5:
                break
        else:
            raise RuntimeError('Queued input did not finish processing')
        summary['status'] = 'completed'
    except BaseException as exc:
        summary.update(status='failed', error=str(exc))
        raise
    finally:
        for child in reversed(children):
            try:
                stop(child)
            except (OSError, subprocess.TimeoutExpired) as exc:
                summary.update(status='failed', error=str(exc))
        for handle in logs:
            handle.close()
        summary['child_exit_codes'] = [c.returncode for c in children]
        match = re.search(r'INPUT_COUNTS lidar=(\d+) imu=(\d+)', (out / 'mapping.log').read_text())
        if match:
            summary['received_sensor_messages'] = {'lidar': int(match[1]), 'imu': int(match[2])}
            summary['all_sensor_messages_received'] = int(match[1]) == expected[raw_topic] and int(match[2]) == expected[imu_topic]
        else:
            summary['all_sensor_messages_received'] = False
        if summary['status'] == 'completed' and (not summary['all_sensor_messages_received'] or any(summary['child_exit_codes'])):
            summary.update(status='failed', error='Input coverage or shutdown check failed')
        (out / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    if summary['status'] != 'completed':
        raise RuntimeError(summary['error'])
    print(out)


if __name__ == '__main__':
    main()
