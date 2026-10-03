#!/usr/bin/env python3
"""Resume bag playback after publisher and subscriber discovery."""
import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rosbag2_interfaces.srv import Resume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('topics', nargs='+')
    args = parser.parse_args()
    rclpy.init()
    node = Node('hilti2022_playback_ready')
    client = node.create_client(Resume, '/rosbag2_player/resume')
    start, stable = time.monotonic(), None
    try:
        while time.monotonic() - start < 40:
            rclpy.spin_once(node, timeout_sec=0.1)
            seen = {t: {'publishers': len(node.get_publishers_info_by_topic(t)),
                        'subscribers': len(node.get_subscriptions_info_by_topic(t))} for t in args.topics}
            ready = all(v['publishers'] and v['subscribers'] for v in seen.values()) and client.service_is_ready()
            if ready:
                if stable is None:
                    stable = time.monotonic()
                if time.monotonic() - stable > 5:
                    break
            else:
                stable = None
        else:
            raise RuntimeError('Playback endpoints not ready')
        future = client.call_async(Resume.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=10)
        if not future.done() or future.exception():
            raise RuntimeError('Could not resume playback')
        args.report.write_text(json.dumps({'endpoints': seen, 'discovery_wait_s': time.monotonic() - start}, indent=2) + '\n')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
