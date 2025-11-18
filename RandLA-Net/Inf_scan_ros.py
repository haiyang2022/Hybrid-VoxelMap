#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import warnings
import pickle
import random
from sklearn.neighbors import KDTree

import rospy
from sensor_msgs.msg import PointCloud2
import ros_numpy
import hashlib

def pack_rgb(r, g, b):
    rgb_uint32 = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)
    return rgb_uint32.view(np.float32)

def compute_hash(np_array):
    m = hashlib.md5()
    m.update(np_array.tobytes())
    return m.hexdigest()

def save_ply(filename, xyz, colors):
    num_points = xyz.shape[0]
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(num_points):
            x, y, z = xyz[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

class SemanticSegmentationNode(object):
    def __init__(self, gpu, scene, mapsize=0.5):
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

        rospy.init_node('semantic_segmentation_node', anonymous=True)
                 
        self.scene = scene
        self.gpu = gpu
        self.mapsize = mapsize
                 
        if self.scene == 'indoor':
            self.checkpoint_path = 'output/checkpoint_Area_5.tar'
            from helper_tool import ConfigS3DIS, DataProcessing as DP
            self.cfg = ConfigS3DIS()
            self.cfg.name = 'S3DIS'
            self.colormap = {
                0: [0,255,255],  # ceiling
                1: [0,204,0],    # floor
                2: [0,102,204],  # wall
                3: [200,50,50],  # beam
                4: [255,204,0],  # column
                5: [204,0,204],  # window
                6: [0,128,128],  # door
                7: [128,0,128],  # table
                8: [128,128,0],  # chair
                9: [0,128,0],    # sofa
                10:[128,0,0],    # bookcase
                11:[0,0,128],    # board
                12:[128,128,128] # clutter
            }
            self.sub_grid_size = 0.04
        elif self.scene == 'outdoor':
            self.checkpoint_path = 'output/checkpoint_SemanticKITTI.tar'
            from helper_tool import ConfigSemanticKITTI, DataProcessing as DP
            self.cfg = ConfigSemanticKITTI()
            self.cfg.name = 'SemanticKITTI'
            self.colormap = {
                0: [245,150,100],  # car
                1: [245,230,100],  # bicycle
                2: [150,60,30],    # motorcycle
                3: [180,30,80],    # truck
                4: [255,0,0],      # other-vehicle
                5: [30,30,255],    # person
                6: [200,40,255],   # bicyclist
                7: [90,30,150],    # motorcyclist
                8: [255,255,255],  # road
                9: [255,255,0],    # parking
                10:[0,255,0],      # sidewalk
                11:[255,0,255],    # other-ground
                12:[0,0,255],      # building
                13:[255,100,100],  # fence
                14:[255,200,100],  # vegetation
                15:[150,150,150],  # trunk
                16:[200,200,200],  # terrain
                17:[50,50,50],     # pole
                18:[150,150,0]     # traffic-sign
            }
            self.sub_grid_size = self.cfg.sub_grid_size
        else:
            rospy.logerr("Invalid scene parameter. Should be 'indoor' or 'outdoor'.")
            rospy.signal_shutdown("Invalid scene parameter")
            return
                 
        self.num_vote = 5
                 
        if self.gpu >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.gpu}')
        else:
            self.device = torch.device("cpu")
                 
        rospy.loginfo("Using checkpoint: " + self.checkpoint_path)
                 
        from RandLANet import Network
        self.net = Network(self.cfg)
        self.net.to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)
        if os.path.isfile(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            rospy.loginfo("Model restored from " + self.checkpoint_path)
        else:
            rospy.logerr("Checkpoint not found!")
            rospy.signal_shutdown("No checkpoint")
        self.net.eval()
                 
        self.DP = DP
                 
        self.pub_voxel = rospy.Publisher('/segmented_cloud', PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber('/input_cloud', PointCloud2, self.callback, queue_size=1)
        rospy.loginfo("Semantic segmentation node started. Scene: " + self.scene)
        
        self.preheat()
        
    def preheat(self):
        rospy.loginfo("Preheating the network with dummy inference...")
        num_points = self.cfg.num_points if hasattr(self.cfg, 'num_points') else 4096
        dummy_xyz = np.random.rand(num_points, 3).astype(np.float32)
        dummy_colors = np.random.rand(num_points, 3).astype(np.float32)

        inputs = self.build_network_input_block(dummy_xyz, dummy_colors)

        for key in inputs:
            if isinstance(inputs[key], list):
                inputs[key] = [x.to(self.device) for x in inputs[key]]
            else:
                inputs[key] = inputs[key].to(self.device)

        with torch.no_grad():
            _ = self.net(inputs)
        rospy.loginfo("Preheat completed.")
                 
    def callback(self, msg):
        t0 = time.time()
        
        from importlib import reload
        import helper_tool
        self.DP = reload(helper_tool).DataProcessing
                 
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)

        if 'r' in pc_array.dtype.names and 'g' in pc_array.dtype.names and 'b' in pc_array.dtype.names:
            raw_xyz = np.vstack((pc_array['x'], pc_array['y'], pc_array['z'])).T.astype(np.float32)
            raw_colors = np.vstack((pc_array['r'], pc_array['g'], pc_array['b'])).T.astype(np.float32)
        elif 'rgb' in pc_array.dtype.names:
            raw_xyz = np.vstack((pc_array['x'], pc_array['y'], pc_array['z'])).T.astype(np.float32)
            s = pc_array['rgb'].view(np.uint32)
            raw_colors = np.empty((pc_array.shape[0], 3), dtype=np.float32)
            raw_colors[:,0] = ((s >> 16) & 0x0000ff).astype(np.float32)
            raw_colors[:,1] = ((s >> 8) & 0x0000ff).astype(np.float32)
            raw_colors[:,2] = (s & 0x0000ff).astype(np.float32)
            if raw_colors.max() > 1.0:
                raw_colors = raw_colors / 255.0
        else:
            raw_xyz = np.vstack((pc_array['x'], pc_array['y'], pc_array['z'])).T.astype(np.float32)
            raw_colors = np.ones((raw_xyz.shape[0], 3), dtype=np.float32) * 0.5
        t1 = time.time()
        rospy.loginfo("PointCloud2 conversion time: {:.1f} ms".format((t1-t0)*1000))
                 
        hash_xyz = compute_hash(raw_xyz)
        hash_colors = compute_hash(raw_colors)
        current_hash = hash_xyz + hash_colors
        if hasattr(self, 'last_msg_hash') and self.last_msg_hash == current_hash:
            rospy.loginfo("Same point cloud received, skipping segmentation.")
            return
        self.last_msg_hash = current_hash

        offset = np.mean(raw_xyz, axis=0)
        raw_xyz_centered = raw_xyz - offset

        if self.scene == 'indoor':
            result = self.DP.grid_sub_sampling(raw_xyz_centered, raw_colors, labels=None, grid_size=self.sub_grid_size)
            if isinstance(result, tuple) and len(result) == 3:
                sub_xyz, sub_colors, _ = result
            elif isinstance(result, tuple) and len(result) == 2:
                sub_xyz, sub_colors = result
            else:
                rospy.logerr("Unexpected return format from grid_sub_sampling")
                return

            restored_xyz = sub_xyz + offset

            points = sub_xyz
        else:
            result = self.DP.grid_sub_sampling(raw_xyz_centered, raw_colors, labels=None, grid_size=self.cfg.sub_grid_size)
            if isinstance(result, tuple) and (len(result) == 3 or len(result) == 2):
                if len(result) == 3:
                    sub_xyz, sub_colors, _ = result
                else:
                    sub_xyz, sub_colors = result
            else:
                rospy.logerr("Unexpected return format from grid_sub_sampling for outdoor")
                return

            restored_xyz = sub_xyz + offset
            points = sub_xyz
                 
        t2 = time.time()
        rospy.loginfo("Preprocessing (subsampling) time: {:.1f} ms".format((t2-t1)*1000))
                 

        N = points.shape[0]
        num_classes = self.cfg.num_classes
        accumulated_probs = np.zeros((N, num_classes), dtype=np.float32)
        vote_count = np.zeros((N,), dtype=np.int32)
                 

        if self.scene == 'outdoor':
            kd_tree = KDTree(points)
                 

        for vote in range(self.num_vote):
            if self.scene == 'indoor':

                if points.shape[0] >= self.cfg.num_points:
                    block_idx = np.random.choice(points.shape[0], self.cfg.num_points, replace=False)
                else:
                    block_idx = np.random.choice(points.shape[0], self.cfg.num_points, replace=True)
            else:

                center_index = np.random.randint(0, N)
                center_point = points[center_index].reshape(1, -1)
                num_available = kd_tree.data.shape[0]
                k_val = min(self.cfg.num_points, num_available)
                block_idx = kd_tree.query(center_point, k=k_val)[1][0]
                block_idx = self.DP.shuffle_idx(block_idx)
                 
            block_xyz = points[block_idx]
            block_colors = sub_colors[block_idx]
            inputs = self.build_network_input_block(block_xyz, block_colors)
            for key in inputs:
                if isinstance(inputs[key], list):
                    inputs[key] = [x.to(self.device) for x in inputs[key]]
                else:
                    inputs[key] = inputs[key].to(self.device)
            with torch.no_grad():
                out = self.net(inputs)
            if 'logits' in out:
                logits = out['logits']
            else:
                rospy.logerr("Model output missing logits")
                return
            logits = logits.transpose(1, 2)
            probs = F.softmax(logits, dim=2).cpu().numpy()[0]
            accumulated_probs[block_idx] += probs
            vote_count[block_idx] += 1
        
        if self.scene == 'outdoor':
            del kd_tree
        final_prob = accumulated_probs / np.maximum(vote_count[:, None], 1)
        final_pred = np.argmax(final_prob, axis=1)
                 
        unique_labels = np.unique(final_pred)
        rospy.loginfo("Unique predicted labels: " + str(unique_labels))
                 
        t_end = time.time()
        rospy.loginfo("Total processing time: {:.1f} ms".format((t_end-t0)*1000))
                 
        def normalize_label(label):
            if self.scene == 'indoor':
                if label in [0, 1, 2]:
                    return 1  # planar
                elif label in [3, 4, 5, 6, 7, 10, 11]:
                    return 2  # stable
                elif label in [8, 9, 12]:
                    return 3  # unstable
                else:
                    return label
            elif self.scene == 'outdoor':
                if label in [8, 10, 11, 12]:
                    return 1
                elif label in [9, 13, 14, 15, 16, 17, 18]:
                    return 2
                elif label in [0, 1, 2, 3, 4, 5, 6, 7]:
                    return 3
                else:
                    return label
            else:
                return label

        kdt = KDTree(restored_xyz)
        dists, nn_idx = kdt.query(raw_xyz, k=1)
        del kdt
        raw_pred = final_pred[nn_idx.flatten()]
 
        normalized_pred = np.array([normalize_label(int(lbl)) for lbl in raw_pred])
        new_color_map = {1: [0, 0, 255], 2: [255, 255, 0], 3: [255, 255, 0]}
        raw_pred_colors = np.array(
            [new_color_map.get(int(lbl), [255, 255, 255]) for lbl in normalized_pred],
            dtype=np.uint8
        )
        # orig_colors = np.array([self.colormap.get(int(lbl), [255, 255, 255]) for lbl in raw_pred], dtype=np.uint8)
        # ply_filename = "colored_cloud_orig_{}.ply".format(int(time.time()))
        # save_ply(ply_filename, raw_xyz, orig_colors)
        # rospy.loginfo("Saved original colored point cloud to " + ply_filename)

        voxel_size = self.mapsize  

        voxel_idx = np.floor(raw_xyz / voxel_size).astype(np.int32)

        voxel_dict = {}
        for i, idx in enumerate(voxel_idx):
            key = tuple(idx.tolist())
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(normalized_pred[i])
        

        voxel_centers = []
        voxel_labels = []
        for key, labels in voxel_dict.items():
            total = len(labels)
            if total < 10:
                continue

            count_label = {}
            for lbl in labels:
                count_label[lbl] = count_label.get(lbl, 0) + 1

            if count_label.get(3, 0) / total > 0.2:
                voxel_label = 3
            else:

                max_count = max(count_label.values())

                winners = [lbl for lbl, cnt in count_label.items() if cnt == max_count]
                if len(winners) > 1:
                    voxel_label = 2 
                else:
                    voxel_label = winners[0]

            center = (np.array(key, dtype=np.float32) + 0.5) * voxel_size
            voxel_centers.append(center)
            voxel_labels.append(voxel_label)
        
        if len(voxel_centers) > 0:
            voxel_centers = np.vstack(voxel_centers)
        else:
            voxel_centers = np.empty((0, 3), dtype=np.float32)
        voxel_labels = np.array(voxel_labels, dtype=np.float32)
        

        voxel_colors = np.array([new_color_map.get(int(lbl), [255, 255, 255]) for lbl in voxel_labels], dtype=np.uint8)

        dtype_voxel = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('rgb', 'f4')])
        voxel_array = np.empty(voxel_centers.shape[0], dtype=dtype_voxel)
        voxel_array['x'] = voxel_centers[:, 0]
        voxel_array['y'] = voxel_centers[:, 1]
        voxel_array['z'] = voxel_centers[:, 2]
        packed_voxel_rgb = pack_rgb(voxel_colors[:,0], voxel_colors[:,1], voxel_colors[:,2])
        voxel_array['rgb'] = packed_voxel_rgb
        
        voxel_msg = ros_numpy.point_cloud2.array_to_pointcloud2(voxel_array, frame_id="camera_init")
        voxel_msg.header.seq = msg.header.seq  

        self.pub_voxel.publish(voxel_msg)

        

        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
    def build_network_input_block(self, pc_xyz, pc_colors):
        
        if self.scene == 'outdoor':

            pc = pc_xyz  # [block_size, 3]
            pc = np.expand_dims(pc, axis=0)  # [1, block_size, 3]
            batch_xyz = pc

            batch_features = np.empty((1, pc.shape[1], 0), dtype=np.float32)
        else:

            pc = np.concatenate([pc_xyz, pc_colors], axis=-1)  # [block_size, 6]
            pc = np.expand_dims(pc, axis=0)  # [1, block_size, 6]
            batch_xyz = pc[:, :, :3]
            batch_features = pc[:, :, 3:6]
                 
        batch_label = np.zeros((1, pc.shape[1]), dtype=np.int32)
        batch_pc_idx = np.expand_dims(np.arange(pc.shape[1], dtype=np.int32), axis=0)
        batch_cloud_idx = np.array([0], dtype=np.int32)
        flat_inputs = self.local_tf_map_block(batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx)
        nl = self.cfg.num_layers
        inputs = {}
        inputs['xyz'] = [torch.from_numpy(tmp).float() for tmp in flat_inputs[:nl]]
        inputs['neigh_idx'] = [torch.from_numpy(tmp).long() for tmp in flat_inputs[nl:2*nl]]
        inputs['sub_idx'] = [torch.from_numpy(tmp).long() for tmp in flat_inputs[2*nl:3*nl]]
        inputs['interp_idx'] = [torch.from_numpy(tmp).long() for tmp in flat_inputs[3*nl:4*nl]]
        inputs['features'] = torch.from_numpy(flat_inputs[4*nl]).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4*nl+1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4*nl+2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4*nl+3]).long()
        return inputs
     
    def local_tf_map_block(self, batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx):

        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        temp_xyz = batch_xyz.copy()
        for i in range(self.cfg.num_layers):
            neighbour_idx = self.DP.knn_search(temp_xyz, temp_xyz, self.cfg.k_n)
            sub_points = temp_xyz[:, :temp_xyz.shape[1] // self.cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :temp_xyz.shape[1] // self.cfg.sub_sampling_ratio[i], :]
            up_i = self.DP.knn_search(sub_points, temp_xyz, 1)
            input_points.append(temp_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            temp_xyz = sub_points
        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_label, batch_pc_idx, batch_cloud_idx]
        return input_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('--scene', type=str, default='indoor', help="Scene type: 'indoor' or 'outdoor' (default: indoor)")

    parser.add_argument('--mapsize', type=float, default=0.5, help="Voxel grid size for voxelization (default: 0.5)")
    args, unknown = parser.parse_known_args()
     
    node = SemanticSegmentationNode(gpu=args.gpu, scene=args.scene, mapsize=args.mapsize)
    rospy.spin()
