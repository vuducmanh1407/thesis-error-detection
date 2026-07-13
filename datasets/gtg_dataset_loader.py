import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import math
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets.loader_graph import GraphLoader
from torch_geometric.data import Data, DataLoader
import jsonlines
import json

max_dist = math.sqrt(2)

def get_data_dict(v_feature_dir, label_dir, bboxes_dir, bboxes_feats_dir, seg_feat_dir,
                  video_list, action2idx, actiontype2idx, addition_name="Error_Addition", suffix=""):
    
        
    data_dict = {k:{
        'v_feature': None,
        'label_seq': None,
        'type_label_seq': None,
        'obj_bboxes': None,
        'graph_data': None,
        } for k in video_list
    }
    
    print(f'Loading Dataset ...')
    
    for video in tqdm(video_list):
        v_feature_file = os.path.join(v_feature_dir, '{}.npy'.format(video+suffix))
        
        event_file = os.path.join(label_dir, '{}.txt'.format(video))

        with open(os.path.join(event_file), 'r') as fp:
            event = fp.readlines()
        
        frame_num = len(event)
                
        label_seq = np.zeros((frame_num,))
        type_label_seq = np.zeros((frame_num,))
        for i in range(frame_num):
            tokens = event[i].split('|')
            if len(tokens) == 2:
                action, action_type = tokens
            elif len(tokens) == 3:
                action, action_type, error_des = tokens
            
            action_type = action_type.strip("\n")
            
            if action_type == addition_name:
                label_seq[i] = -1
            else:
                if action in action2idx:
                    label_seq[i] = action2idx[action]
                else:
                    print(video, action_type, action, "is not in dict, therefore we assign it to %s" % addition_name)
                    label_seq[i] = -1
                    action_type = addition_name

            type_label_seq[i] = actiontype2idx[action_type]
        

        v_feature = np.load(v_feature_file)
        if v_feature.shape[0] != label_seq.shape[0]:
            print(v_feature_file)
            print(label_seq.shape)
            print(v_feature.shape)
        assert(v_feature.shape[0] == label_seq.shape[0])
        l = v_feature.shape[0]

        # # open bbox files
        # with open(os.path.join(bboxes_dir, f'{video+suffix}.pkl'), 'rb') as f:
        #     bboxes_list = pickle.load(f)
        
        # with open(os.path.join(bboxes_feats_dir, f'{video+suffix}.pkl'), 'rb') as f:
        #     bbox_feats_list = pickle.load(f)

        with open(os.path.join(seg_feat_dir, f'{video+suffix}.pkl'), 'rb') as f:
            seg_feats_list = pickle.load(f)
        
        # if len(bboxes_list) != l or len(bbox_feats_list) != l:
        #     print(v_feature_file)
        #     print(len(bboxes_list))
        #     print(len(bbox_feats_list))
        #     print(l)
        # assert len(bboxes_list) == len(bbox_feats_list) == l
        # input_features_list = list()
        # all_edges = list()
        # weight_list = list()
        # data_list = list()
        # for i in range(len(bboxes_list)):
        #     feat = preprocess_bbox_features(bbox_feats_list[i], bboxes_list[i], 1536)
        #     input_features_list.append(torch.from_numpy(feat))

        #     # iou weight
        #     iou_weight = torch.tensor(bboxes_to_iou_weights(bboxes_list[i]))
        #     l2_weight = torch.tensor(bboxes_to_l2_weight(bboxes_list[i], r=0.5))
        #     full_weight = torch.tensor(bboxes_to_full_weight(bboxes_list[i]))
        #     # edge_weight = iou_weight + l2_weight
        #     # edge_weight = l2_weight
        #     edge_weight = full_weight

        #     # edge_weight = 1.0 - torch.eye(len(bboxes_list[i]["hands"]) + len(bboxes_list[i]["objects"]))

        #     # create edge weight and edge matrix
        #     num_nodes = feat.shape[0]
        #     source_nodes = torch.arange(num_nodes)
        #     target_nodes = torch.arange(num_nodes)
        #     weights = torch.tensor(edge_weight.flatten(), dtype=torch.float32)
        #     # This gives pairs like (0,0), (0,1), ..., (1,0), (1,1), ...
        #     all_edges = torch.cartesian_prod(source_nodes, target_nodes).t().contiguous()
        #     data_list.append(Data(x=torch.tensor(feat, dtype=torch.float32), edge_index=all_edges, edge_weights=weights))
        
        # # convert to graph data
        # [Data(x=torch.tensor(input_features_list[i], dtype=torch.float32), edge_index=all_edges[i], edge_weights=weight_list[i]) for i in range(l)]

        if len(seg_feats_list) != l:
            print(v_feature_file)
            print(len(seg_feats_list))
            print(l)
        assert len(seg_feats_list) == l

        data_list = list()
        for i in range(len(seg_feats_list)):
            feats = torch.tensor(seg_feats_list[i], dtype=torch.float32)
            
            # create edge matrix
            num_nodes = feats.shape[0]
            source_nodes = torch.arange(num_nodes)
            target_nodes = torch.arange(num_nodes)
            # This gives pairs like (0,0), (0,1), ..., (1,0), (1,1), ...
            all_edges = torch.cartesian_prod(source_nodes, target_nodes).t().contiguous()

            # create weight matrix
            edge_weight = 1 - torch.eye(num_nodes, dtype=torch.float32)

            # # distance weight
            # r = 0.5
            # coords = feats[:, -2:]
            # edge_weight = torch.zeros(num_nodes, num_nodes, dtype=torch.float32)
            # for i in range(num_nodes):
            #     for j in range(num_nodes):
            #         if i == j:
            #             edge_weight[i, j] = 0.
            #         else:
            #             distance = torch.norm(coords[i] - coords[j])
            #             edge_weight[i, j] = math.exp(-(distance * distance) / (2 * r * r))


            weights = torch.tensor(edge_weight.flatten(), dtype=torch.float32)
            data_list.append(Data(x=feats, edge_index=all_edges, edge_weights=weights))

        data_dict[video]['v_feature'] = torch.from_numpy(v_feature).float()
        data_dict[video]['label_seq'] = torch.from_numpy(label_seq).long()
        data_dict[video]['type_label_seq'] = torch.from_numpy(type_label_seq).long()
        # data_dict[video]['obj_bboxes'] = bboxes_list
        data_dict[video]['graph_data'] = data_list

    return data_dict

def calculate_bbox_iou(box1, box2):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Boxes are expected in [x1, y1, x2, y2] format.
    """
    # Get the coordinates of the intersection rectangle
    x_intersect_min = max(box1[0], box2[0])
    y_intersect_min = max(box1[1], box2[1])
    x_intersect_max = min(box1[2], box2[2])
    y_intersect_max = min(box1[3], box2[3])

    # Calculate the area of intersection
    # Check if there is any overlap
    if x_intersect_max < x_intersect_min or y_intersect_max < y_intersect_min:
        intersection_area = 0
    else:
        intersection_area = (x_intersect_max - x_intersect_min) * (y_intersect_max - y_intersect_min)

    # Calculate the area of the individual boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of union (Area1 + Area2 - Intersection Area)
    union_area = area_box1 + area_box2 - intersection_area

    # Avoid division by zero if both boxes have zero area
    if union_area == 0:
        return 0.0

    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def calculate_bbox_distance(box1, box2, r=0.5):
    """
    Calculates the Euclidean distance between the centers of two bounding boxes.

    Args:
        box1 (list): Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
        box2 (list): Coordinates of the second bounding box [x_min, y_min, x_max, y_max].

    Returns:
        float: The Euclidean distance between the centers.
    """
    
    # 1. Calculate the center coordinates for the first box
    # Center X = (x_min + x_max) / 2
    # Center Y = (y_min + y_max) / 2
    center_x1 = (box1[0] + box1[2]) / 2
    center_y1 = (box1[1] + box1[3]) / 2

    # 2. Calculate the center coordinates for the second box
    center_x2 = (box2[0] + box2[2]) / 2
    center_y2 = (box2[1] + box2[3]) / 2

    # 3. Use the Euclidean distance formula (Pythagorean theorem)
    # distance = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    squared_distance = ((center_x2 - center_x1) ** 2) + ((center_y2 - center_y1) ** 2)
    distance = math.sqrt(((center_x2 - center_x1) ** 2) + ((center_y2 - center_y1) ** 2))
    
    return math.exp(-squared_distance / (2 * r * r))  
    # return max_dist - distance # distance -> weight

def bboxes_to_l2_weight(bboxes_dict, r=0.5):
    hand_bboxes = bboxes_dict["hands"]
    obj_bboxes = bboxes_dict["objects"]

    bboxes = hand_bboxes + obj_bboxes

    num_hands = len(hand_bboxes)
    num_objs = len(obj_bboxes)
    l = len(bboxes)

    total_iou = np.zeros((l, l), dtype=float)
    for i in range(l):
        for j in range(l):
            if i == j:
                total_iou[i, j] = 0.
            else:
                total_iou[i, j] = calculate_bbox_distance(bboxes[i], bboxes[j], r=r)
    
    return total_iou

def bboxes_to_full_weight(bboxes_dict):
    hand_bboxes = bboxes_dict["hands"]
    obj_bboxes = bboxes_dict["objects"]

    bboxes = hand_bboxes + obj_bboxes

    num_hands = len(hand_bboxes)
    num_objs = len(obj_bboxes)
    l = len(bboxes)

    total_iou = np.zeros((l, l), dtype=float)
    for i in range(l):
        for j in range(l):
            if i == j:
                total_iou[i, j] = 0.0
            else:
                total_iou[i, j] = 1.0
    
    return total_iou


def bboxes_to_iou_weights(bboxes_dict):
    hand_bboxes = bboxes_dict["hands"]
    if len(hand_bboxes) == 0:
        hand_bboxes.append([0., 0., 0., 0.])
    obj_bboxes = bboxes_dict["objects"]
    if len(obj_bboxes) == 0:
        obj_bboxes.append([0., 0., 0., 0.])

    bboxes = hand_bboxes + obj_bboxes

    num_hands = len(hand_bboxes)
    num_objs = len(obj_bboxes)
    l = len(bboxes)

    total_iou = np.zeros((l, l), dtype=float)
    for i in range(l):
        for j in range(l):
            if i == j:
                total_iou[i, j] = 0.
            else:
                total_iou[i, j] = calculate_bbox_iou(bboxes[i], bboxes[j])
    
    return total_iou

# def process_graph_data(bboxes_list, bboxes_feats_list, dim):
#     assert len(bboxes_list) == len(bboxes_feats_list)
#     l = len(bboxes_list)

#     data_list = list()
#     features_list = list()

#     for i in range(l):
#         bboxes_feats_dict = bboxes_feats_list[i]
#         bboxes_dict = bboxes_list[i]
#         hand_bboxes_feat = bboxes_feats_dict["hands"]
#         if hand_bboxes_feat.shape[0] == 0:
#             hand_bboxes_feat = np.zeros((1, dim), dtype=float)
#         obj_bboxes_feat = bboxes_feats_dict["objects"]
#         if obj_bboxes_feat.shape[0] == 0:
#             obj_bboxes_feat = np.zeros((1, dim), dtype=float)

#         hand_bboxes = bboxes_dict["hands"]
#         if len(hand_bboxes) == 0:
#             hand_bboxes.append([0., 0., 0., 0.])
#         obj_bboxes = bboxes_dict["objects"]
#         if len(obj_bboxes) == 0:
#             obj_bboxes.append([0., 0., 0., 0.])
#         hand_bboxes = np.array(hand_bboxes)
#         obj_bboxes = np.array(obj_bboxes)

#         num_hands = len(hand_bboxes)
#         num_objs = len(obj_bboxes)

#         one_hot_hands = np.zeros((num_hands, 2), dtype=float)
#         one_hot_hands[:, 0] = 1.
#         one_hot_objs = np.zeros((num_objs, 2), dtype=float)
#         one_hot_objs[:, 1] = 1.

#         hand_features = np.concatenate([hand_bboxes_feat, hand_bboxes, one_hot_hands], axis=1)
#         obj_features = np.concatenate([obj_bboxes_feat, obj_bboxes, one_hot_objs], axis=1)

#         features_list.append(torch.tensor(np.concatenate([hand_features, obj_features], axis=0)))

#     return features_list

def preprocess_bbox_features(bboxes_feats_list, bboxes_dict, dim):
    hand_bboxes_feat = bboxes_feats_list["hands"]
    if hand_bboxes_feat.shape[0] == 0:
        hand_bboxes_feat = np.zeros((1, dim), dtype=float)
    obj_bboxes_feat = bboxes_feats_list["objects"]
    if obj_bboxes_feat.shape[0] == 0:
        obj_bboxes_feat = np.zeros((1, dim), dtype=float)

    hand_bboxes = bboxes_dict["hands"]
    if len(hand_bboxes) == 0:
        hand_bboxes.append([0., 0., 0., 0.])
    obj_bboxes = bboxes_dict["objects"]
    if len(obj_bboxes) == 0:
        obj_bboxes.append([0., 0., 0., 0.])
    hand_bboxes = np.array(hand_bboxes)
    obj_bboxes = np.array(obj_bboxes)

    num_hands = len(hand_bboxes)
    num_objs = len(obj_bboxes)

    one_hot_hands = np.zeros((num_hands, 2), dtype=float)
    one_hot_hands[:, 0] = 1.
    one_hot_objs = np.zeros((num_objs, 2), dtype=float)
    one_hot_objs[:, 1] = 1.

    hand_features = np.concatenate([hand_bboxes_feat, hand_bboxes, one_hot_hands], axis=1)
    obj_features = np.concatenate([obj_bboxes_feat, obj_bboxes, one_hot_objs], axis=1)

    return np.concatenate([hand_features, obj_features], axis=0)

class VideoDataset(Dataset):
    def __init__(self, root_data_dir, data_dict, class_num, mode, naming=None, dataset_name=None):
        super(VideoDataset, self).__init__()
        
        assert(mode in ['train', 'test'])
        
        self.data_dict = data_dict
        self.class_num = class_num
        self.mode = mode
        self.video_list = [i for i in self.data_dict.keys()]

        self.G = GraphLoader(naming, dataset_name, self.class_num)

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):

        video = self.video_list[idx]
        v_feature = self.data_dict[video]['v_feature']
        label = self.data_dict[video]['label_seq']
        type_label = self.data_dict[video]['type_label_seq']
        graph_data = self.data_dict[video]['graph_data']
        v_feature = v_feature.T   # F x T

        boundaries = torch.cat([torch.tensor([0]), torch.logical_or((label[:-1] != label[1:]), (type_label[:1] !=type_label[1:]))]).long()

        seg_label = torch.cat([label[torch.argwhere(boundaries == 1).squeeze(1) - 1], label[-1:]]).long()

        # mark the last segment's boundary
        boundaries[-1] = 1

        # print(len(graph_data))

        return v_feature, graph_data, label, seg_label, type_label, boundaries, video
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))