import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from sklearn.cluster import KMeans

from .utils import detect_boundaries, boundaries_to_idx, data_to_device
from .abd import ABD
from .gcn_bb import GCNWithWeights
# from torch_geometric.nn import GCNConv
# from torch_geometric.nn import global_mean_pool # For graph-level tasks
from torch_geometric.data import DataLoader

tol = 1e-8
class ASDiffusionBackbone(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 real_num_classes,
                 num_types,
                 addition_idx,
                 device,
                 loss_weights,
                 bg_w=1.0,
                 window_size=1,
                 temperature=1.0,
                 margin=torch.arccos(torch.zeros(1)).item() * 2/18, # 10 degrees
                #  margin=0.0, # 10 degrees
                #  threshold=torch.cos(torch.tensor(torch.pi / 18)),
                 threshold=1.0,
                 ema_weight=0.99,
                 smooth_feature=True,
                 mode="normal",
                 feature_mode="seg",
                 fusion_mode="early",
                 update_prototypes=True,
                 use_2_towers=False
                 ):
        super(ASDiffusionBackbone, self).__init__()
        

        self.device = device
        self.addition_idx = addition_idx
        self.num_classes = num_classes
        self.num_types = num_types #4 # normal, modification, slip, correction
        self.real_num_classes = real_num_classes
        self.window_size = window_size
        self.cls_w = 0.0
        self.input_dim = input_dim
        self.mode = mode
        assert mode in ["gtg", "normal"]
        assert feature_mode in ["seg", "frame"]
        self.feature_mode = feature_mode

        self.rescale_factor = 10
        action_weights = []
        for i in range(num_classes):
            if i == 0 or i >= real_num_classes:
                action_weights.append(bg_w)
            else:
                action_weights.append(1.0)

        self.cosine_similarity = nn.CosineSimilarity()
        self.mse = nn.MSELoss(reduction='none')
        self.temperature = temperature
        self.num_f_maps = 128
        kernel_size = 5
        dropout_rate = 0.1

        self.num_stages = 1
        self.num_layers = 8

        self.out_dim = 512
        self.prototype_dim = self.out_dim

        self.final_layer = nn.Conv1d(self.num_f_maps, self.out_dim, 1, 1, 0)
        
        self.boundary_head = nn.Conv1d(self.out_dim, 1, 1)

        self.loss_weights = loss_weights
        self.boundary_loss = torch.nn.BCEWithLogitsLoss()

        self.var_infonce = Var_InfoNCE(temperature, margin)
        self.smooth_feature = smooth_feature
        
        self.register_buffer('threshold', torch.tensor(threshold))
        self.register_buffer('score_threshold', torch.tensor(threshold))

        self.entropy_weight = 1.0
        self.cosine_weight = 0.0

        self.update_prototypes = update_prototypes
        self.num_clusters = 1

        self.abd = ABD(self.window_size, smooth_feature=smooth_feature, to_merge_boundaries=True)

        self.ema_weight = ema_weight
        self.beta = Beta(torch.tensor([10.0]), torch.tensor([10.0]))
        self.topk = 3

        self.num_gcn_layers = 8
        assert fusion_mode in ["early", "late", "graph_only", "none"]
        # for graph nn
        if fusion_mode != "none":
            self.gcn = GCNWithWeights(770, self.num_f_maps, input_dim, num_layers=self.num_gcn_layers)
        
        self.fusion_mode = fusion_mode
        self.use_2_towers = fusion_mode == "late" and use_2_towers == True

        if self.fusion_mode == "early":
            self.conv_in = nn.Conv1d(2 * input_dim, self.num_f_maps, 1)
            self.module = nn.Sequential(*[MixedConvAttModuleV2(self.num_layers, self.num_f_maps, 2 * input_dim, self.num_f_maps, kernel_size, dropout_rate) for _ in range(self.num_stages)])
        else:
            self.conv_in = nn.Conv1d(input_dim, self.num_f_maps, 1)
            self.module = nn.Sequential(*[MixedConvAttModuleV2(self.num_layers, self.num_f_maps, input_dim, self.num_f_maps, kernel_size, dropout_rate) for _ in range(self.num_stages)])
            
        if self.fusion_mode != "late":
            self.action_head = nn.Conv1d(self.out_dim, num_classes, 1)
            self.real_action_head = nn.Conv1d(self.out_dim, real_num_classes, 1)
        else:
            self.action_head = nn.Conv1d(2 * self.out_dim, num_classes, 1)
            self.real_action_head = nn.Conv1d(2 * self.out_dim, real_num_classes, 1)

        if self.use_2_towers == True:
            self.gfeat_in = nn.Conv1d(input_dim, self.num_f_maps, 1)
            gfeat_tower = [MixedConvAttModuleV2(self.num_layers, self.num_f_maps, input_dim, self.num_f_maps, kernel_size, dropout_rate) for _ in range(self.num_stages)]
            self.gfeat_tower = nn.Sequential(*gfeat_tower)
            self.gfeat_out = nn.Conv1d(self.num_f_maps, self.out_dim, 1, 1, 0)
        else:
            self.gfeat_in = nn.Conv1d(input_dim, self.out_dim, 1)

        out_dim = self.out_dim if self.fusion_mode != "late" else 2 * self.out_dim
        self.out_dim = out_dim
        if update_prototypes == True:
            if self.mode == "normal":
                self.action_ce = nn.CrossEntropyLoss()
                # self.register_buffer('prototypes', F.normalize(torch.randn(real_num_classes, self.out_dim), p=2, dim=1))
                self.register_buffer('prototypes', torch.randn(real_num_classes * self.num_clusters, out_dim))
            else:
                self.action_ce = nn.CrossEntropyLoss(weight=torch.tensor(action_weights))
                # self.register_buffer('prototypes', F.normalize(torch.randn(num_classes, self.out_dim), p=2, dim=1))
                self.register_buffer('prototypes', torch.randn(num_classes * self.num_clusters, out_dim))
        else:
            self.action_ce = nn.CrossEntropyLoss()
            prototypes = self.initialize_mcs_prototypes()
            self.register_buffer('prototypes', prototypes)

    def forward(self, in_feat, graph_data):
        if self.fusion_mode != "none":
            loader = DataLoader(graph_data, batch_size=len(graph_data), shuffle=False)

            gcn_feat = list()
            for batch in loader:
                x = batch.x
                edge_index = batch.edge_index
                edge_weight = batch.edge_weights
                batch = batch.batch
                out = self.gcn(x, edge_index, edge_weight, batch)
                gcn_feat.append(out)

            gcn_feat = torch.cat(gcn_feat, dim=0)       

            if self.fusion_mode == "early":
                in_feat = torch.cat([in_feat, gcn_feat.unsqueeze(0)], dim=-1)
            elif self.fusion_mode == "graph_only":
                in_feat = gcn_feat.unsqueeze(0)

        in_feat = in_feat.permute(0, 2, 1)
        x_ = self.conv_in(in_feat)
        for i in range(self.num_stages):
            x_ = self.module[i](x_, in_feat)

        features = self.final_layer(x_) # B, D, L

        if self.use_2_towers is True:
            gcn_feat = gcn_feat.unsqueeze(0).permute(0, 2, 1)
            gcn_feat_ = self.gfeat_in(gcn_feat)
            for i in range(self.num_stages):
                gcn_feat_ = self.gfeat_tower[i](gcn_feat_, gcn_feat)
            gcn_feat_ = self.gfeat_out(gcn_feat_)
            features = torch.cat([features, gcn_feat_], dim=1)
        elif self.fusion_mode == "late":
            gcn_feat = gcn_feat.unsqueeze(0).permute(0, 2, 1)
            gcn_feat_ = self.gfeat_in(gcn_feat)
            features = torch.cat([features, gcn_feat_], dim=1)           

        action_logits = self.action_head(features)
        real_action_logits = self.real_action_head(features)
        # action_boundaries_logits = self.boundary_head(features)
        return real_action_logits, action_logits, None, features

    def segment_loss(self, samples):
        seg_loss = 0.0
        cl_loss = 0.0
        for sample in samples:
            features = sample["frame_features"]
            boundary_gt = sample["boundary_labels"]
            boundary_gt[-1] = 1 # last idx is also a boundary
            s_ids, e_ids = boundaries_to_idx(boundary_gt)
            
            agg_features = torch.cat([torch.mean(features[:, s_ids[i]:e_ids[i]], dim=-1, keepdim=True) for i in range(len(s_ids))], dim=-1) # D, L'
            if self.mode == "normal":
                num_classes = self.real_num_classes
                agg_logits = self.real_action_head(agg_features)
                seg_label = sample["seg_labels"]
            else:
                num_classes = self.num_classes
                agg_logits = self.action_head(agg_features)
                seg_label = sample["relabeled_seg_labels"]

            agg_cos = F.cosine_similarity(agg_features.unsqueeze(1), self.prototypes.transpose(0, 1).unsqueeze(-1), dim=0) / self.temperature # k*C, L
            agg_cos = agg_cos.transpose(0, 1) # L, k*C

            mask = seg_label != -1

            agg_logits = agg_logits.permute(1, 0)[mask]
            agg_cos = agg_cos[mask]
            seg_label = seg_label[mask]
            agg_features = agg_features.transpose(0, 1)[mask]

            # select correct clusters
            temp = torch.cat([agg_cos[i:i+1, seg_label[i]::num_classes] for i in range(len(seg_label))], dim=0)
            ids = torch.argmax(temp, dim=1) # L, k
            refined_label = ids * num_classes + seg_label
            
            seg_loss += self.action_ce(agg_logits, seg_label.to(self.device))
            cl_loss += self.var_infonce(agg_features, refined_label.to(self.device), self.prototypes)["cl_loss"]

        return seg_loss / len(samples), cl_loss / len(samples)
    
    def action_seg_loss(self, samples):
        ce_loss = 0
        for i, sample in enumerate(samples):
            
            # features = sample["frame_features"]
            # logits = F.cosine_similarity(features.unsqueeze(1), self.prototypes.transpose(0, 1).unsqueeze(-1), dim=0) / self.temperature # C, L
            if self.mode == "normal":
                logits = sample["action_logits"]
                labels = sample["framewise_labels"]
            else:
                logits = sample["relabeled_action_logits"]
                labels = sample["relabeled_framewise_labels"]
            mask = labels != -1
            ce_loss += self.action_ce(logits.permute(1, 0)[mask], labels[mask].to(self.device))
            # smoothing loss
            ce_loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(logits.permute(1, 0)[1:, :], dim=1), F.log_softmax(logits.detach().permute(1, 0)[:-1, :], dim=1)), min=0, max=16))
            add_mask = labels == -1
            if add_mask.sum() > 0:
                ce_loss += torch.sum(- torch.softmax(logits.permute(1, 0)[add_mask], dim=1) * torch.log(torch.softmax(logits.permute(1, 0)[add_mask], dim=1)), dim=1).mean()
        return ce_loss / len(samples)

    def frame_csct_variance_loss(self, samples):
        num_classes = self.real_num_classes if self.mode == "normal" else self.num_classes
        total_clusters = self.num_clusters * num_classes
        features_dict = {i: [] for i in range(total_clusters)}
        for sample in samples:
            features = sample["frame_features"] # D, L
            # features = F.normalize(features, dim=0, p=2)
            if self.mode == "normal":
                labels = sample["framewise_labels"]
            else:
                labels = sample["relabeled_framewise_labels"]
            for i in range(num_classes):
                prototypes = self.prototypes[i::num_classes, :]
                cur_features = features[:, labels == i]

                ids = torch.argmax(F.cosine_similarity(cur_features.unsqueeze(1), prototypes.transpose(0, 1).unsqueeze(-1), dim=0), dim=0) # L'

                cluster_labels = ids * num_classes + i
                unique_labels = torch.unique(cluster_labels).detach().cpu().tolist()
                for l in unique_labels:                    
                    features_dict[l].append(cur_features[:, cluster_labels == l])

        if self.update_prototypes == True:
            for i in range(total_clusters):
                if len(features_dict[i]) == 0:
                    continue
                feat = torch.cat(features_dict[i], dim=1)
                if feat.shape[1] == 0: # no feature of class i exist in the batch
                    continue
                # feat = F.normalize(feat, p=2, dim=0)
                # feat = F.normalize(torch.mean(feat, dim=1), p=2, dim=-1).detach()
                feat = torch.mean(feat, dim=1).detach()

                # self.prototypes[i] = F.normalize(self.ema_weight * self.prototypes[i] + (1 - self.ema_weight) * feat, p=2, dim=-1)
                self.prototypes[i] = self.ema_weight * self.prototypes[i] + (1 - self.ema_weight) * feat

        total_loss_dict = {
            "var_loss": 0.0,
            "cl_loss": 0.0
        }
        # perform infonce loss for all features
        for i in range(total_clusters):
            if len(features_dict[i]) == 0:
                continue
            features = torch.cat(features_dict[i], dim=1)
            if features.shape[1] == 0:
                continue
            loss_dict = self.var_infonce(features.transpose(0, 1), i, self.prototypes)
            total_loss_dict["var_loss"] = total_loss_dict["var_loss"] + loss_dict["var_loss"]
            total_loss_dict["cl_loss"] = total_loss_dict["cl_loss"] + loss_dict["cl_loss"]

        return total_loss_dict

    def mixup_loss(self):
        num_classes = self.real_num_classes if self.mode == "normal" else self.num_classes
        total_clusters = self.num_clusters * num_classes
        distance = torch.cdist(self.prototypes, self.prototypes) # C, C
        distance = torch.eye(total_clusters, device=self.device) * torch.inf + distance
        loss = 0.0
        for i in range(total_clusters):
            _, indices = torch.topk(distance[i], self.topk, largest=False)
            alpha = self.beta.sample().to(self.device)
            mixed_features = (1 - alpha) * self.prototypes[indices] + alpha * self.prototypes[i:i+1, :]
            mixed_features = mixed_features.transpose(0, 1)
            if self.mode == "normal":
                logits = self.real_action_head(mixed_features)
            else:
                logits = self.action_head(mixed_features)
            neg_entropy = torch.sum(torch.softmax(logits, dim=0) * torch.log_softmax(logits, dim=0))
            loss = loss + neg_entropy
        
        return loss / (self.topk * num_classes)

    def compute_loss(self, samples):
        if self.loss_weights["cl_loss"] > 0.0:
            var_cl_loss = self.frame_csct_variance_loss(samples)
            seg_loss, seg_cl_loss = self.segment_loss(samples)
            mixup_loss = self.mixup_loss()
        else:
            var_cl_loss = dict(
                cl_loss=torch.tensor(0.0, device=self.device),
                var_loss=torch.tensor(0.0, device=self.device),
            )
            mixup_loss = torch.tensor(0.0, device=self.device)
        action_ce_loss = self.action_seg_loss(samples)

        cos = F.cosine_similarity(self.prototypes.unsqueeze(0), self.prototypes.unsqueeze(1), dim=-1) # C, C
        cos.fill_diagonal_(-1.0)
        self.threshold = torch.cos(torch.arccos(torch.max(cos).detach()) / 2.0)

        if self.feature_mode == "seg":
            return {
                "action_ce_loss": self.loss_weights["action_ce_loss"] * action_ce_loss,
                "seg_cl_loss": self.loss_weights["seg_cl_loss"] * seg_cl_loss,
                "seg_loss": self.loss_weights["seg_loss"] * seg_loss,
                "cl_loss": self.loss_weights["cl_loss"] * var_cl_loss["cl_loss"],
                "var_loss": self.loss_weights["var_loss"] * var_cl_loss["var_loss"],
                "mixup_loss": self.loss_weights["mixup_loss"] * mixup_loss
            }
        else:
            return {
                "action_ce_loss": action_ce_loss,
                "cl_loss": self.loss_weights["cl_loss"] * var_cl_loss["cl_loss"],
                "var_loss": self.loss_weights["var_loss"] * var_cl_loss["var_loss"],
                "mixup_loss": self.loss_weights["mixup_loss"] * mixup_loss
            }            

    def detect_and_aggregate_with_abd(self, samples):
        """
        Docstring for detect_and_aggregate_with_abd
        
        :param self: Description
        """

        agg_logit_list = list()
        agg_features_list = list()
        boundaries_list = list()
        frame_cos_list = list()
        smooth_cos_list = list()
        for sample in samples:
            features = sample["frame_features"] # (D, L)
            # features = F.normalize(features, p=2, dim=0)
            start_ids, end_ids, score = self.abd.detect_boundary(features.transpose(0, 1), threshold=self.threshold)
            agg_features = torch.cat([torch.mean(features[:, start_ids[i]:end_ids[i]], dim=-1, keepdim=True) for i in range(len(start_ids))], dim=-1)

            if self.mode == "normal":
                agg_logits = self.real_action_head(agg_features)
            else:
                agg_logits = self.action_head(agg_features)
            # agg_logits = F.cosine_similarity(agg_features.unsqueeze(1), self.prototypes.transpose(0, 1).unsqueeze(-1), dim=0) / self.temperature # C, L

            agg_logit_list.append(agg_logits)
            boundaries_list.append(torch.cat([start_ids[:, None], end_ids[:, None]], dim=1))
            frame_cos_list.append(F.cosine_similarity(features.transpose(0, 1)[1:, :], features.transpose(0, 1)[:-1, :], dim=1)) # L - 1
            smooth_cos_list.append(score)
            agg_features_list.append(agg_features)
        
        error_list = self.detect_error(agg_features_list, agg_logit_list)
        return {
            "boundaries": boundaries_list,
            "agg_features": agg_features_list,
            "agg_logits": agg_logit_list,
            "frame_cos": frame_cos_list,
            "smooth_cos": smooth_cos_list,
            "error_list": error_list
        }

    def detect_error(self, agg_features_list, agg_logit_list):
        error_list = list()
        threshold = self.score_threshold
        B = len(agg_features_list)
        for i in range(B): 
            agg_features = agg_features_list[i] # D, L'
            agg_logits = agg_logit_list[i] # C, L'
            agg_entropy = -torch.sum(torch.softmax(agg_logits, dim=0) * torch.log_softmax(agg_logits, dim=0), dim=0) # L'

            score = F.cosine_similarity(agg_features.unsqueeze(1), self.prototypes.transpose(0, 1).unsqueeze(-1), dim=0) # C, L
            score, _ = torch.max(score, dim=0) # nearest prototypes
            score = 1.0 - score
            score = self.cosine_weight * score + self.entropy_weight * agg_entropy

            error = score >= threshold

            error_list.append(error.detach().cpu())
        return error_list

    @torch.no_grad()
    def estimate_prototypes(self, train_loader):
        num_classes = self.real_num_classes if self.mode == "normal" else self.num_classes
        features_dict = {i: [] for i in range(num_classes)}
        list_frame_scores = list()
        list_seg_scores = list()
        for video_idx, data in enumerate(train_loader):

            v_feature, frame_label, seg_label, _, boundaries, _  = data
            s_ids, e_ids = boundaries_to_idx(boundaries.squeeze(0))

            feature = v_feature.to(self.device)
            seg_label = seg_label.squeeze(0)
            frame_label = frame_label.squeeze(0)

            action_logits, relabeled_action_logits, boundaries_logits, frame_features = self.forward(feature.permute(0, 2, 1))
            frame_features = frame_features.squeeze(0) # (D, L)

            # if self.feature_mode == "seg":
            #     agg_features = torch.cat([torch.mean(frame_features[:, s_ids[i]:e_ids[i]], dim=-1, keepdim=True) for i in range(len(s_ids))], dim=-1) # D, L'

            for i in range(num_classes):
                prototypes = self.prototypes[i::num_classes, :]
                cur_features = frame_features[:, frame_label == i]
                if cur_features.shape[1] > 0:
                    features_dict[i].append(cur_features.transpose(0, 1))

        for i in range(num_classes):
            cur_features = torch.cat(features_dict[i], dim=0).detach().cpu().numpy()
            if cur_features.shape[0] == 0:
                continue
            # k-means
            kmeans = KMeans(self.num_clusters).fit(cur_features)
            cluster_centers = kmeans.cluster_centers_
            for j in range(self.num_clusters):
                self.prototypes[i + num_classes * j] = torch.tensor(cluster_centers[j], device=self.device)

    def initialize_mcs_prototypes(self):
        # Proof from https://arxiv.org/pdf/2206.08704
        num_classes = self.real_num_classes if self.mode == "normal" else self.num_classes
        assert num_classes * self.num_clusters <= (self.out_dim + 1)

        prototypes = np.array([[1.0, -1.0]])
        for k in range(2, self.out_dim + 1):
            new_prototypes = np.zeros([prototypes.shape[0] + 1, prototypes.shape[1] + 1], dtype=float)
            new_prototypes[0, 0] = 1.0
            new_prototypes[0, 1:] = -1.0 / k
            new_prototypes[1:, 1:] = np.sqrt(1 - (1 / (k * k))) * prototypes
            prototypes = new_prototypes

        prototypes = prototypes.T    # prototypes: (D + 1, D)
        return torch.tensor(prototypes[-num_classes * self.num_clusters:, :], dtype=torch.float)
    
    @torch.no_grad()
    def estimate_thresholds(self, train_loader):
        num_classes = self.real_num_classes if self.mode == "normal" else self.num_classes
        list_frame_scores = list()
        list_seg_scores = list()
        for video_idx, data in enumerate(train_loader):

            v_feature, graph_data, frame_label, seg_label, _, boundaries, _  = data
            s_ids, e_ids = boundaries_to_idx(boundaries[0])

            feature = v_feature[0].unsqueeze(0).to(self.device)
            graph_data = data_to_device(graph_data[0], self.device)
            seg_label = seg_label[0]
            frame_label = frame_label[0]

            action_logits, relabeled_action_logits, boundaries_logits, frame_features = self.forward(feature.permute(0, 2, 1), graph_data)
            frame_features = frame_features.squeeze(0) # (D, L)

            if self.feature_mode == "seg":
                agg_features = torch.cat([torch.mean(frame_features[:, s_ids[i]:e_ids[i]], dim=-1, keepdim=True) for i in range(len(s_ids))], dim=-1) # D, L'
                agg_logits = self.real_action_head(agg_features)
                seg_entropy = -torch.sum(torch.softmax(agg_logits, dim=0) * torch.log_softmax(agg_logits, dim=0), dim=0) # L'

                seg_cos_score = F.cosine_similarity(agg_features.unsqueeze(1), self.prototypes.transpose(0, 1).unsqueeze(-1), dim=0) # k*C, L'
                seg_cos_score = seg_cos_score.transpose(0, 1) # L', k*C
                mask = seg_label != -1

                seg_cos_score = seg_cos_score[mask]
                seg_label = seg_label[mask]

                # select correct clusters
                seg_cos_score, _ = torch.max(torch.cat([seg_cos_score[i:i+1, seg_label[i]::num_classes] for i in range(len(seg_label))], dim=0), dim=1) # L'
                seg_score = (1.0 - seg_cos_score) + self.entropy_weight * seg_entropy
                # seg_cos_score = torch.gather(seg_cos_score.transpose(0, 1), 1, seg_label.unsqueeze(1).to(self.device)).squeeze()

            action_logits = action_logits.squeeze(0)
            action_entropy = -torch.sum(torch.softmax(action_logits, dim=0) * torch.log_softmax(action_logits, dim=0), dim=0) # L
            frame_cos_score = F.cosine_similarity(frame_features.unsqueeze(1), self.prototypes.transpose(0, 1).unsqueeze(-1), dim=0) # C, L
            frame_cos_score = frame_cos_score.transpose(0, 1) # L, k*C
            mask = frame_label != -1

            frame_cos_score = frame_cos_score[mask]
            frame_label = frame_label[mask]

            # select correct clusters
            frame_cos_score, _ = torch.max(torch.cat([frame_cos_score[i:i+1, frame_label[i]::num_classes] for i in range(len(frame_label))], dim=0), dim=1) # L
            # frame_cos_score = torch.gather(frame_cos_score.transpose(0, 1), 1, frame_label.unsqueeze(1).to(self.device)).squeeze()
            frame_score = self.cosine_weight * (1.0 - frame_cos_score) + self.entropy_weight * action_entropy

            if self.feature_mode == "frame":
                seg_score = frame_score
            
            seg_score = frame_score

            # seg_cos_score = frame_cos_score

            list_seg_scores.append(seg_score.detach())

            list_frame_scores.append(frame_cos_score.detach())
        
        all_seg_scores = torch.cat(list_seg_scores, dim=0)
        L = len(all_seg_scores)
        for value in torch.arange(0.0, 10000.0, 0.001):
            T = torch.sum(all_seg_scores < value)
            if T / L > 0.95:
                self.score_threshold = value.to(self.device)
                break
        
        # all_frame_scores = torch.cat(list_frame_scores, dim=0)
        # L = len(all_frame_scores)
        # for value in torch.arange(1.0, -1.0, -0.001):
        #     T = torch.sum(all_frame_scores > value)
        #     if T / L > 0.95:
        #         self.threshold = value.to(self.device)
        #         break

# adapted from DPU https://github.com/lili0415/DPU-OOD-Detection
class Var_InfoNCE(nn.Module):
    def __init__(self, temperature, margin):
        super(Var_InfoNCE, self).__init__()

        self.temperature = temperature
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1, eps=tol)
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, features, label, prototypes):
        """
        Docstring for forward
        
        :param self: Description
        :param features: (L, D)
        :param label: int
        :param prototypes: (C, D)
        """
        L = features.shape[0]
        C = prototypes.shape[0]
        
        feat_cos = self.cos(features.unsqueeze(1), prototypes.unsqueeze(0)) # L, C
        feat_cos = torch.clamp(feat_cos, min=-1 + (tol), max=1 - (tol))
        feat_cos = torch.arccos(feat_cos)

        if type(label) is int:
            label_all = torch.ones(L).long().to(feat_cos.device) * label
        elif type(label) is torch.Tensor and label.shape[0] == 1:
            label_all = torch.ones(L).long().to(feat_cos.device) * label
        else:
            label_all = label
        mask = F.one_hot(label_all, C).float()
        feat_cos = torch.cos(torch.add(feat_cos, mask * self.margin))
        feat_cos = torch.div(feat_cos, self.temperature)
        # feat_cos = torch.exp(feat_cos)

        loss_per_frame = self.ce(feat_cos, label_all)

        var_loss = torch.var(loss_per_frame, correction=0)
        return {'cl_loss': torch.mean(loss_per_frame),
                'var_loss': var_loss,
                'cl_loss_per_frame': loss_per_frame}


class MixedConvAttModuleV2(nn.Module): # for decoder
    def __init__(self, num_layers, num_f_maps, input_dim_cross, out_dim, kernel_size, dropout_rate, time_emb_dim=None):
        super(MixedConvAttModuleV2, self).__init__()

        if time_emb_dim is not None:
            self.time_proj = nn.Linear(time_emb_dim, num_f_maps)

        self.layers = nn.ModuleList([copy.deepcopy(
            MixedConvAttentionLayerV2(num_f_maps, input_dim_cross, kernel_size, 2 ** i, dropout_rate)
        ) for i in range(num_layers)])  #2 ** i

        # self.final_layer = nn.Conv1d(num_f_maps, out_dim, 1, 1, 0)
    
    def forward(self, x, x_cross, time_emb=None):

        if time_emb is not None:
            x = x + self.time_proj(swish(time_emb))[:,:,None]

        for layer in self.layers:
            x = layer(x, x_cross)

        return x
        # return self.final_layer(x)

class MixedConvAttentionLayerV2(nn.Module):
    
    def __init__(self, d_model, d_cross, kernel_size, dilation, dropout_rate):
        super(MixedConvAttentionLayerV2, self).__init__()
        
        self.d_model = d_model
        self.d_cross = d_cross
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout_rate
        self.padding = (self.kernel_size // 2) * self.dilation 
        
        assert(self.kernel_size % 2 == 1)

        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=self.padding, dilation=dilation),
        )

        self.att_linear_q = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_k = nn.Conv1d(d_model + d_cross, d_model, 1)
        self.att_linear_v = nn.Conv1d(d_model, d_model, 1)

        self.ffn_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, 1),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.norm = nn.InstanceNorm1d(d_model, track_running_stats=False)

        self.attn_indices = None


    def get_attn_indices(self, l, device):
            
        attn_indices = []
                
        for q in range(l):
            s = q - self.padding
            e = q + self.padding + 1
            step = max(self.dilation // 1, 1)  
            # 1  2  4   8  16  32  64  128  256  512  # self.dilation
            # 1  1  1   2  4   8   16   32   64  128  # max(self.dilation // 4, 1)  
            # 3  3  3 ...                             (k=3, //1)          
            # 3  5  5  ....                           (k=3, //2)
            # 3  5  9   9 ...                         (k=3, //4)
                        
            indices = [i + self.padding for i in range(s,e,step)]

            attn_indices.append(indices)
        
        attn_indices = np.array(attn_indices)
            
        self.attn_indices = torch.from_numpy(attn_indices).long()
        self.attn_indices = self.attn_indices.to(device)
        
        
    def attention(self, x, x_cross):
        
        if self.attn_indices is None:
            self.get_attn_indices(x.shape[2], x.device)
        else:
            if self.attn_indices.shape[0] < x.shape[2]:
                self.get_attn_indices(x.shape[2], x.device)
                                
        flat_indicies = torch.reshape(self.attn_indices[:x.shape[2],:], (-1,))
        
        x_q = self.att_linear_q(torch.cat([x, x_cross], 1))
        x_k = self.att_linear_k(torch.cat([x, x_cross], 1))
        x_v = self.att_linear_v(x)

        x_k = torch.index_select(
            F.pad(x_k, (self.padding, self.padding), 'constant', 0),
            2, flat_indicies)  
        x_v = torch.index_select(
            F.pad(x_v, (self.padding, self.padding), 'constant', 0), 
            2, flat_indicies)  
                        
        x_k = torch.reshape(x_k, (x_k.shape[0], x_k.shape[1], x_q.shape[2], self.attn_indices.shape[1]))
        x_v = torch.reshape(x_v, (x_v.shape[0], x_v.shape[1], x_q.shape[2], self.attn_indices.shape[1])) 
        
        att = torch.einsum('n c l, n c l k -> n l k', x_q, x_k)
        
        padding_mask = torch.logical_and(
            self.attn_indices[:x.shape[2],:] >= self.padding,
            self.attn_indices[:x.shape[2],:] < att.shape[1] + self.padding
        ) # 1 keep, 0 mask
        
        att = att / np.sqrt(self.d_model)
        att = att + torch.log(padding_mask + 1e-6)
        att = F.softmax(att, 2)
        att = att * padding_mask

        r = torch.einsum('n l k, n c l k -> n c l', att, x_v)
        
        return r
    
                
    def forward(self, x, x_cross):
        
        x_drop = self.dropout(x)
        x_cross_drop = self.dropout(x_cross)

        out1 = self.conv_block(x_drop)
        out2 = self.attention(x_drop, x_cross_drop)
                
        out = self.ffn_block(self.norm(out1 + out2))

        return x + out