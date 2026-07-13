import torch
import torch.nn as nn
import torch.nn.functional as F

def temporal_non_minimum_surpression(scores, w_size, minima=True):
    """
    scores: (L)
    """
    if minima == False: # detecting maxima
        scores = -scores

    L = len(scores)

    candidate_minima = torch.argwhere((scores[1:-1] <= scores[:-2]) * (scores[1:-1] <= scores[2:])).squeeze() + 1
    candidate_start = torch.clamp(candidate_minima - w_size, min=0).tolist()
    candidate_end = torch.clamp(candidate_minima + w_size + 1, max=L).tolist()
    candidate_minima = candidate_minima.tolist()

    non_nms_result = torch.zeros(L, dtype=bool)
    non_nms_result[candidate_minima] = 1
    
    # nms_1d
    nms_result = torch.zeros(L, dtype=bool)
    for i, idx in enumerate(candidate_minima):
        start = candidate_start[i]
        end = candidate_end[i]
        if (scores[idx] == torch.min(scores[start:end])) and (torch.max(nms_result[start:end]) == 0):
            nms_result[idx] = 1

    non_nms_result[-1] = 1
    nms_result[-1] = 1

    return nms_result.cpu(), non_nms_result.cpu()


def merge_boundaries(features, start_ids, end_ids, threshold):
    if len(start_ids) == 1:
        return start_ids, end_ids
    
    start_ids = start_ids.tolist()
    end_ids = end_ids.tolist()
    while True:
        agg_features = torch.cat([torch.mean(features[start_ids[i]:end_ids[i], :], dim=0, keepdim=True) for i in range(len(start_ids))], dim=0)
        cos_score = F.cosine_similarity(agg_features[1:], agg_features[:-1])
        max_cos, max_idx = torch.max(cos_score, dim=0)
        if max_cos < threshold:
            break
        max_idx = max_idx.item()        
        start_ids.pop(max_idx + 1)
        end_ids.pop(max_idx)
    
    return torch.tensor(start_ids), torch.tensor(end_ids)

class ABD():
    def __init__(self, window_size, smooth_feature, to_merge_boundaries=True):
        self.smooth_feature = smooth_feature
        self.window_size = window_size
        self.kernel = None
        self.merge_boundaries = to_merge_boundaries
        if smooth_feature is True:
            self.kernel = nn.AvgPool1d(2 * window_size + 1, stride=1, padding=window_size)

    def detect_boundary(self, features, threshold):
        """
        Docstring for detect_boundary
        
        :param self: Description
        :param features: (L, D)
        :param threshold: Description
        """
        with torch.no_grad():
            ori_features = features
            if self.kernel is not None:
                features = self.kernel(features.transpose(0, 1)).transpose(0, 1)
            
            score = F.cosine_similarity(features[1:], features[:-1], dim=1).squeeze() # L - 1

            nms_boundaries, _ = temporal_non_minimum_surpression(score, self.window_size)

            boundaries_indices = torch.squeeze(torch.argwhere(nms_boundaries) + 2, 1)

            end_ids = boundaries_indices
            start_ids = torch.cat([torch.tensor([0]), boundaries_indices[:-1]])

            if self.merge_boundaries:
                start_ids, end_ids = merge_boundaries(ori_features, start_ids, end_ids, threshold)

            return start_ids, end_ids, score