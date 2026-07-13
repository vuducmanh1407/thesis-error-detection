import numpy as np
import torch
import torch_geometric

def detect_boundaries(scores, w_size, minima=True):
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

    return nms_result, non_nms_result

def boundaries_to_idx(boundaries):
    """
    boundaries: (L)
    """
    boundaries_indices = torch.argwhere(boundaries).cpu()


    boundaries_indices = boundaries_indices.squeeze(1)

    end_ids = (boundaries_indices + 1)

    if boundaries_indices.shape[0] > 1:
        start_ids = torch.cat([torch.tensor([0]), (boundaries_indices + 1)[:-1]], dim=0)
    else:
        start_ids = torch.tensor([0])

    return start_ids, end_ids

def data_to_device(data, output_device):
    if isinstance(data, torch.FloatTensor):
        return data.to(output_device)
    elif isinstance(data, torch.DoubleTensor):
        return data.float().to(output_device)
    elif isinstance(data, torch.ByteTensor):
        return data.long().to(output_device)
    elif isinstance(data, torch.LongTensor):
        return data.to(output_device)
    elif isinstance(data, list) or isinstance(data, tuple):
        return [data_to_device(d, output_device) for d in data]
    elif isinstance(data, dict):
        new_data = dict()
        for k, v in data.items():
            new_data[k] = data_to_device(v, output_device)
        return new_data
    elif isinstance(data, torch_geometric.data.Data):
        return data.to(output_device)
    elif isinstance(data, (str, int, float)):
        return data
    else:
        raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))