# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All Rights Reserved.
# from: https://github.com/Qualcomm-AI-research/outlier-free-transformers/blob/main/transformers_language/models/softmax.py
import torch


def clipped_softmax(data, dim=1, eta=1.1, gamma=-0.1, **kw):
    sm_out = torch.nn.functional.softmax(data, dim=dim, **kw)
    stretched_out = sm_out * (eta - gamma) + gamma
    return torch.clip(stretched_out, 0, 1)