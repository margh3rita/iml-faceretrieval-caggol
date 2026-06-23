import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.3):
        super().__init__()
        # initialising parameters
        self.s     = s    # scaling
        self.m     = m    # angular margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # boundary terms for numerical stability
        self.th    = math.cos(math.pi - m)    
        self.mm    = math.sin(math.pi - m) * m
        # learnable weight matrix with xavier uniform distr.
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # normalizing both embedding and weights
        emb_norm    = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight, dim=1)
        # computing cosine similarity and then sine
        cosine = F.linear(emb_norm, weight_norm)
        sine   = (1.0 - cosine.pow(2)).clamp(0).sqrt()
        # applying angular margin to reinforce similarity and dissimilarity
        phi    = cosine * self.cos_m - sine * self.sin_m
        # boundary check to avoid num instability
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)
        # one-hot target (penalizes the model not being confident enough)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # finally compute cross entropy between scaled arcface loss and gt
        return F.cross_entropy(output * self.s, labels.long())
