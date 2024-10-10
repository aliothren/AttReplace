import torch.nn as nn


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosineSimilarityLoss, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)
    
    def forward(self, x1, x2):
        return 1 - self.cos_sim(x1, x2).mean()
