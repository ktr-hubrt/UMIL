import torch
from torch import nn
import math
import torch.nn.functional as F
import numpy as np

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def reduce_dimension(features, mode='umap', dim=50):
    if mode == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=dim)
        transformed_features = pca.fit_transform(features)
        fit_score = pca.explained_variance_ratio_.sum()
    elif mode == 'umap':
        import umap
        fit = umap.UMAP(n_components=dim)
        transformed_features = fit.fit_transform(features)
        fit_score = 0.0
    return transformed_features, fit_score


def PairEnum(x,mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1,1).repeat(1,x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1,x.size(1))
        x2 = x2[xmask].view(-1,x.size(1))
    return x1,x2

class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()

class ClusterLoss():
    def __init__(self, num_classes, bce_type, cosine_threshold, topk):
        self.num_classes = num_classes
        self.bce_type = bce_type
        self.costhre = cosine_threshold
        self.topk = topk
        self.bce = BCE()

    def compute_losses(self, inputs, include_label=False, unlabel_only=True):
        assert (include_label == False) or (unlabel_only == False)
        bce_loss = 0.0
        feat, feat_q, output2 = \
            inputs["x1"], inputs["x1_norm"], inputs["preds1_u"]
        feat_bar, feat_k, output2_bar = \
            inputs["x2"], inputs["x2_norm"], inputs["preds2_u"]
        label = inputs["labels"]

        if unlabel_only:
            mask_lb = inputs["mask"]
        else:
            mask_lb = torch.zeros_like(inputs["mask"]).bool()
        
        prob2, prob2_bar = F.softmax(output2, dim=1), F.softmax(output2_bar, dim=1)

        rank_feat = (feat[~mask_lb]).detach()
        if self.bce_type == 'cos':
            # default: cosine similarity with threshold
            feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
            tmp_distance_ori = torch.bmm(
                feat_row.view(feat_row.size(0), 1, -1),
                feat_col.view(feat_row.size(0), -1, 1)
            )
            tmp_distance_ori = tmp_distance_ori.squeeze()
            target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
            target_ulb[tmp_distance_ori > self.costhre] = 1
        elif self.bce_type == 'RK':
            # top-k rank statics
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :self.topk], rank_idx2[:, :self.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float().cuda()
            target_ulb[rank_diff > 0] = -1

        if include_label:
            # use source domain label for similar/dissimilar
            labels = labels_s.contiguous().view(-1, 1)
            mask_l = torch.eq(labels, labels.T).float().to(device)
            mask_l = (mask_l - 0.5) * 2.0
            target_ulb_t = target_ulb.view(feat.size(0), -1)
            target_ulb_t[:num_s, :num_s] = mask_l
            target_ulb = target_ulb_t.flatten()

        prob1_ulb, _ = PairEnum(prob2[~mask_lb])
        _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

        bce_loss = self.bce(prob1_ulb, prob2_ulb, target_ulb)
        return bce_loss, target_ulb
