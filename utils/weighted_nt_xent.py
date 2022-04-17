import torch
from torch import nn
import numpy as np
from rdkit import DataStructs, Chem
import torch.nn.functional as F
from rdkit.Chem import AllChem


class WeightedNTXentLoss(torch.nn.Module):
    def __init__(self, device, temperature=0.1, use_cosine_similarity=True, lambda_1=0.5, **kwargs):
        super(WeightedNTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.lambda_1 = lambda_1
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self, batch_size):
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, x1, x2, mols):
        assert x1.size(0) == x2.size(0)
        batch_size = x1.size(0)

        fp_score = np.zeros((batch_size, batch_size-1))
        fps = [AllChem.GetMorganFingerprint(Chem.AddHs(x), 2, useFeatures=True) for x in mols]

        for i in range(len(mols)):
            for j in range(i+1, len(mols)):
                fp_sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                fp_score[i,j-1] = fp_sim
                fp_score[j,i] = fp_sim

        fp_score = 1 - self.lambda_1 * torch.tensor(fp_score, dtype=torch.float).to(x1.device)
        fp_score = fp_score.repeat(2, 2)

        representations = torch.cat([x2, x1], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        mask_samples_from_same_repr = self._get_correlated_mask(batch_size).type(torch.bool)
        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * batch_size, -1)
        negatives *= fp_score

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * batch_size)
