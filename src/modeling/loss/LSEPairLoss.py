import math
import torch.nn.functional


# Compute the LSEPair (Log-Sum-Exp Pairwise) loss.
class LSEPairLoss:
    @staticmethod
    def forward(q_embedding: torch.Tensor, d_embedding: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) \
            -> torch.Tensor:
        """
        Compute the LSEPairLoss: log(1 + sum{d+} sum{d-} exp(s(d-)-s(d+))).
        :param q_embedding: Query embedding (shape: BxH).
        :param d_embedding: Document embedding (shape: NxH).
        :param labels: Labels (shape: BxN) indicating whether document[i][j] is positive (1) or negative (0).
        :param temperature: Scaling factor (default: 1.0).
        :return: The LSEPairLoss.
        """
        # Compute the similarity matrix, and rescale the values using the temperature scaling factor.
        similarity = torch.matmul(q_embedding, d_embedding.transpose(0, 1)) * (1.0 / temperature) # BxN

        # Compute the mask for positive and negative documents.
        p_mask = (labels != 0).bool()  # BxN
        n_mask = (labels == 0).bool()  # BxN

        # Fill the similarity matrix using the masks for positive and negative documents.
        p_similarity = torch.masked_fill(-similarity, n_mask, -math.inf)
        n_similarity = torch.masked_fill(similarity, p_mask, -math.inf)
        del similarity, p_mask, n_mask

        # sum{d+} sum{d-} exp(s(d-)-s(d+))
        # = sum{d-} exp(s(d-)) * sum{d+} -exp(s(d+))
        # = exp[ log( sum{d-} exp(s(d-)) ) + log( sum{d+} -exp(s(d+)) ) ]
        p_logsumexp = torch.logsumexp(p_similarity, dim=1)
        n_logsumexp = torch.logsumexp(n_similarity, dim=1)
        del p_similarity, n_similarity

        loss = torch.log1p(torch.exp(p_logsumexp + n_logsumexp))
        del p_logsumexp, n_logsumexp

        return torch.mean(loss)
