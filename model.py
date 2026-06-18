import torch
import torch.nn as nn

class KGCModel(nn.Module):
    def __init__(
        self,
        nentity,
        nrelation,
        dim,
        gamma
    ):
        super().__init__()
        self.nentity = nentity
        self.nrelation = nrelation
        self.dim = dim
        self.gamma = gamma

        # 实体embedding
        self.entity_embedding = nn.Embedding(nentity, dim)
        self.relation_embedding = nn.Embedding(nrelation, dim)

        self.reset_parameters()

    def reset_parameters(self):
        embedding_range = (self.gamma + 2.0) / self.dim
        # 均匀分布初始化实体向量
        nn.init.uniform_(
            self.entity_embedding.weight.data,
            a=-embedding_range,
            b=embedding_range
        )
        # 均匀分布初始化关系向量
        nn.init.uniform_(
            self.relation_embedding.weight.data,
            a=-embedding_range,
            b=embedding_range
        )

    def forward(self, triples, model="single"):
        h = triples[:, 0]
        r = triples[:, 1]
        t = triples[:, 2]

        h_emb = self.entity_embedding(h)
        r_emb = self.relation_embedding(r)
        t_emb = self.entity_embedding(t)

        # TransE: h + r ≈ t
        score = self.gamma - torch.norm(h_emb + r_emb - t_emb, p=1, dim=-1)

        return score

