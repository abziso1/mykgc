import random
import torch
from typing import List, Tuple, Set
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import logging

Triple = Tuple[int, int, int]

def build_dataloaders(args):
    """
     创建 train_loader / valid_loader / test_loader
    """
    data_path = Path(args.data_root) / args.dataset

    entity2id = {}
    relation2id = {}

    # entity -> id
    with open(data_path / "entities.dict", "r", encoding="utf-8") as f:
        for line in f:
            eid, entity = line.strip().split("\t")
            entity2id[entity] = int(eid)

    # relation -> id
    with open(data_path / "relations.dict", "r", encoding="utf-8") as f:
        for line in f:
            rid, relation = line.strip().split("\t")
            relation2id[relation] = int(rid)

    train_triples = []
    valid_triples = []
    test_triples = []

    # train
    with open(data_path / "train.txt" , "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            train_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # valid
    with open(data_path / "valid.txt" , "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            valid_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # test
    with open(data_path / "test.txt" , "r", encoding="utf-8") as f:
        for line in f:
            h, r, t = line.strip().split("\t")
            test_triples.append((entity2id[h], relation2id[r], entity2id[t]))

    # 用于过滤假负样本
    all_true_triples = set(train_triples + valid_triples + test_triples)

    train_dataset = KGDataset(train_triples)
    valid_dataset = KGDataset(valid_triples)
    test_dataset = KGDataset(test_triples)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    logging.info(f"entity2id: {nentity}, relation2id: {nrelation}, train_triples: {len(train_triples)}, valid_triples: {len(valid_triples)}, test_triples: {len(test_triples)} all_true_triples: {len(all_true_triples)}")

    train_collator = NegativeSamplingCollator(
        nentity=nentity,
        neg_num=args.neg_num,
        all_true_triples=all_true_triples
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        collate_fn=lambda batch: torch.LongTensor(batch),
        batch_size = args.batch_size,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        collate_fn=lambda batch: torch.LongTensor(batch),
        batch_size = args.batch_size,
    )

    return train_loader, valid_loader, test_loader, nentity, nrelation,   all_true_triples


class KGDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int):
        return self.triples[idx]

class NegativeSamplingCollator:
    def __init__(self, nentity: int, neg_num: int, all_true_triples: Set[Triple]):
        self.nentity = nentity
        self.neg_num = neg_num
        self.all_true_triples = all_true_triples

    def _sample_negative_entity(
        self,
        h: int,
        r: int,
        t: int,
        corrupt_head: bool
    ) -> Triple:
        """
        对一个正样本构造一个负样本
        """

        while True:
            candidate = random.randint(0, self.nentity - 1)

            if corrupt_head:
                neg_triple = (candidate, r, t)
            else:
                neg_triple = (h, r, candidate)

            if neg_triple not in self.all_true_triples:
                return neg_triple

    def __call__(self, batch: List[Triple]):
        """
        batch是Dataset的__getitem__返回的一批正样本
        [
            (h1, r1, t1),
            (h2, r2, t2),
            ...
        ]
        """
        pos_triples = []
        neg_triples = []

        for h, r, t in batch:
            pos_triples.append((h, r, t))

            for _ in range(self.neg_num):
                corrupt_head = random.random() < 0.5

                neg_triple = self._sample_negative_entity(
                    h, r, t,
                    corrupt_head
                )
                neg_triples.append(neg_triple)

        pos_triples = torch.LongTensor(pos_triples)
        neg_triples = torch.LongTensor(neg_triples)

        return pos_triples, neg_triples