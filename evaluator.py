import logging
from typing import Dict, Set, Tuple
import torch

from data import Triple

@torch.no_grad()
def evaluate(
    model,
    data_loader,
    all_true_triples: Set[Triple],
    nentity: int,
    device: torch.device,
    chunk_size: int = 1024
) -> Dict[str, float]:
    """
    对每个三元组 (h, r, t)，分别做：
    1. tail prediction: (h, r, ?)
    2. head prediction: (?, r, t)
    返回:
    MR / MRR / Hits@1 / Hits@3 / Hits@10
    """

    model.eval()
    ranks = []

    for batch in data_loader:

        for h, r, t in batch.tolist():
            # tail prediction: (h, r, ?)
            tail_rank = _evaluate_tail_prediction(
                model=model,
                h=h,
                r=r,
                t=t,
                all_true_triples=all_true_triples,
                nentity=nentity,
                device=device,
                chunk_size=chunk_size,
            )

            # head prediction: (?, r, t)
            head_rank = _evaluate_head_prediction(
                model=model,
                h=h,
                r=r,
                t=t,
                all_true_triples=all_true_triples,
                nentity=nentity,
                device=device,
                chunk_size=chunk_size,
            )

            ranks.append(tail_rank)
            ranks.append(head_rank)

    ranks = torch.tensor(ranks, dtype=torch.float)

    metrics = {
        "MR": ranks.mean().item(),
        "MRR": (1.0 / ranks).mean().item(),
        "Hits@1": (ranks <= 1).float().mean().item(),
        "Hits@3": (ranks <= 3).float().mean().item(),
        "Hits@10": (ranks <= 10).float().mean().item(),
    }

    return metrics

@torch.no_grad()
def _evaluate_tail_prediction(
    model,
    h: int,
    r: int,
    t: int,
    all_true_triples: Set[Triple],
    nentity: int,
    device: torch.device,
    chunk_size: int
) -> int:
    """
    给定 (h, r, ?)，预测 tail。
    """

    scores = torch.empty(nentity, device=device)

    for start in range(0, nentity, chunk_size):
        end = min(start + chunk_size, nentity)

        candidates = torch.arange(start, end, device=device)

        h_tensor = torch.full_like(candidates, fill_value=h)
        r_tensor = torch.full_like(candidates, fill_value=r)
        t_tensor = candidates

        candidate_triples = torch.stack(
            [h_tensor, r_tensor, t_tensor],
            dim = 1
        )

        scores[start: end] = model(candidate_triples)

    # 如果 (h, r, candidate) 是真实三元组，并且 candidate 不是当前目标 t，
    # 那么它不应该参与排名，需要过滤掉。
    for candidate in range(nentity):
        if candidate != t and (h, r, candidate) in all_true_triples:
            scores[candidate] = -1e6

    true_score = scores[t]

    rank = (scores > true_score).sum().item() + 1

    return rank

@torch.no_grad()
def _evaluate_head_prediction(
    model,
    h: int,
    r: int,
    t: int,
    all_true_triples: Set[Triple],
    nentity: int,
    device: torch.device,
    chunk_size: int
) -> int:
    """
    给定 (?, r, t)，预测 head。
    """

    scores = torch.empty(nentity, device=device)

    for start in range(0, nentity, chunk_size):
        end = min(start + chunk_size, nentity)

        candidates = torch.arange(start, end, device=device)

        h_tensor = candidates
        r_tensor = torch.full_like(candidates, fill_value=r)
        t_tensor = torch.full_like(candidates, fill_value=t)

        candidate_triples = torch.stack(
            [h_tensor, r_tensor, t_tensor],
            dim=1
        )

        scores[start:end] = model(candidate_triples)

    # 如果 (h, r, candidate) 是真实三元组，并且 candidate 不是当前目标 t，
    # 那么它不应该参与排名，需要过滤掉。
    for candidate in range(nentity):
        if candidate != h and (candidate, r, t) in all_true_triples:
            scores[candidate] = -1e6

    true_score = scores[h]

    rank = (scores > true_score).sum().item() + 1

    return rank

def log_metrics(metrics: Dict[str, float], prefix: str = "Valid"):
    """
    打印评估指标
    """
    logging.info(
        f"{prefix} | "
        f"MR: {metrics['MR']:.4f} | "
        f"MRR: {metrics['MRR']:.6f} | "
        f"Hits@1: {metrics['Hits@1']:.6f} | "
        f"Hits@3: {metrics['Hits@3']:.6f} | "
        f"Hits@10: {metrics['Hits@10']:.6f}"
    )


if __name__ == '__main__':
    pass