import torch
import torch.nn.functional as F

from args import get_args
from pathlib import Path
from utils import set_logger
from datetime import datetime
import logging
from model import KGCModel
from data import build_dataloaders
from evaluator import evaluate, log_metrics

def train_one_epoch(model, train_loader, optimizer, device, epoch, epochs):
    model.train()

    total_loss = 0

    for pos_triples, neg_triples in train_loader:
        pos_triples = pos_triples.to(device)
        neg_triples = neg_triples.to(device)

        pos_score = model(pos_triples)
        neg_score = model(neg_triples)

        loss = (
            -F.logsigmoid(pos_score).mean()
            -F.logsigmoid(-neg_score).mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss

if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)

    # 设置日志
    log_path = Path("logs") / args.dataset / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    set_logger(log_path)
    logging.info(args)

    train_loader, valid_loader, test_loader, nentity, nrelation, all_true_triples = build_dataloaders(args)

    model = KGCModel(
        nentity=nentity,
        nrelation=nrelation,
        dim=args.dim,
        gamma=args.gamma
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=args.epochs
        )

        logging.info(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")

        if epoch % args.eval_interval == 0:
            valid_metrics = evaluate(
                model=model,
                data_loader=valid_loader,
                all_true_triples=all_true_triples,
                nentity=nentity,
                device=device,
            )
            log_metrics(valid_metrics, prefix="valid")














