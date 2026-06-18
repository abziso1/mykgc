import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wn18")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument('--neg_num', default=128, type=int)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--gamma', default=12.0, type=float)
    parser.add_argument('-d', '--dim', default=500, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()

