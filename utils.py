import os
import random
from pathlib import Path

import numpy as np
import torch
import logging
from datetime import datetime
from args import get_args

def set_seed(seed):
    random.seed(seed) # 自带随机数
    np.random.seed(seed) # numpy随机数
    torch.manual_seed(seed) # pytorch在CPU端的随机数生成器
    torch.cuda.manual_seed_all(seed) # pytorch GPU端的随机生成器
    os.environ["PYTHONHASHSEED"] = str(seed) # python解释器的哈希种子

def set_logger(log_path: Path):
    log_path.parent.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8")
        ]
    )


if __name__ == '__main__':
    args = get_args()
    filename = f"{datetime.now().strftime("%Y%m%d_%H%M%S")}.log"
    log_path = Path("logs") / args.dataset / filename
    print(log_path)
    set_logger(log_path)
    logging.info("Training started")