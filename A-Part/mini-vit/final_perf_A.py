from train import train, seed
from config import configs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=int, default=0, help='0 1 2 3 4')
args = parser.parse_args()

def main():
    seed()
    for config in [configs[args.case]]:
        config["lr"] = config["batch_size"] * 1e-4
        config["log_interval"] = 1
        train(config=config)

if __name__ == "__main__":
    main()
