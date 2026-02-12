"""Training script for LLaMA model using Weigou Trainer."""
import argparse

from weigou.config import WeigouConfig
from weigou.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = WeigouConfig.load(args.config)
    trainer = Trainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
