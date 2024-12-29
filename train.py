import os
import torch
from argparse import ArgumentParser


def main():
  parser = ArgumentParser()
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--data_dir", type=str, default="./data")
  parser.add_argument("--decay", type=float, default=1e-4)
  parser.add_argument("--epochs", type=int, default=5)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--momentum", type=float, default=0.9)
  parser.add_argument("--name", type=str, default="name")
  parser.add_argument("--output_dir", type=str, default="./output")
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()


  device = "cuda" if torch.cuda.is_available() else "mps"

  # Create model output path
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  # Create data path
  if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

  # Prepares train, val datasets

  # Init models

  # Load checkpoint if exists

  # Init lr scheduler

  # Train

if __name__ == "__main__":
  main() 