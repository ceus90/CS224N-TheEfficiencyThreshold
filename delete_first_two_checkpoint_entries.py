#!/usr/bin/env python3
import argparse


def main():
    parser = argparse.ArgumentParser(description="Delete the first two entries from a JSONL checkpoint file.")
    parser.add_argument("checkpoint", help="Path to the JSONL checkpoint file")
    args = parser.parse_args()

    with open(args.checkpoint, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) <= 2:
        remaining = []
    else:
        remaining = lines[2:]

    with open(args.checkpoint, "w", encoding="utf-8") as f:
        f.writelines(remaining)


if __name__ == "__main__":
    main()
