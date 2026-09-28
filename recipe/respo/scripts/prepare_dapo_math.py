#!/usr/bin/env python3
# Copyright 2026 ReSPO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Prepare the DAPO-MATH-17k training split used by ReSPO."""

import argparse
from pathlib import Path

from datasets import Features, Value, load_dataset

SOURCE = "open-r1/DAPO-Math-17k-Processed"
OLD_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
OLD_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
BOX_INSTRUCTION = " Please reason step by step, and put your final answer within \\boxed{}."


def convert(example, index):
    question = example["prompt"].replace(OLD_PREFIX, "").replace(OLD_SUFFIX, "").strip()
    reward_model = example.get("reward_model", {})
    ground_truth = reward_model.get("ground_truth", "") if isinstance(reward_model, dict) else ""
    if not ground_truth:
        ground_truth = example.get("solution", "")

    extra_info = example.get("extra_info", {})
    if not isinstance(extra_info, dict):
        extra_info = {}

    return {
        "id": f"dapo_math_{index}",
        "data_source": example.get("data_source", SOURCE),
        "prompt": [{"role": "user", "content": question + BOX_INSTRUCTION}],
        "ability": example.get("ability", "math").lower(),
        "reward_model": {"style": "rule", "ground_truth": str(ground_truth)},
        "extra_info": {
            "index": str(extra_info.get("index", "")),
            "processed_index": index,
            "split": "train",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/dapo_math/train.parquet"))
    args = parser.parse_args()

    dataset = load_dataset(SOURCE, split="train")
    dataset = dataset.map(convert, with_indices=True, remove_columns=dataset.column_names)
    features = Features(
        {
            "id": Value("string"),
            "data_source": Value("string"),
            "prompt": [{"content": Value("string"), "role": Value("string")}],
            "ability": Value("string"),
            "reward_model": {"style": Value("string"), "ground_truth": Value("string")},
            "extra_info": {
                "index": Value("string"),
                "processed_index": Value("int64"),
                "split": Value("string"),
            },
        }
    )
    dataset = dataset.cast(features)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(args.output))
    print(f"Wrote {len(dataset):,} examples to {args.output}")


if __name__ == "__main__":
    main()
