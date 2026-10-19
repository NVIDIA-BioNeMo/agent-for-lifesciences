#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert DPO training data to SFT format for NeMo RL.

This script transforms the DPO pairs exported by scripts/export_training_data.py
into the SFT messages format required by NeMo RL's run_sft.py.

Supports two input formats:

New format (model-agnostic messages):
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
     "chosen_response": "...", "rejected_response": "...", "metadata": {...}}

Legacy format (Nemotron-templated prompt):
    {"prompt": "<extra_id_0>System\n...<extra_id_1>User\n...",
     "chosen_response": "...", "rejected_response": "...", "metadata": {...}}

SFT format (output):
    {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

Usage:
    # Basic conversion (uses chosen responses only)
    python convert_to_sft.py input_dpo.jsonl output_sft.jsonl

    # Split into train/val sets (90/10 split)
    python convert_to_sft.py input_dpo.jsonl --split --train-output train.jsonl --val-output val.jsonl

    # Preview without writing
    python convert_to_sft.py input_dpo.jsonl --dry-run
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional


def parse_nemotron_prompt(prompt: str) -> tuple[str, str]:
    """Parse a Nemotron-style prompt into system and user messages.

    Handles two formats:

    Format 1 (Nemotron special tokens):
        <extra_id_0>System
        <system content>
        <extra_id_1>User
        <user content>
        <extra_id_1>Assistant

    Format 2 (Plain text):
        System: <system content>
        User: <user content>
        Assistant:

    Args:
        prompt: The formatted prompt string.

    Returns:
        Tuple of (system_content, user_content).
    """
    system_content = ""
    user_content = ""

    # Format 1: Nemotron special tokens (<extra_id_0>System, <extra_id_1>User)
    if "<extra_id_0>System" in prompt:
        # Extract system content (between <extra_id_0>System and <extra_id_1>User)
        system_match = re.search(
            r"<extra_id_0>System\n?(.*?)<extra_id_1>User", prompt, re.DOTALL
        )
        if system_match:
            system_content = system_match.group(1).strip()

        # Extract user content (between <extra_id_1>User and <extra_id_1>Assistant)
        user_match = re.search(
            r"<extra_id_1>User\n?(.*?)<extra_id_1>Assistant", prompt, re.DOTALL
        )
        if user_match:
            user_content = user_match.group(1).strip()

        return system_content, user_content

    # Format 2: Plain text (System: ... User: ... Assistant:)
    if prompt.startswith("System:"):
        # Find where User: starts
        user_match = re.search(r"\n\s*User:", prompt)
        if user_match:
            system_content = prompt[7 : user_match.start()].strip()
            # Extract user content (everything between User: and Assistant:)
            user_start = user_match.end()
            assistant_match = re.search(r"\n\s*Assistant:", prompt[user_start:])
            if assistant_match:
                user_content = prompt[
                    user_start : user_start + assistant_match.start()
                ].strip()
            else:
                user_content = prompt[user_start:].strip()
        else:
            # No User: found, treat everything after System: as system content
            system_content = prompt[7:].strip()
    else:
        # No System: prefix, try to find User: directly
        user_match = re.search(r"User:", prompt)
        if user_match:
            user_start = user_match.end()
            assistant_match = re.search(r"\n\s*Assistant:", prompt[user_start:])
            if assistant_match:
                user_content = prompt[
                    user_start : user_start + assistant_match.start()
                ].strip()
            else:
                user_content = prompt[user_start:].strip()
        else:
            # Fallback: treat entire prompt as user content
            user_content = prompt.strip()

    return system_content, user_content


def convert_dpo_to_sft(dpo_entry: dict) -> Optional[dict]:
    """Convert a single DPO entry to SFT format.

    Uses only the chosen_response (positive feedback) for SFT training.

    Supports two input formats:
    - New format: has 'messages' array directly (model-agnostic)
    - Legacy format: has 'prompt' with Nemotron tokens that need parsing

    Args:
        dpo_entry: A DPO format entry with messages or prompt, and chosen_response.

    Returns:
        SFT format entry with messages array, or None if conversion fails.
    """
    chosen_response = dpo_entry.get("chosen_response", "")

    if not chosen_response:
        return None

    # New format: has 'messages' array directly
    if "messages" in dpo_entry:
        messages = dpo_entry["messages"]
        if not messages:
            return None
        # Make a copy to avoid modifying the original
        messages = [dict(m) for m in messages]
    # Legacy format: has 'prompt' with Nemotron tokens
    elif "prompt" in dpo_entry:
        prompt = dpo_entry["prompt"]
        if not prompt:
            return None

        # Parse the prompt into system and user messages
        system_content, user_content = parse_nemotron_prompt(prompt)

        if not user_content:
            return None

        # Build messages array
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})
    else:
        # Neither format found
        return None

    # Append the assistant response (the chosen response we're training on)
    messages.append({"role": "assistant", "content": chosen_response})

    return {"messages": messages}


def convert_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    train_output: Optional[Path] = None,
    val_output: Optional[Path] = None,
    split: bool = False,
    val_ratio: float = 0.1,
    seed: int = 42,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Convert a DPO JSONL file to SFT format.

    Args:
        input_path: Path to input DPO JSONL file.
        output_path: Path for output SFT JSONL (if not splitting).
        train_output: Path for training set (if splitting).
        val_output: Path for validation set (if splitting).
        split: Whether to split into train/val sets.
        val_ratio: Fraction of data for validation (default 0.1).
        seed: Random seed for reproducible splits.
        dry_run: If True, only preview without writing.

    Returns:
        Tuple of (total_converted, total_skipped).
    """
    converted = []
    skipped = 0

    # Read and convert
    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                dpo_entry = json.loads(line)
                sft_entry = convert_dpo_to_sft(dpo_entry)

                if sft_entry:
                    converted.append(sft_entry)
                else:
                    skipped += 1
                    print(f"  Skipped line {line_num}: missing required fields")

            except json.JSONDecodeError as e:
                skipped += 1
                print(f"  Skipped line {line_num}: JSON parse error - {e}")

    print(f"\nConverted: {len(converted)}, Skipped: {skipped}")

    if dry_run:
        print("\n[DRY RUN] Sample output:")
        if converted:
            print(json.dumps(converted[0], indent=2))
        return len(converted), skipped

    # Write output
    if split:
        # Shuffle and split
        random.seed(seed)
        random.shuffle(converted)

        val_size = int(len(converted) * val_ratio)
        val_data = converted[:val_size]
        train_data = converted[val_size:]

        print(f"\nSplit: {len(train_data)} train, {len(val_data)} val")

        # Write train set
        train_path = train_output or Path("train.jsonl")
        with open(train_path, "w", encoding="utf-8") as f:
            for entry in train_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(train_data)} entries to {train_path}")

        # Write val set
        val_path = val_output or Path("val.jsonl")
        with open(val_path, "w", encoding="utf-8") as f:
            for entry in val_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(val_data)} entries to {val_path}")

    else:
        # Write single file
        out_path = output_path or input_path.with_suffix(".sft.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in converted:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(converted)} entries to {out_path}")

    return len(converted), skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert DPO training data to SFT format for NeMo RL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input DPO JSONL file (from export_training_data.py)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output SFT JSONL file (default: input.sft.jsonl)",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split into train/val sets instead of single output",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=None,
        help="Output path for training set (default: train.jsonl)",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=None,
        help="Output path for validation set (default: val.jsonl)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview conversion without writing files",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    converted, skipped = convert_file(
        input_path=args.input,
        output_path=args.output,
        train_output=args.train_output,
        val_output=args.val_output,
        split=args.split,
        val_ratio=args.val_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    if converted == 0:
        print("\nWarning: No entries were converted!", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
