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

import streamlit as st
import json
import pandas as pd
import argparse
from pathlib import Path

import tiktoken


# Parse CLI arguments before Streamlit takes over
# Streamlit passes its own args, so we need to handle this carefully
def get_cli_args():
    """Parse CLI arguments, filtering out Streamlit's args."""
    parser = argparse.ArgumentParser(description="JSONL Training Data Viewer")
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default="/workspaces/aevom/training/data",
        help="Directory containing JSONL files (default: /workspaces/aevom/training/data)",
    )
    # Only parse known args to avoid conflicts with Streamlit
    args, _ = parser.parse_known_args()
    return args


args = get_cli_args()
data_directory = Path(args.directory)

st.set_page_config(page_title="JSONL Training Data Viewer", layout="wide")

st.title("📊 Training Data Viewer")

# Display current directory
st.caption(f"📁 Directory: `{data_directory}`")


def get_jsonl_files(directory: Path) -> list[Path]:
    """Get all JSONL files in the directory."""
    if not directory.exists():
        return []
    return sorted(directory.glob("*.jsonl"))


@st.cache_resource
def get_tokenizer(model: str = "cl100k_base"):
    """Get a tiktoken tokenizer (cached)."""
    return tiktoken.get_encoding(model)


def count_tokens(text: str, tokenizer=None) -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def count_messages_tokens(messages: list[dict], tokenizer=None) -> int:
    """Count tokens in a list of messages (input sequence)."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    total = 0
    for msg in messages:
        # Account for message formatting overhead (~4 tokens per message)
        total += 4
        total += count_tokens(msg.get("role", ""), tokenizer)
        total += count_tokens(msg.get("content", ""), tokenizer)
    return total


@st.cache_data
def load_jsonl(path: str):
    """Load JSONL file and return list of records."""
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                records.append({"index": i, "data": data})
    return records


# Get available JSONL files
jsonl_files = get_jsonl_files(data_directory)

if not jsonl_files:
    st.error(f"No JSONL files found in: {data_directory}")
    st.info(
        "Use `--directory` or `-d` to specify a different directory when launching."
    )
    st.code("streamlit run scripts/jsonl_viewer.py -- --directory /path/to/data")
    st.stop()

# File selector dropdown
selected_file = st.selectbox(
    "📄 Select JSONL File", options=jsonl_files, format_func=lambda x: x.name
)

if selected_file is None:
    st.stop()

try:
    records = load_jsonl(str(selected_file))
except FileNotFoundError:
    st.error(f"File not found: {selected_file}")
    st.stop()

if not records:
    st.warning(f"No records found in: {selected_file.name}")
    st.stop()

# Create summary table
table_data = []
for rec in records:
    data = rec["data"]
    messages = data.get("messages", [])

    # Extract summary info
    system_preview = ""
    user_preview = ""
    for msg in messages:
        if msg["role"] == "system":
            system_preview = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )
        elif msg["role"] == "user":
            user_preview = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )

    metadata = data.get("metadata", {})

    table_data.append(
        {
            "Index": rec["index"],
            "Query ID": metadata.get("query_id", "N/A"),
            "Chosen Mode": metadata.get("chosen_mode", "N/A"),
            "Rejected Mode": metadata.get("rejected_mode", "N/A"),
            "System Prompt": system_preview,
            "User Query": user_preview,
        }
    )

df = pd.DataFrame(table_data)

st.subheader("Data Overview")
st.dataframe(df, use_container_width=True, height=200)

# Row selection
st.subheader("Select a Row to View Details")
selected_index = st.selectbox(
    "Row Index",
    options=[r["index"] for r in records],
    format_func=lambda x: f"Row {x}: {table_data[x]['Query ID']}",
)

if selected_index is not None:
    selected_record = records[selected_index]["data"]

    st.divider()

    # Display metadata
    metadata = selected_record.get("metadata", {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Query ID", metadata.get("query_id", "N/A"))
    with col2:
        st.metric("Chosen Mode", metadata.get("chosen_mode", "N/A"))
    with col3:
        st.metric("Rejected Mode", metadata.get("rejected_mode", "N/A"))
    with col4:
        st.metric("Chosen Steps", metadata.get("chosen_steps", "N/A"))

    st.divider()

    # Calculate token counts
    messages = selected_record.get("messages", [])
    tokenizer = get_tokenizer()
    input_tokens = count_messages_tokens(messages, tokenizer)
    chosen = selected_record.get("chosen_response", "")
    rejected = selected_record.get("rejected_response", "")
    chosen_tokens = count_tokens(chosen, tokenizer)
    rejected_tokens = count_tokens(rejected, tokenizer)

    # Display input (messages)
    st.subheader("📥 Input")
    st.write(f"**Input Tokens:** {input_tokens:,}")

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            with st.expander("🤖 System Prompt", expanded=False):
                st.markdown(content)
        elif role == "user":
            with st.expander("👤 User Query", expanded=True):
                st.markdown(content)
        elif role == "assistant":
            with st.expander("🤖 Assistant Response", expanded=False):
                st.markdown(content)

    st.divider()

    # Display chosen response
    st.subheader("✅ Chosen Response")
    st.write(f"**Output Tokens:** {chosen_tokens:,}")
    with st.expander("View Chosen Response", expanded=True):
        st.markdown(chosen)

    st.divider()

    # Display rejected response
    st.subheader("❌ Rejected Response")
    st.write(f"**Output Tokens:** {rejected_tokens:,}")
    with st.expander("View Rejected Response", expanded=False):
        st.markdown(rejected)
