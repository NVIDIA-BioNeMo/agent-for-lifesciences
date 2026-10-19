#!/bin/sh
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

# Print a parallel job count for the C++ builds in this image.
#
# One job per core runs the RDKit build out of memory on a Docker Desktop VM,
# which gets half the Mac's RAM by default: the Boost.Python wrapper units peak
# near 3 GB each. So the count is the smaller of the core count and usable
# memory / MB_PER_JOB, where usable memory is the smaller of MemAvailable and
# the cgroup limit. Cores are the ones this process may run on (nproc honours a
# cpuset), not the machine's total.
#
# A limit set on an enclosing cgroup -- a buildx docker-container builder
# started with --driver-opt memory=... -- is not visible from inside the build
# step, which sees the host's RAM; set BUILD_JOBS there.
#
# BUILD_JOBS, when set to a positive integer, overrides the calculation.

MB_PER_JOB="${MB_PER_JOB:-3072}"

case "${BUILD_JOBS:-}" in
    ''|0|*[!0-9]*) ;;
    *) echo "$BUILD_JOBS"; exit 0 ;;
esac

cores="$(nproc 2>/dev/null || echo 1)"

mem_mb="$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo 2>/dev/null)"
for limit_file in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$limit_file" ]; then
        limit="$(cat "$limit_file")"
        case "$limit" in
            ''|max|*[!0-9]*) ;;
            *)
                limit_mb=$((limit / 1024 / 1024))
                if [ -z "$mem_mb" ] || [ "$limit_mb" -lt "$mem_mb" ]; then
                    mem_mb="$limit_mb"
                fi
                ;;
        esac
        break
    fi
done

jobs="$cores"
if [ -n "$mem_mb" ]; then
    by_mem=$((mem_mb / MB_PER_JOB))
    [ "$by_mem" -lt "$jobs" ] && jobs="$by_mem"
fi
[ "$jobs" -lt 1 ] && jobs=1

echo "build-jobs: cores=$cores usable_mem_mb=${mem_mb:-unknown} -> jobs=$jobs" >&2
echo "$jobs"
