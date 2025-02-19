#!/bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# Utilize docker-compose to run beam-pipeline locally in the same environment
# as the remote workers.
#
# This file is for testing the joint learning scheme for SSL methods
#
cd ..
BUILD_NAME="graphworld"
while getopts b: flag
do
    case "${flag}" in
        b) BUILD_NAME=${OPTARG};;
    esac
done

echo ${OPTARG}

docker-compose run \
  --entrypoint "python3 /app/beam_benchmark_main.py \
  --output /app/out \
  --gin_files /app/configs/SSL_nodeclassification/mwe/nodeclassification_mwe_jl.gin \
  --runner DirectRunner \
  --direct_num_workers 10 \
  --direct_running_mode multi_processing" \
  ${BUILD_NAME}



