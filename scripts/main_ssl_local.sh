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
start=`date +%s`

BUILD_NAME="graphworld"
while getopts b: flag
do
    case "${flag}" in
        b) BUILD_NAME=${OPTARG};;
    esac
done

# OUTPUT_DIR="out_$1"
OUTPUT_DIR="R3_direct2worker_inmemory_10samples"
OUTPUT_PATH="/app/out/"${OUTPUT_DIR}

exec > out/${OUTPUT_DIR}/log.txt 2>&1

rm -rf "${OUTPUT_PATH}"
mkdir -p ${OUTPUT_PATH}

echo ${OPTARG}

docker compose run \
  --entrypoint "python3 /app/beam_benchmark_main.py \
  --output ${OUTPUT_PATH} \
  --gin_files /app/configs/SSL_nodeclassification/nodeclassification.gin \
  --runner DirectRunner \
  --direct_num_workers 2 \
  --direct_running_mode in_memory"\
  ${BUILD_NAME}
# docker compose run \
#   --entrypoint "python3 /app/beam_benchmark_main.py \
#   --output ${OUTPUT_PATH} \
#   --gin_files /app/configs/SSL_nodeclassification/nodeclassification.gin \
#   --runner PortableRunner \
#   --job_endpoint localhost:8099 \
#   --environment_type LOOPBACK"\
#   ${BUILD_NAME}

end=`date +%s`
runtime=$((end-start))
echo "runtime: ${runtime}"
