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

# Kick off a build on GCP.
#
cd ..
PROJECT_NAME="project"
BUILD_NAME="graphworld"
while getopts p:b: flag
do
    case "${flag}" in
        p) PROJECT_NAME=${OPTARG};;
        b) BUILD_NAME=${OPTARG};;
    esac
done

gcloud builds submit --tag gcr.io/${PROJECT_NAME}/${BUILD_NAME} --timeout=3600
