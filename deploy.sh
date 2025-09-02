#!/usr/bin/env bash
# USING ACR - apitesting

set -euo pipefail

docker build -t testing-agent .
docker tag testing-agent apitesting.azurecr.io/testingagent:latest
docker push apitesting.azurecr.io/testingagent:latest
