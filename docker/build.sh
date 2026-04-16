#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="rl-control-lazy"
IMAGE_TAG="latest"
DOCKERFILE="Dockerfile"
CONTEXT_DIR="."

BUILD_UID="${BUILD_UID:-1006}"
BUILD_GID="${BUILD_GID:-1006}"
BUILD_USERNAME="${BUILD_USERNAME:-flocking}"
BUILD_PASSWORD="${BUILD_PASSWORD:-lazy}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker not found"
  exit 1
fi

if [ ! -f "${DOCKERFILE}" ]; then
  echo "Error: ${DOCKERFILE} not found in current directory"
  exit 1
fi

if [ ! -f "${CONTEXT_DIR}/requirements.txt" ]; then
  echo "Error: requirements.txt not found in build context"
  exit 1
fi

echo "Building image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  UID: ${BUILD_UID}, GID: ${BUILD_GID}, USERNAME: ${BUILD_USERNAME}"

docker build \
  --build-arg UID="${BUILD_UID}" \
  --build-arg GID="${BUILD_GID}" \
  --build-arg USERNAME="${BUILD_USERNAME}" \
  --build-arg PASSWORD="${BUILD_PASSWORD}" \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  -f "${DOCKERFILE}" \
  "${CONTEXT_DIR}"

echo "Build complete."
docker images "${IMAGE_NAME}:${IMAGE_TAG}"

