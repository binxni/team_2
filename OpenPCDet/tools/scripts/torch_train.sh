#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

PYTHON_BIN=${PYTHON_BIN:-python}

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
TOOLS_DIR="$(cd "${SCRIPT_DIR}/.."; pwd)"
TRAIN_PY="${TOOLS_DIR}/train.py"

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

if "${PYTHON_BIN}" -m torch.distributed.run --help >/dev/null 2>&1; then
    "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} "${TRAIN_PY}" --launcher pytorch ${PY_ARGS}
else
    "${PYTHON_BIN}" -m torch.distributed.launch --nproc_per_node=${NGPUS} --master_port=${PORT} "${TRAIN_PY}" --launcher pytorch ${PY_ARGS}
fi
