#!/usr/bin/env bash
# Sequential per-category Isaac character smoke driver.
#
# Runs four FRESH Isaac Python processes (pedestrian, doctor, construction_worker,
# police) under ONE shared flock.  Each process spawns exactly one pedestrian,
# writes its own pre-ledger/receipt/PTS/MP4, and exits before the next starts.
#
# Engineering smoke only: no Qwen, no Nav2, no scientific episode.
set -u

OUT_ROOT=${1:?usage: run_character_category_smoke.sh <out-root> [seed] [steps]}
SEED=${2:-20261103}
STEPS=${3:-120}

CONTAINER=${ARENA_ISAAC_CONTAINER:-arena-arena_jazzy_ws-isaac-1}
REPO_IN_CONTAINER=${ARENA_ISAAC_REPO_IN_CONTAINER:-/home/ubuntu/arena_jazzy_ws/outputs/arena_isaac_clean_integration_2116}
LOCK_FILE=${ARENA_ISAAC_LOCK_FILE:-/tmp/arena_jazzy_ws_isaac_gpu_runtime.lock}
PYTHONPATH_IN_CONTAINER=${ARENA_ISAAC_PYTHONPATH:-${REPO_IN_CONTAINER}/arena_isaac:/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/rclpy}
LD_LIBRARY_PATH_IN_CONTAINER=${ARENA_ISAAC_LD_LIBRARY_PATH:-/isaac-sim/exts/isaacsim.ros2.bridge/jazzy/lib}
ROS_DOMAIN_ID_VALUE=${ARENA_ISAAC_ROS_DOMAIN_ID:-107}
SOURCE_COMMIT_VALUE=${ARENA_ISAAC_SOURCE_COMMIT:-UNKNOWN}
EXPECTED_SOURCE_ROOT_VALUE=${ARENA_EXPECTED_SOURCE_ROOT:-${REPO_IN_CONTAINER}/arena_isaac}
PROVENANCE_MANIFEST_VALUE=${ARENA_PROVENANCE_MANIFEST:-}
CONTAINER_IMAGE_VALUE=${ARENA_CONTAINER_IMAGE:-UNKNOWN}

# Concrete models are the deterministic seed=20261103 selection recorded in the
# first four-category engineering run; they are frozen here so each fresh
# process reproduces the same identity without relying on cross-category RNG
# consumption order.
declare -A MODEL=(
  [pedestrian]=classic_female_business_01
  [doctor]=classic_female_medical_01
  [police]=classic_male_police_01
  [construction_worker]=classic_male_construction_04
)
CATEGORIES=(pedestrian doctor construction_worker police)

mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/DRIVER.log"
: > "$LOG"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "LOCK_BUSY" | tee -a "$LOG"
  exit 2
fi
date -u +"LOCK_ACQUIRED=%Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"

OVERALL=0
for category in "${CATEGORIES[@]}"; do
  cat_out="$OUT_ROOT/$category"
  mkdir -p "$cat_out"
  echo "START category=$category model=${MODEL[$category]} utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  docker exec "$CONTAINER" bash -lc "
    export PYTHONPATH=$PYTHONPATH_IN_CONTAINER
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_IN_CONTAINER
    export ROS_DOMAIN_ID=$ROS_DOMAIN_ID_VALUE
    export ARENA_ISAAC_SOURCE_COMMIT=$SOURCE_COMMIT_VALUE
    export ARENA_EXPECTED_SOURCE_ROOT=$EXPECTED_SOURCE_ROOT_VALUE
    export ARENA_PROVENANCE_MANIFEST=$PROVENANCE_MANIFEST_VALUE
    export ARENA_CONTAINER_IMAGE=$CONTAINER_IMAGE_VALUE
    export ARENA_CONTAINER_ID=$CONTAINER
    cd $REPO_IN_CONTAINER
    /isaac-sim/python.sh arena_isaac/scripts/validate_character_video_smoke.py \
      --output-dir $cat_out \
      --category $category \
      --model ${MODEL[$category]} \
      --seed $SEED \
      --steps $STEPS \
      --ros-domain-id $ROS_DOMAIN_ID_VALUE
  " >> "$LOG" 2>&1
  rc=$?
  echo "END category=$category rc=$rc utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  if [ "$rc" -ne 0 ]; then
    OVERALL=$rc
    echo "STOPPING_AFTER_FAILURE category=$category rc=$rc" | tee -a "$LOG"
    break
  fi
done

date -u +"LOCK_RELEASED=%Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"
exit "$OVERALL"
