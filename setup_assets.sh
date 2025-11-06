#!/bin/bash

BASE_URL="https://sdk.deepx.ai/"

# default value
OBB_TEST_IMG_PATH="res/assets/deepx_yolo/obb-images.tar.gz"
TEST_VIDEO_PATH="res/assets/deepx_yolo/videos.tar.gz"
OUTPUT_DIR="./assets"
SYMLINK_TARGET_PATH=""
SYMLINK_ARGS=""

# parse args
while [ "$#" -gt 0 ]; do
    case "$1" in
        --src_path=*)
            OBB_TEST_IMG_PATH="${1#*=}"
            ;;
        --output=*)
            OUTPUT_DIR="${1#*=}"

            # Symbolic link cannot be created when output_dir is the current directory.
            OUTPUT_REAL_DIR=$(readlink -f "$OUTPUT_DIR")
            CURRENT_REAL_DIR=$(readlink -f "./")
            if [ "$OUTPUT_REAL_DIR" == "$CURRENT_REAL_DIR" ]; then
                echo "'--output' is the same as the current directory. Please specify a different directory."
                exit 1
            fi
            ;;
        --symlink_target_path=*)
            SYMLINK_TARGET_PATH="${1#*=}"
            SYMLINK_ARGS="--symlink_target_path=$SYMLINK_TARGET_PATH"
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

SCRIPT_DIR=$(realpath "$(dirname "$0")")
GET_RES_CMD="$SCRIPT_DIR/scripts/get_resource.sh --src_path=$OBB_TEST_IMG_PATH --output=$OUTPUT_DIR $SYMLINK_ARGS $FORCE_ARGS --extract"
echo "Get Resources from remote server ..."
echo "$GET_RES_CMD"

$GET_RES_CMD
if [ $? -ne 0 ]; then
    echo "Get resource failed!"
    exit 1
fi

GET_RES_CMD="$SCRIPT_DIR/scripts/get_resource.sh --src_path=$TEST_VIDEO_PATH --output=$OUTPUT_DIR $SYMLINK_ARGS $FORCE_ARGS --extract"
echo "Get Resources from remote server ..."
echo "$GET_RES_CMD"

$GET_RES_CMD
if [ $? -ne 0 ]; then
    echo "Get resource failed!"
    exit 1
fi

echo "Cleaning up temporary files ..."
rm -rf download || true
rm -rf assets/download || true

echo "Setup assets completed."
exit 0
