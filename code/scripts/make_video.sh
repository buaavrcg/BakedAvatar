#!/bin/bash

function make_video() {
    # Parse input arguments
    IMAGE_DIR=$1
    OUTPUT_VIDEO=${2:-"$IMAGE_DIR.mp4"}
    PREFIX=${3:-""}
    FRAME_RATE=${4:-"25"}

    if [ -n "$PREFIX" ]; then
        PREFIX="${PREFIX}_"
    fi

    # Convert images to video using ffmpeg
    ffmpeg -i "$IMAGE_DIR/${PREFIX}%d.png" \
        -framerate $FRAME_RATE \
        -c:v libx264 \
        -pix_fmt yuv420p \
        -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        $OUTPUT_VIDEO -y
}

RESULT_DIR=$1

make_video $RESULT_DIR/rgb
make_video $RESULT_DIR/normal
make_video $RESULT_DIR/rendering $RESULT_DIR/rendering_manifold.mp4 manifold
make_video $RESULT_DIR/rendering $RESULT_DIR/rendering_result.mp4 result