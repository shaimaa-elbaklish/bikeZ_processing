#!/bin/bash

DATE="2025-06-16"
VEH_TYPE="bike"
FLAG="False"
SESSIONS=("AM1" "AM2" "AM3" "AM4" "AM5" "AM6" "PM1" "PM2" "PM3" "PM4" "PM5" "PM6")

for session in "${SESSIONS[@]}"; do
    echo "Running: $session"
    python main_coordinate_transform_D3_E.py "$DATE" "$VEH_TYPE" "$INTERSECTION" "$DIRECTION" "$session" "$FLAG"
done