@echo off

set DATE=2025-09-29
set VEH_TYPE=vehicle
set INTERSECTION=D1
set CODE=C
set FLAG=False

for %%S in (PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_coordinate_transform_Sep.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)