@echo off

set DATE=2025-06-17
set VEH_TYPE=vehicle
set INTERSECTION=D4
set CODE=F
set FLAG=False

for %%S in (AM1 AM2 AM3 AM4 AM5 AM6 PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_coordinate_transform_D4_F.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)