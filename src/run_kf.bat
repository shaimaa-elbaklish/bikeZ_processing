@echo off

set VEH_TYPE=vehicle
set FLAG=False


GOTO:skip1
set DATE=2025-06-16


set INTERSECTION=D1
set CODE=A
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6 PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D2
set CODE=G
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D2
set CODE=C
for %%S in (PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D3
set CODE=E
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6 PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D4
set CODE=F
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6 PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)



set DATE=2025-06-17


set INTERSECTION=D1
set CODE=A
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D1
set CODE=B
for %%S in (PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D2
set CODE=C
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6 PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D3
set CODE=E
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6 PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D4
set CODE=F
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6 PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)
:skip1


set DATE=2025-09-29


GOTO:skip2
set INTERSECTION=D1
set CODE=A
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D1
set CODE=C
for %%S in (PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)
:skip2

set INTERSECTION=D1
set CODE=C
for %%S in (PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D2
set CODE=B
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D2
set CODE=E
for %%S in (PM1 PM2 PM3 PM4 PM5 PM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)



set DATE=2025-09-30


set INTERSECTION=D1
set CODE=H
for %%S in (PM1 PM2 PM3) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D2
set CODE=I
for %%S in (PM1 PM2 PM3) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)


set INTERSECTION=D1
set CODE=G
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

set INTERSECTION=D2
set CODE=F
for %%S in (AM1 AM2 AM3 AM4 AM5 AM6) do (
    echo Running: %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
    python main_kf.py %DATE% %VEH_TYPE% %INTERSECTION% %CODE% %%S %FLAG%
)

