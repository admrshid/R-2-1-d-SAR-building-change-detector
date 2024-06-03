@echo off
setlocal enabledelayedexpansion

rem Set path to desired environment
echo Activating desired environment...
call C:\Users\user\Documents\venv\osm_change\Scripts\activate.bat

rem change current dir to main repository
echo Changing directory...
cd C:\Users\user\Documents\R(2+1)d SAR building change detector

rem Set path to desired python executable
echo Setting python executable...
set "python_executable=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"

set "layer1=201610"
set "layer2=202310"
set "output_path=output_swindon"
set "min_latitude=51.52"
set "max_latitude=51.56"
set "min_longitude=-1.75"
set "max_longitude=-1.71"
set "minsize=80"
set "threshold=250"

rem Initialize counter to 0
set "counter=0"

rem Loop through each character in the location variable
for %%a in (!output_path!) do (
    set /a "counter+=1"
)

echo Length of location: !counter!

for /l %%i in (1,1,!counter!) do (
    set "index=0"
    set "index1=0"
    set "index2=0"
    for %%a in (!layer1!) do (
        set /a "index+=1"
        if !index! equ %%i (
            set "layer1_value=%%a"
            for %%b in (!layer2!) do (
                set /a "index1+=1"
                if !index1! equ %%i (
                    set "layer2_value=%%b"
                    for %%c in (!output_path!) do (
                        set /a "index2+=1"
                        if !index2! equ %%i (
                            set "output_path_value=%%c"
                            rem iterate.py opens config file and rewrites with new values (layer1, layer2, output_dir) for each iteration
                            python iterate.py !layer1_value! !layer2_value! !output_path_value! !min_latitude! !max_latitude! !min_longitude! !max_longitude!
                            rem runs the main script for detecting change
                            python -m osm_changes
                            rem filters out the artefacts according to a fixed threshold
                            python -m noise_remover !output_path_value! !minsize!
                            rem rearranges filtered files into classes
                            python -m osm_changes_class_save !output_path_value! !threshold!
                        )
                    )
                )
            )
        )
    )
)


