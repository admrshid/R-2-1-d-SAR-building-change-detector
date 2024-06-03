@echo off

echo Setting up environment variables...
setlocal enabledelayedexpansion

rem Set path to desired python executable
echo Setting python executable...
set "python_executable=C:\Users\user\anaconda3\envs\MLpractice2\python.exe"

rem Set path to desired environment
echo Activating desired environment...
call C:\Users\user\anaconda3\envs\MLpractice2\Lib\venv\scripts\nt\activate.bat

rem change current dir to main repository
echo Changing directory...
cd C:\Users\user\Documents\R(2+1)d SAR building change detector

rem set path to which SAR data is saved to
echo Setting data path...
set "path=D:\Swindon_geotiffs_2\video_dataset"

rem set decision to form segmented processing
echo Setting segmentation...
set "segment=No"

rem set expected class names
echo Setting class names...
set "classes=change,no_change"

rem set desired frames
echo Setting frame parameters...
set "num_frames=5"
set "frame_step=11"

rem list out metrics to test
echo Setting metrics...
set "metrics=VV,VH,VH_COHERENCE,VV_COHERENCE"

echo Running Python script...
"%python_executable%" -m model_synthetic !path! !classes! !segment! !num_frames! !frame_step! !metrics!
