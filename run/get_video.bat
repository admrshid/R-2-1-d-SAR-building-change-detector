@echo off

echo Setting up environment variables...
setlocal enabledelayedexpansion

rem Set path to desired python executable
echo Setting python executable...
set "python_executable=C:\Users\user\AppData\Local\Programs\Python\Python311\python.exe"

rem Set path to desired environment
echo Activating desired environment...
call C:\Users\user\Documents\venv\osm_change\Scripts\activate.bat

rem change current dir to main repository
echo Changing directory...
cd C:\Users\user\Documents\R(2+1)d SAR building change detector

rem set path to which SAR data is saved to
echo Setting data path...
set "path=D:\Swindon_geotiffs_2"

rem set decision to form segmented processing
echo Setting segmentation...
set "segment=Yes"

"%python_executable%" -m create_video !path! !segment!
