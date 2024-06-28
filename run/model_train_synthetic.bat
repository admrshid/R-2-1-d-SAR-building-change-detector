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

echo Running Python script...
"%python_executable%" -m model_synthetic
