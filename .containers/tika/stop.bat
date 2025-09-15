@echo off
setlocal enabledelayedexpansion
REM Stop and remove the Tika container via docker-compose
cd /d %~dp0
docker-compose down
echo Tika has been stopped.
