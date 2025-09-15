@echo off
setlocal enabledelayedexpansion
REM Run Apache Tika server in detached mode via docker-compose
cd /d %~dp0
docker-compose up -d
echo Tika is starting on http://localhost:9998
