@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%..\.." >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Unable to resolve repository root from %SCRIPT_DIR%
  exit /b 1
)
set "REPO_ROOT=%CD%"
set "CRAWLER_DIR=%REPO_ROOT%\socora-crawler"
set "RUN_ROOT=%REPO_ROOT%\output"

if not exist "%CRAWLER_DIR%\pyproject.toml" (
  echo [ERROR] Could not locate socora-crawler project at %CRAWLER_DIR%
  popd >nul
  exit /b 1
)

set "RUN_DIR="
if not "%~1"=="" (
  set "ARG1=%~1"
  if not "%ARG1:~0,1%"=="-" (
    set "RUN_DIR=%ARG1%"
    shift
  )
)

if defined RUN_DIR (
  if not exist "%RUN_DIR%" (
    if exist "%REPO_ROOT%\%RUN_DIR%" (
      set "RUN_DIR=%REPO_ROOT%\%RUN_DIR%"
    )
  )
)

if not defined RUN_DIR (
  if exist "%RUN_ROOT%" (
    for /f "delims=" %%R in ('dir /b /ad /o-n "%RUN_ROOT%\run-*" 2^>nul') do (
      if not defined RUN_DIR set "RUN_DIR=%RUN_ROOT%\%%R"
    )
  )
)

if not defined RUN_DIR (
  echo Usage: scripts\crawler\normalize_run.bat [output\run-YYYYmmdd-HHMMSS] [--force-commonalities] [--common-threshold 0.5]
  popd >nul
  exit /b 1
)

for %%F in ("%RUN_DIR%") do set "RUN_DIR=%%~fF"
if not exist "%RUN_DIR%" (
  echo [ERROR] Run directory not found: %RUN_DIR%
  popd >nul
  exit /b 1
)

set "EXTRA_ARGS=%*"
set "PYTHONIOENCODING=utf-8"
set "UV_NO_SYNC=1"
set "UV_LINK_MODE=copy"

echo Normalizing run: %RUN_DIR%

pushd "%CRAWLER_DIR%" >nul
set "CRAWLER_PUSHED=1"

for /d %%D in ("%RUN_DIR%\*") do (
  if exist "%%D\content.json" (
    echo - %%D\content.json
    uv run --no-sync python -m socora_crawler.normalize_content "%%D\content.json" %EXTRA_ARGS% > "%%D\content.md"
  ) else if exist "%%D\content.txt" (
    echo - %%D\content.txt
    uv run --no-sync python -m socora_crawler.normalize_content "%%D\content.txt" %EXTRA_ARGS% > "%%D\content.md"
  ) else (
    echo [WARN] Missing content.json and content.txt: %%D
  )
)

echo Done.

goto :cleanup

:cleanup
if defined CRAWLER_PUSHED popd >nul
popd >nul
endlocal