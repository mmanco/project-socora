@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Resolve repository root two levels up from script directory
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

REM Determine run directory from args or pick latest run
set "RUN_DIR="
if not "%~1"=="" (
  set "RUN_DIR=%~1"
  shift
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
  echo Usage: scripts\crawler\extract_links_run.bat [output\run-YYYYmmdd-HHMMSS]
  popd >nul
  exit /b 1
)

for %%F in ("%RUN_DIR%") do set "RUN_DIR=%%~fF"
if not exist "%RUN_DIR%" (
  echo [ERROR] Run directory not found: %RUN_DIR%
  popd >nul
  exit /b 1
)

echo Extracting links for run: %RUN_DIR%
set "UV_NO_SYNC=1"
set "UV_LINK_MODE=copy"
pushd "%CRAWLER_DIR%" >nul
set "CRAWLER_PUSHED=1"

for /d %%D in ("%RUN_DIR%\*") do (
  if exist "%%D\content.json" (
    echo - %%D\content.json
    uv run --no-sync python -m socora_crawler.extract_links "%%D\content.json" --write >nul
  ) else if exist "%%D\content.txt" (
    echo - %%D\content.txt
    uv run --no-sync python -m socora_crawler.extract_links "%%D\content.txt" --write >nul
  ) else (
    echo - %%D (no content files; using metadata)
    uv run --no-sync python -m socora_crawler.extract_links "%%D" --write >nul
  )
)

uv run --no-sync python -m socora_crawler.aggregate_links "%RUN_DIR%"

goto :cleanup

:cleanup
if defined CRAWLER_PUSHED popd >nul
popd >nul
endlocal