@echo off
setlocal enabledelayedexpansion

REM cd to project root
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "DID_PUSHD="
if exist "%ROOT_DIR%\pyproject.toml" (
  pushd "%ROOT_DIR%"
  set "DID_PUSHD=1"
) else if exist "pyproject.toml" (
  REM already in root
) else (
  for /f "delims=" %%i in ('git rev-parse --show-toplevel 2^>nul') do set "ROOT_DIR=%%i"
  if defined ROOT_DIR if exist "%ROOT_DIR%\pyproject.toml" (
    pushd "%ROOT_DIR%"
    set "DID_PUSHD=1"
  ) else (
    echo [WARN] Could not locate project root. Running from current dir.
  )
)

REM First arg is run dir; otherwise pick latest
set "RUN_DIR="
if not "%~1"=="" (
  set "RUN_DIR=%~1"
  shift
)
if "%RUN_DIR%"=="" (
  if exist "output" (
    for /f "delims=" %%R in ('dir /b /ad /o-n output\run-* 2^>nul') do (
      if not defined RUN_DIR set "RUN_DIR=output\%%R"
    )
  )
)
if "%RUN_DIR%"=="" (
  echo Usage: scripts\extract_links_run.bat [output\run-YYYYmmdd-HHMMSS]
  if defined DID_PUSHD popd
  exit /b 1
)

echo Extracting links for run: %RUN_DIR%
for /d %%D in ("%RUN_DIR%\*") do (
  if exist "%%D\content.json" (
    echo - %%D\content.json
    uv run python -m socora_crawler.extract_links "%%D\content.json" --write >nul
  ) else if exist "%%D\content.txt" (
    echo - %%D\content.txt
    uv run python -m socora_crawler.extract_links "%%D\content.txt" --write >nul
  ) else (
    echo - %%D (no content files; using metadata)
    uv run python -m socora_crawler.extract_links "%%D" --write >nul
  )
)

REM Aggregate JSONL
uv run python -m socora_crawler.aggregate_links "%RUN_DIR%"

if defined DID_PUSHD popd
endlocal
