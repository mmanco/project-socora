@echo off
setlocal enabledelayedexpansion

REM Find project root (folder containing pyproject.toml)
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "DID_PUSHD="

if exist "%ROOT_DIR%\pyproject.toml" (
  pushd "%ROOT_DIR%"
  set "DID_PUSHD=1"
) else if exist "pyproject.toml" (
  REM Already in project root
) else (
  for /f "delims=" %%i in ('git rev-parse --show-toplevel 2^>nul') do set "ROOT_DIR=%%i"
  if defined ROOT_DIR if exist "%ROOT_DIR%\pyproject.toml" (
    pushd "%ROOT_DIR%"
    set "DID_PUSHD=1"
  ) else (
    echo [WARN] Could not locate project root containing pyproject.toml. Running from current directory.
  )
)

REM Base directory to scan (default: output)
set "BASE_DIR=%~1"
if "%BASE_DIR%"=="" set "BASE_DIR=output"

if not exist "%BASE_DIR%" (
  echo [ERROR] Base directory not found: %BASE_DIR%
  if defined DID_PUSHD popd
  exit /b 2
)

REM Keep uv from re-syncing per invocation; avoid hardlink warnings
set "UV_NO_SYNC=1"
set "UV_LINK_MODE=copy"
set "PYTHONIOENCODING=utf-8"

uv run --no-sync python scripts/find_empty_content_md.py "%BASE_DIR%"

if defined DID_PUSHD popd
endlocal

