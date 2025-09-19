@echo off
setlocal enabledelayedexpansion

REM Determine project root (directory containing pyproject.toml)
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

REM Optional first arg is run directory; otherwise pick latest under output\run-*
set "RUN_DIR="
set "ARG1=%~1"
if not "%ARG1%"=="" (
  set "FIRSTCHAR=%ARG1:~0,1%"
  if not "%FIRSTCHAR%"=="-" (
    set "RUN_DIR=%ARG1%"
    shift
  )
)

if "%RUN_DIR%"=="" (
  if exist "output" (
    for /f "delims=" %%R in ('dir /b /ad /o-n output\run-* 2^>nul') do (
      if not defined RUN_DIR set "RUN_DIR=output\%%R"
    )
  )
)

if "%RUN_DIR%"=="" (
  echo Usage: scripts\normalize_run.bat [output\run-YYYYmmdd-HHMMSS] [--force-commonalities] [--common-threshold 0.5]
  if defined DID_PUSHD popd
  exit /b 1
)

if not exist "%RUN_DIR%" (
  echo [ERROR] Run directory not found: %RUN_DIR%
  if defined DID_PUSHD popd
  exit /b 1
)

set "EXTRA_ARGS=%*"
set "PYTHONIOENCODING=utf-8"

echo Normalizing run: %RUN_DIR%

REM Iterate page directories once; prefer text_content.json over content.txt; warn if both missing
for /d %%D in ("%RUN_DIR%\*") do (
  if exist "%%D\content.json" (
    echo - %%D\content.json
    uv run python -m socora_crawler.normalize_content "%%D\content.json" %EXTRA_ARGS% > "%%D\content.md"
  ) else if exist "%%D\content.txt" (
    echo - %%D\content.txt
    uv run python -m socora_crawler.normalize_content "%%D\content.txt" %EXTRA_ARGS% > "%%D\content.md"
  ) else (
    echo [WARN] Missing content.json and content.txt: %%D
  )
)

echo Done.
if defined DID_PUSHD popd
endlocal
