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
  REM Try to detect git root as fallback
  for /f "delims=" %%i in ('git rev-parse --show-toplevel 2^>nul') do set "ROOT_DIR=%%i"
  if defined ROOT_DIR if exist "%ROOT_DIR%\pyproject.toml" (
    pushd "%ROOT_DIR%"
    set "DID_PUSHD=1"
  ) else (
    echo [WARN] Could not locate project root containing pyproject.toml. Running from current directory.
  )
)

REM Allow overrides via args: %1=start_urls, %2=allowed, %3=max_depth
set "START_URLS=%~1"
if "%START_URLS%"=="" set "START_URLS=https://www.cresskillboro.com"

set "ALLOWED=%~2"
if "%ALLOWED%"=="" set "ALLOWED=edmundsassoc.com,edmundsgovtech.cloud,cresskilllibrary.org,bergencountynj.gov"

set "MAX_DEPTH=%~3"
if "%MAX_DEPTH%"=="" set "MAX_DEPTH=8"

REM Run Scrapy Universal spider via uv on Windows
uv run scrapy crawl universal ^
  -a start_urls="%START_URLS%" ^
  -a allowed="%ALLOWED%" ^
  -a follow_links=true -a max_depth=%MAX_DEPTH% ^
  -s ROBOTSTXT_OBEY=true ^
  -s SCRAPY_OUTPUT_DIR=./output

if defined DID_PUSHD popd
endlocal
