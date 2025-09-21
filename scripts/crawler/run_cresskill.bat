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

if not exist "%CRAWLER_DIR%\pyproject.toml" (
  echo [ERROR] Could not locate socora-crawler project at %CRAWLER_DIR%
  popd >nul
  exit /b 1
)

set "START_URLS=%~1"
if "%START_URLS%"=="" set "START_URLS=https://www.cresskillboro.com"

set "ALLOWED=%~2"
if "%ALLOWED%"=="" set "ALLOWED=edmundsassoc.com,edmundsgovtech.cloud,cresskilllibrary.org,bergencountynj.gov"

set "MAX_DEPTH=%~3"
if "%MAX_DEPTH%"=="" set "MAX_DEPTH=8"

set "UV_NO_SYNC=1"
set "UV_LINK_MODE=copy"

pushd "%CRAWLER_DIR%" >nul
set "CRAWLER_PUSHED=1"

uv run --no-sync scrapy crawl universal ^
  -a start_urls="%START_URLS%" ^
  -a allowed="%ALLOWED%" ^
  -a follow_links=true ^
  -a max_depth=%MAX_DEPTH% ^
  -a extractors="extractors.cresskill.meeting_documents" ^
  -s ROBOTSTXT_OBEY=true ^
  -s SCRAPY_OUTPUT_DIR=../output

goto :cleanup

:cleanup
if defined CRAWLER_PUSHED popd >nul
popd >nul
endlocal