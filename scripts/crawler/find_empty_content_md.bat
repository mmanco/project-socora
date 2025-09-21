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
set "HELPER_PATH=%REPO_ROOT%\scripts\crawler\find_empty_content_md.py"

if not exist "%CRAWLER_DIR%\pyproject.toml" (
  echo [ERROR] Could not locate socora-crawler project at %CRAWLER_DIR%
  popd >nul
  exit /b 1
)

set "BASE_DIR=%~1"
if "%BASE_DIR%"=="" set "BASE_DIR=%REPO_ROOT%\output"

if not "%BASE_DIR:~1,1%"==":" (
  if exist "%REPO_ROOT%\%BASE_DIR%" (
    set "BASE_DIR=%REPO_ROOT%\%BASE_DIR%"
  )
)

if not exist "%BASE_DIR%" (
  echo [ERROR] Base directory not found: %BASE_DIR%
  popd >nul
  exit /b 2
)

set "UV_NO_SYNC=1"
set "UV_LINK_MODE=copy"
set "PYTHONIOENCODING=utf-8"

pushd "%CRAWLER_DIR%" >nul
set "CRAWLER_PUSHED=1"

uv run --no-sync python "%HELPER_PATH%" "%BASE_DIR%"

goto :cleanup

:cleanup
if defined CRAWLER_PUSHED popd >nul
popd >nul
endlocal