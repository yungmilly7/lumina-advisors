@echo off
echo ================================================
echo   Lumina Advisors - expanded universe (port 8011)
echo   448 companies, retry/backoff, incremental
echo   ingestion, scheduled background refresh,
echo   per-field data health/live-vs-synthetic badges
echo ================================================
echo.
cd /d "%~dp0"
set STOCKGRAPH_PORT=8011
for %%p in (python py) do (
    where %%p >nul 2>nul && set PYEXE=%%p
)
echo Using %PYEXE%, listening on http://localhost:8011
echo Leave this window OPEN while you use the site.
echo.
%PYEXE% run.py
pause
