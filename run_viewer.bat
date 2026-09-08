@echo off
REM DepthWizard viewer launcher.
REM
REM Why this file exists: index.html loads as an ES module and fetches the scene binaries.
REM Both are blocked by the browser under file://, so double-clicking index.html shows an
REM empty page. This serves the folder over http and adds the upload endpoint.
REM
REM NOTE FOR EDITORS: this file must keep CRLF line endings. cmd.exe misparses a .bat
REM saved with Unix newlines -- the parenthesised blocks below break and it exits with
REM "not recognized as an internal or external command".

setlocal
cd /d "%~dp0"
set "PORT=8080"
set "PY="

REM Prefer the project virtualenv. The system Python on PATH will run the server (it is
REM stdlib only) but has no torch or rasterio, so uploads would fail at inference with a
REM confusing traceback instead of working.
if exist "..\.venv\Scripts\python.exe" set "PY=..\.venv\Scripts\python.exe"
if defined PY goto haspy

where python >nul 2>nul
if not errorlevel 1 set "PY=python"
if defined PY goto haspy

where py >nul 2>nul
if not errorlevel 1 set "PY=py"
if defined PY goto haspy

goto nopython

:haspy
echo.
echo   DepthWizard viewer
echo   ------------------
echo   Serving on http://localhost:%PORT%/
echo   Python: %PY%
echo   Leave this window open. Press Ctrl+C here to stop.
echo.

start "" "http://localhost:%PORT%/"
"%PY%" tools\serve_viewer.py --port %PORT%
if errorlevel 1 goto fallback
goto done

:fallback
echo.
echo   The upload server did not start. Serving the built-in scenes only.
echo.
"%PY%" -m http.server %PORT% -d viewer
goto done

:nopython
echo.
echo   Python was not found on this machine.
echo.
echo   Either install Python 3 from https://python.org and run this file again,
echo   or open the single-file build instead:  viewer_standalone.html
echo.
pause

:done
endlocal
