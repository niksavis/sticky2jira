@echo off
REM Download Frontend Libraries Script
REM Downloads Bootstrap, DataTables, jQuery, and Socket.IO for offline use

echo ============================================================
echo Sticky2Jira - Download Frontend Libraries
echo ============================================================
echo.

REM Create directories
if not exist "static\libs\bootstrap-5.3.0" mkdir "static\libs\bootstrap-5.3.0\css"
if not exist "static\libs\bootstrap-5.3.0\js" mkdir "static\libs\bootstrap-5.3.0\js"
if not exist "static\libs\datatables-1.13" mkdir "static\libs\datatables-1.13\css"
if not exist "static\libs\datatables-1.13\js" mkdir "static\libs\datatables-1.13\js"
if not exist "static\libs\jquery-3.7.0" mkdir "static\libs\jquery-3.7.0"
if not exist "static\libs\socket.io-4.5.4" mkdir "static\libs\socket.io-4.5.4"

echo [1/6] Downloading Bootstrap 5.3.0...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css' -OutFile 'static\libs\bootstrap-5.3.0\css\bootstrap.min.css' }"
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js' -OutFile 'static\libs\bootstrap-5.3.0\js\bootstrap.bundle.min.js' }"

if %errorlevel% neq 0 (
    echo ERROR: Bootstrap download failed
    goto :error
)
echo Bootstrap 5.3.0 downloaded successfully

echo [2/6] Downloading jQuery 3.7.0...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://code.jquery.com/jquery-3.7.0.min.js' -OutFile 'static\libs\jquery-3.7.0\jquery.min.js' }"

if %errorlevel% neq 0 (
    echo ERROR: jQuery download failed
    goto :error
)
echo jQuery 3.7.0 downloaded successfully

echo [3/6] Downloading DataTables 1.13.4 CSS...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://cdn.datatables.net/1.13.4/css/dataTables.bootstrap5.min.css' -OutFile 'static\libs\datatables-1.13\css\dataTables.bootstrap5.min.css' }"

if %errorlevel% neq 0 (
    echo ERROR: DataTables CSS download failed
    goto :error
)
echo DataTables CSS downloaded successfully

echo [4/6] Downloading DataTables 1.13.4 JS...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://cdn.datatables.net/1.13.4/js/jquery.dataTables.min.js' -OutFile 'static\libs\datatables-1.13\js\jquery.dataTables.min.js' }"
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://cdn.datatables.net/1.13.4/js/dataTables.bootstrap5.min.js' -OutFile 'static\libs\datatables-1.13\js\dataTables.bootstrap5.min.js' }"

if %errorlevel% neq 0 (
    echo ERROR: DataTables JS download failed
    goto :error
)
echo DataTables JS downloaded successfully

echo [5/6] Downloading Socket.IO Client 4.5.4...
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://cdn.socket.io/4.5.4/socket.io.min.js' -OutFile 'static\libs\socket.io-4.5.4\socket.io.min.js' }"

if %errorlevel% neq 0 (
    echo ERROR: Socket.IO download failed
    goto :error
)
echo Socket.IO 4.5.4 downloaded successfully

echo [6/6] Verifying downloads...
set MISSING=0

if not exist "static\libs\bootstrap-5.3.0\css\bootstrap.min.css" set MISSING=1
if not exist "static\libs\bootstrap-5.3.0\js\bootstrap.bundle.min.js" set MISSING=1
if not exist "static\libs\jquery-3.7.0\jquery.min.js" set MISSING=1
if not exist "static\libs\datatables-1.13\css\dataTables.bootstrap5.min.css" set MISSING=1
if not exist "static\libs\datatables-1.13\js\jquery.dataTables.min.js" set MISSING=1
if not exist "static\libs\datatables-1.13\js\dataTables.bootstrap5.min.js" set MISSING=1
if not exist "static\libs\socket.io-4.5.4\socket.io.min.js" set MISSING=1

if %MISSING% equ 1 (
    echo ERROR: Some files are missing
    goto :error
)

echo.
echo ============================================================
echo SUCCESS: All frontend libraries downloaded successfully!
echo ============================================================
echo.
echo Libraries installed:
echo   - Bootstrap 5.3.0 (CSS + JS)
echo   - jQuery 3.7.0
echo   - DataTables 1.13.4 (CSS + JS)
echo   - Socket.IO Client 4.5.4
echo.
echo Location: static\libs\
echo.
pause
exit /b 0

:error
echo.
echo ============================================================
echo ERROR: Frontend library download failed
echo ============================================================
echo.
echo Please check your internet connection and try again.
echo Alternatively, download libraries manually:
echo.
echo Bootstrap: https://getbootstrap.com/docs/5.3/getting-started/download/
echo jQuery: https://jquery.com/download/
echo DataTables: https://datatables.net/download/
echo Socket.IO: https://cdn.socket.io/4.5.4/socket.io.min.js
echo.
pause
exit /b 1
