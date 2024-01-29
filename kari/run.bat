call .\.venv\Scripts\activate
start jupyter lab

if %errorlevel% neq 0 (

    echo ***************************
    echo Error!!
    echo ***************************
    
    pause
    exit /b
)

echo ***************************
echo Success!!
echo ***************************

pause