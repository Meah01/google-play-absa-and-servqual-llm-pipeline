@echo off
cd /d "C:\Users\User\Desktop\ABSA project"
echo Starting ABSA Pipeline...
docker-compose up -d
echo Waiting for services...
timeout /t 45 /nobreak > nul
echo Services ready. You can now run: python main.py run
echo.
echo Press any key to start the full pipeline...
pause > nul
python main.py run