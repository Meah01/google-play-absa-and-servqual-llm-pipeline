@echo off
cd /d "C:\Users\User\Desktop\ABSA project"
echo Stopping ABSA Pipeline...
docker-compose down
echo Pipeline stopped successfully
pause