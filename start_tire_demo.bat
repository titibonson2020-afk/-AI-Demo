@echo off
echo Starting Tire Industry AI Demo...
cd /d "%~dp0"
streamlit run tire_tech_writing_demo.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
pause