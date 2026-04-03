@echo off
cd /d E:\python_projects\mianbeifen
echo Starting evaluation...
E:\anaconda\envs\mianbeifen_env\python.exe -u run_lm_eval.py
echo.
echo === EVALUATION COMPLETE ===
echo Results saved to logs/lm_eval_results_sampled.json
pause
