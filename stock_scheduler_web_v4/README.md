# Stock Scheduler Web v4

一个最小可用的可视化定时任务管理台，基于 FastAPI + APScheduler。

## 启动
pip install -r requirements.txt
export BATCH_API_URL=http://127.0.0.1:8000/batch/sync/daily
uvicorn main:app --host 0.0.0.0 --port 8010
