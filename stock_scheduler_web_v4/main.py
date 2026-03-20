
import json
import os
import sqlite3
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, Form, HTTPException, Request, Query
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

APP_DIR = Path(__file__).resolve().parent
DB_FILE = APP_DIR / "scheduler.db"

BATCH_API_URL = os.getenv("BATCH_API_URL", "http://127.0.0.1:8000/batch/sync/daily")
STOCK_API_BASE_URL = os.getenv("STOCK_API_BASE_URL", "http://127.0.0.1:8000")
CN_TZ = ZoneInfo("Asia/Shanghai")

app = FastAPI(title="Stock Scheduler Web")
static_dir = APP_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

scheduler = BackgroundScheduler(timezone="Asia/Shanghai")


def cn_now() -> datetime:
    return datetime.now(CN_TZ)


def cn_now_str() -> str:
    return cn_now().strftime("%Y-%m-%d %H:%M:%S")


def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        symbols_json TEXT NOT NULL,
        window_days INTEGER NOT NULL DEFAULT 7,
        sync_adj_factor INTEGER NOT NULL DEFAULT 1,
        hour INTEGER NOT NULL DEFAULT 18,
        minute INTEGER NOT NULL DEFAULT 30,
        weekdays TEXT NOT NULL DEFAULT 'mon-fri',
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS run_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL,
        run_at TEXT NOT NULL,
        status TEXT NOT NULL,
        message TEXT,
        response_text TEXT
    )
    """)
    conn.commit()
    conn.close()


def log_run(task_id: int, status: str, message: str, response_text: str = ""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO run_logs (task_id, run_at, status, message, response_text) VALUES (?, ?, ?, ?, ?)",
        (task_id, cn_now_str(), status, message, response_text[:5000]),
    )
    conn.commit()
    conn.close()


def load_task(task_id: int):
    conn = get_conn()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return row


def parse_symbols_text(symbols_text: str):
    normalized_text = (
        symbols_text
        .replace("，", ",")
        .replace("；", ",")
        .replace(";", ",")
        .replace("\t", ",")
        .replace("\n", ",")
    )
    raw_symbols = [s.strip() for s in normalized_text.split(",")]
    symbols = [s.strip().strip('"').strip("'").upper() for s in raw_symbols if s.strip()]
    if not symbols:
        raise HTTPException(status_code=400, detail="symbols 不能为空")

    symbol_pattern = re.compile(r"^\d{6}\.(SZ|SH|BJ)$")
    invalid_symbols = [s for s in symbols if not symbol_pattern.match(s)]
    if invalid_symbols:
        raise HTTPException(
            status_code=400,
            detail=(
                "股票代码格式错误，需与 Batch Sync Daily 保持一致，例如 "
                '"000001.SZ", "000002.SZ", "000004.SZ"；错误项: ' + ", ".join(invalid_symbols[:20])
            ),
        )
    return symbols


def format_response_summary(payload):
    if isinstance(payload, dict):
        summary_keys = ["message", "total", "success", "no_data", "error", "sync_adj_factor"]
        summary = {k: payload.get(k) for k in summary_keys if k in payload}
        if summary:
            return json.dumps(summary, ensure_ascii=False)
        payload = {k: v for k, v in payload.items() if k != "results"}
        return json.dumps(payload, ensure_ascii=False)
    if isinstance(payload, list):
        return json.dumps(payload, ensure_ascii=False)
    return str(payload)



def run_task(task_id: int):
    task = load_task(task_id)
    if not task:
        return
    symbols = json.loads(task["symbols_json"])
    end_date = cn_now().strftime("%Y%m%d")
    start_date = (cn_now() - timedelta(days=int(task["window_days"]))).strftime("%Y%m%d")
    payload = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "sync_adj_factor": bool(task["sync_adj_factor"]),
    }
    try:
        resp = requests.post(BATCH_API_URL, json=payload, timeout=3600)
        status = "ok" if resp.ok else "error"
        message = f"HTTP {resp.status_code}"
        try:
            response_text = format_response_summary(resp.json())
        except Exception:
            response_text = resp.text
        log_run(task_id, status, message, response_text)
    except Exception as e:
        log_run(task_id, "error", str(e), "")


def schedule_task(task):
    job_id = f"task_{task['id']}"
    try:
        scheduler.remove_job(job_id)
    except Exception:
        pass

    if not int(task["enabled"]):
        return

    scheduler.add_job(
        run_task,
        trigger="cron",
        id=job_id,
        replace_existing=True,
        args=[int(task["id"])],
        day_of_week=task["weekdays"],
        hour=int(task["hour"]),
        minute=int(task["minute"]),
    )


def reload_all_jobs():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM tasks").fetchall()
    conn.close()
    for row in rows:
        schedule_task(row)


@app.on_event("startup")
def startup_event():
    init_db()
    if not scheduler.running:
        scheduler.start()
    reload_all_jobs()


@app.on_event("shutdown")
def shutdown_event():
    if scheduler.running:
        scheduler.shutdown(wait=False)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    conn = get_conn()
    tasks = conn.execute("SELECT * FROM tasks ORDER BY id DESC").fetchall()
    logs = conn.execute("""
        SELECT rl.*, t.name
        FROM run_logs rl
        LEFT JOIN tasks t ON rl.task_id = t.id
        ORDER BY rl.id DESC
        LIMIT 30
    """).fetchall()
    conn.close()

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tasks": tasks,
            "logs": logs,
            "batch_api_url": BATCH_API_URL,
            "stock_api_base_url": STOCK_API_BASE_URL,
        },
    )


@app.get("/screen/new-high", response_class=HTMLResponse)
def screen_new_high_page(request: Request):
    return templates.TemplateResponse(
        "screen_new_high.html",
        {
            "request": request,
            "stock_api_base_url": STOCK_API_BASE_URL,
        },
    )


@app.post("/api/proxy/screen/new-high")
def proxy_screen_new_high(
    trade_date: str = Query(...),
    trading_days: int = Query(250),
    recent_days: int = Query(5),
    adj: str = Query("qfq"),
    price_field: str = Query("high"),
    include_name: bool = Query(True),
):
    url = f"{STOCK_API_BASE_URL}/screen/new-high"
    params = {
        "trade_date": trade_date,
        "trading_days": trading_days,
        "recent_days": recent_days,
        "adj": adj,
        "price_field": price_field,
        "include_name": str(include_name).lower(),
    }
    try:
        resp = requests.get(url, params=params, timeout=300)
        try:
            payload = resp.json()
        except Exception:
            payload = {"detail": resp.text or "upstream returned non-json response"}
        return JSONResponse(status_code=resp.status_code, content=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"proxy request failed: {e}")


@app.get("/tasks/new", response_class=HTMLResponse)
def new_task_page(request: Request):
    return templates.TemplateResponse(
        "new_task.html",
        {
            "request": request,
            "page_title": "新建任务",
            "form_action": "/tasks/new",
            "submit_label": "保存任务",
            "task": None,
            "symbols_text": "",
        },
    )


@app.post("/tasks/new")
def create_task(
    name: str = Form(...),
    symbols_text: str = Form(...),
    window_days: int = Form(7),
    sync_adj_factor: Optional[str] = Form(None),
    hour: int = Form(18),
    minute: int = Form(30),
    weekdays: str = Form("mon-fri"),
    enabled: Optional[str] = Form(None),
):
    symbols = parse_symbols_text(symbols_text)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO tasks (name, symbols_json, window_days, sync_adj_factor, hour, minute, weekdays, enabled, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name.strip(),
            json.dumps(symbols, ensure_ascii=False),
            window_days,
            1 if sync_adj_factor else 0,
            hour,
            minute,
            weekdays,
            1 if enabled else 0,
            cn_now_str(),
        ),
    )
    task_id = cur.lastrowid
    conn.commit()
    conn.close()

    task = load_task(task_id)
    schedule_task(task)
    return RedirectResponse("/", status_code=303)




@app.get("/tasks/{task_id}/edit", response_class=HTMLResponse)
def edit_task_page(task_id: int, request: Request):
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    symbols = json.loads(task["symbols_json"])
    return templates.TemplateResponse(
        "new_task.html",
        {
            "request": request,
            "page_title": "编辑任务",
            "form_action": f"/tasks/{task_id}/edit",
            "submit_label": "保存修改",
            "task": task,
            "symbols_text": ", ".join([f'"{s}"' for s in symbols]),
        },
    )


@app.post("/tasks/{task_id}/edit")
def update_task(
    task_id: int,
    name: str = Form(...),
    symbols_text: str = Form(...),
    window_days: int = Form(7),
    sync_adj_factor: Optional[str] = Form(None),
    hour: int = Form(18),
    minute: int = Form(30),
    weekdays: str = Form("mon-fri"),
    enabled: Optional[str] = Form(None),
):
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")

    symbols = parse_symbols_text(symbols_text)

    conn = get_conn()
    conn.execute(
        """
        UPDATE tasks
        SET name = ?, symbols_json = ?, window_days = ?, sync_adj_factor = ?, hour = ?, minute = ?, weekdays = ?, enabled = ?
        WHERE id = ?
        """,
        (
            name.strip(),
            json.dumps(symbols, ensure_ascii=False),
            window_days,
            1 if sync_adj_factor else 0,
            hour,
            minute,
            weekdays,
            1 if enabled else 0,
            task_id,
        ),
    )
    conn.commit()
    conn.close()

    task = load_task(task_id)
    schedule_task(task)
    return RedirectResponse("/", status_code=303)

@app.post("/tasks/{task_id}/toggle")
def toggle_task(task_id: int):
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    new_enabled = 0 if int(task["enabled"]) else 1
    conn = get_conn()
    conn.execute("UPDATE tasks SET enabled = ? WHERE id = ?", (new_enabled, task_id))
    conn.commit()
    conn.close()
    task = load_task(task_id)
    schedule_task(task)
    return RedirectResponse("/", status_code=303)


@app.post("/tasks/{task_id}/run")
def run_task_now(task_id: int):
    task = load_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    run_task(task_id)
    return RedirectResponse("/", status_code=303)


@app.post("/tasks/{task_id}/delete")
def delete_task(task_id: int):
    try:
        scheduler.remove_job(f"task_{task_id}")
    except Exception:
        pass
    conn = get_conn()
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.execute("DELETE FROM run_logs WHERE task_id = ?", (task_id,))
    conn.commit()
    conn.close()
    return RedirectResponse("/", status_code=303)


@app.get("/screen/limit-up-high-shrink-volume", response_class=HTMLResponse)
def screen_limit_up_high_shrink_volume_page(request: Request):
    return templates.TemplateResponse(
        "screen_limit_up_high_shrink_volume.html",
        {
            "request": request,
            "stock_api_base_url": STOCK_API_BASE_URL,
        },
    )


@app.post("/api/proxy/screen/limit-up-high-shrink-volume")
def proxy_screen_limit_up_high_shrink_volume(
    trade_date: str = Query(...),
    trading_days: int = Query(60),
    adj: str = Query("qfq"),
    include_name: bool = Query(True),
    limit_up_lookback_days: int = Query(20),
):
    url = f"{STOCK_API_BASE_URL}/screen/limit-up-high-shrink-volume"
    params = {
        "trade_date": trade_date,
        "trading_days": trading_days,
        "adj": adj,
        "include_name": str(include_name).lower(),
        "limit_up_lookback_days": limit_up_lookback_days,
    }
    try:
        resp = requests.get(url, params=params, timeout=300)
        try:
            payload = resp.json()
        except Exception:
            payload = {"detail": resp.text or "upstream returned non-json response"}
        return JSONResponse(status_code=resp.status_code, content=payload)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"proxy request failed: {e}")
