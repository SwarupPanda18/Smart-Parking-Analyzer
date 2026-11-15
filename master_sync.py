"""
Master Supabase Synchronization Module for Mother's Hut Traffic & Gate Operations.
Syncs 5-minute interval average vehicle counts to the Master Supabase database:
    postgresql://postgres.vmdxwszdawcdluaignpl:The_Neotia_Uni@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres

Table: public.master_metrics
Key: window_start (timestamp without time zone)
Target column: vehicle_count (integer)

Rule:
- Windows align to 5-minute boundaries ending in 0 or 5 (e.g., 12:00, 12:05, 12:10, 12:15, ...).
- Average vehicle count in that 5-minute window from the actual Supabase ledger is written to vehicle_count.
- All other columns (hygiene_score, total_orders, food_quality_score, positive_sentiment, etc.) remain untouched.
- If window_start already exists: updates vehicle_count (replacing 0 with the actual average).
- If window_start does not exist: creates the interval row with vehicle_count and default values for other modules.
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv
import zoneinfo

load_dotenv()

logger = logging.getLogger("master_sync")

# India Standard Time (UTC+5:30) for Mother's Hut operational schedule
IST = zoneinfo.ZoneInfo("Asia/Kolkata")

# Master Supabase PostgreSQL Connection URL
DEFAULT_MASTER_URL = "postgresql://postgres.vmdxwszdawcdluaignpl:The_Neotia_Uni@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres"
MASTER_DB_URL = os.getenv("MASTER_DB_URL", DEFAULT_MASTER_URL).strip()

_daemon_started = False
_last_synced_window = None


def get_master_connection():
    """Returns a direct psycopg2 connection to Master Supabase."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    clean_url = MASTER_DB_URL
    if "@" in clean_url:
        # Check if password needs URL encoding
        pass

    conn = psycopg2.connect(
        clean_url,
        sslmode="require",
        connect_timeout=10
    )
    conn.autocommit = False
    return conn


def calculate_window_start(dt=None):
    """
    Calculates the 5-minute interval start timestamp ending in 0 or 5.
    Returns a naive datetime representing IST local time (matches Postgres timestamp without time zone).
    """
    if dt is None:
        dt = datetime.now(timezone.utc).astimezone(IST)
    elif dt.tzinfo is not None:
        dt = dt.astimezone(IST)
    else:
        # Assume local IST if naive
        dt = dt.replace(tzinfo=IST)

    b_min = (dt.minute // 5) * 5
    w_start = dt.replace(minute=b_min, second=0, microsecond=0)
    return w_start.replace(tzinfo=None)


def compute_average_vehicle_count(window_start_naive):
    """
    Queries traffic_volume_ledger in the actual Supabase database for the given 5-minute interval.
    Returns (avg_count, count_observations, counts_list).
    """
    import ledger_db
    window_end_naive = window_start_naive + timedelta(minutes=5)

    # Convert IST window to UTC for created_at query
    w_start_utc = window_start_naive.replace(tzinfo=IST).astimezone(timezone.utc)
    w_end_utc = window_end_naive.replace(tzinfo=IST).astimezone(timezone.utc)

    conn = None
    try:
        conn = ledger_db.get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT total_vehicles
                FROM public.traffic_volume_ledger
                WHERE (created_at >= %s AND created_at < %s)
                   OR (timestamp >= %s AND timestamp < %s);
            """, (w_start_utc, w_end_utc, window_start_naive, window_end_naive))
            rows = cur.fetchall()
            counts = [r[0] for r in rows if r[0] is not None]

            if counts:
                avg_count = int(round(sum(counts) / float(len(counts))))
                return avg_count, len(counts), counts
            else:
                return 0, 0, []
    except Exception as e:
        logger.error(f"[Master Sync] Error querying actual traffic volume: {e}")
        return 0, 0, []
    finally:
        if conn:
            conn.close()


def upsert_master_window(window_start_naive, avg_count=None):
    """
    Upserts or updates the 5-minute interval into public.master_metrics in Master Supabase.
    - If a row for window_start_naive already exists: updates vehicle_count.
    - If a team member created a row within +-5 minutes: updates that team row directly!
      (Never creates duplicate rows alongside team member models).
    - If no row exists and avg_count > 0: inserts the new interval.
    - If avg_count == 0 and no row exists: skips insertion to prevent empty ghost rows.
    """
    if avg_count is None:
        avg_count, _, _ = compute_average_vehicle_count(window_start_naive)

    conn = None
    try:
        conn = get_master_connection()
        with conn.cursor() as cur:
            # 1. Check for exact matching window
            cur.execute("""
                SELECT window_start, vehicle_count
                FROM public.master_metrics
                WHERE window_start = %s;
            """, (window_start_naive,))
            exact_row = cur.fetchone()

            target_window = window_start_naive
            if exact_row:
                cur.execute("""
                    UPDATE public.master_metrics
                    SET vehicle_count = %s
                    WHERE window_start = %s;
                """, (int(avg_count), window_start_naive))
            else:
                # 2. Check if a team row was created within +-5 minutes
                near_start = window_start_naive - timedelta(minutes=5)
                near_end = window_start_naive + timedelta(minutes=5)
                cur.execute("""
                    SELECT window_start, vehicle_count
                    FROM public.master_metrics
                    WHERE window_start >= %s AND window_start <= %s
                    ORDER BY ABS(EXTRACT(EPOCH FROM (window_start - %s))) ASC
                    LIMIT 1;
                """, (near_start, near_end, window_start_naive))
                near_row = cur.fetchone()

                if near_row:
                    target_window = near_row[0]
                    cur.execute("""
                        UPDATE public.master_metrics
                        SET vehicle_count = %s
                        WHERE window_start = %s;
                    """, (int(avg_count), target_window))
                elif avg_count > 0:
                    cur.execute("""
                        INSERT INTO public.master_metrics (
                            window_start,
                            vehicle_count,
                            total_orders,
                            hygiene_score,
                            food_quality_score,
                            positive_sentiment,
                            negative_sentiment,
                            customer_count
                        ) VALUES (
                            %s, %s, 0, 0.0, 0.0, 0.0, 0.0, 0
                        )
                        ON CONFLICT (window_start) DO UPDATE
                        SET vehicle_count = EXCLUDED.vehicle_count;
                    """, (window_start_naive, int(avg_count)))
                else:
                    # avg_count == 0 and no row exists: skip inserting ghost rows
                    return {
                        "window_start": window_start_naive.isoformat(),
                        "vehicle_count": 0,
                        "status": "skipped_empty"
                    }

            conn.commit()

        logger.info(f"[Master Sync] Updated Master DB: window={target_window.isoformat()} vehicle_count={avg_count}")
        return {
            "window_start": target_window.isoformat(),
            "vehicle_count": int(avg_count),
            "status": "synced"
        }
    except Exception as e:
        logger.error(f"[Master Sync] Failed to upsert window {window_start_naive}: {e}")
        if conn:
            conn.rollback()
        return {
            "window_start": window_start_naive.isoformat(),
            "vehicle_count": int(avg_count) if avg_count is not None else 0,
            "status": "error",
            "error": str(e)
        }
    finally:
        if conn:
            conn.close()


def sync_all_history():
    """
    Performs full catchup synchronization:
    Groups all traffic records from actual Supabase into 5-minute intervals and updates Master Supabase.
    """
    import ledger_db
    conn = None
    try:
        conn = ledger_db.get_db_connection()
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT created_at, total_vehicles
                FROM public.traffic_volume_ledger
                ORDER BY created_at ASC;
            """)
            rows = cur.fetchall()
    except Exception as e:
        logger.error(f"[Master Sync] Error reading actual volume ledger: {e}")
        return []
    finally:
        if conn:
            conn.close()

    if not rows:
        return []

    # Group into 5-minute buckets
    buckets = {}
    for r in rows:
        dt_ist = r["created_at"].astimezone(IST)
        b_min = (dt_ist.minute // 5) * 5
        w_start = dt_ist.replace(minute=b_min, second=0, microsecond=0).replace(tzinfo=None)
        if w_start not in buckets:
            buckets[w_start] = []
        buckets[w_start].append(r["total_vehicles"])

    results = []
    for w_start, counts in sorted(buckets.items()):
        avg_count = int(round(sum(counts) / float(len(counts))))
        if avg_count > 0:
            res = upsert_master_window(w_start, avg_count)
            results.append(res)

    logger.info(f"[Master Sync] Completed full history sync: {len(results)} intervals processed.")
    return results


def sync_current_and_previous_window():
    """
    Syncs the current active 5-min window and the immediately preceding completed window.
    Ensures that when a 5-minute boundary is crossed, the completed window is finalized.
    """
    now_ist = datetime.now(timezone.utc).astimezone(IST)
    current_w = calculate_window_start(now_ist)
    prev_w = current_w - timedelta(minutes=5)

    res_prev = upsert_master_window(prev_w)
    res_curr = upsert_master_window(current_w)

    return [res_prev, res_curr]


def notify_traffic_recorded():
    """
    Non-blocking notification hook called whenever new traffic is detected and recorded.
    Updates the current 5-minute window in a background worker thread.
    """
    def _async_worker():
        try:
            now_ist = datetime.now(timezone.utc).astimezone(IST)
            current_w = calculate_window_start(now_ist)
            upsert_master_window(current_w)
        except Exception as e:
            logger.warning(f"[Master Sync] Async traffic update note: {e}")

    t = threading.Thread(target=_async_worker, daemon=True)
    t.start()


def _daemon_loop():
    """
    Continuous background loop that checks every 10 seconds.
    Triggers synchronization at every 5-minute boundary ending in 0 or 5 (:00, :05, :10, :15, ...).
    """
    global _last_synced_window
    logger.info("[Master Sync] Daemon started. Monitoring 5-minute wall clock boundaries (:00, :05, :10, ...)")

    # Run initial catchup sync on startup
    try:
        sync_all_history()
    except Exception as e:
        logger.warning(f"[Master Sync] Initial catchup note: {e}")

    while True:
        try:
            now_ist = datetime.now(timezone.utc).astimezone(IST)
            current_window = calculate_window_start(now_ist)

            # Check if this 5-minute interval has already been synchronized
            if _last_synced_window != current_window:
                # Trigger sync for both current and previous window
                sync_current_and_previous_window()
                _last_synced_window = current_window
                logger.info(f"[Master Sync] 5-minute boundary sync executed for window: {current_window.isoformat()}")

        except Exception as e:
            logger.error(f"[Master Sync] Error in daemon loop: {e}")

        # Sleep for 10 seconds before checking wall clock again
        time.sleep(10)


def start_master_sync_daemon():
    """Starts the background 5-minute synchronization daemon if not already running."""
    global _daemon_started
    if _daemon_started:
        return
    _daemon_started = True
    thread = threading.Thread(target=_daemon_loop, daemon=True, name="MasterSyncDaemon")
    thread.start()


def get_master_status():
    """Returns status, connection health, and latest rows from Master Supabase."""
    conn = None
    try:
        conn = get_master_connection()
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT window_start, vehicle_count, hygiene_score, total_orders, food_quality_score
                FROM public.master_metrics
                ORDER BY window_start DESC
                LIMIT 12;
            """)
            recent_rows = cur.fetchall()

            cur.execute("SELECT COUNT(*) as total_intervals FROM public.master_metrics;")
            total_count = cur.fetchone()["total_intervals"]

        return {
            "status": "connected",
            "master_db_configured": True,
            "total_intervals": total_count,
            "recent_windows": [
                {
                    "window_start": r["window_start"].isoformat() if hasattr(r["window_start"], "isoformat") else str(r["window_start"]),
                    "vehicle_count": r["vehicle_count"],
                    "hygiene_score": float(r["hygiene_score"]) if r["hygiene_score"] is not None else 0.0,
                    "total_orders": r["total_orders"],
                    "food_quality_score": float(r["food_quality_score"]) if r["food_quality_score"] is not None else 0.0
                }
                for r in recent_rows
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "master_db_configured": True,
            "error": str(e)
        }
    finally:
        if conn:
            conn.close()
