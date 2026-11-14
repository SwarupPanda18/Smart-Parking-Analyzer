"""
Autonomous PostgreSQL Cloud Ledger Database for Mother's Hut Traffic & Gate Operations.
Connected to Supabase PostgreSQL:
    postgresql://postgres.fksxkdabjoihtoebpuxq:vir@tKohli18@aws-1-ap-south-1.pooler.supabase.com:6543/postgres

Permanently archives all vehicle detections, license plate records, and traffic metrics in PostgreSQL.
Users can query, view, and export CSV logs for any date even days or weeks later.
"""

import os
import csv
import io
import time
import logging
from datetime import datetime, date
from urllib.parse import urlparse, unquote
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ledger_db")

# Default Supabase PostgreSQL connection details
DEFAULT_DB_URL = "postgresql://postgres.fksxkdabjoihtoebpuxq:vir%40tKohli18@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL).strip()

# Parsed connection parameters
DB_USER = os.getenv("POSTGRES_USER", "postgres.fksxkdabjoihtoebpuxq")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "vir@tKohli18")
DB_HOST = os.getenv("POSTGRES_HOST", "aws-1-ap-south-1.pooler.supabase.com")
DB_PORT = int(os.getenv("POSTGRES_PORT", "6543"))
DB_NAME = os.getenv("POSTGRES_DB", "postgres")

# If DATABASE_URL is set, parse it
if DATABASE_URL and "://" in DATABASE_URL:
    try:
        # Handle @ in password by replacing vir@tKohli18 with vir%40tKohli18 if needed
        clean_url = DATABASE_URL
        if "vir@tKohli18" in clean_url:
            clean_url = clean_url.replace("vir@tKohli18", "vir%40tKohli18")
        parsed = urlparse(clean_url)
        if parsed.username:
            DB_USER = unquote(parsed.username)
        if parsed.password:
            DB_PASS = unquote(parsed.password)
        if parsed.hostname:
            DB_HOST = parsed.hostname
        if parsed.port:
            DB_PORT = parsed.port
        if parsed.path:
            DB_NAME = parsed.path.lstrip("/")
    except Exception as e:
        logger.warning(f"Error parsing DATABASE_URL: {e}")

_db_initialized = False


def get_db_connection():
    """Returns a direct PostgreSQL connection with RealDictCursor."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        sslmode="require",
        connect_timeout=8
    )
    conn.autocommit = False
    return conn


def init_db():
    """Initializes tables, columns, and indexes in Supabase PostgreSQL."""
    global _db_initialized
    if _db_initialized:
        return True

    try:
        import psycopg2
        conn = get_db_connection()
        with conn.cursor() as cur:
            # 1. Plate ledger table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.plate_ledger (
                    id BIGSERIAL PRIMARY KEY,
                    log_date VARCHAR(16) NOT NULL,
                    timestamp VARCHAR(64) NOT NULL,
                    plate_text VARCHAR(32) NOT NULL,
                    state_code VARCHAR(8),
                    state_name VARCHAR(64),
                    is_valid_format BOOLEAN DEFAULT FALSE,
                    confidence_score NUMERIC(5, 2) DEFAULT 90.0,
                    vehicle_type VARCHAR(32) DEFAULT 'vehicle',
                    facility VARCHAR(64) DEFAULT 'Mother''s Hut Main Gate',
                    source VARCHAR(32) DEFAULT 'desktop',
                    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::TEXT, NOW())
                );
            """)

            # 2. Traffic volume table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS public.traffic_volume_ledger (
                    id BIGSERIAL PRIMARY KEY,
                    log_date VARCHAR(16) NOT NULL,
                    timestamp VARCHAR(64) NOT NULL,
                    cars_count INTEGER DEFAULT 0,
                    buses_count INTEGER DEFAULT 0,
                    bikes_count INTEGER DEFAULT 0,
                    total_vehicles INTEGER DEFAULT 0,
                    fps NUMERIC(5, 2) DEFAULT 0.0,
                    source VARCHAR(32) DEFAULT 'desktop',
                    created_at TIMESTAMPTZ DEFAULT TIMEZONE('utc'::TEXT, NOW())
                );
            """)

            # Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_ledger_date ON public.plate_ledger(log_date);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_plate_ledger_plate ON public.plate_ledger(plate_text);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_traffic_volume_date ON public.traffic_volume_ledger(log_date);")
            conn.commit()
            _db_initialized = True
            logger.info("[Database] Supabase PostgreSQL tables verified successfully.")
            return True
    except Exception as e:
        logger.warning(f"[Database] PostgreSQL init note: {e}")
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


# Try initializing on module import
try:
    init_db()
except Exception:
    pass


def record_plate(plate_text, state_code=None, state_name=None, is_valid_format=False,
                 confidence_score=90.0, vehicle_type="vehicle", source="desktop"):
    """
    Permanently records an authentic license plate detection into Supabase PostgreSQL.
    Includes smart debouncing to avoid duplicating the same vehicle plate within 25 seconds.
    """
    if not plate_text or len(plate_text.strip()) < 4:
        return False

    clean_text = plate_text.strip().upper()
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    iso_timestamp = now.isoformat()

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Smart debounce: check if the exact same plate was recorded in the last 25 seconds
            cur.execute("""
                SELECT timestamp FROM public.plate_ledger 
                WHERE plate_text = %s AND log_date = %s 
                ORDER BY id DESC LIMIT 1
            """, (clean_text, today_str))
            
            row = cur.fetchone()
            if row:
                try:
                    last_time = datetime.fromisoformat(row[0])
                    if (now - last_time).total_seconds() < 25:
                        return False
                except Exception:
                    pass

            cur.execute("""
                INSERT INTO public.plate_ledger (
                    log_date, timestamp, plate_text, state_code, state_name,
                    is_valid_format, confidence_score, vehicle_type, facility,
                    source, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                today_str,
                iso_timestamp,
                clean_text,
                (state_code or "")[:8],
                (state_name or "Standard Registration")[:64],
                bool(is_valid_format),
                float(confidence_score),
                str(vehicle_type or "vehicle")[:32],
                "Mother's Hut Main Gate",
                str(source)[:32]
            ))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"[PostgreSQL Record Plate Error]: {e}")
        return False
    finally:
        if conn:
            conn.close()


def record_traffic_volume(cars=0, buses=0, bikes=0, total=None, fps=0.0, source="desktop"):
    """
    Permanently records aggregate traffic counts into PostgreSQL volume ledger.
    """
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    iso_timestamp = now.isoformat()
    if total is None:
        total = cars + buses + bikes

    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.traffic_volume_ledger (
                    log_date, timestamp, cars_count, buses_count, bikes_count,
                    total_vehicles, fps, source, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                today_str,
                iso_timestamp,
                int(cars),
                int(buses),
                int(bikes),
                int(total),
                float(round(fps, 1)),
                str(source)[:32]
            ))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"[PostgreSQL Volume Error]: {e}")
        return False
    finally:
        if conn:
            conn.close()


def get_available_dates():
    """
    Returns a sorted list of all dates that have recorded ledger data in PostgreSQL,
    including the total count of plates detected on each date.
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT log_date, COUNT(*) as plate_count,
                       MIN(timestamp) as first_seen,
                       MAX(timestamp) as last_seen
                FROM public.plate_ledger
                GROUP BY log_date
                ORDER BY log_date DESC
            """)
            rows = cur.fetchall()
            return [
                {
                    "date": r[0],
                    "plate_count": r[1],
                    "first_seen": r[2],
                    "last_seen": r[3]
                }
                for r in rows
            ]
    except Exception as e:
        logger.error(f"[PostgreSQL Query Dates Error]: {e}")
        return []
    finally:
        if conn:
            conn.close()


def get_ledger_entries(date_filter=None, search=None, limit=200):
    """
    Fetches ledger entries from PostgreSQL.
    If date_filter is provided (e.g. '2026-11-15'), returns that date's records.
    If date_filter is None, defaults to today's date (or all recent if today is empty).
    """
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            query = """
                SELECT id, log_date, timestamp, plate_text, state_code, state_name,
                       is_valid_format, confidence_score, vehicle_type, facility, source
                FROM public.plate_ledger
                WHERE 1=1
            """
            params = []

            if date_filter and date_filter.lower() != "all":
                query += " AND log_date = %s"
                params.append(date_filter)

            if search:
                query += " AND (plate_text ILIKE %s OR state_name ILIKE %s)"
                params.extend([f"%{search}%", f"%{search}%"])

            query += " ORDER BY id DESC LIMIT %s"
            params.append(limit)

            cur.execute(query, tuple(params))
            rows = cur.fetchall()

            # If today has no records yet and no specific date was asked, load most recent records
            if not rows and date_filter is None:
                cur.execute("""
                    SELECT id, log_date, timestamp, plate_text, state_code, state_name,
                           is_valid_format, confidence_score, vehicle_type, facility, source
                    FROM public.plate_ledger
                    ORDER BY id DESC LIMIT %s
                """, (limit,))
                rows = cur.fetchall()

            return [
                {
                    "id": r[0],
                    "date": r[1],
                    "timestamp": r[2],
                    "plate_text": r[3],
                    "state_code": r[4] or "",
                    "state_name": r[5] or "Standard Registration",
                    "is_valid_format": bool(r[6]),
                    "confidence_score": float(r[7]) if r[7] is not None else 90.0,
                    "vehicle_type": r[8] or "vehicle",
                    "facility": r[9] or "Mother's Hut Main Gate",
                    "source": r[10] or "desktop"
                }
                for r in rows
            ]
    except Exception as e:
        logger.error(f"[PostgreSQL Query Entries Error]: {e}")
        return []
    finally:
        if conn:
            conn.close()


def export_ledger_csv(date_filter=None):
    """
    Generates a full CSV string of the ledger from PostgreSQL for any date (or 'all' for full history).
    Guarantees that records can be exported anytime even days or weeks later.
    """
    entries = get_ledger_entries(date_filter=date_filter, limit=10000)

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Index",
        "Registration Number",
        "State Code",
        "State / Jurisdiction",
        "RTO Format",
        "Confidence Score",
        "Vehicle Type",
        "Log Date",
        "Timestamp",
        "Facility",
        "Capture Source"
    ])

    for idx, e in enumerate(entries, 1):
        writer.writerow([
            idx,
            e["plate_text"],
            e["state_code"] or "IND",
            e["state_name"] or "Standard Registration",
            "Verified" if e["is_valid_format"] else "Unverified",
            f"{e['confidence_score']:.1f}%",
            e["vehicle_type"],
            e["date"],
            e["timestamp"],
            e["facility"],
            e["source"]
        ])

    return output.getvalue()


def get_ledger_stats():
    """
    Returns summary statistics from PostgreSQL: total lifetime plates,
    today's plates, total distinct dates logged, and latest recorded plate.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Lifetime count
            cur.execute("SELECT COUNT(*) FROM public.plate_ledger")
            total_lifetime = cur.fetchone()[0]

            # Today's count
            cur.execute("SELECT COUNT(*) FROM public.plate_ledger WHERE log_date = %s", (today_str,))
            today_total = cur.fetchone()[0]

            # Total distinct days
            cur.execute("SELECT COUNT(DISTINCT log_date) FROM public.plate_ledger")
            total_days = cur.fetchone()[0]

            # Latest plate
            cur.execute("SELECT plate_text, timestamp FROM public.plate_ledger ORDER BY id DESC LIMIT 1")
            latest_row = cur.fetchone()
            latest_plate = latest_row[0] if latest_row else "None"
            latest_time = latest_row[1] if latest_row else "None"

            return {
                "db_type": "Supabase PostgreSQL Cloud Database",
                "db_host": DB_HOST,
                "db_status": "Active & Connected",
                "today_date": today_str,
                "today_plates_count": today_total,
                "lifetime_plates_count": total_lifetime,
                "days_archived_count": total_days,
                "latest_plate": latest_plate,
                "latest_time": latest_time
            }
    except Exception as e:
        return {
            "db_type": "Supabase PostgreSQL Cloud Database",
            "db_host": DB_HOST,
            "db_status": f"Connection Error: {e}",
            "today_date": today_str,
            "today_plates_count": 0,
            "lifetime_plates_count": 0,
            "days_archived_count": 0,
            "latest_plate": "None",
            "latest_time": "None"
        }
    finally:
        if conn:
            conn.close()
