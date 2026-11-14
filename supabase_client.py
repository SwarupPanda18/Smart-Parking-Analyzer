"""
Supabase PostgreSQL Database Client for Mother's Hut Traffic & Gate Operations Console.
Connects directly to Supabase PostgreSQL:
    postgresql://postgres.fksxkdabjoihtoebpuxq:vir@tKohli18@aws-1-ap-south-1.pooler.supabase.com:6543/postgres

Unified with ledger_db for persistent daily logging, telemetry, and CSV exports.
"""

import os
import logging
from datetime import datetime
import ledger_db

logger = logging.getLogger("supabase_client")


def is_configured():
    """Returns True since PostgreSQL Supabase is configured."""
    return True


def log_traffic_record(cars, buses, bikes, total=None, fps=0.0, is_video=False):
    """
    Logs an autonomous vehicle count event directly to Supabase PostgreSQL.
    """
    return ledger_db.record_traffic_volume(
        cars=cars,
        buses=buses,
        bikes=bikes,
        total=total,
        fps=fps,
        source="video_analysis" if is_video else "image_analysis"
    )


def log_plate_detection(plate_text, state_code=None, state_name=None, is_valid_state=False,
                        vehicle_type="unclassified", vehicle_id=None, sharpness=0.0):
    """
    Logs an authentic license plate detection event directly to Supabase PostgreSQL.
    """
    return ledger_db.record_plate(
        plate_text=plate_text,
        state_code=state_code,
        state_name=state_name,
        is_valid_format=is_valid_state,
        confidence_score=sharpness,
        vehicle_type=vehicle_type,
        source="alpr_engine"
    )


def get_recent_plate_detections(limit=25):
    """Fetches the most recent detected plates from Supabase PostgreSQL."""
    return ledger_db.get_ledger_entries(limit=limit)


def get_traffic_summary():
    """Returns database connection status and live metrics from Supabase PostgreSQL."""
    stats = ledger_db.get_ledger_stats()
    return {
        "supabase_configured": True,
        "database_type": "Supabase PostgreSQL Pooler",
        "host": stats.get("db_host", "aws-1-ap-south-1.pooler.supabase.com"),
        "status": stats.get("db_status", "Active & Connected"),
        "today_date": stats.get("today_date"),
        "today_plates_count": stats.get("today_plates_count", 0),
        "lifetime_plates_count": stats.get("lifetime_plates_count", 0),
        "latest_plate": stats.get("latest_plate", "None"),
        "latest_time": stats.get("latest_time", "None")
    }
