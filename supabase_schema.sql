-- ====================================================================
-- MOTHER'S HUT SMART TRAFFIC & GATE OPERATIONS CONSOLE
-- SUPABASE DATABASE SCHEMA (PostgreSQL)
-- ====================================================================
-- Instructions:
-- 1. Log in to your Supabase project dashboard (https://supabase.com).
-- 2. Open the "SQL Editor" tab from the left sidebar.
-- 3. Click "New Query", paste this entire script, and click "Run".
-- ====================================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- --------------------------------------------------------------------
-- 1. TABLE: traffic_logs (Real-time vehicle counts & traffic metrics)
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.traffic_logs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    cars_count INTEGER NOT NULL DEFAULT 0,
    buses_count INTEGER NOT NULL DEFAULT 0,
    bikes_count INTEGER NOT NULL DEFAULT 0,
    total_vehicles INTEGER NOT NULL DEFAULT 0,
    fps NUMERIC(5, 2) DEFAULT 0.0,
    is_video BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT TIMEZONE('utc'::TEXT, NOW())
);

-- Index for real-time telemetry queries
CREATE INDEX IF NOT EXISTS idx_traffic_logs_created_at 
    ON public.traffic_logs (created_at DESC);

-- --------------------------------------------------------------------
-- 2. TABLE: plate_detections (ALPR License Plate Records & Jurisdiction)
-- --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.plate_detections (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    plate_text VARCHAR(32) NOT NULL,
    state_code VARCHAR(8),
    state_name VARCHAR(64) DEFAULT 'Standard Registration',
    is_valid_state BOOLEAN DEFAULT FALSE,
    vehicle_type VARCHAR(32) DEFAULT 'unclassified',
    vehicle_id INTEGER,
    sharpness NUMERIC(7, 2) DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT TIMEZONE('utc'::TEXT, NOW())
);

-- Indexes for rapid plate search & time-series retrieval
CREATE INDEX IF NOT EXISTS idx_plate_detections_created_at 
    ON public.plate_detections (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_plate_detections_plate_text 
    ON public.plate_detections (plate_text);
CREATE INDEX IF NOT EXISTS idx_plate_detections_state_code 
    ON public.plate_detections (state_code);

-- --------------------------------------------------------------------
-- 3. ROW LEVEL SECURITY (RLS) POLICIES
-- --------------------------------------------------------------------
-- Enable RLS on both tables
ALTER TABLE public.traffic_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.plate_detections ENABLE ROW LEVEL SECURITY;

-- Allow anonymous inserts & reads (via Supabase anon public key)
CREATE POLICY "Allow public read on traffic_logs" 
    ON public.traffic_logs FOR SELECT 
    USING (true);

CREATE POLICY "Allow public insert on traffic_logs" 
    ON public.traffic_logs FOR INSERT 
    WITH CHECK (true);

CREATE POLICY "Allow public read on plate_detections" 
    ON public.plate_detections FOR SELECT 
    USING (true);

CREATE POLICY "Allow public insert on plate_detections" 
    ON public.plate_detections FOR INSERT 
    WITH CHECK (true);

-- --------------------------------------------------------------------
-- 4. REALTIME NOTIFICATIONS (Optional - for live dashboard sync)
-- --------------------------------------------------------------------
-- Add tables to supabase_realtime publication so web clients get live pushes
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_publication WHERE pubname = 'supabase_realtime') THEN
        ALTER PUBLICATION supabase_realtime ADD TABLE public.traffic_logs;
        ALTER PUBLICATION supabase_realtime ADD TABLE public.plate_detections;
    END IF;
EXCEPTION WHEN OTHERS THEN
    -- Fall through safely if publication is already subscribed
    NULL;
END $$;
