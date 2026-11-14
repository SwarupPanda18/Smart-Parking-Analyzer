# ====================================================================
# MOTHER'S HUT SMART TRAFFIC & GATE OPERATIONS CONSOLE
# PRODUCTION CLOUD DEPLOYMENT & SUPABASE DATABASE GUIDE
# ====================================================================

This guide details how to deploy the traffic and license plate recognition system to a cloud server, connect it to **Supabase** (PostgreSQL cloud database), and link/embed the model icon into your main website.

---

## 📁 Production File Structure

```text
smart_traffic/
├── .env                     # Production environment variables & secrets (ignored by git)
├── .env.example             # Clean template of all environment variables
├── .gitignore               # Production git exclusion rules
├── requirements.txt         # Clean, locked production dependencies
├── supabase_client.py       # Supabase database client (with REST fallback)
├── supabase_schema.sql      # Database schema script for Supabase SQL Editor
├── download_models.py       # Pre-fetches OCR & ALPR model weights
├── app.py                   # FastAPI production backend & inference pipeline
├── license_plate_detector.pt# YOLOv8 license plate localization model
├── yolov8n.pt               # YOLOv8 multi-class vehicle detector
├── models/
│   └── easyocr/             # CRAFT & English BiLSTM weights (100% offline)
├── static/                  # Desktop operations console web application
├── mobile/                  # Mobile back-camera gate officer web app
└── website_integration/     # Website embed widgets & launcher portal
    ├── index.html           # Live portal & demonstration page
    ├── model_embed_widget.html # Copy-pasteable HTML/CSS model card for your website
    └── ModelLauncher.jsx    # Production React / Next.js component
```

---

## 🗄️ Step 1: Set Up Supabase Cloud Database

1. Create a free account at [supabase.com](https://supabase.com).
2. Click **New Project** and name it (e.g. `mothers-hut-traffic`). Choose a region close to your users.
3. In the Supabase sidebar, click on the **SQL Editor** tab (`>_`).
4. Click **New Query**, open [supabase_schema.sql](file:///c:/Users/winw4/Desktop/smart_traffic/supabase_schema.sql), copy its contents, paste into the query box, and click **Run**.
   * This creates the `traffic_logs` and `plate_detections` tables with indexes and Row-Level Security (RLS) enabled.
5. In the Supabase sidebar, go to **Project Settings** (gear icon) $\rightarrow$ **API**.
6. Copy the following credentials:
   * **Project URL**: (e.g. `https://xyzproject.supabase.co`)
   * **Project API keys (`anon` / `public`)**: (e.g. `eyJhbGciOi...`)
7. Paste them into your `.env` file:
   ```env
   SUPABASE_URL=https://xyzproject.supabase.co
   SUPABASE_KEY=eyJhbGciOi...
   ```

*(Note: If Supabase credentials are left empty, the application automatically falls back to in-memory local logging so the camera/video stream never freezes).*

---

## ☁️ Step 2: Cloud Deployment Options (Python Server)

### Option A: Railway (Direct Python Git Deployment)

1. Push your repository to GitHub:
   ```powershell
   git init
   git add .
   git commit -m "Production release: ALPR + Supabase"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/smart_traffic.git
   git push -u origin main
   ```
2. Log in to [railway.app](https://railway.app) and click **+ New Project** $\rightarrow$ **Deploy from GitHub repo**.
3. Select your repository. Railway automatically detects Python via `requirements.txt`.
4. Go to the project **Settings** $\rightarrow$ **Deploy** and set the Start Command:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port $PORT
   ```
5. Go to the project **Variables** tab and add:
   * `SUPABASE_URL` = `https://xyzproject.supabase.co`
   * `SUPABASE_KEY` = `your-anon-key`
6. Go to **Settings** $\rightarrow$ **Networking** $\rightarrow$ Click **Generate Domain**.
   * Your project is now live at: `https://your-app-name.up.railway.app`!

---

### Option B: Render (Python Web Service)

1. Log in to [render.com](https://render.com) and click **New +** $\rightarrow$ **Web Service**.
2. Connect your GitHub repository.
3. Select **Python 3** as the runtime environment:
   * **Build Command**: `pip install -r requirements.txt && python download_models.py`
   * **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Under **Environment Variables**, add `SUPABASE_URL` and `SUPABASE_KEY`.
5. Click **Create Web Service**. Render builds the service and assigns your URL (e.g. `https://mothers-hut-traffic.onrender.com`).

---

### Option C: VPS / AWS EC2 / DigitalOcean Droplet (Native Python / Systemd)

1. Connect to your server via SSH:
   ```bash
   git clone https://github.com/YOUR_USERNAME/smart_traffic.git
   cd smart_traffic
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python download_models.py
   cp .env.example .env
   nano .env # Paste your Supabase credentials
   ```
2. Run with Uvicorn:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

---

## 🌐 Step 3: Implement the Model Link & Icon on Your Website

When visitors click the model icon or button on your main website, they will be redirected directly to your deployed cloud console.

### Method 1: Using the Pre-Built Embed Widget (Any HTML Website)

Open [website_integration/model_embed_widget.html](file:///c:/Users/winw4/Desktop/smart_traffic/website_integration/model_embed_widget.html):
1. Replace `YOUR_CLOUD_SERVER_URL` with your live cloud domain (e.g. `https://your-app.up.railway.app`).
2. Paste the HTML snippet into your website's source code where you want the model card to appear.
3. Visitors will see the model card with glowing status ("LIVE ON CLOUD"), and clicking the icon or "Launch Model Console" redirects directly to your cloud server!

### Method 2: Using the Direct Redirect Endpoints

You can link any icon or image directly to these clean endpoints:
* **Desktop Console**: `https://YOUR_CLOUD_SERVER_URL/launch`
* **Mobile Officer App**: `https://YOUR_CLOUD_SERVER_URL/launch-mobile`

Example standard HTML button:
```html
<a href="https://your-cloud-server.railway.app/launch" target="_blank">
  <img src="your-model-icon.png" alt="Traffic AI Model" width="48" height="48">
  <span>Launch Traffic AI Console ↗</span>
</a>
```

### Method 3: React / Next.js Component

Import [website_integration/ModelLauncher.jsx](file:///c:/Users/winw4/Desktop/smart_traffic/website_integration/ModelLauncher.jsx) into your React or Next.js app:
```jsx
import ModelLauncher from './website_integration/ModelLauncher';

export default function HomePage() {
  return (
    <section>
      <h2>Our AI Systems</h2>
      <ModelLauncher cloudUrl="https://your-cloud-server.railway.app" />
    </section>
  );
}
```

---

## 🔍 Verification & Healthcheck

You can verify your cloud deployment at any time by pinging the healthcheck endpoint:

```bash
curl https://YOUR_CLOUD_SERVER_URL/health
```

Expected Response:
```json
{
  "status": "healthy",
  "service": "Mother's Hut Traffic & Gate Operations Console",
  "environment": "production",
  "database": {
    "supabase_configured": true,
    "supabase_url": "https://xyzproject.supabase.co",
    "mode": "Cloud Supabase Database",
    "recent_plates_count": 5
  }
}
```
