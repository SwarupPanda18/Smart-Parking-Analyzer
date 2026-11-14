import React from 'react';

/**
 * Production Model Launcher Component for React / Next.js Websites
 * 
 * Usage:
 *   import ModelLauncher from './ModelLauncher';
 *   <ModelLauncher cloudUrl="https://smart-traffic.up.railway.app" />
 */
export default function ModelLauncher({ cloudUrl = "http://localhost:8000" }) {
  const cleanUrl = cloudUrl.replace(/\/$/, "");

  return (
    <div className="max-w-md w-full bg-slate-900 border border-slate-800 hover:border-emerald-500/40 rounded-2xl p-6 shadow-2xl transition-all duration-300">
      {/* Top Header Badge */}
      <div className="flex items-center justify-between mb-4">
        <span className="text-xs font-bold tracking-wider uppercase text-slate-400">
          AI Vision Engine
        </span>
        <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
          ONLINE ON CLOUD
        </span>
      </div>

      {/* Model Clickable Banner */}
      <a
        href={`${cleanUrl}/launch`}
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-4 text-left group mb-4"
      >
        <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 border border-emerald-500/30 flex items-center justify-center shrink-0 shadow-lg group-hover:scale-105 transition-transform">
          <svg className="w-7 h-7 text-emerald-400" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 15a3 3 0 100-6 3 3 0 000 6z" />
            <path d="M20 4h-3.17L15 2H9L7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm-8 13c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z" />
          </svg>
        </div>
        <div>
          <h3 className="text-lg font-bold text-white group-hover:text-emerald-400 transition-colors">
            Mother's Hut Traffic AI
          </h3>
          <p className="text-xs text-slate-400 line-clamp-1">
            Autonomous Gate Vision & ALPR Recognition
          </p>
        </div>
      </a>

      {/* Action Buttons */}
      <div className="flex flex-col gap-2 mt-4">
        <a
          href={`${cleanUrl}/launch`}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-white text-sm font-bold shadow-lg shadow-emerald-500/20 transition-all"
        >
          <span>Launch Model Console</span>
          <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
            <path d="M19 19H5V5h7V3H5c-1.11 0-2 .9-2 2v14c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2v-7h-2v7zM14 3v2h3.59l-9.83 9.83 1.41 1.41L19 6.41V10h2V3h-7z" />
          </svg>
        </a>

        <a
          href={`${cleanUrl}/launch-mobile`}
          target="_blank"
          rel="noopener noreferrer"
          className="text-center text-xs text-cyan-400 hover:underline font-medium py-1"
        >
          📱 Open Mobile Rear Camera App →
        </a>
      </div>
    </div>
  );
}
