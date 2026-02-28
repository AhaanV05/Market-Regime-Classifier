import React, { useCallback, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

const RANGE_OPTIONS = [
  { key: 'max', label: 'Max', days: 0 },
  { key: '5y', label: '5Y', days: 1260 },
  { key: '1y', label: '1Y', days: 252 },
  { key: '6m', label: '6M', days: 126 },
  { key: '1m', label: '1M', days: 21 },
  { key: '1w', label: '1W', days: 5 },
];

const REGIME_COLORS = {
  'Durable-Calm': '#22c55e',
  'Durable-Choppy': '#0ea5e9',
  'Durable-Stress': '#fb923c',
  'Fragile-Calm': '#8b5cf6',
  'Fragile-Choppy': '#ec4899',
  'Fragile-Stress': '#ef4444',
};

const FAST_COLORS = {
  Calm: '#22c55e',
  Choppy: '#06b6d4',
  Stress: '#ef4444',
};

function normalizeStateLabel(value) {
  if (!value) return 'Unknown';
  const txt = String(value)
    .replaceAll('Ã¢â‚¬â€œ', '-')
    .replaceAll('â€“', '-')
    .replaceAll('â€”', '-')
    .replaceAll('–', '-')
    .replaceAll('—', '-')
    .trim();
  const parts = txt.split('-').map((p) => p.trim());
  if (parts.length >= 2 && parts[0] && parts[1]) return `${parts[0]}-${parts[1]}`;
  return txt;
}

function formatPct(value, decimals = 1) {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(decimals)}%` : 'N/A';
}

function formatNumber(value, decimals = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(decimals) : 'N/A';
}

function normalizedEntropy(values) {
  const arr = (values || []).map((v) => Math.max(Number(v) || 0, 1e-12));
  const s = arr.reduce((a, b) => a + b, 0);
  if (s <= 0) return 0;
  const p = arr.map((v) => v / s);
  const h = -p.reduce((acc, pi) => acc + pi * Math.log(pi), 0);
  const hmax = Math.log(p.length || 1);
  if (!Number.isFinite(h) || !Number.isFinite(hmax) || hmax <= 0) return 0;
  return h / hmax;
}

function levelClass(level) {
  const l = String(level || '').toUpperCase();
  if (l === 'RETRAIN' || l === 'CRITICAL') return 'bg-red-900/40 text-red-300 border-red-500';
  if (l === 'WARN' || l === 'WARNING') return 'bg-amber-900/40 text-amber-300 border-amber-500';
  if (l === 'WATCH') return 'bg-yellow-900/40 text-yellow-300 border-yellow-500';
  return 'bg-emerald-900/30 text-emerald-300 border-emerald-500';
}

function formatTimestamp(isoString) {
  if (!isoString) return 'N/A';
  try {
    const date = new Date(isoString);
    const dateStr = date.toLocaleDateString('en-US', { month: '2-digit', day: '2-digit', year: 'numeric' });
    const timeStr = date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
    return `${dateStr} - ${timeStr}`;
  } catch {
    return isoString;
  }
}

function App() {
  const [rangeKey, setRangeKey] = useState('1y');
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [refreshError, setRefreshError] = useState(null);
  const [lastFetchedAt, setLastFetchedAt] = useState(null);
  const [hasLoadedOnce, setHasLoadedOnce] = useState(false);
  const [showConfidence, setShowConfidence] = useState(true);
  const [showEntropy, setShowEntropy] = useState(false);
  const [showPDurable, setShowPDurable] = useState(false);
  const [showPFragile, setShowPFragile] = useState(true);
  const [showPCalm, setShowPCalm] = useState(true);
  const [showPChoppy, setShowPChoppy] = useState(true);
  const [showPStress, setShowPStress] = useState(true);
  const [showSmooth, setShowSmooth] = useState(true);
  const [showRaw, setShowRaw] = useState(false);
  const [showTrend, setShowTrend] = useState(false);
  const [regimeView, setRegimeView] = useState('smoothed'); // 'smoothed' | 'raw' | 'combined'

  const [currentRegime, setCurrentRegime] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [probabilities, setProbabilities] = useState(null);
  const [health, setHealth] = useState(null);
  const [ops, setOps] = useState(null);
  const [backfill, setBackfill] = useState(null);
  const [changes, setChanges] = useState([]);
  const [modelDiag, setModelDiag] = useState(null);

  const selectedRange = useMemo(
    () => RANGE_OPTIONS.find((r) => r.key === rangeKey) || RANGE_OPTIONS[2],
    [rangeKey]
  );

  const fetchData = useCallback(
    async ({ silent = false } = {}) => {
      try {
        if (!silent && !hasLoadedOnce) setLoading(true);
        setRefreshing(true);
        const days = selectedRange.days;
        const timelineUrl = days === 0 ? `${API_BASE}/timeline` : `${API_BASE}/timeline?days=${days}`;
        const diagUrl = days === 0 ? `${API_BASE}/model-diagnostics?view=${regimeView}` : `${API_BASE}/model-diagnostics?days=${days}&view=${regimeView}`;
        const [currentRes, timelineRes, metricsRes, probsRes, healthRes, opsRes, backfillRes, changesRes, modelRes] =
          await Promise.all([
            axios.get(`${API_BASE}/current-regime`),
            axios.get(timelineUrl),
            axios.get(`${API_BASE}/metrics`),
            axios.get(`${API_BASE}/probabilities`),
            axios.get(`${API_BASE}/health`),
            axios.get(`${API_BASE}/ops`),
            axios.get(`${API_BASE}/backfill?limit=30`),
            axios.get(`${API_BASE}/regime-changes?limit=20`),
            axios.get(diagUrl),
          ]);

        setCurrentRegime(currentRes?.data?.data ?? null);
        setTimeline(timelineRes?.data?.data ?? []);
        setMetrics(metricsRes?.data?.data ?? {});
        setProbabilities(probsRes?.data?.data ?? null);
        setHealth(healthRes?.data ?? null);
        setOps(opsRes?.data?.data ?? null);
        setBackfill(backfillRes?.data?.data ?? null);
        setChanges(changesRes?.data?.data ?? []);
        setModelDiag(modelRes?.data?.data ?? null);
        setLastFetchedAt(new Date());
        setHasLoadedOnce(true);
        setError(null);
        setRefreshError(null);
      } catch (err) {
        const msg = err?.response?.data?.detail || err?.message || 'Unknown error';
        if (!hasLoadedOnce) {
          setError(msg);
        } else {
          setRefreshError(msg);
        }
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [selectedRange.days, hasLoadedOnce]
  );

  // Separate fetch for diagnostics when regime view tab changes
  useEffect(() => {
    if (!hasLoadedOnce) return;
    const days = selectedRange.days;
    const diagUrl = days === 0 ? `${API_BASE}/model-diagnostics?view=${regimeView}` : `${API_BASE}/model-diagnostics?days=${days}&view=${regimeView}`;
    axios.get(diagUrl).then((res) => {
      setModelDiag(res?.data?.data ?? null);
    }).catch(() => {});
  }, [regimeView]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    fetchData({ silent: false });
  }, [fetchData]);

  useEffect(() => {
    if (!autoRefresh) return undefined;
    const onTick = () => {
      if (document.visibilityState === 'visible') {
        fetchData({ silent: true });
      }
    };
    const onVisibility = () => {
      if (document.visibilityState === 'visible') {
        fetchData({ silent: true });
      }
    };
    const id = setInterval(onTick, 5 * 60 * 1000);
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      clearInterval(id);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [autoRefresh, fetchData]);

  const currentCombined = normalizeStateLabel(currentRegime?.combined_state);
  const currentCombinedColor = REGIME_COLORS[currentCombined] || '#e5e7eb';
  const currentFastColor = FAST_COLORS[currentRegime?.fast_state] || '#e5e7eb';

  // Derive regime info per selected view tab
  const viewRegime = useMemo(() => {
    if (!currentRegime) return null;
    if (regimeView === 'raw') {
      const macro = currentRegime.raw_macro_state || 'Unknown';
      const fast = currentRegime.raw_fast_state || 'Unknown';
      const combined = normalizeStateLabel(currentRegime.raw_combined_state);
      return {
        macro, fast, combined,
        macroColor: macro === 'Durable' ? '#22c55e' : '#ef4444',
        fastColor: FAST_COLORS[fast] || '#e5e7eb',
        combinedColor: REGIME_COLORS[combined] || '#e5e7eb',
        macroBars: { Durable: currentRegime.p_durable_raw ?? 0.5, Fragile: currentRegime.p_fragile_raw ?? 0.5 },
        fastBars: { Calm: currentRegime.p_calm_raw ?? 1/3, Choppy: currentRegime.p_choppy_raw ?? 1/3, Stress: currentRegime.p_stress_raw ?? 1/3 },
        label: 'Raw (no smoothing)',
      };
    }
    if (regimeView === 'combined') {
      const macro = currentRegime.adaptive_macro_state || 'Unknown';
      const fast = currentRegime.adaptive_fast_state || 'Unknown';
      const combined = normalizeStateLabel(currentRegime.adaptive_combined_state);
      return {
        macro, fast, combined,
        macroColor: macro === 'Durable' ? '#22c55e' : '#ef4444',
        fastColor: FAST_COLORS[fast] || '#e5e7eb',
        combinedColor: REGIME_COLORS[combined] || '#e5e7eb',
        macroBars: { Durable: currentRegime.p_durable_adaptive ?? 0.5, Fragile: currentRegime.p_fragile_adaptive ?? 0.5 },
        fastBars: { Calm: currentRegime.p_calm_adaptive ?? 1/3, Choppy: currentRegime.p_choppy_adaptive ?? 1/3, Stress: currentRegime.p_stress_adaptive ?? 1/3 },
        label: 'Adaptive-\u03b1 (divergence-driven)',
      };
    }
    // Default: smoothed
    return {
      macro: currentRegime.macro_state,
      fast: currentRegime.fast_state,
      combined: currentCombined,
      macroColor: currentRegime.macro_state === 'Durable' ? '#22c55e' : '#ef4444',
      fastColor: currentFastColor,
      combinedColor: currentCombinedColor,
      macroBars: { Durable: currentRegime.p_durable_smooth ?? 0.5, Fragile: currentRegime.p_fragile_smooth ?? 0.5 },
      fastBars: { Calm: currentRegime.p_calm_smooth ?? 1/3, Choppy: currentRegime.p_choppy_smooth ?? 1/3, Stress: currentRegime.p_stress_smooth ?? 1/3 },
      label: 'Fixed-\u03b1 EMA + Hysteresis',
    };
  }, [currentRegime, regimeView, currentCombined, currentCombinedColor, currentFastColor]);

  const metricsRows = useMemo(() => Object.entries(metrics || {}), [metrics]);

  const timelineData = useMemo(
    () =>
      (timeline || []).map((r, i) => {
        const pDur = Number(r.p_durable_smooth ?? 1 - Number(r.p_fragile_smooth ?? 0.5));
        const pFra = Number(r.p_fragile_smooth ?? 0.5);
        const pCalm = Number(r.p_calm_smooth ?? 1 / 3);
        const pChoppy = Number(r.p_choppy_smooth ?? 1 / 3);
        const pStress = Number(r.p_stress_smooth ?? 1 / 3);
        const macroEntropy = normalizedEntropy([pDur, pFra]);
        const fastEntropy = normalizedEntropy([pCalm, pChoppy, pStress]);
        const macroConf = Number(r.macro_confidence ?? Math.max(pDur, pFra));
        const fastConf = Number(r.fast_confidence ?? Math.max(pCalm, pChoppy, pStress));
        const combinedConf =
          Number(r.confidence ?? r.combined_confidence ?? r.combined_confidence_smooth ?? Math.min(macroConf, fastConf));
        return {
          ...r,
          idx: i,
          p_durable_smooth: pDur,
          zero_line: 0,
          combined_state: normalizeStateLabel(r.combined_state),
          macro_confidence: macroConf,
          fast_confidence: fastConf,
          confidence: combinedConf,
          macro_entropy: macroEntropy,
          fast_entropy: fastEntropy,
          combined_entropy: (macroEntropy + fastEntropy) / 2,
          regime_strip: 1,
        };
      }),
    [timeline]
  );

  const timelineMeta = useMemo(() => {
    if (!timelineData || timelineData.length === 0) {
      return { count: 0, start: null, end: null, insufficient: false };
    }
    const start = timelineData[0]?.date || null;
    const end = timelineData[timelineData.length - 1]?.date || null;
    const target = selectedRange.days;
    const insufficient = target > 0 && timelineData.length < target;
    return { count: timelineData.length, start, end, insufficient };
  }, [timelineData, selectedRange.days]);
  const uniqueCombinedStates = useMemo(() => {
    const s = new Set((timelineData || []).map((r) => normalizeStateLabel(r.combined_state)));
    s.delete('Unknown');
    return Array.from(s);
  }, [timelineData]);

  const opsPayload = ops?.ops ?? ops ?? {};
  const systemHealth = ops?.system_health ?? opsPayload?.system_health ?? {};
  const driftAlertCount = Number(opsPayload?.drift?.alert_features?.length || 0);
  const bocpdLevel = String(opsPayload?.bocpd?.level || 'DISABLED').toUpperCase();
  const emergencyFlag = Boolean(opsPayload?.emergency?.emergency_retrain);
  const retrainDecision = opsPayload?.decision?.primary_reason || 'no_trigger';
  const shouldRetrain = Boolean(opsPayload?.decision?.should_retrain);

  const latestBackfillRun = backfill?.last_backfill_run || null;
  const latestRun = backfill?.latest_run || null;
  const backfillReason = useMemo(() => {
    if (backfill?.summary_reason) return backfill.summary_reason;
    if (!latestRun) return 'No orchestrator run found yet.';
    if (latestRun.mode === 'backfill') return 'Backfill mode executed.';
    const miss = Number(latestRun.auto_backfill_missing_days || 0);
    if (miss <= 0) return 'No missing trailing trading days after latest timeline date.';
    if (miss < 2) return `Missing trailing days (${miss}) are below auto-backfill trigger threshold (default=2).`;
    return `Missing trailing days detected (${miss}), but current run did not enter backfill mode.`;
  }, [latestRun]);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-white text-xl">Loading regime dashboard...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="text-red-400 text-lg">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Regime Ops Dashboard</h1>
          <p className="text-slate-400 text-sm">
            Range: {selectedRange.label} | API: {API_BASE}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {RANGE_OPTIONS.map((r) => (
            <button
              key={r.key}
              onClick={() => setRangeKey(r.key)}
              className={`px-3 py-1.5 rounded border text-sm ${
                r.key === rangeKey ? 'bg-cyan-600 border-cyan-400 text-white' : 'bg-slate-800 border-slate-700 text-slate-300'
              }`}
            >
              {r.label}
            </button>
          ))}

          <button
            onClick={() => fetchData({ silent: true })}
            className="px-3 py-1.5 rounded border border-slate-600 bg-slate-800 text-slate-200"
          >
            {refreshing ? 'Refreshing...' : 'Refresh'}
          </button>

          <label className="flex items-center gap-2 text-sm px-3 py-1.5 rounded border border-slate-700 bg-slate-900">
            <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
            Auto refresh (5m)
          </label>
        </div>
      </div>

      <div className="mb-6 text-sm text-slate-400">
        Last fetched: {lastFetchedAt ? lastFetchedAt.toLocaleString() : 'N/A'} {refreshing ? '| Refreshing...' : ''}
      </div>
      {refreshError && (
        <div className="mb-4 p-3 rounded border border-amber-500 bg-amber-900/30 text-amber-200 text-sm">
          Last refresh failed: {refreshError}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-6">
        <div className={`p-3 rounded border ${health?.status === 'healthy' ? 'bg-emerald-900/30 border-emerald-500' : 'bg-yellow-900/30 border-yellow-500'}`}>
          <div className="text-xs text-slate-300">System Health</div>
          <div className="text-lg font-semibold">{String(health?.status || 'unknown').toUpperCase()}</div>
        </div>
        <div className={`p-3 rounded border ${driftAlertCount > 0 ? 'bg-amber-900/40 border-amber-500' : 'bg-emerald-900/30 border-emerald-500'}`}>
          <div className="text-xs text-slate-300">Drift</div>
          <div className="text-lg font-semibold">{driftAlertCount > 0 ? `ALERT (${driftAlertCount})` : 'OK'}</div>
        </div>
        <div className={`p-3 rounded border ${levelClass(bocpdLevel)}`}>
          <div className="text-xs text-slate-300">BOCPD</div>
          <div className="text-lg font-semibold">{String(bocpdLevel).toUpperCase()}</div>
        </div>
          <div className={`p-3 rounded border ${emergencyFlag ? 'bg-red-900/40 border-red-500' : 'bg-emerald-900/30 border-emerald-500'}`}>
            <div className="text-xs text-slate-300">Retrain Decision</div>
            <div className="text-lg font-semibold">
              {shouldRetrain ? `TRIGGERED (${retrainDecision})` : `NO_TRIGGER (${retrainDecision})`}
            </div>
          </div>
      </div>

      {currentRegime && viewRegime && (
        <div className="mb-6 p-6 rounded-xl bg-slate-900 border border-slate-700">
          <div className="flex flex-wrap justify-between items-start gap-4 mb-3">
            <div>
              <h2 className="text-xl font-bold">Current Regime</h2>
              <p className="text-slate-400 text-sm">
                asof: {currentRegime.date || 'N/A'} | model version: {systemHealth?.model?.version_tag || 'N/A'}
              </p>
            </div>
            <div className="text-right">
              <div className="text-xs text-slate-400">Combined Confidence</div>
              <div className="text-3xl font-bold">{formatPct(currentRegime.confidence, 0)}</div>
            </div>
          </div>

          {/* ── 3-Tab Switcher ── */}
          <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 p-0.5 gap-0.5 mb-4 w-fit">
            {[
              { key: 'smoothed', label: 'Smoothed Regime' },
              { key: 'raw', label: 'Raw Regime' },
              { key: 'combined', label: 'Combined Regime' },
            ].map((tab) => (
              <button
                key={tab.key}
                onClick={() => setRegimeView(tab.key)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                  regimeView === tab.key
                    ? 'bg-sky-600 text-white shadow'
                    : 'text-slate-400 hover:text-white hover:bg-slate-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="text-xs text-slate-500 mb-4">{viewRegime.label}</div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded bg-slate-800/70">
              <div className="text-xs text-slate-400">Macro</div>
              <div className="text-2xl font-bold" style={{ color: viewRegime.macroColor }}>
                {viewRegime.macro}
              </div>
            </div>
            <div className="p-4 rounded bg-slate-800/70">
              <div className="text-xs text-slate-400">Fast</div>
              <div className="text-2xl font-bold" style={{ color: viewRegime.fastColor }}>
                {viewRegime.fast}
              </div>
            </div>
            <div className="p-4 rounded bg-slate-800/70">
              <div className="text-xs text-slate-400">Combined</div>
              <div className="text-2xl font-bold" style={{ color: viewRegime.combinedColor }}>
                {viewRegime.combined}
              </div>
            </div>
          </div>
        </div>
      )}

      {probabilities && viewRegime && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div className="bg-slate-900 border border-slate-700 rounded-xl p-5">
            <h3 className="text-lg font-semibold mb-4">Macro Probabilities</h3>
            {['Durable', 'Fragile'].map((k) => {
              const v = Number(viewRegime.macroBars?.[k] ?? probabilities?.macro?.[k] ?? 0);
              const barColor = k === 'Durable' ? '#22c55e' : '#ef4444';
              return (
                <div key={k} className="mb-3">
                  <div className="flex justify-between text-sm mb-1">
                    <span>{k}</span>
                    <span>{formatPct(v)}</span>
                  </div>
                  <div className="h-2 rounded bg-slate-700 overflow-hidden">
                    <div className="h-full" style={{ width: `${Math.max(0, Math.min(100, v * 100))}%`, backgroundColor: barColor }} />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="bg-slate-900 border border-slate-700 rounded-xl p-5">
            <h3 className="text-lg font-semibold mb-4">Fast Probabilities</h3>
            {['Calm', 'Choppy', 'Stress'].map((k) => {
              const v = Number(viewRegime.fastBars?.[k] ?? probabilities?.fast?.[k] ?? 0);
              return (
                <div key={k} className="mb-3">
                  <div className="flex justify-between text-sm mb-1">
                    <span style={{ color: FAST_COLORS[k] }}>{k}</span>
                    <span>{formatPct(v)}</span>
                  </div>
                  <div className="h-2 rounded bg-slate-700 overflow-hidden">
                    <div className="h-full" style={{ width: `${Math.max(0, Math.min(100, v * 100))}%`, backgroundColor: FAST_COLORS[k] }} />
                  </div>
                </div>
              );
            })}
            <div className="mt-3 text-xs text-slate-400">
              Macro confidence: {formatPct(probabilities?.macro_confidence || currentRegime?.macro_confidence || 0)} | Fast confidence:{' '}
              {formatPct(probabilities?.fast_confidence || currentRegime?.fast_confidence || 0)}
            </div>
          </div>
        </div>
      )}

      {timelineData.length > 0 && (
        <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 mb-6">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold">Timeline ({selectedRange.label})</h3>
              <div className="flex items-center bg-slate-800 rounded-lg border border-slate-700 p-0.5 gap-0.5">
                {[
                  { key: 'smooth', label: 'Smooth', active: showSmooth, toggle: () => setShowSmooth(v => !v) },
                  { key: 'raw', label: 'Raw', active: showRaw, toggle: () => setShowRaw(v => !v) },
                  { key: 'trend', label: 'Trend', active: showTrend, toggle: () => setShowTrend(v => !v) },
                ].map((m) => (
                  <button
                    key={m.key}
                    onClick={m.toggle}
                    className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
                      m.active
                        ? 'bg-cyan-600 text-white'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2 text-xs border-l border-slate-700 pl-3">
                <span className="text-slate-500 font-medium">Macro:</span>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showPDurable} onChange={(e) => setShowPDurable(e.target.checked)} />
                  <span style={{ color: '#22c55e' }}>Durable</span>
                </label>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showPFragile} onChange={(e) => setShowPFragile(e.target.checked)} />
                  <span style={{ color: '#ef4444' }}>Fragile</span>
                </label>
              </div>
              <div className="flex items-center gap-2 text-xs border-l border-slate-700 pl-3">
                <span className="text-slate-500 font-medium">Fast:</span>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showPCalm} onChange={(e) => setShowPCalm(e.target.checked)} />
                  <span style={{ color: '#22c55e' }}>Calm</span>
                </label>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showPChoppy} onChange={(e) => setShowPChoppy(e.target.checked)} />
                  <span style={{ color: '#06b6d4' }}>Choppy</span>
                </label>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showPStress} onChange={(e) => setShowPStress(e.target.checked)} />
                  <span style={{ color: '#f59e0b' }}>Stress</span>
                </label>
              </div>
              <div className="flex items-center gap-2 text-xs border-l border-slate-700 pl-3">
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showConfidence} onChange={(e) => setShowConfidence(e.target.checked)} />
                  Confidence
                </label>
                <label className="flex items-center gap-1">
                  <input type="checkbox" checked={showEntropy} onChange={(e) => setShowEntropy(e.target.checked)} />
                  Entropy
                </label>
              </div>
            </div>
          </div>
          <div className="text-xs text-slate-400 mb-2">
            Points: {timelineMeta.count} | Range shown: {timelineMeta.start || 'N/A'} to {timelineMeta.end || 'N/A'}
            {timelineMeta.insufficient ? ' | Note: dataset has fewer points than selected preset.' : ''}
            {showSmooth && ' | Smooth: solid lines'}
            {showRaw && (
              timelineData[0]?.p_fragile_raw != null
                ? ' | Raw: dashed lines'
                : ' | Raw: requires re-backfill'
            )}
            {showTrend && ' | Trend: 5-day Δ (below)'}
          </div>
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 11 }} />
              <YAxis stroke="#94a3b8" domain={[0, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
              <Legend />

              {/* ── Smoothed lines (solid) ── */}
              {showSmooth && showPDurable && (
                <Line type="monotone" dataKey="p_durable_smooth" stroke="#22c55e" name="P(Durable)" dot={false} strokeWidth={2} />
              )}
              {showSmooth && showPFragile && (
                <Line type="monotone" dataKey="p_fragile_smooth" stroke="#ef4444" name="P(Fragile)" dot={false} strokeWidth={2} />
              )}
              {showSmooth && showPCalm && (
                <Line type="monotone" dataKey="p_calm_smooth" stroke="#22c55e" name="P(Calm)" dot={false} strokeWidth={2} />
              )}
              {showSmooth && showPChoppy && (
                <Line type="monotone" dataKey="p_choppy_smooth" stroke="#06b6d4" name="P(Choppy)" dot={false} strokeWidth={2} />
              )}
              {showSmooth && showPStress && (
                <Line type="monotone" dataKey="p_stress_smooth" stroke="#f59e0b" name="P(Stress)" dot={false} strokeWidth={2} />
              )}

              {/* ── Raw lines (dashed, thinner) ── */}
              {showRaw && showPDurable && timelineData[0]?.p_durable_raw != null && (
                <Line type="monotone" dataKey="p_durable_raw" stroke="#22c55e" name="P(Durable) Raw" dot={false} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.7} />
              )}
              {showRaw && showPFragile && timelineData[0]?.p_fragile_raw != null && (
                <Line type="monotone" dataKey="p_fragile_raw" stroke="#ef4444" name="P(Fragile) Raw" dot={false} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.7} />
              )}
              {showRaw && showPCalm && timelineData[0]?.p_calm_raw != null && (
                <Line type="monotone" dataKey="p_calm_raw" stroke="#22c55e" name="P(Calm) Raw" dot={false} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.7} />
              )}
              {showRaw && showPChoppy && timelineData[0]?.p_choppy_raw != null && (
                <Line type="monotone" dataKey="p_choppy_raw" stroke="#06b6d4" name="P(Choppy) Raw" dot={false} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.7} />
              )}
              {showRaw && showPStress && timelineData[0]?.p_stress_raw != null && (
                <Line type="monotone" dataKey="p_stress_raw" stroke="#f59e0b" name="P(Stress) Raw" dot={false} strokeWidth={1.5} strokeDasharray="6 3" opacity={0.7} />
              )}

              {/* ── Overlays ── */}
              {showConfidence && (
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="#a78bfa"
                  name="Combined Confidence"
                  dot={false}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                />
              )}
              {showEntropy && (
                <Line
                  type="monotone"
                  dataKey="combined_entropy"
                  stroke="#f97316"
                  name="Entropy"
                  dot={false}
                  strokeWidth={1.5}
                  isAnimationActive={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>

          {/* ── Trend chart (separate, shown when Trend toggle is on) ── */}
          {showTrend && (
            <div className="mt-4">
              <div className="text-xs text-slate-400 mb-1">5-Day Rate of Change (Δ)</div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" stroke="#94a3b8" tick={{ fontSize: 10 }} />
                  <YAxis stroke="#94a3b8" domain={[-0.5, 0.5]} />
                  <Tooltip contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }} />
                  <Legend />
                  {showPDurable && (
                    <Line type="monotone" dataKey="d_durable" stroke="#22c55e" name="Δ Durable" dot={false} strokeWidth={1.5} />
                  )}
                  {showPFragile && (
                    <Line type="monotone" dataKey="d_fragile" stroke="#ef4444" name="Δ Fragile" dot={false} strokeWidth={1.5} />
                  )}
                  {showPCalm && (
                    <Line type="monotone" dataKey="d_calm" stroke="#22c55e" name="Δ Calm" dot={false} strokeWidth={1.5} />
                  )}
                  {showPChoppy && (
                    <Line type="monotone" dataKey="d_choppy" stroke="#06b6d4" name="Δ Choppy" dot={false} strokeWidth={1.5} />
                  )}
                  {showPStress && (
                    <Line type="monotone" dataKey="d_stress" stroke="#f59e0b" name="Δ Stress" dot={false} strokeWidth={1.5} />
                  )}
                  <Line type="monotone" dataKey="zero_line" stroke="#475569" name="Zero" dot={false} strokeWidth={1} strokeDasharray="4 4" legendType="none" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}

          <div className="mt-2 text-xs text-slate-400">Regime Strip</div>
          {uniqueCombinedStates.length === 1 && (
            <div className="mb-1 text-xs text-slate-500">
              Uniform strip: all points in this window are `{uniqueCombinedStates[0]}`.
            </div>
          )}
          {uniqueCombinedStates.length === 1 ? (
            <div
              className="h-11 rounded border border-slate-700"
              style={{ backgroundColor: REGIME_COLORS[uniqueCombinedStates[0]] || '#64748b' }}
            />
          ) : (
            <ResponsiveContainer width="100%" height={44}>
              <BarChart data={timelineData} barCategoryGap="0%" barGap={0} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                <XAxis dataKey="date" hide />
                <YAxis hide domain={[0, 1]} />
                <Tooltip
                  cursor={false}
                  contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                  formatter={(value, name, props) => [
                    normalizeStateLabel(props?.payload?.combined_state || 'Unknown'),
                    'State',
                  ]}
                />
                <Bar dataKey="regime_strip" isAnimationActive={false} stroke="none">
                  {timelineData.map((entry, idx) => (
                    <Cell key={`strip-${entry.date}-${idx}`} fill={REGIME_COLORS[normalizeStateLabel(entry.combined_state)] || '#64748b'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
        <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 flex flex-col">
          <h3 className="text-lg font-semibold mb-4">Regime Change Feed</h3>
          <div className="space-y-2 flex-1 overflow-auto" style={{ maxHeight: '37rem' }}>
            {(changes || []).length === 0 && <div className="text-slate-400 text-sm">No recent regime transitions in selected range.</div>}
            {(changes || []).map((c, idx) => (
              <div key={`${c.date}-${idx}`} className="p-3 rounded bg-slate-800/60 border border-slate-700 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-slate-300">{c.date}</span>
                  <span className={`px-2 py-0.5 rounded text-xs border ${levelClass(c.severity)}`}>{c.severity}</span>
                </div>
                <div className="mt-1">
                  <span className="text-slate-400">{normalizeStateLabel(c.from)}</span> {'->'}{' '}
                  <span className="font-semibold" style={{ color: REGIME_COLORS[normalizeStateLabel(c.to)] || '#e2e8f0' }}>
                    {normalizeStateLabel(c.to)}
                  </span>
                </div>
                <div className="text-xs text-slate-400 mt-1">Confidence: {formatPct(c.confidence || 0)}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900 border border-slate-700 rounded-xl p-5">
          <h3 className="text-lg font-semibold mb-4">Backfill Monitor</h3>
          
          {/* Current Status Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
            <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
              <div className="text-xs text-slate-400 mb-1">Latest Mode</div>
              <div className="text-sm font-semibold text-slate-200">{latestRun?.mode || 'N/A'}</div>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
              <div className="text-xs text-slate-400 mb-1">Auto-Backfill</div>
              <div className="flex items-center gap-2">
                <span className={`px-2 py-0.5 rounded text-xs font-semibold ${latestRun?.auto_backfill_triggered ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' : latestRun?.mode === 'backfill' ? 'bg-slate-600 text-slate-300' : 'bg-slate-700 text-slate-400'}`}>
                  {latestRun?.auto_backfill_triggered ? 'TRIGGERED' : latestRun?.mode === 'backfill' ? 'MANUAL' : 'STANDBY'}
                </span>
              </div>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
              <div className="text-xs text-slate-400 mb-1">Missing Days</div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold text-slate-200">{latestRun?.auto_backfill_missing_days ?? 'N/A'}</span>
                {(latestRun?.auto_backfill_missing_days || 0) === 0 && (
                  <span className="text-xs text-green-400">✓ Complete</span>
                )}
              </div>
            </div>
            <div className="bg-slate-800/60 rounded-lg p-3 border border-slate-700">
              <div className="text-xs text-slate-400 mb-1">Processed Dates</div>
              <div className="text-lg font-bold text-slate-200">{latestRun?.processed_dates_count ?? 0}</div>
            </div>
          </div>

          {backfillReason && (
            <div className="bg-slate-800/40 border border-slate-700 rounded-lg p-3 mb-4">
              <div className="text-xs text-slate-400 mb-1">Status</div>
              <div className="text-sm text-slate-300">{backfillReason}</div>
            </div>
          )}

          {/* Latest Backfill Run Details */}
          {latestBackfillRun && (
            <div className="bg-slate-800/40 border border-slate-700 rounded-lg p-3 mb-4">
              <div className="text-xs text-slate-400 mb-2">Last Backfill Execution</div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs text-slate-300">
                <div><span className="text-slate-500">Timestamp:</span> {formatTimestamp(latestBackfillRun.timestamp)}</div>
                <div><span className="text-slate-500">Written Days:</span> {latestBackfillRun.processed_dates_count ?? 0}</div>
                <div><span className="text-slate-500">Start Date:</span> {latestBackfillRun.requested_start_date || 'N/A'}</div>
                <div><span className="text-slate-500">End Date:</span> {latestBackfillRun.requested_end_date || 'N/A'}</div>
              </div>
            </div>
          )}

          {/* Data Source Max Dates */}
          {!!backfill?.source_max_dates && (
            <details className="bg-slate-800/40 border border-slate-700 rounded-lg p-3 mb-4">
              <summary className="text-xs text-slate-400 cursor-pointer hover:text-slate-300">Data Source Max Dates</summary>
              <div className="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-1 text-xs text-slate-400">
                <div><span className="text-slate-500">Market:</span> {backfill.source_max_dates.market || 'N/A'}</div>
                <div><span className="text-slate-500">Final Features:</span> {backfill.source_max_dates.final_features || 'N/A'}</div>
                <div><span className="text-slate-500">Slow Features:</span> {backfill.source_max_dates.slow_features || 'N/A'}</div>
                <div><span className="text-slate-500">Fast Features:</span> {backfill.source_max_dates.fast_features || 'N/A'}</div>
                <div><span className="text-slate-500">Timeline:</span> {backfill.source_max_dates.timeline || 'N/A'}</div>
                <div><span className="text-slate-500">Timeline Current:</span> {backfill.source_max_dates.timeline_current || 'N/A'}</div>
              </div>
            </details>
          )}

          {/* Recent Runs Table */}
          <div className="bg-slate-800/40 border border-slate-700 rounded-lg overflow-hidden">
            <div className="px-3 py-2 bg-slate-800 border-b border-slate-700">
              <div className="text-xs font-semibold text-slate-400">Recent Runs</div>
            </div>
            <div className="max-h-52 overflow-auto">
              <table className="w-full text-xs">
                <thead className="bg-slate-800 sticky top-0 z-10">
                  <tr className="text-left text-slate-400">
                    <th className="px-3 py-2 font-medium bg-slate-800">Timestamp</th>
                    <th className="px-3 py-2 font-medium bg-slate-800">Mode</th>
                    <th className="px-3 py-2 font-medium bg-slate-800">Status</th>
                    <th className="px-3 py-2 font-medium text-right bg-slate-800">Missing</th>
                  </tr>
                </thead>
                <tbody>
                  {(backfill?.recent_runs || []).slice(-10).reverse().map((r, idx) => (
                    <tr key={`${r.timestamp}-${idx}`} className="border-b border-slate-800 hover:bg-slate-800/30">
                      <td className="px-3 py-2 text-slate-300">{formatTimestamp(r.timestamp)}</td>
                      <td className="px-3 py-2 text-slate-400">{r.mode}</td>
                      <td className="px-3 py-2">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          r.status === 'SUCCESS' ? 'bg-green-500/20 text-green-400 border border-green-500/50' :
                          r.status === 'HARD_FAIL' ? 'bg-red-500/20 text-red-400 border border-red-500/50' :
                          r.status === 'SOFT_FAIL' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50' :
                          'bg-slate-700 text-slate-400'
                        }`}>
                          {r.status}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right">
                        <span className={`font-mono ${(r.auto_backfill_missing_days ?? 0) === 0 ? 'text-green-400' : 'text-yellow-400'}`}>
                          {r.auto_backfill_missing_days ?? 0}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {modelDiag?.available && (
        <div className="bg-slate-900 border border-slate-700 rounded-xl p-5 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">Model Diagnostics</h3>
            <span className="text-xs px-2 py-1 rounded bg-slate-700 text-slate-300">
              Source: {regimeView === 'smoothed' ? 'Smoothed Regime' : regimeView === 'raw' ? 'Raw Regime' : 'Combined (Adaptive) Regime'}
            </span>
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-4 mb-4">
            <div className="p-3 rounded bg-slate-800/60">
              <div className="text-xs text-slate-400">Macro switch risk (next day)</div>
              <div className="text-2xl font-bold">{formatPct(modelDiag?.switch_risk?.macro_switch_next_day || 0, 1)}</div>
            </div>
            <div className="p-3 rounded bg-slate-800/60">
              <div className="text-xs text-slate-400">Fast switch risk (next day)</div>
              <div className="text-2xl font-bold">{formatPct(modelDiag?.switch_risk?.fast_switch_next_day || 0, 1)}</div>
            </div>
            <div className="p-3 rounded bg-slate-800/60">
              <div className="text-xs text-slate-400">Model states</div>
              <div className="text-sm">
                SLOW={modelDiag?.model_info?.slow_states} | FAST(D/F)=
                {modelDiag?.model_info?.fast_durable_states}/{modelDiag?.model_info?.fast_fragile_states}
              </div>
            </div>
            <div className="p-3 rounded bg-slate-800/60">
              <div className="text-xs text-slate-400">Flip rate 60d (Combined)</div>
              <div className="text-2xl font-bold">{formatNumber(modelDiag?.flip_rates_60?.combined || 0, 1)}</div>
              <div className="text-xs text-slate-500">per year</div>
            </div>
            <div className="p-3 rounded bg-slate-800/60">
              <div className="text-xs text-slate-400">Min occupancy 60d</div>
              <div className="text-2xl font-bold">{formatPct(modelDiag?.occupancy_60?.min_occupancy || 0, 1)}</div>
              <div className="text-xs text-slate-500">
                {modelDiag?.occupancy_60?.collapse_detected ? 'Collapse risk' : 'No collapse'}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div>
              <div className="text-sm font-semibold mb-2">Avg Duration (empirical)</div>
              <div className="text-sm text-slate-300 space-y-1">
                {Object.entries(modelDiag?.empirical_duration_days?.macro || {}).map(([state, stats]) => (
                  <div key={state}>
                    {state}: <span className="font-mono">{stats.mean}d</span>
                    <span className="text-xs text-slate-500 ml-1">(med {stats.median}d, {stats.min}-{stats.max}d, n={stats.n})</span>
                  </div>
                ))}
                <div className="mt-1 text-xs text-slate-500 border-t border-slate-700 pt-1">Fast layer:</div>
                {Object.entries(modelDiag?.empirical_duration_days?.fast || {}).map(([state, stats]) => (
                  <div key={state}>
                    {state}: <span className="font-mono">{stats.mean}d</span>
                    <span className="text-xs text-slate-500 ml-1">(med {stats.median}d, {stats.min}-{stats.max}d, n={stats.n})</span>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="text-sm font-semibold mb-2">If Regime Changes, Next Most Likely</div>
              <div className="text-xs text-slate-500 mb-1">Self-transition excluded &amp; renormalized</div>
              {(modelDiag?.next_regime_if_change || []).length === 0 ? (
                <div className="text-sm text-slate-400">No transition data available.</div>
              ) : (
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart
                    data={(modelDiag?.next_regime_if_change || []).filter((x) => x.probability > 0.001).map((x) => ({
                      state: normalizeStateLabel(x.state),
                      probability: Number(x.probability || 0),
                    }))}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="state" stroke="#94a3b8" tick={{ fontSize: 11 }} />
                    <YAxis stroke="#94a3b8" domain={[0, 1]} />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #334155' }}
                      formatter={(v) => formatPct(v)}
                    />
                    <Bar dataKey="probability" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>

          <div className="mt-4 text-xs text-slate-400">
            Features: SLOW {modelDiag?.model_info?.slow_feature_count} | FAST {modelDiag?.model_info?.fast_feature_count}
          </div>

          <div className="mt-5 grid grid-cols-1 xl:grid-cols-2 gap-6">
            <div className="bg-slate-800/40 rounded p-3">
              <div className="text-sm font-semibold mb-2">Occupancy (252d)</div>
              <div className="space-y-1 text-sm">
                {Object.entries(modelDiag?.occupancy_252?.occupancy || {}).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span style={{ color: REGIME_COLORS[normalizeStateLabel(k)] || '#cbd5e1' }}>{normalizeStateLabel(k)}</span>
                    <span className="font-mono">{formatPct(v, 1)}</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="bg-slate-800/40 rounded p-3">
              <div className="text-sm font-semibold mb-2">Flip Rates (252d, annualized)</div>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span>Combined</span>
                  <span className="font-mono">{formatNumber(modelDiag?.flip_rates_252?.combined || 0, 2)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Macro</span>
                  <span className="font-mono">{formatNumber(modelDiag?.flip_rates_252?.macro || 0, 2)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Fast</span>
                  <span className="font-mono">{formatNumber(modelDiag?.flip_rates_252?.fast || 0, 2)}</span>
                </div>
                <div className="text-xs text-slate-400 mt-2">
                  Degenerate: {modelDiag?.occupancy_252?.degenerate_detected ? 'Yes' : 'No'} | Collapse:{' '}
                  {modelDiag?.occupancy_252?.collapse_detected ? 'Yes' : 'No'}
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 grid grid-cols-1 xl:grid-cols-3 gap-4">
            <div className="bg-slate-800/40 rounded p-3 overflow-auto">
              <div className="text-sm font-semibold mb-2">Emission Sanity: Slow</div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400">
                    <th className="text-left">Feature</th>
                    <th className="text-right">Dur</th>
                    <th className="text-right">Fra</th>
                  </tr>
                </thead>
                <tbody>
                  {(modelDiag?.emission_summary?.slow || []).slice(0, 8).map((r) => (
                    <tr key={r.feature}>
                      <td>{r.feature}</td>
                      <td className="text-right font-mono">{formatNumber(r.Durable, 2)}</td>
                      <td className="text-right font-mono">{formatNumber(r.Fragile, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="bg-slate-800/40 rounded p-3 overflow-auto">
              <div className="text-sm font-semibold mb-2">Emission Sanity: Fast Durable</div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400">
                    <th className="text-left">Feature</th>
                    <th className="text-right">Calm</th>
                    <th className="text-right">Stress</th>
                  </tr>
                </thead>
                <tbody>
                  {(modelDiag?.emission_summary?.fast_durable || []).slice(0, 8).map((r) => (
                    <tr key={r.feature}>
                      <td>{r.feature}</td>
                      <td className="text-right font-mono">{formatNumber(r.Calm, 2)}</td>
                      <td className="text-right font-mono">{formatNumber(r.Stress, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="bg-slate-800/40 rounded p-3 overflow-auto">
              <div className="text-sm font-semibold mb-2">Emission Sanity: Fast Fragile</div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-400">
                    <th className="text-left">Feature</th>
                    <th className="text-right">Calm</th>
                    <th className="text-right">Stress</th>
                  </tr>
                </thead>
                <tbody>
                  {(modelDiag?.emission_summary?.fast_fragile || []).slice(0, 8).map((r) => (
                    <tr key={r.feature}>
                      <td>{r.feature}</td>
                      <td className="text-right font-mono">{formatNumber(r.Calm, 2)}</td>
                      <td className="text-right font-mono">{formatNumber(r.Stress, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {metricsRows.length > 0 && (
        <div className="bg-slate-900 border border-slate-700 rounded-xl p-5">
          <h3 className="text-lg font-semibold mb-4">Economic Metrics by Regime</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700">
                  <th className="text-left p-2">Regime</th>
                  <th className="text-right p-2">Ann Return</th>
                  <th className="text-right p-2">Vol</th>
                  <th className="text-right p-2">Sharpe</th>
                  <th className="text-right p-2">MaxDD</th>
                  <th className="text-right p-2">VaR95</th>
                  <th className="text-right p-2">%Neg Days</th>
                  <th className="text-right p-2">Days</th>
                </tr>
              </thead>
              <tbody>
                {metricsRows.map(([rawRegime, data]) => {
                  const regime = normalizeStateLabel(rawRegime);
                  const annReturn = data?.return != null ? Number(data.return) : Number(data?.['Ann. Return (%)'] ?? 0) / 100;
                  const vol = data?.volatility != null ? Number(data.volatility) : Number(data?.['Volatility (%)'] ?? 0) / 100;
                  const sharpe = Number(data?.sharpe ?? data?.Sharpe ?? 0);
                  const maxDD =
                    data?.max_dd != null
                      ? Number(data.max_dd)
                      : Number(data?.['MaxDD on regime-days (%)'] ?? data?.['MaxDD (%)'] ?? 0) / 100;
                  const var95Raw = data?.var95 ?? data?.['VaR 95% (%)'] ?? data?.['VaR95 (%)'];
                  const negDaysRaw = data?.neg_days ?? data?.['% Negative Days'] ?? data?.['Negative Days (%)'];
                  const var95 = var95Raw == null ? null : data?.var95 != null ? Number(var95Raw) : Number(var95Raw) / 100;
                  const negDays = negDaysRaw == null ? null : data?.neg_days != null ? Number(negDaysRaw) : Number(negDaysRaw) / 100;
                  const count = Number(data?.count ?? data?.Days ?? 0);
                  return (
                    <tr key={rawRegime} className="border-b border-slate-800 hover:bg-slate-800/40">
                      <td className="p-2 font-semibold" style={{ color: REGIME_COLORS[regime] || '#e2e8f0' }}>
                        {regime}
                      </td>
                      <td className={`p-2 text-right font-mono ${annReturn >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {formatPct(annReturn, 2)}
                      </td>
                      <td className="p-2 text-right font-mono">{formatPct(vol, 2)}</td>
                      <td className={`p-2 text-right font-mono ${sharpe >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {formatNumber(sharpe, 2)}
                      </td>
                      <td className="p-2 text-right font-mono text-red-300">{formatPct(maxDD, 2)}</td>
                      <td className="p-2 text-right font-mono text-red-200">{var95 == null || Number.isNaN(var95) ? 'N/A' : formatPct(var95, 2)}</td>
                      <td className="p-2 text-right font-mono">{negDays == null || Number.isNaN(negDays) ? 'N/A' : formatPct(negDays, 1)}</td>
                      <td className="p-2 text-right font-mono">{count}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
