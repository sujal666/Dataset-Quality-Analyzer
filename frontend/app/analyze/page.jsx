"use client";

import { useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

function apiBaseUrl() {
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
}

async function postCsv(file, { textCol, labelCol, noHfModels }) {
  const form = new FormData();
  form.append("file", file);
  if (textCol) form.append("text_col", textCol);
  if (labelCol) form.append("label_col", labelCol);
  if (noHfModels) form.append("no_hf_models", "true");

  const res = await fetch(`${apiBaseUrl()}/analyze/csv`, { method: "POST", body: form });
  if (!res.ok) {
    let msg = await res.text();
    try {
      const j = JSON.parse(msg);
      msg = j?.detail?.error || j?.detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

async function postHf(payload) {
  const res = await fetch(`${apiBaseUrl()}/analyze/hf`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    let msg = await res.text();
    try {
      const j = JSON.parse(msg);
      msg = j?.detail?.error || j?.detail || msg;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

function SeverityBadge({ severity }) {
  const cls =
    severity === "critical" ? "sev-critical" : severity === "warning" ? "sev-warning" : severity === "info" ? "sev-info" : "";
  return <span className={`pill ${cls}`}>{severity}</span>;
}

export default function Page() {
  const [mode, setMode] = useState("csv");

  const [csvFile, setCsvFile] = useState(null);
  const [textCol, setTextCol] = useState("");
  const [labelCol, setLabelCol] = useState("");
  const [noHfModels, setNoHfModels] = useState(false);

  const [hfDataset, setHfDataset] = useState("imdb");
  const [hfSplit, setHfSplit] = useState("train");
  const [hfSubset, setHfSubset] = useState("");
  const [hfMaxRows, setHfMaxRows] = useState(500);
  const [hfTextCol, setHfTextCol] = useState("");
  const [hfLabelCol, setHfLabelCol] = useState("");

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState(null);

  const subscoresChart = useMemo(() => {
    if (!report?.score?.subscores) return [];
    return Object.entries(report.score.subscores).map(([k, v]) => ({ name: k, score: v }));
  }, [report]);

  async function runAnalysis() {
    setError("");
    setLoading(true);
    setReport(null);
    try {
      if (mode === "csv") {
        if (!csvFile) throw new Error("Select a CSV file first.");
        const rep = await postCsv(csvFile, { textCol: textCol.trim() || null, labelCol: labelCol.trim() || null, noHfModels });
        setReport(rep);
      } else {
        const payload = {
          dataset: hfDataset.trim(),
          split: hfSplit.trim() || "train",
          subset: hfSubset.trim() || null,
          text_col: hfTextCol.trim() || null,
          label_col: hfLabelCol.trim() || null,
          max_rows: hfMaxRows ? Number(hfMaxRows) : null,
          no_hf_models: !!noHfModels
        };
        if (!payload.dataset) throw new Error("Enter a Hugging Face dataset name.");
        const rep = await postHf(payload);
        setReport(rep);
      }
    } catch (e) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <h1 className="title">Dataset Quality Analyzer</h1>
      <p className="subtitle">
        Upload a CSV or analyze a Hugging Face dataset, then get duplicates / labels / toxicity / domain / leakage signals + a 0-100 score.
      </p>

      <div className="grid">
        <div className="panel">
          <h2>Input</h2>

          <div className="row">
            <div style={{ flex: 1 }}>
              <label>Mode</label>
              <select value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="csv">CSV upload</option>
                <option value="hf">Hugging Face dataset</option>
              </select>
            </div>
            <div style={{ width: 160 }}>
              <label>Models</label>
              <select value={noHfModels ? "lite" : "hf"} onChange={(e) => setNoHfModels(e.target.value === "lite")}>
                <option value="hf">HF (default)</option>
                <option value="lite">Lite fallback</option>
              </select>
            </div>
          </div>

          {mode === "csv" ? (
            <>
              <div className="row">
                <div style={{ flex: 1 }}>
                  <label>CSV file</label>
                  <input type="file" accept=".csv,text/csv" onChange={(e) => setCsvFile(e.target.files?.[0] || null)} />
                </div>
              </div>
              <div className="row">
                <div style={{ flex: 1 }}>
                  <label>Text column (optional)</label>
                  <input type="text" placeholder="e.g. text" value={textCol} onChange={(e) => setTextCol(e.target.value)} />
                </div>
                <div style={{ flex: 1 }}>
                  <label>Label column (optional)</label>
                  <input type="text" placeholder="e.g. label" value={labelCol} onChange={(e) => setLabelCol(e.target.value)} />
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="row">
                <div style={{ flex: 1 }}>
                  <label>Dataset name</label>
                  <input type="text" value={hfDataset} onChange={(e) => setHfDataset(e.target.value)} placeholder="imdb" />
                </div>
              </div>
              <div className="row">
                <div style={{ flex: 1 }}>
                  <label>Split</label>
                  <input type="text" value={hfSplit} onChange={(e) => setHfSplit(e.target.value)} placeholder="train" />
                </div>
                <div style={{ flex: 1 }}>
                  <label>Subset/config (optional)</label>
                  <input type="text" value={hfSubset} onChange={(e) => setHfSubset(e.target.value)} placeholder="plain_text" />
                </div>
              </div>
              <div className="row">
                <div style={{ flex: 1 }}>
                  <label>Max rows</label>
                  <input type="number" value={hfMaxRows} onChange={(e) => setHfMaxRows(e.target.value)} min={50} step={50} />
                </div>
                <div style={{ flex: 1 }}>
                  <label>Text column (optional)</label>
                  <input type="text" value={hfTextCol} onChange={(e) => setHfTextCol(e.target.value)} placeholder="text" />
                </div>
                <div style={{ flex: 1 }}>
                  <label>Label column (optional)</label>
                  <input type="text" value={hfLabelCol} onChange={(e) => setHfLabelCol(e.target.value)} placeholder="label" />
                </div>
              </div>
            </>
          )}

          <div className="row" style={{ marginTop: 12 }}>
            <button onClick={runAnalysis} disabled={loading}>
              {loading ? "Analyzing..." : "Run analysis"}
            </button>
            <span className="mono">API: {apiBaseUrl()}</span>
          </div>

          {error ? (
            <div className="row" style={{ marginTop: 12 }}>
              <span className="pill sev-critical">Error</span>
              <span className="mono" style={{ whiteSpace: "pre-wrap" }}>
                {error}
              </span>
            </div>
          ) : null}
        </div>

        <div className="panel">
          <h2>Report</h2>
          {!report ? (
            <div className="mono">Run an analysis to see results.</div>
          ) : (
            <>
              <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
                <div>
                  <p className="score">{report.summary?.quality_score ?? report.score?.overall}</p>
                  <div className="verdict">
                    Verdict: <span className="pill">{report.summary?.verdict ?? report.score?.verdict}</span>
                    {"  "}
                    <span className="pill">flagged rows: {report.summary?.flagged_rows ?? 0}</span>
                  </div>
                </div>
                <div style={{ width: 420, height: 220 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={subscoresChart} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="name" tick={{ fill: "rgba(231,239,255,0.7)", fontSize: 11 }} />
                      <YAxis domain={[0, 100]} tick={{ fill: "rgba(231,239,255,0.7)", fontSize: 11 }} />
                      <Tooltip contentStyle={{ background: "#0b1220", border: "1px solid rgba(255,255,255,0.12)", borderRadius: 10 }} />
                      <Bar dataKey="score" fill="rgba(106,163,255,0.85)" radius={[10, 10, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="row" style={{ gap: 8, marginTop: 10 }}>
                <SeverityBadge severity="critical" />
                <span className="mono">{report.summary?.critical_issues ?? 0}</span>
                <SeverityBadge severity="warning" />
                <span className="mono">{report.summary?.warnings ?? 0}</span>
                <SeverityBadge severity="info" />
                <span className="mono">{report.summary?.info ?? 0}</span>
              </div>

              <div style={{ marginTop: 14 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="pill">Issues</span>
                  <span className="mono">generated_at: {report.generated_at}</span>
                </div>
                <ul className="issues">
                  {(report.issues || []).slice(0, 10).map((i, idx) => (
                    <li key={idx}>
                      <span className={`mono ${i.severity === "critical" ? "sev-critical" : i.severity === "warning" ? "sev-warning" : "sev-info"}`}>
                        [{i.severity}] {i.module}/{i.code}:
                      </span>{" "}
                      {i.message}
                    </li>
                  ))}
                </ul>
                {(report.issues || []).length > 10 ? <div className="mono">Showing first 10 issues.</div> : null}
              </div>

              <div style={{ marginTop: 14 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="pill">Recommendations</span>
                </div>
                <ul className="issues">
                  {(report.recommendations || []).slice(0, 10).map((r, idx) => (
                    <li key={idx}>{r}</li>
                  ))}
                </ul>
              </div>

              <div style={{ marginTop: 14 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="pill">Row flags (sample)</span>
                  <span className="mono">up to 50</span>
                </div>
                <div className="mono" style={{ marginTop: 8, overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr>
                        <th align="left">row_id</th>
                        <th align="left">severity</th>
                        <th align="left">flags</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(report.row_flags || []).slice(0, 50).map((r) => (
                        <tr key={r.row_id}>
                          <td style={{ padding: "6px 0", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>{r.row_id}</td>
                          <td style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>{r.severity}</td>
                          <td style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>{(r.flags || []).join(", ")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
