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
  const label = severity === "critical" ? "High risk" : severity === "warning" ? "Needs attention" : "FYI";
  return <span className={`pill ${cls}`}>{label}</span>;
}

function verdictLabel(verdict) {
  if (verdict === "production_ready") return "Ready to use";
  if (verdict === "needs_cleanup") return "Needs cleanup";
  if (verdict === "high_risk") return "High risk";
  return verdict || "Unknown";
}

function statusClass(status) {
  const norm = String(status || "").toLowerCase();
  if (norm === "excellent" || norm === "good" || norm === "low") return "sev-info";
  if (norm === "moderate" || norm === "elevated") return "sev-warning";
  if (norm === "high" || norm === "needs work") return "sev-critical";
  return "";
}

function IssueItem({ issue }) {
  return (
    <li>
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
        <div>
          <div>
            <span className="mono">{issue.headline || issue.code}</span>
          </div>
          <div className="issue-plain">{issue.what_we_found || issue.explanation || issue.message}</div>
        </div>
        <SeverityBadge severity={issue.severity} />
      </div>
      {issue.how_serious ? <div className="issue-why">Why this matters: {issue.how_serious}</div> : null}
      {issue.what_you_can_do ? <div className="issue-why">What you can do: {issue.what_you_can_do}</div> : null}
      {Array.isArray(issue.examples) && issue.examples.length > 0 ? (
        <ul className="issue-examples">
          {issue.examples.slice(0, 3).map((example, exampleIdx) => (
            <li key={exampleIdx}>
              <span className="mono">{example.title ? `${example.title}: ` : "Example: "}</span>
              {example.details || ""}
            </li>
          ))}
        </ul>
      ) : null}
      {issue.technical_details ? (
        <details className="issue-tech">
          <summary className="mono">Show technical details</summary>
          <pre className="mono issue-tech-pre">{JSON.stringify(issue.technical_details, null, 2)}</pre>
        </details>
      ) : null}
    </li>
  );
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
    const labels = {
      missing: "Missing",
      exact_duplicates: "Duplicates",
      label_issues: "Labels",
      toxicity: "Toxicity",
      semantic_warnings: "Semantic",
      domain_mixing: "Domain",
      modality_warning: "Modality"
    };
    return Object.entries(report.score.subscores).map(([k, v]) => ({
      key: k,
      name: labels[k] || String(k).replaceAll("_", " "),
      score: v
    }));
  }, [report]);

  const issueGroups = useMemo(() => {
    if (!report) return { needsAttention: [], optional: [] };
    const grouped = report.issue_groups || {};
    if (Array.isArray(grouped.needs_attention) || Array.isArray(grouped.optional_improvements)) {
      return {
        needsAttention: grouped.needs_attention || [],
        optional: grouped.optional_improvements || []
      };
    }
    const all = report.issues || [];
    return {
      needsAttention: all.filter((issue) => issue.severity !== "info"),
      optional: all.filter((issue) => issue.severity === "info")
    };
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
    <div className="container analyze-container">
      <h1 className="title">Dataset Quality Analyzer</h1>
      <p className="subtitle">Upload a dataset and get a plain-language quality report with concrete cleanup actions.</p>

      <div className="analyze-layout">
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

        <div className="panel report-panel">
          <h2>Report</h2>
          {!report ? (
            <div className="mono">Run an analysis to see results.</div>
          ) : (
            <>
              <div className="report-top">
                <div className="report-kpis">
                  <p className="score">{report.summary?.quality_score ?? report.score?.overall}</p>
                  <div className="verdict">
                    Status: <span className="pill">{verdictLabel(report.summary?.verdict ?? report.score?.verdict)}</span>
                    {"  "}
                    <span className="pill">flagged rows: {report.summary?.flagged_rows ?? 0}</span>
                    {"  "}
                    <span className="pill">modality: {report.meta?.modality?.modality ?? "unknown"}</span>
                  </div>
                </div>
                <div className="report-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={subscoresChart} margin={{ top: 8, right: 8, left: -12, bottom: 0 }} barCategoryGap="18%">
                      <CartesianGrid stroke="rgba(17,19,26,0.09)" vertical={false} />
                      <XAxis
                        dataKey="name"
                        tick={{ fill: "#4f5564", fontSize: 12 }}
                        axisLine={{ stroke: "rgba(17,19,26,0.2)" }}
                        tickLine={{ stroke: "rgba(17,19,26,0.2)" }}
                      />
                      <YAxis
                        domain={[0, 100]}
                        width={28}
                        tick={{ fill: "#4f5564", fontSize: 12 }}
                        axisLine={{ stroke: "rgba(17,19,26,0.2)" }}
                        tickLine={{ stroke: "rgba(17,19,26,0.2)" }}
                      />
                      <Tooltip
                        formatter={(value) => [Number(value).toFixed(2), "Score"]}
                        labelFormatter={(value, payload) => payload?.[0]?.payload?.key || value}
                        contentStyle={{
                          background: "rgba(255,255,255,0.96)",
                          border: "1px solid rgba(17,19,26,0.12)",
                          borderRadius: 10,
                          color: "#11131a"
                        }}
                      />
                      <Bar dataKey="score" fill="rgba(106,163,255,0.85)" radius={[10, 10, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="row" style={{ gap: 8, marginTop: 10 }}>
                <SeverityBadge severity="critical" />
                <span className="mono">{report.summary?.status_counts?.high_risk ?? report.summary?.critical_issues ?? 0}</span>
                <SeverityBadge severity="warning" />
                <span className="mono">
                  {report.summary?.status_counts?.needs_attention ?? report.summary?.warnings ?? 0}
                </span>
                <SeverityBadge severity="info" />
                <span className="mono">{report.summary?.status_counts?.fyi ?? report.summary?.info ?? 0}</span>
              </div>

              {report.modality_explanation ? (
                <div className="issue-why" style={{ marginTop: 10 }}>
                  Modality: {report.modality_explanation}
                </div>
              ) : null}

              {Array.isArray(report.score_breakdown?.cards) && report.score_breakdown.cards.length > 0 ? (
                <div style={{ marginTop: 14 }}>
                  <div className="row" style={{ justifyContent: "space-between" }}>
                    <span className="pill">Score Breakdown</span>
                    <span className="pill">{report.score_breakdown?.overall_status}</span>
                  </div>
                  <div className="breakdown-grid">
                    {report.score_breakdown.cards.map((card) => (
                      <div className="breakdown-card" key={card.key}>
                        <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
                          <span className="mono">{card.title}</span>
                          <span className={`pill ${statusClass(card.status)}`}>{card.status}</span>
                        </div>
                        <div className="breakdown-score">{card.score == null ? "N/A" : card.score}</div>
                        <div className="issue-why">{card.details}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              <div style={{ marginTop: 14 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="pill">Needs Attention</span>
                  <span className="mono">generated_at: {report.generated_at}</span>
                </div>
                <ul className="issues">
                  {(issueGroups.needsAttention || []).slice(0, 12).map((issue, idx) => (
                    <IssueItem issue={issue} key={`${issue.code}-${idx}`} />
                  ))}
                </ul>
                {(issueGroups.needsAttention || []).length === 0 ? <div className="mono">No urgent issues found.</div> : null}
              </div>

              <div style={{ marginTop: 14 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="pill">Everything Else Looks Good</span>
                </div>
                <ul className="issues">
                  {(report.good_to_go || []).map((message, idx) => (
                    <li key={idx}>{message}</li>
                  ))}
                </ul>
                {(report.good_to_go || []).length === 0 ? (
                  <div className="mono">No additional positive checks to display for this run.</div>
                ) : null}
              </div>

              <div style={{ marginTop: 14 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="pill">Optional Improvements</span>
                </div>
                <ul className="issues">
                  {(issueGroups.optional || []).slice(0, 8).map((issue, idx) => (
                    <IssueItem issue={issue} key={`${issue.code}-${idx}`} />
                  ))}
                </ul>
                {(issueGroups.optional || []).length === 0 ? <div className="mono">None.</div> : null}
              </div>

              <div style={{ marginTop: 14 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span className="pill">Recommendations</span>
                </div>
                <ul className="issues">
                  {(report.recommendations || []).slice(0, 12).map((r, idx) => (
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
                          <td style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>{r.severity_label || r.severity}</td>
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
