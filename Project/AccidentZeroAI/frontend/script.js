const API_BASE =
    (window.__ACCIDENTZERO_CONFIG__ && window.__ACCIDENTZERO_CONFIG__.API_BASE) ||
    "http://127.0.0.1:8000";
const TABLEAU_EMBED_URL =
    (window.__ACCIDENTZERO_CONFIG__ && window.__ACCIDENTZERO_CONFIG__.TABLEAU_EMBED_URL) || "";
const TABLEAU_AUTO_REFRESH_MS = Number(
    (window.__ACCIDENTZERO_CONFIG__ && window.__ACCIDENTZERO_CONFIG__.TABLEAU_AUTO_REFRESH_MS) || 0
);
const FEATURE_COLUMNS = [
    "shift_hours",
    "overtime_hours",
    "worker_experience",
    "equipment_age",
    "maintenance_score",
    "temperature",
    "humidity",
    "inspection_score",
];
const CHART_ROW_CAP = 500;
/** Matches Gemini / server batch limit for scene analysis. */
const MAX_ANALYSIS_IMAGES = 16;

/** Last tabular prediction snapshot for reverting image blend (manual or Excel). */
let tabularOutputSnapshot = null;
let imageBlendApplied = false;

/** Human-readable Y-axis text for each safety feature column. */
const FEATURE_AXIS_META = {
    shift_hours: { label: "Shift hours", yUnit: "h" },
    overtime_hours: { label: "Overtime hours", yUnit: "h" },
    worker_experience: { label: "Worker experience", yUnit: "years" },
    equipment_age: { label: "Equipment age", yUnit: "years" },
    maintenance_score: { label: "Maintenance score", yUnit: "0–100" },
    temperature: { label: "Temperature", yUnit: "°C" },
    humidity: { label: "Relative humidity", yUnit: "%" },
    inspection_score: { label: "Inspection score", yUnit: "0–100" },
};

function axisStyle() {
    return {
        ticks: { color: "#9ca3af", font: { size: 10 } },
        grid: { color: "rgba(255,255,255,0.06)" },
    };
}

function axisTitlesForColumnChart(col, chartType) {
    const m = FEATURE_AXIS_META[col] || { label: col, yUnit: "" };
    const yFull = m.yUnit ? `${m.label} (${m.yUnit})` : m.label;
    if (chartType === "histogram") {
        return { x: `${m.label} — value bin`, y: "Number of rows" };
    }
    return { x: "Row index (sample order)", y: yFull };
}

let modelOverviewChart = null;
let riskDistributionChart = null;
let rowModelChart = null;
let sourceRiskChart = null;
let lastBatchRows = [];
let corrAnimationHandle = null;

let lastOriginalRows = [];
let lastCleanedRows = [];
let lastPredictionRows = [];
let lastImageAnalysis = null;
let columnChartInstances = [];
let columnChartRaf = null;
let excelSourceRiskCtx = null;
let imageSourceRiskCtx = null;
let tableauRefreshTimer = null;

function withTableauQuery(url) {
    if (!url) return "";
    const hasQuery = url.includes("?");
    const base = hasQuery ? `${url}&:showVizHome=no` : `${url}?:showVizHome=no`;
    return `${base}&_ts=${Date.now()}`;
}

function initTableauEmbed() {
    const frame = document.getElementById("tableau_embed");
    const status = document.getElementById("tableau_status");
    if (!frame || !status) return;

    if (!TABLEAU_EMBED_URL) {
        status.textContent = "Tableau URL not configured. Set TABLEAU_EMBED_URL in frontend/config.js.";
        frame.style.display = "none";
        return;
    }

    frame.style.display = "block";
    status.textContent = "Loading Tableau dashboard...";
    frame.onload = () => {
        status.textContent = "Tableau dashboard loaded.";
    };
    frame.src = withTableauQuery(TABLEAU_EMBED_URL);

    if (tableauRefreshTimer) {
        clearInterval(tableauRefreshTimer);
        tableauRefreshTimer = null;
    }
    if (TABLEAU_AUTO_REFRESH_MS > 0) {
        tableauRefreshTimer = setInterval(() => {
            refreshTableauEmbed(false);
        }, TABLEAU_AUTO_REFRESH_MS);
        status.textContent = `Tableau dashboard loaded. Auto-refresh: ${Math.round(TABLEAU_AUTO_REFRESH_MS / 1000)}s.`;
    }
}

function refreshTableauEmbed(showMessage = true) {
    const frame = document.getElementById("tableau_embed");
    const status = document.getElementById("tableau_status");
    if (!frame || !status) return;
    if (!TABLEAU_EMBED_URL) {
        status.textContent = "Tableau URL not configured.";
        return;
    }
    if (showMessage) status.textContent = "Refreshing Tableau dashboard...";
    frame.src = withTableauQuery(TABLEAU_EMBED_URL);
}

/** FastAPI may return detail as a string or a validation error array. */
function formatFastApiDetail(detail) {
    if (detail == null) return "";
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
        return detail
            .map((d) => (d && typeof d === "object" && d.msg != null ? d.msg : String(d)))
            .join("; ");
    }
    if (typeof detail === "object") {
        if (detail.msg != null) return String(detail.msg);
        if (detail.message != null) return String(detail.message);
        try {
            return JSON.stringify(detail);
        } catch {
            return String(detail);
        }
    }
    return String(detail);
}

/** Prefer FastAPI detail; fall back to other common error JSON shapes. */
function extractHttpErrorMessage(data, rawText) {
    if (data && typeof data === "object") {
        const fromDetail = formatFastApiDetail(data.detail);
        if (fromDetail) return fromDetail;
        if (typeof data.message === "string" && data.message) return data.message;
        const err = data.error;
        if (err && typeof err === "object" && typeof err.message === "string") return err.message;
    }
    if (rawText && typeof rawText === "string" && rawText.trim()) return rawText.trim();
    return "";
}

function imageAnalysisFailureHint(message) {
    const m = (message || "").toLowerCase();
    const tail =
        message && message.length > 0
            ? "\n\n" + message.slice(0, 500) + (message.length > 500 ? "…" : "")
            : "";
    if (
        m.includes("api_key_invalid") ||
        m.includes("expired") ||
        m.includes("renew the api key") ||
        m.includes("invalid api key") ||
        m.includes("api key not valid") ||
        m.includes("pass a valid api key")
    ) {
        return (
            "Your Gemini API key is invalid or expired. Create a new key at https://aistudio.google.com/apikey , set GEMINI_API_KEY in the project .env, then restart the API." +
            tail
        );
    }
    if (m.includes("not set") && m.includes("gemini")) {
        return "Set GEMINI_API_KEY in the project .env and restart the API." + tail;
    }
    if (m.trim() === "not found" || (m.includes("not found") && m.includes("404"))) {
        return (
            "The API returned 404—often the running server is an older build without /analyze/images. Restart Uvicorn from this project folder, or reload script v1.1.10+ to use automatic one-by-one fallback." +
            tail
        );
    }
    if (m.includes("failed to fetch") || m.includes("networkerror")) {
        return (
            "Could not reach the API at " +
            API_BASE +
            ". Start the backend (uvicorn) and refresh the page." +
            tail
        );
    }
    if (message && message.length > 0) {
        return "Image analysis failed." + tail;
    }
    return "Image analysis failed. Check the result panel below and the browser console (F12) for details.";
}

/** Same thresholds as `models/ensemble_engine.classify_risk_level` (risk score 0–100). */
function riskLevelFromScore(score) {
    const s = Number(score);
    if (Number.isNaN(s)) return "MODERATE";
    if (s < 30) return "LOW";
    if (s < 60) return "MODERATE";
    if (s < 80) return "HIGH";
    return "CRITICAL";
}

/** Batch aggregate level from API `risk_level_aggregate` or mean `risk_score` (same thresholds as single-row). */
function batchRiskLevelFromSummary(summary) {
    if (!summary || typeof summary !== "object") return "--";
    if (summary.risk_level_aggregate) return String(summary.risk_level_aggregate);
    const rs = Number(summary.risk_score);
    if (!Number.isNaN(rs)) return riskLevelFromScore(rs);
    return "--";
}

function safeNum(v, d = 0) {
    const n = Number(v);
    return Number.isNaN(n) ? d : n;
}

function updateSourceAnalysisPanel() {
    const excelEl = document.getElementById("source_excel_risk");
    const imageEl = document.getElementById("source_image_risk");
    const fusedEl = document.getElementById("source_fused_risk");
    const noteEl = document.getElementById("source_analysis_note");
    if (!excelEl || !imageEl || !fusedEl || !noteEl) return;

    const excelRisk = excelSourceRiskCtx ? safeNum(excelSourceRiskCtx.riskScore, NaN) : NaN;
    const imageRisk = imageSourceRiskCtx ? safeNum(imageSourceRiskCtx.riskScore, NaN) : NaN;

    excelEl.textContent = Number.isNaN(excelRisk)
        ? "--"
        : `${excelRisk.toFixed(2)} (${riskLevelFromScore(excelRisk)})`;
    imageEl.textContent = Number.isNaN(imageRisk)
        ? "--"
        : `${imageRisk.toFixed(2)} (${riskLevelFromScore(imageRisk)})`;

    let fusedRisk = NaN;
    let note = "Run Excel prediction and image analysis to populate individual source risks.";
    if (!Number.isNaN(excelRisk) && !Number.isNaN(imageRisk)) {
        const excelW = Math.max(1, safeNum(excelSourceRiskCtx.weight, 1));
        const imageW = Math.max(1, safeNum(imageSourceRiskCtx.weight, 1));
        fusedRisk = (excelRisk * excelW + imageRisk * imageW) / (excelW + imageW);
        note =
            `Fused risk uses weighted mean by data volume: Excel weight=${excelW}, Image weight=${imageW}. ` +
            `Individual risks remain shown separately.`;
    } else if (!Number.isNaN(excelRisk)) {
        fusedRisk = excelRisk;
        note = "Only Excel analysis is available right now.";
    } else if (!Number.isNaN(imageRisk)) {
        fusedRisk = imageRisk;
        note = "Only image analysis is available right now.";
    }

    fusedEl.textContent = Number.isNaN(fusedRisk)
        ? "--"
        : `${fusedRisk.toFixed(2)} (${riskLevelFromScore(fusedRisk)})`;
    noteEl.textContent = note;

    ensureCharts();
    if (sourceRiskChart) {
        sourceRiskChart.data.datasets[0].data = [
            Number.isNaN(excelRisk) ? 0 : excelRisk,
            Number.isNaN(imageRisk) ? 0 : imageRisk,
            Number.isNaN(fusedRisk) ? 0 : fusedRisk,
        ];
        sourceRiskChart.update();
    }
}

function captureTabularOutputFromPredict(result, mode) {
    imageBlendApplied = false;
    const blendHint = document.getElementById("image_blend_hint");
    const mainNote = document.getElementById("main_output_blend_note");
    if (blendHint) blendHint.innerHTML = "";
    if (mainNote) mainNote.textContent = "";
    if (mode === "single") {
        tabularOutputSnapshot = {
            mode: "single",
            ensemble_probability: result.ensemble_probability,
            risk_score: result.risk_score,
            risk_level: result.risk_level,
        };
    } else {
        tabularOutputSnapshot = {
            mode: "batch",
            summary: result.summary || {},
            count: result.count ?? (result.rows ? result.rows.length : 0),
        };
    }
    updateImageBlendButtons();
}

function updateImageBlendButtons() {
    const addBtn = document.getElementById("btn_add_image_risk");
    const revBtn = document.getElementById("btn_revert_image_risk");
    const canBlend = Boolean(tabularOutputSnapshot && lastImageAnalysis);
    if (addBtn) addBtn.disabled = !canBlend || imageBlendApplied;
    if (revBtn) revBtn.disabled = !imageBlendApplied;
}

/**
 * Blend tabular risk score (0–100) with mean image risk % using an equal-weight average.
 * This is a deterministic convex combination on the same scale as `risk_score`.
 */
function applyImageRiskToMainOutput() {
    if (!tabularOutputSnapshot) {
        alert("Run a tabular prediction (manual form or Excel batch) first.");
        return;
    }
    if (!lastImageAnalysis) {
        alert("Run scene image analysis first.");
        return;
    }
    const imgPctRaw =
        lastImageAnalysis.aggregate_risk_probability_percent != null
            ? lastImageAnalysis.aggregate_risk_probability_percent
            : lastImageAnalysis.risk_probability_percent;
    const imgPct = Number(imgPctRaw);
    if (Number.isNaN(imgPct)) {
        alert("No numeric image risk score available.");
        return;
    }
    let tabRisk = 0;
    if (tabularOutputSnapshot.mode === "single") {
        tabRisk = Number(tabularOutputSnapshot.risk_score);
    } else {
        tabRisk = Number(tabularOutputSnapshot.summary.risk_score);
    }
    if (Number.isNaN(tabRisk)) {
        alert("Could not read tabular risk score.");
        return;
    }
    const combined = Math.min(100, Math.max(0, (tabRisk + imgPct) / 2));
    const combinedProb = combined / 100;
    const level = riskLevelFromScore(combined);

    document.getElementById("probability").innerText = "Probability: " + combinedProb.toFixed(4);
    document.getElementById("risk_score").innerText = "Risk Score: " + combined.toFixed(2);
    document.getElementById("risk_level").innerText = "Risk Level: " + level;

    imageBlendApplied = true;
    const hint = document.getElementById("image_blend_hint");
    if (hint) {
        hint.innerHTML =
            `<span class="muted small">Main metrics use <strong>½·tabular + ½·image</strong> mean risk (0–100). ` +
            `Tabular ${tabRisk.toFixed(1)} + image ${imgPct.toFixed(1)} → <strong>${combined.toFixed(2)}</strong>. ` +
            `Charts still show tabular batch features only.</span>`;
    }
    const mainNote = document.getElementById("main_output_blend_note");
    if (mainNote) {
        mainNote.textContent =
            "Showing blended output (tabular + image). Use Revert to restore tabular-only metrics.";
    }
    updateImageBlendButtons();
}

function revertImageBlendFromMainOutput() {
    if (!tabularOutputSnapshot || !imageBlendApplied) return;
    if (tabularOutputSnapshot.mode === "single") {
        const r = tabularOutputSnapshot;
        document.getElementById("probability").innerText = "Probability: " + r.ensemble_probability;
        document.getElementById("risk_score").innerText = "Risk Score: " + r.risk_score;
        document.getElementById("risk_level").innerText = "Risk Level: " + r.risk_level;
    } else {
        const s = tabularOutputSnapshot.summary;
        const avgProb = s.ensemble_probability ?? "--";
        const avgRisk = s.risk_score ?? "--";
        document.getElementById("probability").innerText = "Probability: " + avgProb;
        document.getElementById("risk_score").innerText = "Risk Score: " + avgRisk;
        document.getElementById("risk_level").innerText =
            "Risk Level: " + batchRiskLevelFromSummary(tabularOutputSnapshot.summary);
    }
    imageBlendApplied = false;
    const hint = document.getElementById("image_blend_hint");
    if (hint) hint.innerHTML = "";
    const mainNote = document.getElementById("main_output_blend_note");
    if (mainNote) mainNote.textContent = "";
    updateImageBlendButtons();
}

function normalizeImageAnalysisForInsights(img) {
    if (!img) return null;
    if (img.per_image && img.per_image.length) {
        return {
            ...img,
            risk_probability_percent: img.aggregate_risk_probability_percent,
            risk_level: img.aggregate_risk_level,
            accident_risk_factors: img.accident_risk_factors || [],
        };
    }
    return img;
}

async function predictRisk() {
    try {
        const data = {
            shift_hours: parseFloat(document.getElementById("shift_hours").value),
            overtime_hours: parseFloat(document.getElementById("overtime_hours").value),
            worker_experience: parseFloat(document.getElementById("worker_experience").value),
            equipment_age: parseFloat(document.getElementById("equipment_age").value),
            maintenance_score: parseFloat(document.getElementById("maintenance_score").value),
            temperature: parseFloat(document.getElementById("temperature").value),
            humidity: parseFloat(document.getElementById("humidity").value),
            inspection_score: parseFloat(document.getElementById("inspection_score").value)
        };

        const response = await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const t = await response.text();
            throw new Error(t || response.statusText);
        }

        const result = await response.json();

        document.getElementById("probability").innerText =
            "Probability: " + result.ensemble_probability;

        document.getElementById("risk_score").innerText =
            "Risk Score: " + result.risk_score;

        document.getElementById("risk_level").innerText =
            "Risk Level: " + result.risk_level;

        excelSourceRiskCtx = {
            riskScore: safeNum(result.risk_score, 0),
            weight: 1,
            mode: "single",
        };
        updateSourceAnalysisPanel();

        captureTabularOutputFromPredict(result, "single");

        setBatchSummary(null);
        renderBatchTable([]);

        lastOriginalRows = pickOriginalRowsPreview(result);
        lastCleanedRows = pickCleanedRowsPreview(result);
        if (!lastCleanedRows.length && result.cleaned_inputs) {
            lastCleanedRows = [result.cleaned_inputs];
        }
        if (!lastOriginalRows.length && lastCleanedRows.length) {
            lastOriginalRows = lastCleanedRows.map((r) => ({ ...r }));
        }
        lastPredictionRows = [result];
        renderImputationPanel(null, result.imputation);
        const note = document.getElementById("manual_imputation_note");
        if (note && result.imputation && result.imputation.note) {
            note.textContent = result.imputation.note;
        }

        updateChartsFromSingle(result);
        if (columnChartRaf) cancelAnimationFrame(columnChartRaf);
        columnChartRaf = null;
        refreshColumnCharts();
        await runAccidentInsights();
    } catch (error) {
        console.error(error);
        alert("Prediction failed. Check console for details.");
    }
}

function setBatchSummary(summary) {
    const el = document.getElementById("batch_summary");
    if (!summary) {
        el.innerHTML = "";
        return;
    }
    const ew = summary.ensemble_weighting;
    let formulaLine = "";
    if (ew && ew.weights) {
        const w = ew.weights;
        formulaLine = `<div class="small muted" style="margin-top:8px;line-height:1.35">Ensemble probability is a <strong>fixed weighted sum</strong> P = Σ wᵢ·pᵢ (eight models, Σwᵢ = 1). Weights: xgb ${w.xgb}, lgbm ${w.lgbm}, cat ${w.cat}, hgb ${w.hgb}, extra_trees ${w.extra_trees}, lstm ${w.lstm}, iso ${w.iso}, stacking ${w.stacking}.</div>`;
    }
    el.innerHTML = `
        <div class="metric">
            <span>Batch rows</span>
            <strong>${summary.count}</strong>
        </div>
        <div class="metric">
            <span>Avg ensemble P</span>
            <strong>${summary.ensemble_probability}</strong>
        </div>
        <div class="metric">
            <span>Avg risk score</span>
            <strong>${summary.risk_score}</strong>
        </div>
        <div class="metric">
            <span>Aggregate risk level</span>
            <strong>${batchRiskLevelFromSummary(summary)}</strong>
        </div>
        ${formulaLine}
    `;
}

function renderBatchTable(rows) {
    const tbody = document.querySelector("#batch_table tbody");
    tbody.innerHTML = "";
    if (!rows || rows.length === 0) return;

    lastBatchRows = rows;

    rows.forEach((r, idx) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${r.xgb_probability ?? "--"}</td>
            <td>${r.lgbm_probability ?? "--"}</td>
            <td>${r.cat_probability ?? "--"}</td>
            <td>${r.hgb_probability ?? "--"}</td>
            <td>${r.extra_trees_probability ?? "--"}</td>
            <td>${r.lstm_probability ?? "--"}</td>
            <td>${r.iso_anomaly_score ?? "--"}</td>
            <td>${r.stacking_probability ?? "--"}</td>
            <td>${r.ensemble_tree_probability ?? "--"}</td>
            <td>${r.ensemble_probability ?? "--"}</td>
            <td>${r.risk_score ?? "--"}</td>
            <td>${r.risk_level ?? "--"}</td>
        `;
        tr.addEventListener("click", () => {
            updateRowModelChart(r, idx);
        });
        tbody.appendChild(tr);
    });
}

async function predictFromExcel() {
    const statusEl = document.getElementById("excel_status");
    if (statusEl) statusEl.textContent = "Reading Excel and imputing missing values…";

    try {
        const input = document.getElementById("excel_file");
        if (!input.files || input.files.length === 0) {
            alert("Please choose an Excel (.xlsx) file first.");
            if (statusEl) statusEl.textContent = "";
            return;
        }

        const form = new FormData();
        form.append("file", input.files[0]);

        const response = await fetch(`${API_BASE}/predict/excel`, {
            method: "POST",
            body: form
        });

        if (!response.ok) {
            const t = await response.text();
            throw new Error(t || response.statusText);
        }

        const result = await response.json();

        const summ = result.summary || {};
        const avgProb = summ.ensemble_probability ?? "--";
        const avgRisk = summ.risk_score ?? "--";
        const avgLevel = batchRiskLevelFromSummary(summ);

        document.getElementById("probability").innerText =
            "Probability: " + avgProb;
        document.getElementById("risk_score").innerText =
            "Risk Score: " + avgRisk;
        document.getElementById("risk_level").innerText =
            "Risk Level: " + avgLevel;

        excelSourceRiskCtx = {
            riskScore: safeNum(avgRisk, 0),
            weight: Math.max(1, safeNum(result.count, (result.rows || []).length || 1)),
            mode: "batch",
        };
        updateSourceAnalysisPanel();

        captureTabularOutputFromPredict(result, "batch");

        setBatchSummary({
            count: result.count ?? (result.rows ? result.rows.length : 0),
            ensemble_probability: avgProb,
            risk_score: avgRisk,
            ensemble_weighting: summ.ensemble_weighting,
            risk_level_aggregate: summ.risk_level_aggregate,
        });

        const rows = result.rows || [];
        renderBatchTable(rows);
        updateChartsFromBatch(rows, result.summary || {});

        lastOriginalRows = pickOriginalRowsPreview(result);
        lastCleanedRows = pickCleanedRowsPreview(result);
        lastPredictionRows = rows;
        renderImputationPanel(result.imputation, null);
        if (columnChartRaf) cancelAnimationFrame(columnChartRaf);
        columnChartRaf = null;
        refreshColumnCharts();
        await runAccidentInsights();

        if (statusEl) statusEl.textContent = "Done. Feature charts above show data before and after cleaning.";
    } catch (error) {
        console.error(error);
        alert("Batch prediction failed. Check console for details.");
        const statusEl = document.getElementById("excel_status");
        if (statusEl) statusEl.textContent = "Failed.";
    }
}

function ensureCharts() {
    try {
        const overviewCanvas = document.getElementById("modelOverviewChart");
        const riskCanvas = document.getElementById("riskDistributionChart");
        const rowCanvas = document.getElementById("rowModelChart");
        const sourceCanvas = document.getElementById("sourceRiskChart");

        if (!overviewCanvas || !riskCanvas || !rowCanvas || !sourceCanvas) {
            return;
        }

        const overviewCtx = overviewCanvas.getContext("2d");
        const riskCtx = riskCanvas.getContext("2d");
        const rowCtx = rowCanvas.getContext("2d");
        const sourceCtx = sourceCanvas.getContext("2d");

        if (!modelOverviewChart) {
            modelOverviewChart = new Chart(overviewCtx, {
                type: "bar",
                data: {
                    labels: ["XGB", "LGBM", "CAT", "HistGBM", "ExtraTrees", "LSTM", "ISO", "Stacking", "Ensemble"],
                    datasets: [{
                        label: "Average probability / score",
                        backgroundColor: [
                            "#38bdf8",
                            "#22c55e",
                            "#eab308",
                            "#4ade80",
                            "#fb923c",
                            "#a855f7",
                            "#f97316",
                            "#f472b6",
                            "#facc15"
                        ],
                        data: [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: "Average model output (batch or single row)",
                            color: "#9ca3af",
                            font: { size: 12 },
                        },
                    },
                    scales: {
                        x: {
                            ...axisStyle(),
                            title: {
                                display: true,
                                text: "Model",
                                color: "#9ca3af",
                                font: { size: 11 },
                            },
                        },
                        y: {
                            beginAtZero: true,
                            max: 1,
                            ...axisStyle(),
                            title: {
                                display: true,
                                text: "Probability / score (0–1)",
                                color: "#9ca3af",
                                font: { size: 11 },
                            },
                        },
                    },
                },
            });
        }

        if (!riskDistributionChart) {
            riskDistributionChart = new Chart(riskCtx, {
                type: "bar",
                data: {
                    labels: ["0–30 (LOW)", "30–60 (MODERATE)", "60–80 (HIGH)", "80–100 (CRITICAL)"],
                    datasets: [{
                        label: "Row count",
                        backgroundColor: "#38bdf8",
                        data: [0, 0, 0, 0]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: "How many rows fall in each risk band",
                            color: "#9ca3af",
                            font: { size: 12 },
                        },
                    },
                    scales: {
                        x: {
                            ...axisStyle(),
                            title: {
                                display: true,
                                text: "Risk score band",
                                color: "#9ca3af",
                                font: { size: 11 },
                            },
                        },
                        y: {
                            beginAtZero: true,
                            precision: 0,
                            ...axisStyle(),
                            title: {
                                display: true,
                                text: "Number of rows",
                                color: "#9ca3af",
                                font: { size: 11 },
                            },
                        },
                    },
                },
            });
        }

        if (!rowModelChart) {
            rowModelChart = new Chart(rowCtx, {
                type: "radar",
                data: {
                    labels: ["XGB", "LGBM", "CAT", "HistGBM", "ExtraTrees", "LSTM", "ISO", "Stacking", "Ensemble"],
                    datasets: [{
                        label: "Selected row",
                        backgroundColor: "rgba(56, 189, 248, 0.2)",
                        borderColor: "#38bdf8",
                        pointBackgroundColor: "#38bdf8",
                        data: [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: "Per-model scores for the selected row (0–1 scale)",
                            color: "#9ca3af",
                            font: { size: 12 },
                        },
                    },
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 1,
                            ticks: { color: "#9ca3af", backdropColor: "transparent" },
                            grid: { color: "rgba(255,255,255,0.08)" },
                            pointLabels: {
                                color: "#9ca3af",
                                font: { size: 10 },
                            },
                        },
                    },
                },
            });
        }

        if (!sourceRiskChart) {
            sourceRiskChart = new Chart(sourceCtx, {
                type: "bar",
                data: {
                    labels: ["Excel", "Image", "Fused"],
                    datasets: [{
                        label: "Risk score (0-100)",
                        data: [0, 0, 0],
                        backgroundColor: ["#22c55e", "#a855f7", "#f59e0b"],
                    }],
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: "Individual source risk vs fused risk",
                            color: "#9ca3af",
                            font: { size: 12 },
                        },
                    },
                    scales: {
                        x: {
                            ...axisStyle(),
                            title: {
                                display: true,
                                text: "Data source",
                                color: "#9ca3af",
                                font: { size: 11 },
                            },
                        },
                        y: {
                            beginAtZero: true,
                            max: 100,
                            ...axisStyle(),
                            title: {
                                display: true,
                                text: "Risk score",
                                color: "#9ca3af",
                                font: { size: 11 },
                            },
                        },
                    },
                },
            });
        }
    } catch (error) {
    }
}

function updateChartsFromSingle(row) {
    ensureCharts();
    if (!modelOverviewChart || !riskDistributionChart) return;

    const summary = {
        xgb_probability: row.xgb_probability ?? row.ensemble_probability ?? 0,
        lgbm_probability: row.lgbm_probability ?? row.ensemble_probability ?? 0,
        cat_probability: row.cat_probability ?? row.ensemble_probability ?? 0,
        hgb_probability: row.hgb_probability ?? row.ensemble_probability ?? 0,
        extra_trees_probability: row.extra_trees_probability ?? row.ensemble_probability ?? 0,
        lstm_probability: row.lstm_probability ?? row.ensemble_probability ?? 0,
        iso_anomaly_score: row.iso_anomaly_score ?? 0,
        stacking_probability: row.stacking_probability ?? row.ensemble_probability ?? 0,
        ensemble_probability: row.ensemble_probability ?? 0
    };

    modelOverviewChart.data.datasets[0].data = [
        summary.xgb_probability,
        summary.lgbm_probability,
        summary.cat_probability,
        summary.hgb_probability,
        summary.extra_trees_probability,
        summary.lstm_probability,
        summary.iso_anomaly_score,
        summary.stacking_probability,
        summary.ensemble_probability
    ];
    modelOverviewChart.update();

    const buckets = [0, 0, 0, 0];
    const score = row.risk_score ?? (row.ensemble_probability ?? 0) * 100;
    if (score < 30) buckets[0] = 1;
    else if (score < 60) buckets[1] = 1;
    else if (score < 80) buckets[2] = 1;
    else buckets[3] = 1;
    riskDistributionChart.data.datasets[0].data = buckets;
    riskDistributionChart.update();

    updateRowModelChart(row, null);
}

function updateChartsFromBatch(rows, summary) {
    ensureCharts();
    if (!modelOverviewChart || !riskDistributionChart) return;

    const avgXgb = summary.xgb_probability ?? 0;
    const avgLgbm = summary.lgbm_probability ?? 0;
    const avgCat = summary.cat_probability ?? 0;
    const avgHgb = summary.hgb_probability ?? 0;
    const avgExtra = summary.extra_trees_probability ?? 0;
    const avgLstm = summary.lstm_probability ?? 0;
    const avgIso = summary.iso_anomaly_score ?? 0;
    const avgStack = summary.stacking_probability ?? 0;
    const avgEns = summary.ensemble_probability ?? 0;

    modelOverviewChart.data.datasets[0].data = [
        avgXgb,
        avgLgbm,
        avgCat,
        avgHgb,
        avgExtra,
        avgLstm,
        avgIso,
        avgStack,
        avgEns
    ];
    modelOverviewChart.update();

    const buckets = [0, 0, 0, 0];
    (rows || []).forEach(r => {
        const s = r.risk_score ?? (r.ensemble_probability ?? 0) * 100;
        if (s < 30) buckets[0] += 1;
        else if (s < 60) buckets[1] += 1;
        else if (s < 80) buckets[2] += 1;
        else buckets[3] += 1;
    });
    riskDistributionChart.data.datasets[0].data = buckets;
    riskDistributionChart.update();

    if (rows && rows.length > 0) {
        updateRowModelChart(rows[0], 0);
    }
}

function updateRowModelChart(row, idx) {
    ensureCharts();
    if (!rowModelChart) return;

    const caption = document.getElementById("rowModelCaption");
    if (idx == null) {
        caption.textContent = "Single prediction – showing model outputs for this case.";
    } else {
        caption.textContent = `Row ${idx + 1} – model outputs.`;
    }

    const values = [
        row.xgb_probability ?? 0,
        row.lgbm_probability ?? 0,
        row.cat_probability ?? 0,
        row.hgb_probability ?? 0,
        row.extra_trees_probability ?? 0,
        row.lstm_probability ?? 0,
        row.iso_anomaly_score ?? 0,
        row.stacking_probability ?? 0,
        row.ensemble_probability ?? 0
    ];

    rowModelChart.data.datasets[0].data = values;
    rowModelChart.update();
}

function renderImputationPanel(batchMeta, manualImp) {
    const el = document.getElementById("imputation_panel");
    if (!el) return;

    if (batchMeta && batchMeta.method === "empty") {
        el.innerHTML = "<p>No sheets with data found in this Excel file.</p>";
        return;
    }

    if (batchMeta && batchMeta.method === "none" && (batchMeta.total_imputed_cells ?? 0) === 0) {
        el.innerHTML =
            "<p><strong>No missing numeric values</strong> in loaded columns. KNN imputation was not required.</p>";
        return;
    }

    if (batchMeta && batchMeta.method && batchMeta.method !== "none" && batchMeta.method !== "empty") {
        const miss = batchMeta.missing_counts_before || {};
        const imp = batchMeta.imputed_cell_counts || {};
        const parts = [`<p><strong>Method:</strong> ${batchMeta.method || "knn"}</p>`];
        parts.push("<p><strong>Missing (before)</strong></p><ul>");
        FEATURE_COLUMNS.forEach((c) => {
            if (miss[c]) parts.push(`<li>${c}: ${miss[c]} missing</li>`);
        });
        parts.push("</ul><p><strong>Cells imputed (predicted)</strong></p><ul>");
        FEATURE_COLUMNS.forEach((c) => {
            if (imp[c]) parts.push(`<li>${c}: ${imp[c]} cells</li>`);
        });
        parts.push(`</ul><p>Total imputed cells: <strong>${batchMeta.total_imputed_cells ?? 0}</strong></p>`);
        el.innerHTML = parts.join("");
        return;
    }

    if (manualImp && manualImp.fields) {
        const flagged = Object.keys(manualImp.fields).filter((k) => manualImp.fields[k]);
        el.innerHTML =
            `<p><strong>Method:</strong> ${manualImp.method || "training_mean"}</p>` +
            (flagged.length
                ? `<p>Fields filled from training means: <strong>${flagged.join(", ")}</strong></p>`
                : "<p>No missing fields in this submission.</p>");
        return;
    }

    el.textContent = "Run Excel batch or manual predict to see imputation details.";
}

function destroyColumnCharts() {
    columnChartInstances.forEach((ch) => {
        try {
            ch.destroy();
        } catch (e) { }
    });
    columnChartInstances = [];
    ["column_charts_wrap_original", "column_charts_wrap_cleaned"].forEach((id) => {
        const wrap = document.getElementById(id);
        if (wrap) wrap.innerHTML = "";
    });
}

/** Build one grid of per-feature charts for a row set (original or cleaned). */
function appendFeatureCharts(wrap, rows, chartType) {
    if (!wrap) return;
    wrap.innerHTML = "";
    if (!rows || rows.length === 0) {
        const p = document.createElement("p");
        p.className = "chart-caption";
        p.textContent = "No rows to chart for this stage.";
        wrap.appendChild(p);
        return;
    }
    const sampled = rows.slice(0, CHART_ROW_CAP);
    const titleColor = "#9ca3af";
    const titleFont = { size: 11 };

    for (const col of FEATURE_COLUMNS) {
        const values = sampled
            .map((r) => valueForFeatureColumn(r, col))
            .filter((v) => !Number.isNaN(v));

        const card = document.createElement("div");
        card.className = "column-chart-card";
        const h = document.createElement("h4");
        h.textContent = col;
        card.appendChild(h);

        if (values.length === 0) {
            const p = document.createElement("p");
            p.className = "chart-caption";
            p.textContent = "No numeric values for this column (check header names).";
            card.appendChild(p);
            wrap.appendChild(card);
            continue;
        }

        const canvas = document.createElement("canvas");
        card.appendChild(canvas);
        wrap.appendChild(card);

        const ctx = canvas.getContext("2d");
        const titles = axisTitlesForColumnChart(col, chartType);
        let cfg;

        if (chartType === "histogram") {
            const { labels, data } = buildHistogramData(values, 12);
            cfg = {
                type: "bar",
                data: {
                    labels,
                    datasets: [{ label: "Count", data, backgroundColor: "#38bdf8" }],
                },
                options: {
                    responsive: true,
                    animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            ...axisStyle(),
                            title: { display: true, text: titles.x, color: titleColor, font: titleFont },
                        },
                        y: {
                            beginAtZero: true,
                            ...axisStyle(),
                            title: { display: true, text: titles.y, color: titleColor, font: titleFont },
                        },
                    },
                },
            };
        } else if (chartType === "scatter") {
            const pts = values.map((y, x) => ({ x, y }));
            cfg = {
                type: "scatter",
                data: {
                    datasets: [{ label: col, data: pts, backgroundColor: "#38bdf8" }],
                },
                options: {
                    responsive: true,
                    animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            ...axisStyle(),
                            title: { display: true, text: titles.x, color: titleColor, font: titleFont },
                        },
                        y: {
                            ...axisStyle(),
                            title: { display: true, text: titles.y, color: titleColor, font: titleFont },
                        },
                    },
                },
            };
        } else {
            const labels = values.map((_, i) => String(i));
            cfg = {
                type: chartType === "line" ? "line" : "bar",
                data: {
                    labels,
                    datasets: [
                        {
                            label: col,
                            data: values,
                            borderColor: "#38bdf8",
                            backgroundColor: chartType === "line" ? "rgba(56,189,248,0.2)" : "#38bdf8",
                            fill: chartType === "line",
                        },
                    ],
                },
                options: {
                    responsive: true,
                    animation: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            ...axisStyle(),
                            title: { display: true, text: titles.x, color: titleColor, font: titleFont },
                        },
                        y: {
                            beginAtZero: false,
                            ...axisStyle(),
                            title: { display: true, text: titles.y, color: titleColor, font: titleFont },
                        },
                    },
                },
            };
        }

        try {
            columnChartInstances.push(new Chart(ctx, cfg));
        } catch (e) {
            console.error("Column chart error", col, e);
            const p = document.createElement("p");
            p.className = "chart-caption";
            p.textContent = "Could not render chart for this column.";
            card.appendChild(p);
        }
    }
}

/**
 * Equal-width bins on [min, max]: step = (max − min) / bins.
 * Index k = floor((v − min) / step), capped to [0, bins − 1] so v = max lands in the last bin.
 */
function buildHistogramData(values, bins) {
    const clean = values.filter((v) => typeof v === "number" && !Number.isNaN(v));
    if (clean.length === 0) return { labels: [], data: [] };
    const min = Math.min(...clean);
    const max = Math.max(...clean);
    if (min === max) return { labels: [`${min.toFixed(2)}`], data: [clean.length] };
    const step = (max - min) / bins;
    const counts = new Array(bins).fill(0);
    const labels = [];
    for (let i = 0; i < bins; i++) {
        const lo = min + i * step;
        const hi = i === bins - 1 ? max : min + (i + 1) * step;
        labels.push(`${lo.toFixed(1)}–${hi.toFixed(1)}`);
    }
    clean.forEach((v) => {
        let idx = Math.floor((v - min) / step);
        if (idx >= bins) idx = bins - 1;
        if (idx < 0) idx = 0;
        counts[idx] += 1;
    });
    return { labels, data: counts };
}

/** Match backend _slugify_header so JSON rows with alternate keys still chart. */
function pickOriginalRowsPreview(result) {
    const o = result.original_rows_preview;
    if (Array.isArray(o) && o.length) return o;
    return [];
}

/** Cleaned / imputed preview rows only (for “after cleaning” charts). */
function pickCleanedRowsPreview(result) {
    const cleaned = result.cleaned_rows_preview;
    if (Array.isArray(cleaned) && cleaned.length) return cleaned;
    const alt = result.cleaned_rows;
    if (Array.isArray(alt) && alt.length) return alt;
    return [];
}

function slugifyHeaderKey(name) {
    return String(name)
        .trim()
        .replace(/\s+/g, " ")
        .replace(/([a-z])([A-Z])/g, "$1_$2")
        .toLowerCase()
        .replace(/[\s-]+/g, "_")
        .replace(/[^a-z0-9_]/g, "")
        .replace(/_+/g, "_")
        .replace(/^_|_$/g, "");
}

function valueForFeatureColumn(row, col) {
    if (!row || typeof row !== "object") return NaN;
    const direct = row[col];
    if (direct !== undefined && direct !== null && direct !== "") {
        const n = parseFloat(direct);
        if (!Number.isNaN(n)) return n;
    }
    for (const k of Object.keys(row)) {
        if (k.endsWith("_value_predicted")) continue;
        if (slugifyHeaderKey(k) === col) {
            const n = parseFloat(row[k]);
            if (!Number.isNaN(n)) return n;
        }
    }
    return NaN;
}

function scheduleColumnChartsRefresh() {
    if (columnChartRaf) cancelAnimationFrame(columnChartRaf);
    columnChartRaf = requestAnimationFrame(() => {
        columnChartRaf = null;
        refreshColumnCharts();
    });
}

function refreshColumnCharts() {
    const status = document.getElementById("column_charts_status");
    const typeSel = document.getElementById("column_chart_type");
    const wrapOrig = document.getElementById("column_charts_wrap_original");
    const wrapClean = document.getElementById("column_charts_wrap_cleaned");
    if (!typeSel || !wrapOrig || !wrapClean) return;

    destroyColumnCharts();

    const nOrig = (lastOriginalRows || []).length;
    const nClean = (lastCleanedRows || []).length;
    if (!nOrig && !nClean) {
        if (status) {
            status.textContent = "Run manual predict or Excel batch to see feature charts.";
        }
        return;
    }

    const chartType = typeSel.value || "bar";
    if (status) {
        status.textContent = `Rendering before/after feature charts (${chartType})…`;
    }

    appendFeatureCharts(wrapOrig, lastOriginalRows, chartType);
    appendFeatureCharts(wrapClean, lastCleanedRows, chartType);

    if (status) {
        status.textContent =
            `${FEATURE_COLUMNS.length} features × 2 stages (${chartType}). Rows — before cleaning: ${nOrig}, after cleaning: ${nClean}.`;
    }
}

async function runAccidentInsights() {
    const causeEl = document.getElementById("cause_panel");
    const prevEl = document.getElementById("prevention_panel");
    if (!causeEl || !prevEl) return;

    const hasData = lastCleanedRows.length && lastPredictionRows.length;
    if (!hasData && !lastImageAnalysis) {
        causeEl.textContent = "Provide predictions (manual or Excel) to rank contributing factors.";
        prevEl.textContent = "Recommendations appear after analysis runs.";
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/insights/accident`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                cleaned_rows: lastCleanedRows,
                prediction_rows: lastPredictionRows,
                image_analysis: normalizeImageAnalysisForInsights(lastImageAnalysis),
            }),
        });

        if (!res.ok) {
            const t = await res.text();
            throw new Error(t || res.statusText);
        }

        const data = await res.json();
        const ranked = data.ranked_factors || [];
        const lines = ranked.slice(0, 8).map((r, i) => {
            const isTop = i === 0 && r.factor === data.top_cause;
            const label = `${i + 1}. ${r.factor} (score ${(r.score || 0).toFixed(4)})`;
            return isTop ? `<li><strong>${label}</strong></li>` : `<li>${label}</li>`;
        });

        const top = data.top_cause || "";
        causeEl.innerHTML =
            `<p>Most contributing cause: ${top ? `<strong>${top}</strong>` : "—"}</p>` +
            `<ul>${lines.join("")}</ul>`;

        const bullets = data.prevention_recommendations || [];
        prevEl.innerHTML =
            "<ul>" + bullets.map((b) => `<li>${b}</li>`).join("") + "</ul>";
    } catch (e) {
        console.error(e);
        causeEl.textContent = "Could not compute causes. See console.";
        prevEl.textContent = "—";
    }
}

/**
 * Same aggregation as api/app.py when /analyze/images is unavailable (old server).
 */
function buildMultiImageResultFromSingles(perResults, names) {
    const per_image = perResults.map((r, i) => ({
        ...r,
        source_filename: names[i] || `Image ${i + 1}`,
    }));
    const pcts = per_image.map((p) => Number(p.risk_probability_percent));
    const mean = pcts.reduce((a, b) => a + (Number.isNaN(b) ? 0 : b), 0) / Math.max(1, pcts.length);
    const agg_pct = Math.min(100, Math.max(0, Math.round(mean * 100) / 100));
    const seen = new Set();
    const accident_risk_factors = [];
    for (const p of per_image) {
        for (const f of p.accident_risk_factors || []) {
            const s = String(f).trim();
            if (s && !seen.has(s.toLowerCase())) {
                seen.add(s.toLowerCase());
                accident_risk_factors.push(s);
            }
        }
    }
    return {
        image_count: per_image.length,
        per_image,
        aggregate_risk_probability_percent: agg_pct,
        aggregate_risk_level: riskLevelFromScore(agg_pct),
        aggregate_explanation: `Arithmetic mean of ${per_image.length} independent scene risk scores (each image analyzed separately).`,
        accident_risk_factors: accident_risk_factors.slice(0, 24),
    };
}

async function analyzeRiskImage() {
    const input = document.getElementById("risk_image");
    const status = document.getElementById("image_analysis_status");
    const out = document.getElementById("image_analysis_result");
    if (!input || !input.files || input.files.length === 0) {
        alert("Choose one or more images first.");
        return;
    }
    const n = input.files.length;
    if (n > MAX_ANALYSIS_IMAGES) {
        alert(`Maximum ${MAX_ANALYSIS_IMAGES} images at once.`);
        return;
    }

    if (status) status.textContent = `Analyzing ${n} scene(s)…`;
    if (out) out.innerHTML = "";

    const fd = new FormData();
    for (let i = 0; i < n; i++) {
        const f = input.files[i];
        fd.append("files", f, f.name || `image-${i + 1}.bin`);
    }

    try {
        let res = await fetch(`${API_BASE}/analyze/images`, {
            method: "POST",
            body: fd,
        });

        let text = await res.text();
        let data;

        if (res.status === 404) {
            if (status) {
                status.textContent =
                    "/analyze/images not on this API build — analyzing each image via /analyze/image…";
            }
            const names = [];
            const perResults = [];
            for (let i = 0; i < n; i++) {
                names.push(input.files[i].name || `Image ${i + 1}`);
                if (status) {
                    status.textContent = `Image ${i + 1} of ${n} (fallback mode)…`;
                }
                const sf = new FormData();
                sf.append("file", input.files[i], names[i]);
                const r = await fetch(`${API_BASE}/analyze/image`, {
                    method: "POST",
                    body: sf,
                });
                const t = await r.text();
                let one;
                try {
                    one = JSON.parse(t);
                } catch {
                    throw new Error(
                        t.slice(0, 600) || `HTTP ${r.status} on /analyze/image`
                    );
                }
                if (!r.ok) {
                    throw new Error(
                        extractHttpErrorMessage(one, t) ||
                        `HTTP ${r.status} for ${names[i]}`
                    );
                }
                perResults.push(one);
            }
            data = buildMultiImageResultFromSingles(perResults, names);
        } else {
            try {
                data = JSON.parse(text);
            } catch {
                throw new Error(
                    text.slice(0, 800) || `HTTP ${res.status} ${res.statusText || ""}`.trim()
                );
            }

            if (!res.ok) {
                const errMsg =
                    extractHttpErrorMessage(data, text) ||
                    `HTTP ${res.status} ${res.statusText || ""}`.trim();
                throw new Error(errMsg);
            }
        }

        if (imageBlendApplied && tabularOutputSnapshot) {
            revertImageBlendFromMainOutput();
        }
        lastImageAnalysis = data;
        imageSourceRiskCtx = {
            riskScore: safeNum(
                data.aggregate_risk_probability_percent != null
                    ? data.aggregate_risk_probability_percent
                    : data.risk_probability_percent,
                0
            ),
            weight: Math.max(1, safeNum(data.image_count, (data.per_image || []).length || 1)),
        };
        updateSourceAnalysisPanel();

        if (status) status.textContent = "Analysis complete.";
        if (out) {
            if (data.per_image && data.per_image.length) {
                const parts = data.per_image.map((p, i) => {
                    const name = p.source_filename || `Image ${i + 1}`;
                    return (
                        `<div class="per-image-block" style="margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.08);">` +
                        `<p><strong>${name}</strong> — ${p.risk_probability_percent}% &nbsp; <em>${p.risk_level}</em></p>` +
                        `<p style="font-size:12px;color:#9ca3af;">${p.explanation || ""}</p></div>`
                    );
                });
                out.innerHTML =
                    `<p><strong>Aggregate (mean of ${data.image_count} scores):</strong> ` +
                    `${data.aggregate_risk_probability_percent}% &nbsp; ` +
                    `<strong>Level:</strong> ${data.aggregate_risk_level}</p>` +
                    (data.per_image.some((p) => p.analysis_mode === "local_quota_fallback")
                        ? `<p class="muted small"><strong>Note:</strong> Gemini quota exceeded; one or more images used local fallback analysis.</p>`
                        : "") +
                    `<p class="muted small">${data.aggregate_explanation || ""}</p>` +
                    `<h4 style="margin:14px 0 8px;font-size:13px;">Per image</h4>` +
                    parts.join("") +
                    (data.accident_risk_factors && data.accident_risk_factors.length
                        ? `<p><strong>Merged factors:</strong> ${data.accident_risk_factors.join(", ")}</p>`
                        : "");
            } else {
                out.innerHTML =
                    `<p><strong>Risk probability:</strong> ${data.risk_probability_percent}% &nbsp; ` +
                    `<strong>Level:</strong> ${data.risk_level}</p>` +
                    (data.analysis_mode === "local_quota_fallback"
                        ? `<p class="muted small"><strong>Note:</strong> Gemini quota exceeded; local fallback analysis used.</p>`
                        : "") +
                    `<p><strong>Explanation:</strong> ${data.explanation || ""}</p>` +
                    (data.accident_risk_factors && data.accident_risk_factors.length
                        ? `<p><strong>Factors:</strong> ${data.accident_risk_factors.join(", ")}</p>`
                        : "");
            }
        }

        updateImageBlendButtons();
        await runAccidentInsights();
    } catch (e) {
        console.error(e);
        const msg = e.message || String(e);
        if (status) status.textContent = "Image analysis failed.";
        if (out) {
            const safe = msg.replace(/</g, "&lt;").replace(/>/g, "&gt;");
            out.innerHTML =
                `<p class="image-analysis-error"><strong>Error</strong></p>` +
                `<pre class="image-analysis-error-detail">${safe}</pre>`;
        }
        alert(imageAnalysisFailureHint(msg));
    }
}

document.addEventListener("DOMContentLoaded", () => {
    ensureCharts();
    updateImageBlendButtons();
    updateSourceAnalysisPanel();
    initTableauEmbed();

    const imgInput = document.getElementById("risk_image");
    const prevGrid = document.getElementById("risk_image_previews");
    if (imgInput && prevGrid) {
        imgInput.addEventListener("change", () => {
            prevGrid.innerHTML = "";
            if (!imgInput.files || !imgInput.files.length) return;
            const cap = Math.min(imgInput.files.length, MAX_ANALYSIS_IMAGES);
            for (let i = 0; i < cap; i++) {
                const img = document.createElement("img");
                img.alt = imgInput.files[i].name || `Image ${i + 1}`;
                img.src = URL.createObjectURL(imgInput.files[i]);
                prevGrid.appendChild(img);
            }
        });
    }

    const ct = document.getElementById("column_chart_type");
    if (ct) {
        ct.addEventListener("change", () => scheduleColumnChartsRefresh());
    }
});

async function computeCorrelation() {
    const statusEl = document.getElementById("corrStatus");
    const container = document.getElementById("corrMatrix");

    if (corrAnimationHandle) {
        clearInterval(corrAnimationHandle);
        corrAnimationHandle = null;
    }

    statusEl.textContent = "Computing correlation matrix…";
    container.innerHTML = "";

    try {
        let res;
        if (lastCleanedRows && lastCleanedRows.length >= 2) {
            const sample = lastCleanedRows.slice(0, 8000);
            res = await fetch(`${API_BASE}/correlation`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ rows: sample }),
            });
        } else {
            res = await fetch(`${API_BASE}/correlation`);
        }

        if (!res.ok) {
            const t = await res.text();
            throw new Error(t || res.statusText);
        }

        const data = await res.json();

        const columns = data.columns || [];
        const matrix = data.matrix || [];

        if (!columns.length || !matrix.length) {
            statusEl.textContent = "No correlation data available.";
            return;
        }

        const table = document.createElement("table");
        table.className = "corr-table";

        const headerRow = document.createElement("tr");
        const corner = document.createElement("th");
        corner.textContent = "";
        headerRow.appendChild(corner);
        columns.forEach(col => {
            const th = document.createElement("th");
            th.textContent = col;
            headerRow.appendChild(th);
        });
        table.appendChild(headerRow);

        const cells = [];
        for (let i = 0; i < columns.length; i++) {
            const tr = document.createElement("tr");
            const rowHeader = document.createElement("th");
            rowHeader.textContent = columns[i];
            tr.appendChild(rowHeader);

            for (let j = 0; j < columns.length; j++) {
                const td = document.createElement("td");
                td.className = "corr-cell";
                td.textContent = "";
                tr.appendChild(td);
                cells.push({ td, value: matrix[i][j] });
            }
            table.appendChild(tr);
        }

        container.innerHTML = "";
        container.appendChild(table);

        let idx = 0;
        const total = cells.length;

        function colorFor(v) {
            const clamped = Math.max(-1, Math.min(1, v ?? 0));
            const norm = (clamped + 1) / 2;
            const r = Math.round(255 * norm);
            const b = Math.round(255 * (1 - norm));
            const g = 40;
            return `rgba(${r}, ${g}, ${b}, 0.85)`;
        }

        statusEl.textContent = "Filling matrix cells...";

        corrAnimationHandle = setInterval(() => {
            if (idx >= total) {
                clearInterval(corrAnimationHandle);
                corrAnimationHandle = null;
                statusEl.textContent = "Correlation matrix computed" + (data.source ? ` (${data.source}).` : ".");
                return;
            }

            const { td, value } = cells[idx];
            const v = typeof value === "number" ? value : 0;
            td.textContent = v.toFixed(2);
            td.style.backgroundColor = colorFor(v);
            td.style.opacity = "1";

            if ((idx + 1) % columns.length === 0) {
                const rowNum = Math.floor(idx / columns.length) + 1;
                statusEl.textContent = `Filling row ${rowNum} of ${columns.length}...`;
            }
            idx += 1;
        }, 15);
    } catch (error) {
        statusEl.textContent = "Correlation computation failed.";
    }
}
