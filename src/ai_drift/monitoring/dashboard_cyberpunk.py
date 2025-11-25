# src/ai_drift/monitoring/dashboard_cyberpunk.py
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import json
import numpy as np

# Background image (user-uploaded screenshot). We pass this file path in the HTML as-is.
BACKGROUND_IMAGE_PATH = "/mnt/data/21623917-d285-44b7-8c1e-9814b3f3e67f.png"

NEON_ACCENT = "#00FFDD"
NEON_PINK = "#FF2D95"
NEON_YELLOW = "#FFD166"
CARD_BG = "rgba(10,10,12,0.7)"
TEXT_COLOR = "white"

class CyberpunkDashboard:
    def __init__(self, detector):
        self.detector = detector

    def create_summary_df(self, unified_report, alibi_report):
        rows = []
        features = list(dict.fromkeys(self.detector.numeric_features + self.detector.categorical_features))
        for f in features:
            rows.append({
                "feature": f,
                "z_score_drift": bool(f in unified_report.get("z_score_drift", [])),
                "ks_drift": bool(f in unified_report.get("ks_test_drift", [])),
                "chi_drift": bool(f in unified_report.get("chi_square_drift", [])),
                "alibi_drift": bool(alibi_report.get(f, {}).get("is_drift", False)),
                "p_value": float(alibi_report.get(f, {}).get("p_value", 1.0))
            })
        return pd.DataFrame(rows)

    def neon_card_html(self, title, value, color=NEON_ACCENT, subtitle=""):
        # small neon info card
        return f"""
        <div class="neon-card" style="border-color:{color};">
          <div class="neon-title">{title}</div>
          <div class="neon-value">{value}</div>
          <div class="neon-sub">{subtitle}</div>
        </div>
        """

    def build_pvalue_fig(self, alibi_report):
        features = list(alibi_report.keys())
        pvals = [alibi_report[f]["p_value"] for f in features]
        colors = [NEON_PINK if p < 0.05 else NEON_ACCENT for p in pvals]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=features, y=pvals,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.08)", width=1)),
            hovertemplate="<b>%{x}</b><br>p-value: %{y:.3e}<extra></extra>"
        ))
        fig.add_hline(y=0.05, line_dash="dash", line_color=NEON_YELLOW, annotation_text="α = 0.05", annotation_font_color=NEON_YELLOW)
        fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color=TEXT_COLOR), margin=dict(t=40, b=30))
        fig.update_yaxes(type="log")  # log scale works well for p-values
        return fig

    def build_distribution_fig(self, ref, cur):
        numeric = self.detector.numeric_features
        rows = len(numeric)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=False, subplot_titles=[f"{c}" for c in numeric])
        for i, col in enumerate(numeric):
            fig.add_trace(go.Histogram(x=ref[col], name=f"{col} — ref", opacity=0.6, nbinsx=30,
                                       marker=dict(color="rgba(0,255,221,0.35)")), row=i+1, col=1)
            fig.add_trace(go.Histogram(x=cur[col], name=f"{col} — cur", opacity=0.6, nbinsx=30,
                                       marker=dict(color="rgba(255,45,149,0.35)")), row=i+1, col=1)
            # add density line
            try:
                kde_ref_x = np.linspace(np.min(ref[col]), np.max(ref[col]), 200)
                from scipy.stats import gaussian_kde
                kde_ref = gaussian_kde(ref[col].dropna())
                kde_cur = gaussian_kde(cur[col].dropna())
                fig.add_trace(go.Scatter(x=kde_ref_x, y=kde_ref(kde_ref_x)*len(ref[col].dropna())* (kde_ref_x[1]-kde_ref_x[0]),
                                         mode='lines', name=f"{col} KDE ref", line=dict(color=NEON_ACCENT)), row=i+1, col=1)
                fig.add_trace(go.Scatter(x=kde_ref_x, y=kde_cur(kde_ref_x)*len(cur[col].dropna())*(kde_ref_x[1]-kde_ref_x[0]),
                                         mode='lines', name=f"{col} KDE cur", line=dict(color=NEON_PINK)), row=i+1, col=1)
            except Exception:
                pass

        fig.update_layout(template="plotly_dark", barmode="overlay", height=300*rows, margin=dict(t=40, b=40), legend=dict(orientation="h"))
        return fig

    def build_heatmap(self, unified_report):
        features = self.detector.numeric_features + self.detector.categorical_features
        vals = []
        for f in features:
            score = 0
            if f in unified_report.get("z_score_drift", []): score += 1
            if f in unified_report.get("ks_test_drift", []): score += 2
            if f in unified_report.get("chi_square_drift", []): score += 1
            if score == 0: vals.append(0)
            else: vals.append(score)
        fig = go.Figure(data=go.Heatmap(z=[vals], x=features, y=["drift_score"], colorscale=[[0, 'rgba(0,50,0,0.9)'],
                                                                                             [0.3, 'rgba(20,120,90,0.9)'],
                                                                                             [0.6, 'rgba(255,180,0,0.9)'],
                                                                                             [1, 'rgba(255,20,149,0.95)']]))
        fig.update_layout(template="plotly_dark", height=150, margin=dict(t=10, b=10))
        return fig

    def save_html(self, ref, cur, unified_report, alibi_report, path="artifacts/drift_dashboard_cyberpunk.html"):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        summary_df = self.create_summary_df(unified_report, alibi_report)
        top_cards = []
        # top neon cards: total drifted features, strongest drift, time (static)
        total_drifted = len(unified_report.get("all_drifted_features", []))
        strongest = ", ".join(unified_report.get("all_drifted_features", [])[:2]) or "None"

        top_cards.append(self.neon_card_html("Drifted Features", total_drifted, color=NEON_PINK, subtitle=strongest))
        top_cards.append(self.neon_card_html("Strongest Drift", strongest or "—", color=NEON_YELLOW, subtitle="Investigation"))
        top_cards.append(self.neon_card_html("Mode", "Cyberpunk", color=NEON_ACCENT, subtitle="Dark Monitoring"))

        pval_fig = self.build_pvalue_fig(alibi_report)
        dist_fig = self.build_distribution_fig(ref, cur)
        heat = self.build_heatmap(unified_report)

        # convert plotly to html snippets
        pval_html = pval_fig.to_html(full_html=False, include_plotlyjs="cdn")
        dist_html = dist_fig.to_html(full_html=False, include_plotlyjs=False)
        heat_html = heat.to_html(full_html=False, include_plotlyjs=False)
        table_html = summary_df.to_html(classes="summary-table", index=False)

        # Build the full cyberpunk HTML + inline CSS & JS for neon effects
        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>AI Drift & ADS - Cyberpunk Dashboard</title>
          <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
          <style>
            :root {{
                --bg: #07070a;
                --card: rgba(10,10,12,0.75);
                --neon: {NEON_ACCENT};
                --neon-pink: {NEON_PINK};
                --neon-yellow: {NEON_YELLOW};
                --muted: rgba(255,255,255,0.7);
            }}
            html,body{{height:100%; margin:0; background:#050407; font-family: 'Roboto', sans-serif; color:{TEXT_COLOR}}}
            body {{
              background-image: url('{BACKGROUND_IMAGE_PATH}');
              background-size: cover;
              background-position: center center;
              background-blend-mode: overlay;
            }}
            .overlay {{
              background: linear-gradient(180deg, rgba(0,0,0,0.55), rgba(0,0,0,0.75));
              min-height:100vh;
              padding:24px;
            }}
            header{{text-align:center; padding:18px 0 10px 0}}
            h1{{font-family:'Orbitron',sans-serif; font-weight:700; color:var(--neon-pink); text-shadow:0 0 12px rgba(255,45,149,0.15); margin:0; font-size:40px}}
            h3{{color:var(--neon); margin:6px 0 22px 0; font-weight:400}}

            .top-cards{{display:flex; gap:18px; justify-content:center; margin-bottom:20px}}
            .neon-card {{
              border: 1px solid rgba(0,255,221,0.15);
              background: linear-gradient(135deg, rgba(0,0,0,0.4), rgba(0,0,0,0.2));
              padding:16px 22px;
              min-width:180px;
              text-align:center;
              border-radius:12px;
              box-shadow: 0 6px 30px rgba(0,255,221,0.02), 0 0 18px rgba(0,255,221,0.03) inset;
              transition: transform .18s ease;
            }}
            .neon-card:hover{{ transform: translateY(-6px) scale(1.02); box-shadow:0 8px 40px rgba(0,255,221,0.06)}}
            .neon-title{{font-size:12px; color:var(--muted); letter-spacing:1px}}
            .neon-value{{font-size:28px; font-weight:700; color:var(--neon); margin-top:6px}}
            .neon-sub{{font-size:12px; color:var(--muted); margin-top:6px}}

            .grid {{
              display:grid;
              grid-template-columns: 1fr 450px;
              gap:18px;
              align-items:start;
              padding:10px 24px;
            }}

            .card {{
              background: {CARD_BG};
              border-radius:12px;
              padding:14px;
              border: 1px solid rgba(255,255,255,0.03);
              box-shadow: 0 16px 60px rgba(0,0,0,0.6);
            }}

            .summary-wrap {{padding:12px;}}
            table.summary-table {{width:100%; border-collapse:collapse; color:white; font-size:13px}}
            table.summary-table th, table.summary-table td {{padding:10px; border-bottom:1px solid rgba(255,255,255,0.03); text-align:left; color: #dcdcdc}}
            table.summary-table th {{background:rgba(255,255,255,0.02); font-weight:600; color:var(--neon)}}

            footer{{text-align:center; padding:22px 0 40px 0; color: rgba(255,255,255,0.45)}}
            .small{{font-size:12px; color: rgba(255,255,255,0.5)}}
          </style>
        </head>
        <body>
          <div class="overlay">
            <header>
              <h1>AI Drift & Anomaly Detection Dashboard</h1>
              <h3>Cyberpunk Dark Mode — Live Monitoring Preview</h3>
            </header>

            <div class="top-cards">
              {''.join(top_cards)}
            </div>

            <div class="grid">
              <div class="card">
                <div class="summary-wrap">
                  <h2 style="color:var(--neon); margin:0 0 10px 0; font-family:Orbitron;">Drift Summary</h2>
                  {table_html}
                </div>
              </div>

              <div style="display:flex; flex-direction:column; gap:18px;">
                <div class="card" style="height:260px;">
                  <h3 style="margin:2px 0 10px 0; color:var(--neon-pink)">P-value Overview</h3>
                  {pval_html}
                </div>

                <div class="card" style="height:300px;">
                  <h3 style="margin:2px 0 10px 0; color:var(--neon)">Drift Heatmap</h3>
                  {heat_html}
                </div>
              </div>
            </div>

            <div style="padding:24px;">
              <div class="card">
                <h2 style="color:var(--neon); font-family:Orbitron;">Feature Distributions</h2>
                <div>{dist_html}</div>
              </div>
            </div>

            <footer>
              <div class="small">Generated by AI-DRIFT · Cyberpunk Theme · Interactive plots powered by Plotly</div>
            </footer>
          </div>
        </body>
        </html>
        """

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Cyberpunk dashboard saved at: {path}")
