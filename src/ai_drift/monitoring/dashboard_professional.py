import os
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# If you want a small header image/logo, set this to the path of your uploaded image.
# Use the path from your workspace; change it if necessary.
HEADER_IMAGE_PATH = "/mnt/data/21623917-d285-44b7-8c1e-9814b3f3e67f.png"

# Styling values for the professional (white) theme
ACCENT = "#0074D9"     # blue accent
ACCENT2 = "#2ECC40"    # green accent for "OK"
TEXT = "#222222"
CARD_BG = "#ffffff"
MUTED = "#6b6b6b"

class ProfessionalDashboard:
    def __init__(self, detector):
        self.detector = detector

    def make_summary_df(self, unified_report, alibi_report):
        rows = []
        features = list(dict.fromkeys(self.detector.numeric_features + self.detector.categorical_features))
        for f in features:
            rows.append({
                "feature": f,
                "z_score_drift": "Yes" if f in unified_report.get("z_score_drift", []) else "No",
                "ks_drift": "Yes" if f in unified_report.get("ks_test_drift", []) else "No",
                "chi_drift": "Yes" if f in unified_report.get("chi_square_drift", []) else "No",
                "alibi_drift": "Yes" if alibi_report.get(f, {}).get("is_drift", False) else "No",
                "p_value": float(alibi_report.get(f, {}).get("p_value", 1.0))
            })
        return pd.DataFrame(rows)

    def build_pvalue_chart(self, alibi_report):
        features = list(alibi_report.keys())
        p_values = [alibi_report[f]["p_value"] for f in features]
        colors = [ACCENT2 if p >= 0.05 else "#FF4136" for p in p_values]  # green OK, red drift

        fig = go.Figure()
        fig.add_trace(go.Bar(x=features, y=p_values, marker_color=colors, text=[f"{p:.2e}" for p in p_values]))
        fig.update_layout(
            title="Feature p-values (lower => stronger drift)",
            xaxis_title="Feature",
            yaxis_title="p-value (log scale)",
            template="simple_white",
            height=380,
            margin=dict(l=40, r=20, t=60, b=80)
        )
        fig.update_yaxes(type="log")
        return fig

    def build_distributions(self, ref, cur):
        numeric = self.detector.numeric_features
        rows = len(numeric)
        fig = make_subplots(rows=rows, cols=1, subplot_titles=[f for f in numeric], shared_xaxes=False)
        for i, col in enumerate(numeric):
            fig.add_trace(
                go.Histogram(x=ref[col], name="Reference", opacity=0.6, nbinsx=40,
                             marker=dict(color="rgba(0,116,217,0.6)")),
                row=i+1, col=1
            )
            fig.add_trace(
                go.Histogram(x=cur[col], name="Current", opacity=0.6, nbinsx=40,
                             marker=dict(color="rgba(46,204,64,0.6)")),
                row=i+1, col=1
            )
            # density lines if possible
            try:
                from scipy.stats import gaussian_kde
                xr = np.linspace(min(ref[col].min(), cur[col].min()), max(ref[col].max(), cur[col].max()), 200)
                kde_ref = gaussian_kde(ref[col].dropna())
                kde_cur = gaussian_kde(cur[col].dropna())
                fig.add_trace(go.Scatter(x=xr, y=kde_ref(xr) * len(ref[col].dropna())*(xr[1]-xr[0]),
                                         mode='lines', name='Ref KDE', line=dict(color="rgba(0,116,217,1)")), row=i+1, col=1)
                fig.add_trace(go.Scatter(x=xr, y=kde_cur(xr) * len(cur[col].dropna())*(xr[1]-xr[0]),
                                         mode='lines', name='Cur KDE', line=dict(color="rgba(46,204,64,1)")), row=i+1, col=1)
            except Exception:
                pass

        fig.update_layout(template="simple_white", barmode="overlay", showlegend=True,
                          height=max(380, 250*rows), margin=dict(t=50, b=60))
        return fig

    def build_summary_table_html(self, df):
        # small styling for a professional table
        table_html = df.to_html(index=False, classes="summary-table", float_format="{:.3e}".format, border=0)
        return table_html

    def save_html(self, ref, cur, unified_report, alibi_report, path="artifacts/drift_dashboard_professional.html"):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        summary_df = self.make_summary_df(unified_report, alibi_report)
        pval_fig = self.build_pvalue_chart(alibi_report)
        dist_fig = self.build_distributions(ref, cur)
        table_html = self.build_summary_table_html(summary_df)

        pval_html = pval_fig.to_html(full_html=False, include_plotlyjs="cdn")
        dist_html = dist_fig.to_html(full_html=False, include_plotlyjs=False)

        html = f"""
        <!doctype html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>AI Drift Dashboard - Professional</title>
          <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
          <style>
            body {{ font-family: 'Inter', sans-serif; margin:0; padding:0; background:#ffffff; color:{TEXT}; }}
            .container {{ max-width:1200px; margin:28px auto; padding:24px; }}
            header {{ display:flex; align-items:center; gap:16px; }}
            .logo {{ width:84px; height:56px; object-fit:cover; border-radius:6px; border:1px solid #eee; }}
            h1 {{ margin:0; font-size:24px; color:{TEXT}; }}
            .subtitle {{ margin:6px 0 22px 0; color:{MUTED}; font-size:14px }}

            .cards {{ display:flex; gap:16px; margin:14px 0 22px 0; }}
            .card {{ flex:1; background:{CARD_BG}; border-radius:8px; padding:14px; border:1px solid #efefef }}
            .card h3 {{ margin:0; font-size:14px; color:{MUTED} }}
            .card .val {{ margin-top:8px; font-size:20px; font-weight:600; color:{ACCENT} }}

            .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-top:10px; }}
            .full {{ grid-column: 1 / -1; }}

            .summary-wrap {{ padding:12px; background:#fff; border-radius:6px; border:1px solid #eee }}
            table.summary-table {{ width:100%; border-collapse:collapse; font-size:13px }}
            table.summary-table th, table.summary-table td {{ padding:8px 10px; border-bottom:1px solid #f3f3f3; text-align:left; color:{TEXT} }}
            table.summary-table th {{ background:#fafafa; font-weight:600; color:{MUTED}; font-size:12px }}

            footer {{ margin-top:26px; color:{MUTED}; font-size:13px; text-align:center }}
          </style>
        </head>
        <body>
          <div class="container">
            <header>
              <img src="{HEADER_IMAGE_PATH}" class="logo" alt="logo" onerror="this.style.display='none'"/>
              <div>
                <h1>AI Drift & Anomaly Detection — Professional</h1>
                <div class="subtitle">Reference vs Current — Clean & Minimal</div>
              </div>
            </header>

            <div class="cards">
              <div class="card">
                <h3>Total drifted features</h3>
                <div class="val">{len(unified_report.get("all_drifted_features", []))}</div>
              </div>
              <div class="card">
                <h3>Most affected</h3>
                <div class="val">{', '.join(unified_report.get("all_drifted_features", [])[:2]) or '—'}</div>
              </div>
              <div class="card">
                <h3>Dataset</h3>
                <div class="val">Reference vs Current</div>
              </div>
            </div>

            <div class="grid">
              <div class="card">
                <h3 style="margin-top:0">Drift Summary</h3>
                <div class="summary-wrap">{table_html}</div>
              </div>

              <div class="card">
                <h3 style="margin-top:0">P-value Overview</h3>
                {pval_html}
              </div>

              <div class="card full">
                <h3 style="margin-top:0">Feature Distributions</h3>
                {dist_html}
              </div>
            </div>

            <footer>
              Generated by AI-DRIFT · Clean Professional Theme
            </footer>
          </div>
        </body>
        </html>
        """

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Professional dashboard saved at: {path}")