import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import json


class DriftDashboard:
    def __init__(self, drift_detector):
        self.detector = drift_detector

    def create_summary_table(self, unified_report, alibi_report):
        df = []

        for feature in self.detector.numeric_features + self.detector.categorical_features:
            df.append({
                "feature": feature,
                "z_score_drift": feature in unified_report["z_score_drift"],
                "ks_drift": feature in unified_report["ks_test_drift"],
                "chi_drift": feature in unified_report["chi_square_drift"],
                "alibi_drift": alibi_report.get(feature, {}).get("is_drift", False),
                "p_value": alibi_report.get(feature, {}).get("p_value", 1.0)
            })

        return pd.DataFrame(df)

    def build_dark_table(self, df):
        fig = go.Figure(
            data=[go.Table(
                header=dict(
                    values=list(df.columns),
                    fill_color="#1f1f1f",
                    font=dict(color="white", size=14),
                ),
                cells=dict(
                    values=[df[col] for col in df.columns],
                    fill_color="#2d2d2d",
                    font=dict(color="white", size=12),
                )
            )]
        )

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=10, b=10),
            height=500
        )
        return fig

    def build_pvalue_plot(self, alibi_report):
        features = list(alibi_report.keys())
        p_values = [alibi_report[f]["p_value"] for f in features]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=features,
            y=p_values,
            marker_color=["red" if p < 0.05 else "green" for p in p_values]
        ))

        fig.add_hline(y=0.05, line_dash="dash", line_color="yellow")

        fig.update_layout(
            title="P-value Drift Chart",
            template="plotly_dark",
            height=400
        )
        return fig

    def build_feature_distributions(self, ref, cur):
        numeric = self.detector.numeric_features

        rows = len(numeric)
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=False,
                            subplot_titles=numeric)

        for idx, col in enumerate(numeric):
            fig.add_trace(
                go.Histogram(x=ref[col], name=f"{col} (reference)", opacity=0.5),
                row=idx+1, col=1
            )
            fig.add_trace(
                go.Histogram(x=cur[col], name=f"{col} (current)", opacity=0.5),
                row=idx+1, col=1
            )

        fig.update_layout(
            template="plotly_dark",
            height=300 * rows,
            barmode="overlay"
        )
        return fig

    def save_html_dashboard(self, ref, cur, unified_report, alibi_report,
                            path="artifacts/drift_dashboard.html"):

        table = self.build_dark_table(
            self.create_summary_table(unified_report, alibi_report)
        )
        pval_plot = self.build_pvalue_plot(alibi_report)
        dist_plot = self.build_feature_distributions(ref, cur)

        html = """
        <html>
        <head><title>Drift Dashboard</title></head>
        <body style="background-color:#121212; color:white;">
        <h1 style="text-align:center;">AI Drift & Anomaly Detection Dashboard</h1>
        <h3 style="text-align:center;">Dark Mode Professional Dashboard</h3>
        """

        html += table.to_html(full_html=False, include_plotlyjs="cdn")
        html += pval_plot.to_html(full_html=False, include_plotlyjs=False)
        html += dist_plot.to_html(full_html=False, include_plotlyjs=False)

        html += """
        </body></html>
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Dark Mode Drift Dashboard saved at: {path}")
