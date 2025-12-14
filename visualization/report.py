import os
from typing import Dict, List


def render_report(save_dir: str, context: Dict):
    os.makedirs(save_dir, exist_ok=True)
    html_path = os.path.join(save_dir, "report.html")
    radar = context.get("radar")
    bar = context.get("bar")
    scatter = context.get("scatter")
    safety = context.get("safety")
    metrics_table = context.get("metrics_table", [])
    config_info = context.get("config", {})
    threat_model = context.get("threat_model", {})

    rows = "".join(
        f"<tr><td>{row['name']}</td><td>{row['score']:.4f}</td><td>{row['rank']}</td></tr>" for row in metrics_table
    )
    tm_lines = "<br>".join([f"{k}: {v}" for k, v in (threat_model or {}).items()])
    html = f"""
    <html>
    <head><title>Evaluation Report</title></head>
    <body>
    <h1>Evaluation Summary</h1>
    <p>Algorithm: {config_info.get('algo_name')}</p>
    <p>Dataset: {config_info.get('dataset_name')}</p>
    <h2>Threat Model</h2>
    <pre>{tm_lines}</pre>
    <h2>Scores</h2>
    <table border='1' cellpadding='4'>
        <tr><th>Name</th><th>Final Score</th><th>Rank</th></tr>
        {rows}
    </table>
    <p>Raw metrics: {context.get('raw_csv')}</p>
    <p>Normalized metrics: {context.get('norm_csv')}</p>
    <p>Final score file: {context.get('score_csv')}</p>
    <h2>Figures</h2>
    <div>
        <h3>Radar</h3>
        <img src='{radar}' width='400'>
    </div>
    <div>
        <h3>Bars</h3>
        <img src='{bar}' width='500'>
    </div>
    <div>
        <h3>Accuracy vs Latency</h3>
        <img src='{scatter}' width='500'>
    </div>
    <div>
        <h3>Clean vs Adversarial</h3>
        <img src='{safety}' width='500'>
    </div>
    </body>
    </html>
    """
    with open(html_path, "w") as f:
        f.write(html)
    return html_path
