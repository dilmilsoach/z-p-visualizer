from flask import Flask, render_template, request
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

def create_plot(mu, sigma, x_val):
    # 1. Math Setup
    z_score = (x_val - mu) / sigma
    p_value = stats.norm.sf(abs(z_score)) * 2
    
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 500)
    y = stats.norm.pdf(x, mu, sigma)
    
    # 2. Create Figure
    fig = go.Figure()

    # Main Curve
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Population', line=dict(color='#2c3e50')))

    # 3. Add the "Observed Point"
    # Logic: Red if rare (<0.05), Blue if common
    point_color = 'red' if p_value <= 0.05 else '#2980b9'
    
    fig.add_trace(go.Scatter(
        x=[x_val], 
        y=[stats.norm.pdf(x_val, mu, sigma)],
        mode='markers+text',
        name='Observed Point',
        text=[f"x={x_val}"],
        textposition="top center",
        marker=dict(size=12, color=point_color, symbol='diamond')
    ))

    # 4. Add the 95% Confidence Boundaries (Vertical Dashed Lines)
    upper_bound = mu + 1.96 * sigma
    lower_bound = mu - 1.96 * sigma
    
    for bound, label in [(upper_bound, "Upper 95%"), (lower_bound, "Lower 95%")]:
        fig.add_vline(x=bound, line_dash="dash", line_color="green", opacity=0.5)

    # 5. The "No-Confusion" Header
    # We use update_layout to add a subtitle with the exact parameters
    fig.update_layout(
        title={
            'text': f"Statistical Significance Analysis<br><span style='font-size:14px; color:gray;'>" +
                    f"Inputs: Mean (μ)={mu} | StdDev (σ)={sigma} | Observed (x)={x_val}</span>",
            'y':0.95, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top'
        },
        xaxis_title="Value",
        yaxis_title="Probability Density",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=100) # Give extra space for the header
    )

    return pio.to_html(fig, full_html=False), z_score, p_value

@app.route('/', methods=['GET', 'POST'])
def index():
    graph, z, p = None, None, None
    if request.method == 'POST':
        mu = float(request.form['mu'])
        sigma = float(request.form['sigma'])
        x_val = float(request.form['x_val'])
        graph, z, p = create_plot(mu, sigma, x_val)
    
    return render_template('index.html', graph=graph, z=z, p=p)
