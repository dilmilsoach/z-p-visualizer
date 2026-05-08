from flask import Flask, render_template, request
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.io as pio
import os

app = Flask(__name__)

def create_plot(mu, sigma, x_bar, n):
    # 1. Math Setup - Using Standard Error for Sample Mean
    std_error = sigma / np.sqrt(n)
    z_score = (x_bar - mu) / std_error
    p_value = stats.norm.sf(abs(z_score)) * 2
    
    # Generate distribution data
    x = np.linspace(mu - 4*std_error, mu + 4*std_error, 500)
    y = stats.norm.pdf(x, mu, std_error)
    
    # 2. Create Figure
    fig = go.Figure()

    # Main Curve
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Sampling Distribution', line=dict(color='#2c3e50')))

    # 3. Add the "Observed Point"
    point_color = 'red' if p_value <= 0.05 else '#2980b9'
    
    fig.add_trace(go.Scatter(
        x=[x_bar], 
        y=[stats.norm.pdf(x_bar, mu, std_error)],
        mode='markers+text',
        name='Sample Mean',
        text=[f"x̄={x_bar}"],
        textposition="top center",
        marker=dict(size=12, color=point_color, symbol='diamond')
    ))

    # 4. Add the 95% Confidence Boundaries (Vertical Dashed Lines)
    upper_bound = mu + 1.96 * std_error
    lower_bound = mu - 1.96 * std_error
    
    for bound in [upper_bound, lower_bound]:
        fig.add_vline(x=bound, line_dash="dash", line_color="green", opacity=0.5)

    # 5. Header
    fig.update_layout(
        title={
            'text': f"Statistical Significance Analysis<br><span style='font-size:14px; color:gray;'>" +
                    f"μ={mu} | σ={sigma} | n={int(n)} | x̄={x_bar} | SE={std_error:.2f}</span>",
            'y': 0.95, 
            'x': 0.5, 
            'xanchor': 'center', 
            'yanchor': 'top'
        },
        xaxis=dict(
            title="Measured Value (x̄)",
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Probability Density",
            showgrid=True,
            zeroline=False
        ),
        template="plotly_white",
        showlegend=False,
        margin=dict(t=100, l=50, r=50, b=50) # Added margins for label space
    )

    return pio.to_html(fig, full_html=False), z_score, p_value

@app.route('/', methods=['GET', 'POST'])
def index():
    graph, z, p = None, None, None
    if request.method == 'POST':
        # Using .get() prevents 400 Bad Request errors if a key is missing
        mu = float(request.form.get('mu', 100))
        sigma = float(request.form.get('sigma', 15))
        n = float(request.form.get('n', 1))
        x_bar = float(request.form.get('x_bar', 105)) # Changed from x_val to x_bar to match HTML
        
        graph, z, p = create_plot(mu, sigma, x_bar, n)
    
    return render_template('index.html', graph=graph, z=z, p=p)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
