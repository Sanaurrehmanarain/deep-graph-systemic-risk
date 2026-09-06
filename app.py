from shiny import App, ui, render
from shinywidgets import output_widget, render_widget
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# ==========================================
# 1. LOAD DATA
# ==========================================
PROCESSED_DIR = Path("data/processed") if Path("data/processed").exists() else Path("../data/processed")
anomalies_file = PROCESSED_DIR / "gae_systemic_anomalies.csv"

# Fallback if file isn't found during app initialization
if anomalies_file.exists():
    df_anomalies = pd.read_csv(anomalies_file)
else:
    df_anomalies = pd.DataFrame({'period': ['2000-Q1'], 'reconstruction_error': [0], 'top_anomalous_node': ['N/A'], 'regime': ['Train']})

# ==========================================
# 2. UI DEFINITION
# ==========================================
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Unsupervised Systemic Anomaly Radar",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Deep Learning Radar"),
                ui.p("This dashboard tracks the global banking network's topological drift."),
                ui.hr(),
                ui.p("📈 A spike in the Reconstruction Error indicates the Graph Autoencoder (GAE) can no longer reconstruct the network, signaling a structural break or crisis."),
                ui.hr(),
                ui.h5("Peak Anomaly Node"),
                ui.output_text("top_node_text", inline=True)
            ),
            ui.card(
                ui.card_header("Graph Autoencoder (GAE) Reconstruction Error Over Time"),
                output_widget("anomaly_timeline_plot")
            )
        )
    ),
    title="Deep Graph Systemic Risk EWS",
    id="main_nav"
)

# ==========================================
# 3. SERVER LOGIC
# ==========================================
def server(input, output, session):
    
    @output
    @render.text
    def top_node_text():
        if df_anomalies.empty or 'reconstruction_error' not in df_anomalies.columns:
            return "No data available."
        max_row = df_anomalies.loc[df_anomalies['reconstruction_error'].idxmax()]
        return f"{max_row['top_anomalous_node']} ({max_row['period']})"

    @output
    @render_widget
    def anomaly_timeline_plot():
        fig = go.Figure()
        
        # Add the main error line
        fig.add_trace(go.Scatter(
            x=df_anomalies['period'], 
            y=df_anomalies['reconstruction_error'],
            mode='lines+markers',
            line=dict(color='#2980b9', width=2),
            marker=dict(size=4),
            name="GAE Error"
        ))
        
        # THE FIX: Separate the line drawing from the text annotation
        if 'Val' in df_anomalies['regime'].values:
            val_start = df_anomalies[df_anomalies['regime'] == 'Val']['period'].iloc[0]
            fig.add_vline(x=val_start, line_dash="dash", line_color="orange")
            fig.add_annotation(x=val_start, y=1.05, yref="paper", text="Validation (2019)", showarrow=False, font=dict(color="orange"))
            
        if 'Test' in df_anomalies['regime'].values:
            test_start = df_anomalies[df_anomalies['regime'] == 'Test']['period'].iloc[0]
            fig.add_vline(x=test_start, line_dash="dash", line_color="red")
            fig.add_annotation(x=test_start, y=1.05, yref="paper", text="Test (2022)", showarrow=False, font=dict(color="red"))

        fig.update_layout(
            xaxis_title="Quarter",
            yaxis_title="Reconstruction Error",
            template="plotly_white",
            height=600,
            hovermode="x unified",
            margin=dict(t=60) # Added top margin so the labels fit perfectly
        )
        return fig

app = App(app_ui, server)