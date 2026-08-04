import os
import sys
repo_path =  ".."
os.chdir(repo_path)                 # Move into the repo
sys.path.insert(0, os.getcwd())     # Ensure the repo root is on sys.path

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def plot_asr_decay(turn_numbers, asr_values, defense_asr_values=None, title="ASR Decay Curve"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=turn_numbers, y=asr_values, mode='lines+markers', name='No Defense'))
        if defense_asr_values is not None:
            fig.add_trace(go.Scatter(x=turn_numbers, y=defense_asr_values, mode='lines+markers', name='With Sanitization'))
        fig.update_layout(title=title, xaxis_title="Interaction Turn", yaxis_title="Attack Success Rate")
        return fig

    @staticmethod
    def plot_umap_embeddings(embeddings, labels, title="UMAP Projection"):
        reducer = umap.UMAP(random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
        df = pd.DataFrame(embedding_2d, columns=['UMAP1', 'UMAP2'])
        df['Label'] = labels
        fig = px.scatter(df, x='UMAP1', y='UMAP2', color='Label', title=title)
        return fig

    @staticmethod
    def plot_leakage_heatmap(similarity_matrix, session_ids):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarity_matrix, xticklabels=session_ids, yticklabels=session_ids,
                    annot=True, fmt=".2f", cmap="Reds", ax=ax)
        ax.set_title("Cross-Session Context Leakage Heatmap")
        return fig