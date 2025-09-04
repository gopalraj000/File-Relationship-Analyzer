import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from itertools import combinations
import tempfile
import os

# --- Helper Functions ---

def find_common_columns(df1, df2):
    """Finds common columns between two pandas DataFrames."""
    return list(set(df1.columns).intersection(set(df2.columns)))

def get_relationships_table(data_frames):
    """Generates a DataFrame summarizing file relationships."""
    relationships_data = []
    file_names = list(data_frames.keys())
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            file1 = file_names[i]
            file2 = file_names[j]
            df1 = data_frames[file1]
            df2 = data_frames[file2]
            
            common_cols = find_common_columns(df1, df2)
            if common_cols:
                relationships_data.append({
                    'File 1': file1,
                    'File 2': file2,
                    'Common Columns Count': len(common_cols),
                    'Common Columns': ', '.join(common_cols)
                })
    return pd.DataFrame(relationships_data)

# --- Streamlit Application ---

st.set_page_config(layout="wide")

st.title("CSV File Relationship Analyzer")
st.markdown("Upload multiple CSV files to visualize their column-level relationships.")

# File uploader widget
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    # Use a temporary directory to save the uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        data_frames = {}
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the data frame
            try:
                data_frames[uploaded_file.name] = pd.read_csv(file_path)
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {e}")
                continue

        if len(data_frames) < 2:
            st.warning("Please upload at least two CSV files to analyze relationships.")
        else:
            # Generate the relationships table
            relationships_df = get_relationships_table(data_frames)

            if not relationships_df.empty:
                st.subheader("Pairwise Relationship Summary")
                st.dataframe(relationships_df, use_container_width=True)

                st.subheader("Visual Representation of Pairwise Relationships")
                
                # Create a graph
                G = nx.Graph()
                for file_name in data_frames.keys():
                    G.add_node(file_name, size=len(data_frames[file_name]))

                for _, row in relationships_df.iterrows():
                    G.add_edge(
                        row['File 1'], 
                        row['File 2'], 
                        weight=row['Common Columns Count'],
                        hover_info=f"Common Columns ({row['Common Columns Count']}):<br>{row['Common Columns']}"
                    )

                # Use spring layout for node positions
                pos = nx.spring_layout(G, seed=42, k=0.8)

                # Create edge traces
                edge_x = []
                edge_y = []
                mid_x = []
                mid_y = []
                label_text = []
                label_hover_text = []

                for edge in G.edges(data=True):
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    hover_text = edge[2]['hover_info']
                    
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                    mid_x.append((x0 + x1) / 2)
                    mid_y.append((y0 + y1) / 2)
                    label_text.append(str(edge[2]['weight']))
                    label_hover_text.append(hover_text)

                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )

                edge_label_trace = go.Scatter(
                    x=mid_x, y=mid_y,
                    mode='text',
                    text=label_text,
                    hoverinfo='text',
                    hovertext=label_hover_text,
                    textfont=dict(size=12, color='darkred', family='Arial, sans-serif'),
                )
                
                # Create node traces
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=[f'<b>{node}</b>' for node in G.nodes()],
                    textposition="bottom center",
                    hoverinfo='text',
                    hovertext=[f"<b>{node}</b><br>Rows: {G.nodes[node]['size']}" for node in G.nodes()],
                    marker=dict(
                        color='#A0CBE2',
                        size=[G.nodes[node]['size'] / max(G.nodes[node]['size'] for node in G.nodes()) * 30 + 10 for node in G.nodes()],
                        line_width=2
                    ),
                    textfont=dict(size=14, color='black')
                )

                # Create a separate trace for node labels (numbers)
                node_label_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='text',
                    hoverinfo='text',
                    textposition="top center",
                    textfont=dict(size=12, color='darkblue', family='Arial, sans-serif'),
                )
                
                node_label_text = []
                node_label_hover_text = []

                # Calculate unique common columns for the node number
                for node in G.nodes():
                    neighbors = list(G.neighbors(node))
                    if neighbors:
                        common_cols_in_group = set(data_frames[node].columns)
                        # Find columns common to all neighbors
                        for neighbor in neighbors:
                            common_cols_in_group.intersection_update(data_frames[neighbor].columns)
                        unique_cols_count = len(common_cols_in_group)
                    else:
                        unique_cols_count = 0
                    
                    node_label_text.append(str(unique_cols_count))
                    node_label_hover_text.append(str(unique_cols_count))

                node_label_trace.text = node_label_text
                node_label_trace.hovertext = node_label_hover_text

                # Create figure and layout
                fig = go.Figure(
                    data=[edge_trace, node_trace, edge_label_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
                st.caption("Hover over a number on a node for a summary. Hover over a number on an edge for pairwise details. Node size indicates file size (number of rows).")
            
            else:
                st.warning("No common columns found between any of the uploaded files.")
