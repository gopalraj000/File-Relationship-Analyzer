import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from itertools import combinations
import base64

# --- Core Logic from the original script (adapted for Streamlit) ---

def find_common_columns(df1, df2):
    """Finds common columns between two pandas DataFrames."""
    return list(set(df1.columns).intersection(set(df2.columns)))

def find_common_columns_among_n_files(data_frames, n):
    """Finds common columns among a specified number of files."""
    common_data = []
    file_names = list(data_frames.keys())
    
    for file_group in combinations(file_names, n):
        common_cols = set(data_frames[file_group[0]].columns)
        for file_name in file_group[1:]:
            common_cols.intersection_update(data_frames[file_name].columns)
        
        if common_cols:
            common_data.append({
                f'Files ({n})': ', '.join(file_group),
                'Common Columns Count': len(common_cols),
                'Common Columns': ', '.join(sorted(list(common_cols)))
            })
            
    return pd.DataFrame(common_data)

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
                    'Common Columns': ', '.join(sorted(common_cols))
                })
    return pd.DataFrame(relationships_data)

# --- Streamlit UI ---

st.set_page_config(page_title="File Relationship Analyzer", layout="wide")

st.title("🔗 File Relationship Analyzer")
st.markdown('''
This application analyzes multiple CSV files to discover relationships based on common columns. 
Upload your files, and the app will generate a visual graph and summary tables to illustrate how they are interconnected.
''')

# --- Sidebar for controls ---
with st.sidebar:
    st.header("⚙️ Controls")
    
    uploaded_files = st.file_uploader(
        "Upload your CSV files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload two or more CSV files to see the analysis."
    )
    
    st.subheader("Graph Settings")
    show_node_numbers = st.checkbox('Show common column count on nodes', value=True)
    file_name_font_size = st.slider('File name font size', min_value=8, max_value=24, value=14, step=1)
    node_number_font_size = st.slider('Node number font size', min_value=8, max_value=24, value=12, step=1)

# --- Main Analysis and Visualization Area ---

if uploaded_files and len(uploaded_files) >= 2:
    data_frames = {}
    for uploaded_file in uploaded_files:
        try:
            # Reset buffer to the beginning
            uploaded_file.seek(0)
            data_frames[uploaded_file.name] = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
            continue

    relationships_df = get_relationships_table(data_frames)

    if not relationships_df.empty:
        st.header("📊 Analysis Results")

        # --- Visual Representation ---
        st.subheader("Visual Representation of Pairwise Relationships")
        
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

        pos = nx.spring_layout(G, seed=42, k=0.8)

        edge_x, edge_y, edge_text, mid_x, mid_y, label_text, label_hover_text = [], [], [], [], [], [], []

        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            hover_text = edge[2]['hover_info']
            
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.extend([hover_text, hover_text, None])

            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)
            label_text.append(str(edge[2]['weight']))
            label_hover_text.append(edge[2]['hover_info'])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
        edge_label_trace = go.Scatter(x=mid_x, y=mid_y, mode='text', text=label_text, hoverinfo='text', hovertext=label_hover_text, textfont=dict(size=12, color='darkred'))

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
                size=[G.nodes[node]['size'] / max(1, max(G.nodes[node]['size'] for node in G.nodes())) * 30 + 10 for node in G.nodes()],
                line_width=2
            ),
            textfont=dict(size=file_name_font_size, color='black')
        )

        node_label_trace = go.Scatter(x=node_x, y=node_y, mode='text', hoverinfo='text', textposition="top center", textfont=dict(size=node_number_font_size, color='darkblue'))
        
        node_label_text = [str(len(set.intersection(*[set(data_frames[neighbor].columns) for neighbor in G.neighbors(node)]))) if list(G.neighbors(node)) else '0' for node in G.nodes()]
        node_label_trace.text = node_label_text
        node_label_trace.hovertext = node_label_text

        traces = [edge_trace, node_trace, edge_label_trace]
        if show_node_numbers:
            traces.append(node_label_trace)

        fig = go.Figure(
            data=traces,
            layout=go.Layout(
                title=dict(text='CSV File Relationships', font=dict(size=20)),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(text="Node size indicates file size (number of rows). Hover over edges for details.", showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Data Tables ---
        st.subheader("Relationship Summaries")

        with st.expander("Pairwise Relationship Summary", expanded=True):
            st.dataframe(relationships_df)
            st.download_button(
                label="Download Pairwise Summary as CSV",
                data=relationships_df.to_csv(index=False).encode('utf-8'),
                file_name='pairwise_relationships.csv',
                mime='text/csv',
            )

        if len(data_frames) >= 3:
            for n in range(3, len(data_frames) + 1):
                multi_file_df = find_common_columns_among_n_files(data_frames, n)
                if not multi_file_df.empty:
                    with st.expander(f"Common Columns Among {n} Files"):
                        st.dataframe(multi_file_df)
                        st.download_button(
                            label=f"Download {n}-File Summary as CSV",
                            data=multi_file_df.to_csv(index=False).encode('utf-8'),
                            file_name=f'common_columns_{n}_files.csv',
                            mime='text/csv',
                        )

    else:
        st.warning("No common columns found among the uploaded files. No relationships to analyze.")

elif uploaded_files and len(uploaded_files) < 2:
    st.warning("Please upload at least two CSV files to begin the analysis.")

else:
    st.info("Waiting for CSV file uploads...")
