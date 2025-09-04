import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from itertools import combinations
import base64

# --- Helper Functions (adapted for Streamlit) ---

def find_common_columns(df1, df2):
    """Finds common columns between two pandas DataFrames."""
    return list(set(df1.columns).intersection(set(df2.columns)))

def find_common_columns_among_n_files(data_frames, n):
    """Finds common columns among a specified number of files."""
    common_data = []
    file_names = list(data_frames.keys())
    
    if len(file_names) < n:
        return pd.DataFrame() # Return empty if not enough files
    
    # Iterate through all combinations of n files
    for file_group in combinations(file_names, n):
        # Get the columns from the first file in the group
        common_cols = set(data_frames[file_group[0]].columns)
        
        # Intersect with columns from the remaining files in the group
        for file_name in file_group[1:]:
            common_cols.intersection_update(data_frames[file_name].columns)
        
        if common_cols:
            common_data.append({
                f'Files ({n})': ', '.join(file_group),
                'Common Columns Count': len(common_cols),
                'Common Columns': ', '.join(common_cols)
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
                    'Common Columns': ', '.join(common_cols)
                })
    return pd.DataFrame(relationships_data)

def create_download_link(df, filename, file_format, link_text):
    """Creates a downloadable link for a DataFrame."""
    if file_format == 'csv':
        content = df.to_csv(index=False)
        mime_type = 'text/csv'
    elif file_format == 'html':
        content = df.to_html(index=False)
        mime_type = 'text/html'
    else:
        return None
        
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# --- Main Streamlit App Logic ---

st.title('CSV File Relationship Analyzer')
st.markdown('Upload multiple CSV files to analyze and visualize common columns between them.')

uploaded_files = st.file_uploader(
    "Choose CSV files",
    type="csv",
    accept_multiple_files=True
)

if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning('Please upload at least two CSV files.')
    else:
        # Load all files into a dictionary of DataFrames
        data_frames = {}
        for file in uploaded_files:
            try:
                # Use st.cache_data to cache the DataFrame loading
                @st.cache_data
                def load_data(file):
                    return pd.read_csv(file)
                data_frames[file.name] = load_data(file)
            except Exception as e:
                st.error(f"Error loading {file.name}: {e}")
                
        # --- Sidebar for user settings ---
        with st.sidebar:
            st.header("Visualization Options")
            show_node_numbers = st.checkbox(
                'Show node numbers (common columns for the entire connected group)', 
                value=True
            )
            file_name_font_size = st.slider(
                'File name font size', 
                min_value=8, 
                max_value=24, 
                value=14, 
                step=1
            )
            node_number_font_size = st.slider(
                'Node number font size', 
                min_value=8, 
                max_value=24, 
                value=12, 
                step=1
            )
        
        # --- Analysis Section ---
        st.header("File Relationship Analysis")
        
        relationships_df = get_relationships_table(data_frames)
        
        if not relationships_df.empty:
            st.subheader("Pairwise Relationship Summary:")
            st.dataframe(relationships_df)
            
            # Create a download button for the table
            csv_content = relationships_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download as CSV",
                data=csv_content,
                file_name='relationships.csv',
                mime='text/csv',
            )
            
            # --- Dynamic section for multi-file analysis ---
            st.subheader("Common Columns Among Multiple Files:")
            max_n = len(data_frames)
            # Use st.expander to keep the UI clean
            with st.expander("View common columns for combinations of 3 or more files"):
                for n in range(3, max_n + 1):
                    st.write(f"**Common Columns Among {n} Files:**")
                    multi_file_df = find_common_columns_among_n_files(data_frames, n)
                    if not multi_file_df.empty:
                        st.dataframe(multi_file_df)
                        csv_content_multi = multi_file_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"Download Common Columns ({n}) as CSV",
                            data=csv_content_multi,
                            file_name=f'common_columns_{n}.csv',
                            mime='text/csv',
                            key=f'multi_csv_{n}'
                        )
                    else:
                        st.info(f"No common columns found among any combination of {n} files.")
        else:
            st.info("No relationships found based on common columns. The files may have no columns in common.")

        # --- Visualization Section ---
        if not relationships_df.empty:
            st.header("Visual Representation of Pairwise Relationships:")
            st.markdown(
                """
                Hover over an **edge** to see the shared columns. 
                Hover over a **node's number** for a total summary. 
                Node size indicates file size (number of rows).
                """
            )
            
            G = nx.Graph()
            
            # Add nodes for each file
            for file_name, df in data_frames.items():
                G.add_node(file_name, size=len(df))
            
            # Add edges based on the relationships table
            for _, row in relationships_df.iterrows():
                G.add_edge(
                    row['File 1'], 
                    row['File 2'], 
                    weight=row['Common Columns Count'],
                    hover_info=f"Common Columns ({row['Common Columns Count']}):<br>{row['Common Columns']}"
                )

            # Use spring layout for node positions
            pos = nx.spring_layout(G, seed=42, k=0.8)

            # Create edge traces for Plotly
            edge_x, edge_y, edge_text = [], [], []
            mid_x, mid_y, label_text, label_hover_text = [], [], [], []

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

            # Create node traces for Plotly
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
                    size=[(G.nodes[node]['size'] / max(G.nodes[node]['size'] for node in G.nodes())) * 30 + 10 for node in G.nodes()],
                    line_width=2
                ),
                textfont=dict(size=file_name_font_size, color='black')
            )

            node_label_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='text',
                textposition="top center",
                textfont=dict(size=node_number_font_size, color='darkblue', family='Arial, sans-serif'),
            )
            
            node_label_text = []
            node_label_hover_text = []

            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                if neighbors:
                    # Find the intersection of all common columns for the connected group
                    first_neighbor_cols = set(data_frames[neighbors[0]].columns)
                    common_cols_in_group = set(data_frames[node].columns).intersection(first_neighbor_cols)
                    
                    for neighbor in neighbors[1:]:
                        common_cols_in_group.intersection_update(data_frames[neighbor].columns)
                        
                    unique_cols_count = len(common_cols_in_group)
                else:
                    unique_cols_count = 0
                
                node_label_text.append(str(unique_cols_count))
                node_label_hover_text.append(str(unique_cols_count))

            node_label_trace.text = node_label_text
            node_label_trace.hovertext = node_label_hover_text

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
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            st.plotly_chart(fig)
        else:
            st.info("No common columns were found to create a relationship graph.")