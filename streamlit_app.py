import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Force Wide Layout
st.set_page_config(layout="wide", page_title="Operational Dashboard")

# --- CUSTOM CSS: Orange Shadows, High Contrast, and Sidebar ---
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; }
        [data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #f0f0f0; }
        
        /* The "Squares" with Orange Shadow & Higher Contrast Border */
        .metric-container {
            background-color: #fff4f0; 
            border-radius: 16px;
            padding: 25px;
            border: 2px solid #ffccbc;
            box-shadow: 0 10px 25px rgba(255, 75, 17, 0.1); /* Subtle orange glow */
            margin-bottom: 25px;
        }

        /* Ensure Table Index (First Column) is wide and readable */
        [data-testid="stTable"] th, [data-testid="stTable"] td {
            min-width: 200px !important;
        }
        
        h1, h2, h3, h4 { color: #000000 !important; font-weight: 800 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("Operational Insights Dashboard")

# --- 1. Data Loading ---
st.sidebar.markdown("### 🟠 Data Management")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['day of the week'] = df['start_date'].dt.day_name()

        # --- 2. Filters ---
        city_options = sorted(df['city_name'].dropna().unique().tolist())
        selected_city = st.sidebar.selectbox("City Select", ['All'] + city_options)
        
        graph_df = df.copy()
        if selected_city != 'All': graph_df = graph_df[graph_df['city_name'] == selected_city]

        selected_stage = st.sidebar.selectbox("Stage for Averages", ['All'] + sorted(df['stage'].unique().tolist()))
        avg_df = graph_df.copy()
        if selected_stage != 'All': avg_df = avg_df[avg_df['stage'] == selected_stage]

        dist_df = graph_df[graph_df['stage'] == 1].copy()

        if not avg_df.empty:
            # --- 3. Calculation Helper ---
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            def get_metrics_table(data_avg, data_dist, col_name, prefix):
                stats = data_avg.groupby('day of the week')[col_name].agg(['mean', 'max', 'min']).transpose()
                qs = [0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
                bins = sorted(data_dist[col_name].quantile(qs).unique())
                if len(bins) < 2: return None
                
                labels = [f"Drivers {prefix} {bins[i]:.2f}-{bins[i+1]:.2f} (%)" for i in range(len(bins)-1)]
                data_dist['bin'] = pd.cut(data_dist[col_name], bins=bins, labels=labels, include_lowest=True)
                
                grouped = data_dist.groupby(['day of the week', 'bin'], observed=False)['driver_num_treatment'].sum().reset_index()
                totals = data_dist.groupby('day of the week')['driver_num_treatment'].sum().reset_index().rename(columns={'driver_num_treatment': 'total'})
                merged = pd.merge(grouped, totals, on='day of the week')
                merged['percent'] = (merged['driver_num_treatment'] / merged['total'] * 100).fillna(0)
                
                dist_table = merged.pivot(index='bin', columns='day of the week', values='percent').fillna(0)
                available = [d for d in days_order if d in stats.columns]
                return pd.concat([stats[available].rename(index={'mean':'Avg', 'max':'Max', 'min':'Min'}), dist_table[available]])

            tph_final = get_metrics_table(avg_df, dist_df, 'tph', 'TPH')
            rpt_final = get_metrics_table(avg_df, dist_df, 'rpt', 'RPT')

            # --- 4. Side-by-Side Tables with Contrast ---
            col_left, col_right = st.columns(2)

            def format_table(df, is_money_table=False):
                formatted = df.copy().astype(object)
                for r in df.index:
                    for c in df.columns:
                        val = df.loc[r, c]
                        if r in ['Avg', 'Max', 'Min']:
                            formatted.at[r, c] = f"{int(val)}$" if is_money_table else f"{val:.2f}"
                        else:
                            formatted.at[r, c] = f"{val:.1f}%"
                return formatted

            with col_left:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown("### 📈 TPH Performance")
                # Using st.table for fixed-width columns that don't cut text
                st.table(format_table(tph_final, False))
                st.markdown('</div>', unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                st.markdown("### 💰 RPT Performance")
                st.table(format_table(rpt_final, True))
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")

            # --- 5. Visualizations (Plotly with Axis Labels) ---
            g_col1, g_col2 = st.columns(2)
            stage_stats = graph_df.groupby('stage')[['tph', 'rpt']].mean().reset_index()

            with g_col1:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                fig_tph = px.bar(stage_stats, x='stage', y='tph', 
                                 title="Avg TPH per Stage",
                                 labels={'stage': 'Operational Stage', 'tph': 'Average Trips Per Hour (TPH)'},
                                 color_discrete_sequence=['#FF4B11'])
                fig_tph.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tph, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with g_col2:
                st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                fig_rpt = px.bar(stage_stats, x='stage', y='rpt', 
                                 title="Avg RPT per Stage",
                                 labels={'stage': 'Operational Stage', 'rpt': 'Average Reward Per Trip (RPT)'},
                                 color_discrete_sequence=['#FF4B11'])
                fig_rpt.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_rpt, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload your CSV file to continue.")