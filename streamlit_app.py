import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Force Wide Layout
st.set_page_config(layout="wide", page_title="DIP Deep Dive | Use:107183 ")

# --- CUSTOM CSS: Neutral Text (No Green) & Orange Glow ---
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; }
        
        .header-block {
            background-color: #FF4B11;
            color: white !important;
            padding: 12px 20px;
            border-radius: 8px 8px 0px 0px;
            font-weight: 800;
            display: flex;
            align-items: center;
        }

        .content-container {
            background-color: #fff4f0; 
            border-radius: 0px 0px 12px 12px;
            padding: 20px;
            border: 2px solid #ffccbc;
            box-shadow: 0 10px 20px rgba(255, 75, 17, 0.08);
            margin-bottom: 30px;
        }

        /* Target dataframe cells to ensure black text */
        [data-testid="stTable"] td, [data-testid="stDataFrame"] td {
            color: #000000 !important;
        }
        
        [data-testid="stDataFrame"] th:first-child {
            min-width: 450px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Insights")

# --- 1. Data Loading ---
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df['start_date'] = pd.to_datetime(df['start_date'])
        
        # --- FIXED DAY ORDERING ---
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day of the week'] = pd.Categorical(df['start_date'].dt.day_name(), categories=days_order, ordered=True)

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
            # --- 3. Calculation Logic ---
            def get_metrics_table(data_avg, data_dist, col_name, prefix, is_rpt=False):
                # Ensure grouping respects Categorical order
                stats = data_avg.groupby('day of the week', observed=False)[col_name].agg(['mean', 'max', 'min']).transpose()
                
                qs = [0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
                bins = sorted(data_dist[col_name].quantile(qs).unique())
                if len(bins) < 2: return None
                
                labels = []
                for i in range(len(bins)-1):
                    low, high = bins[i], bins[i+1]
                    if is_rpt:
                        labels.append(f"(%) Drivers inside RPT ({int(low)} - {int(high)})")
                    else:
                        labels.append(f"(%) Drivers inside TPH Req of ({low:.2f} - {high:.2f})")
                
                data_dist['bin'] = pd.cut(data_dist[col_name], bins=bins, labels=labels, include_lowest=True)
                
                # Pivot and reorder columns
                grouped = data_dist.groupby(['day of the week', 'bin'], observed=False)['driver_num_treatment'].sum().reset_index()
                totals = data_dist.groupby('day of the week', observed=False)['driver_num_treatment'].sum().reset_index().rename(columns={'driver_num_treatment': 'total'})
                merged = pd.merge(grouped, totals, on='day of the week')
                merged['percent'] = (merged['driver_num_treatment'] / merged['total'] * 100).fillna(0)
                
                dist_table = merged.pivot(index='bin', columns='day of the week', values='percent').fillna(0)
                
                # Final assembly ensuring day order
                final_df = pd.concat([stats.rename(index={'mean':'Avg', 'max':'Max', 'min':'Min'}), dist_table])
                return final_df[days_order] # Force the column order

            tph_final = get_metrics_table(avg_df, dist_df, 'tph', 'TPH', is_rpt=False)
            rpt_final = get_metrics_table(avg_df, dist_df, 'rpt', 'RPT', is_rpt=True)

            # --- 4. Heatmap Styling Function ---
            def apply_vivid_style(df, is_money=False):
                # We apply heatmap only to the percentile rows (from index 3 onwards)
                percentile_rows = df.index[3:]
                
                return df.style.format(lambda x: f"{x:.2f}" if isinstance(x, float) else x) \
                    .background_gradient(cmap='Oranges', subset=pd.IndexSlice[percentile_rows, :]) \
                    .set_properties(**{'color': 'black'})

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown('<div class="header-block">📈 TPH Performance</div>', unsafe_allow_html=True)
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                st.dataframe(apply_vivid_style(tph_final), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_right:
                st.markdown('<div class="header-block">💰 RPT Performance</div>', unsafe_allow_html=True)
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                # Money symbols removed as requested, just showing values
                st.dataframe(apply_vivid_style(rpt_final), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- 5. Visualizations ---
            g_col1, g_col2 = st.columns(2)
            stage_stats = graph_df.groupby('stage')[['tph', 'rpt']].mean().reset_index()

            with g_col1:
                st.markdown('<div class="header-block">📊 Avg TPH per Stage</div>', unsafe_allow_html=True)
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                fig_tph = px.bar(stage_stats, x='stage', y='tph', labels={'stage': 'Operational Stage', 'tph': 'Average TPH'}, color_discrete_sequence=['#ffccbc'])
                fig_tph.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tph, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with g_col2:
                st.markdown('<div class="header-block">📊 Avg RPT per Stage</div>', unsafe_allow_html=True)
                st.markdown('<div class="content-container">', unsafe_allow_html=True)
                fig_rpt = px.bar(stage_stats, x='stage', y='rpt', labels={'stage': 'Operational Stage', 'rpt': 'Average RPT'}, color_discrete_sequence=['#ffccbc'])
                fig_rpt.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_rpt, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Please upload your CSV file to continue.")