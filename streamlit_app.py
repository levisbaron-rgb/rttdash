import streamlit as st
import pandas as pd
import numpy as np

# Force Wide Layout
st.set_page_config(layout="wide", page_title="DIP Stage Comparison | Use:107183")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; }
        .header-block {
            background-color: #FF4B11;
            color: white !important;
            padding: 12px 20px;
            border-radius: 8px 8px 0px 0px;
            font-weight: 800;
            margin-top: 20px;
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
        [data-testid="stDataFrame"] th:first-child { min-width: 450px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("DIP Deep Dive: Multi-Stage Analysis Use Template: 107183")

# --- 1. Data Loading ---
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx", "xls", "csv"])

if uploaded_file is not None:
    try:
        # Check file type for proper loading
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        
        df.columns = df.columns.str.strip()
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df = df.dropna(subset=['start_date'])
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['day of the week'] = pd.Categorical(df['start_date'].dt.day_name(), categories=days_order, ordered=True)

        city_options = sorted(df['city_name'].dropna().unique().tolist())
        selected_city = st.sidebar.selectbox("City Select", ['All'] + city_options)
        
        working_df = df.copy()
        if selected_city != 'All': 
            working_df = working_df[working_df['city_name'] == selected_city]

        # --- 2. Calculation Logic ---
        def get_metrics_table(stage_data, col_name, is_rpt=False):
            if stage_data.empty: return None
            
            # Base Stats
            stats = stage_data.groupby('day of the week', observed=False)[col_name].agg(['mean', 'max', 'min']).transpose()
            stats.index = ['Avg', 'Max', 'Min']
            
            temp_df = stage_data.copy()

            if not is_rpt:
                # FIXED RANGES FOR TPH
                tph_bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 100] # Added overflow bin
                tph_labels = ["(%) Drivers TPH (0 - 0.5)", "(%) Drivers TPH (0.5 - 1)", 
                              "(%) Drivers TPH (1 - 1.5)", "(%) Drivers TPH (1.5 - 2)", 
                              "(%) Drivers TPH (2 - 2.5)", "(%) Drivers TPH (2.5 - 3)",
                              "(%) Drivers TPH (> 3.0)"]
                
                temp_df['bin'] = pd.cut(temp_df[col_name], bins=tph_bins, labels=tph_labels, include_lowest=True)
                dist_df = temp_df.dropna(subset=['bin'])
            else:
                # PERCENTILE BINS FOR RPT
                try:
                    _, bins = pd.qcut(temp_df[col_name], q=[0, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0], retbins=True, duplicates='drop')
                    if len(bins) < 2: return stats[days_order]
                    
                    labels = [f"(%) Drivers RPT Range ({bins[i]:.2f} - {bins[i+1]:.2f})" for i in range(len(bins)-1)]
                    temp_df['bin'] = pd.cut(temp_df[col_name], bins=bins, labels=labels, include_lowest=True)
                    dist_df = temp_df.dropna(subset=['bin'])
                except:
                    return stats[days_order]

            # Calculate Percentages
            grouped = dist_df.groupby(['day of the week', 'bin'], observed=False)['driver_num_treatment'].sum().reset_index()
            totals = dist_df.groupby('day of the week', observed=False)['driver_num_treatment'].sum().reset_index().rename(columns={'driver_num_treatment': 'total'})
            merged = pd.merge(grouped, totals, on='day of the week')
            merged['percent'] = (merged['driver_num_treatment'] / merged['total'] * 100).fillna(0)
            
            dist_table = merged.pivot(index='bin', columns='day of the week', values='percent').fillna(0)
            
            return pd.concat([stats, dist_table])[days_order]

        def apply_vivid_style(df):
            # na_rep=" " replaces NaNs with blanks in the view
            # subset ensures we only apply gradients to the percentile rows (index 3 and onwards)
            return df.style.format(precision=2, na_rep=" ") \
                .background_gradient(cmap='Oranges', subset=pd.IndexSlice[df.index[3:], :]) \
                .set_properties(**{'color': 'black'})

        # --- 3. Render Stage Tables ---
        stages = sorted(working_df['stage'].unique())
        for stage_num in stages:
            st.subheader(f"📍 Operational Stage: {stage_num}")
            stage_df = working_df[working_df['stage'] == stage_num]
            
            tph_table = get_metrics_table(stage_df, 'tph', is_rpt=False)
            rpt_table = get_metrics_table(stage_df, 'rpt', is_rpt=True)

            col_left, col_right = st.columns(2)
            with col_left:
                if tph_table is not None:
                    st.markdown(f'<div class="header-block">📈 Stage {stage_num}: TPH Performance</div>', unsafe_allow_html=True)
                    st.markdown('<div class="content-container">', unsafe_allow_html=True)
                    st.dataframe(apply_vivid_style(tph_table), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            with col_right:
                if rpt_table is not None:
                    st.markdown(f'<div class="header-block">💰 Stage {stage_num}: RPT Performance</div>', unsafe_allow_html=True)
                    st.markdown('<div class="content-container">', unsafe_allow_html=True)
                    st.dataframe(apply_vivid_style(rpt_table), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

        # --- 4. Final Global Summary Tables ---
        st.divider()
        summary_base = working_df.groupby(['stage', 'day of the week'], observed=False)[['tph', 'rpt']].mean().reset_index()

        col_sum_left, col_sum_right = st.columns(2)

        with col_sum_left:
            st.markdown('<div class="header-block">📋 Global Summary: Avg TPH per Day</div>', unsafe_allow_html=True)
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            tph_summary = summary_base.pivot(index='stage', columns='day of the week', values='tph')
            tph_summary.index = [f"Stage {int(i)}" for i in tph_summary.index]
            # Convert to styled format before table display
            st.table(tph_summary.style.format(precision=2, na_rep="-"))
            st.markdown('</div>', unsafe_allow_html=True)

        with col_sum_right:
            st.markdown('<div class="header-block">📋 Global Summary: Avg RPT per Day</div>', unsafe_allow_html=True)
            st.markdown('<div class="content-container">', unsafe_allow_html=True)
            rpt_summary = summary_base.pivot(index='stage', columns='day of the week', values='rpt')
            rpt_summary.index = [f"Stage {int(i)}" for i in rpt_summary.index]
            st.table(rpt_summary.style.format(precision=2, na_rep="-"))
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing data: {e}")
        st.exception(e) # Show full traceback for debugging
else:
    st.info("Please upload your Excel file to continue.")