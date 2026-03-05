import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPoint, Polygon

st.set_page_config(page_title="Geofence Clustering", layout="centered")
st.title("Acelera Manual Strategies, VROOM VROOM | Use template: 104650")

# Predefined named colors for clusters
named_colors = [
    "red", "blue", "green", "orange", "purple",
    "cyan", "magenta", "darkred", "cadetblue", "darkgreen"
]

# Upload file
uploaded_file = st.file_uploader("Upload your data file CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    # 1. READ DATA
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # 2. DATA CLEANING (Fixed: Removed regex=True to prevent \N interpretation error)
    # We replace common SQL/Data null strings with actual NaN
    df = df.replace([r'\\N', r'\N', 'None', 'null', 'nan'], np.nan)

    # Convert essential columns to numeric (coercing errors to NaN)
    # This is the 'safety net'—anything that isn't a number becomes NaN here
    cols_to_fix = ['lat', 'lng', 'completion_rate', 'orders_initial']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that are missing coordinates or completion rates
    df = df.dropna(subset=['lat', 'lng', 'completion_rate'])

    if df.empty:
        st.error(r"Wait! After cleaning the '\N' values, the dataset is empty. Check your file.")
    else:
        st.success(f"File uploaded and cleaned! {len(df)} rows ready for clustering.")

        # 3. FEATURE SELECTION
        clustering_cols = st.multiselect(
            "Select features to use for clustering :)",
            options=df.columns.tolist(),
            default=["lat", "lng", "orders_initial", "completion_rate"]
        )

        if clustering_cols:
            # Normalize & Cluster
            scaler = MinMaxScaler()
            temp_df = df.dropna(subset=clustering_cols).copy()
            
            scaled_features = scaler.fit_transform(temp_df[clustering_cols])

            n_clusters = st.slider("Select number of clusters | Recommended amount is 4", 2, 10, 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            temp_df['cluster'] = kmeans.fit_predict(scaled_features)

            # Assign colors
            cluster_colors = {i: named_colors[i % len(named_colors)] for i in range(n_clusters)}

            # 4. CLUSTER SUMMARY
            st.subheader("Cluster Summary")
            summary = temp_df.groupby('cluster').agg({
                'completion_rate': 'mean',
                'orders_initial': 'mean'
            }).reset_index()

            summary['color'] = summary['cluster'].map(cluster_colors)
            summary.rename(columns={
                'completion_rate': 'avg_completion_rate',
                'orders_initial': 'avg_orders_initial'
            }, inplace=True)

            st.dataframe(summary)

            # 5. POLYGON GENERATION
            st.subheader("Cluster Polygon Coordinates (Copy and Paste them in Duse-Eye)")
            
            polygons_to_draw = []

            for cluster_id in sorted(temp_df['cluster'].unique()):
                cluster_points_df = temp_df[temp_df['cluster'] == cluster_id][['lat', 'lng']].copy()

                # Calculate centroid and distance for outlier filtering
                centroid = cluster_points_df[['lat', 'lng']].mean().values
                cluster_points_df['dist'] = np.sqrt(
                    (cluster_points_df['lat'] - centroid[0])**2 + 
                    (cluster_points_df['lng'] - centroid[1])**2
                )

                # Remove top 2% farthest points (outliers)
                filtered_cluster = cluster_points_df[
                    cluster_points_df['dist'] <= cluster_points_df['dist'].quantile(0.98)
                ]

                if len(filtered_cluster) < 3:
                    continue

                # Create Convex Hull
                points_array = filtered_cluster[['lng', 'lat']].values
                hull = MultiPoint(points_array).convex_hull
                
                if isinstance(hull, Polygon):
                    coords = list(hull.exterior.coords)
                    polygons_to_draw.append((cluster_id, hull))
                else:
                    coords = list(hull.coords)
                
                coord_text = "\n".join([f"{lat:.6f}, {lng:.6f}" for lng, lat in coords])

                with st.expander(f"Cluster {cluster_id} ({cluster_colors[cluster_id]}) Coordinates"):
                    st.text_area("Copy-paste friendly (lat, lng):", coord_text, height=150, key=f"txt_{cluster_id}")

            # 6. MAP VISUALIZATION
            st.subheader("Cluster Map")
            m = folium.Map(location=[temp_df['lat'].mean(), temp_df['lng'].mean()], zoom_start=12)

            # Add markers
            for _, row in temp_df.iterrows():
                folium.CircleMarker(
                    location=[row['lat'], row['lng']],
                    radius=3,
                    color=cluster_colors[int(row['cluster'])],
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)

            # Add Polygons
            for cluster_id, hull in polygons_to_draw:
                map_coords = [(lat, lng) for lng, lat in hull.exterior.coords]
                folium.Polygon(
                    locations=map_coords,
                    color=cluster_colors[cluster_id],
                    fill=True,
                    fill_color=cluster_colors[cluster_id],
                    fill_opacity=0.3,
                    weight=3,
                    popup=f"Geofence {cluster_id}"
                ).add_to(m)

            st_folium(m, width=700, height=500)