import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import detect_lat_lon


def run_eda(df, target_col):
    st.subheader("Dataset Overview")
    st.write(df.shape)
    st.dataframe(df.head())

    st.subheader("Missing Values")
    st.write(df.isnull().mean().sort_values(ascending=False).head(10))

    st.subheader("Target Distribution")
    st.bar_chart(df[target_col].value_counts())

    st.subheader("Correlation Heatmap (numeric)")
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] > 1:
        fig, ax = plt.subplots()
        sns.heatmap(num_df.corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ✅ CORRECT MAP LOGIC
    lat, lon = detect_lat_lon(df)

    if lat is None or lon is None:
        st.info("No latitude/longitude columns found. Map skipped.")
        return

    map_df = df[[lat, lon]].dropna()

    if map_df.empty:
        st.info("Latitude/Longitude columns exist but contain no valid coordinates.")
        return

    st.subheader("Accident Locations")
    st.map(map_df.rename(columns={lat: "lat", lon: "lon"}))
