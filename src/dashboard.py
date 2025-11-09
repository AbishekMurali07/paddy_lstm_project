"""
dashboard.py
Streamlit dashboard for visualizing crop yield, rainfall, and soil data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

def show_dashboard(df):
    st.subheader("📈 Crop Yield Dashboard (Tamil Nadu Paddy Data)")

    st.sidebar.header("🔍 Filters")
    district = st.sidebar.selectbox("Select District", sorted(df["District"].unique()))
    year = st.sidebar.selectbox("Select Year", sorted(df["Year"].unique()))

    filtered = df[(df["District"] == district) & (df["Year"] == year)]

    st.write(f"### 🏡 Data for {district} ({year})")
    st.dataframe(filtered)

    # Yield Trend
    fig_yield = px.line(df[df["District"] == district], x="Year", y="Yield",
                        title=f"🌾 Yield Trend for {district}")
    st.plotly_chart(fig_yield, use_container_width=True)

    # Rainfall vs Yield
    fig_rf = px.scatter(df[df["District"] == district],
                        x="Annual_Rainfall_mm", y="Yield",
                        color="Year", size="Production",
                        title="🌧️ Rainfall vs Yield")
    st.plotly_chart(fig_rf, use_container_width=True)

    # Soil Composition
    st.write("### 🧪 Soil Composition by District")
    fig_soil = px.bar(df[df["District"] == district],
                      x="Soil_Type", y="Nitrogen",
                      color="Phosphorus", barmode="group",
                      title="Soil Nutrient Comparison")
    st.plotly_chart(fig_soil, use_container_width=True)
