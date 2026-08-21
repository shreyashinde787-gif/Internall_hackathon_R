import streamlit as st
import folium
from streamlit_folium import st_folium

def render_defect_map(defects_list):
    """
    Renders an interactive map from a list of defect dictionaries.
    """
    # Create base map centered on Pune
    city_map = folium.Map(location=[18.5204, 73.8567], zoom_start=13)
    
    # Loop through incoming data and plot points
    for defect in defects_list:
        color = "red" if defect["severity"] == "High Risk" else "orange" if defect["severity"] == "Medium Risk" else "green"
        
        popup_html = f"<b>Type:</b> {defect['type']}<br><b>Severity:</b> {defect['severity']}"
        
        folium.CircleMarker(
            location=[defect["lat"], defect["lng"]],
            radius=9,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{defect['severity']}: {defect['type']}",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(city_map)
        
    st_folium(city_map, width=1000, height=550)

# Local test execution
if __name__ == "__main__":
    st.title("🚦 Smart City Infrastructure Map")
    sample_data = [
        {"lat": 18.5204, "lng": 73.8567, "severity": "High Risk", "type": "Deep Pothole"},
        {"lat": 18.5250, "lng": 73.8580, "severity": "Medium Risk", "type": "Major Crack"},
        {"lat": 18.5180, "lng": 73.8530, "severity": "Low Risk", "type": "Faded Marking"}
    ]
    render_defect_map(sample_data)