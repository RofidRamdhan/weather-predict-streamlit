# -*- coding: utf-8 -*-
"""
Streamlit Weather App for Jakarta
=================================
This Streamlit webapp displays the current weather and a machine learning-based forecast for Jakarta.

Enjoy exploring!
"""

import streamlit as st
import pandas as pd
import requests
import json
import mysql.connector
import joblib
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from datetime import datetime, timezone as tmz
import pytz
from tzwhere import tzwhere
import folium
from streamlit_folium import folium_static

# Load your machine learning model
model = joblib.load("prophet_model_temp.pkl")

# Database connection details
db_config = {
    'user': 'root',
    'password': '',
    'host': 'local_host',
    'database': 'weather-predict'
}

# Function to fetch data from MySQL database
def fetch_data(query):
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return pd.DataFrame(result)

# Coordinates for Jakarta
lat = -6.2088
lng = 106.8456
city = "Jakarta"
country = "Indonesia"

# Title and description for your app
st.title("Weather in Jakarta :sun_behind_rain_cloud:")

# Current weather from Open-Meteo
response_current = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true')

st.subheader("Current weather")
result_current = json.loads(response_current.content)
current = result_current["current_weather"]
temp = current["temperature"]
speed = current["windspeed"]
direction = current["winddirection"]
pressure = current["pressure"]

# Increment added or subtracted from degree values for wind direction
ddeg = 11.25
direction_labels = ["N", "N/NE", "NE", "E/NE", "E", "E/SE", "SE", "S/SE", "S", "S/SW", "SW", "W/SW", "W", "W/NW", "NW", "N/NW", "N"]
common_dir = direction_labels[int((direction + ddeg) % 360 // 22.5)]

st.info(f"The current temperature is {temp} °C. \nThe wind speed is {speed} m/s. \nThe wind is coming from {common_dir}. \nThe air pressure is {pressure} hPa.")

# Fetching data for the machine learning model from MySQL
query = f"SELECT * FROM your_table WHERE city='Jakarta' AND country='Indonesia'"
data_for_model = fetch_data(query)

# Preprocess the data as required by your model
# Example: Assuming 'data_for_model' is already in the correct format
predictions = model.predict(data_for_model)

# Integrate predictions into the Streamlit app
st.subheader("Machine Learning Weather Forecast")
st.write(f"Predicted Temperature for the next week: {predictions}")

st.subheader("Week ahead")
st.write('Temperature, precipitation, and air pressure forecast one week ahead & city location on the map', unsafe_allow_html=True)

with st.spinner('Loading...'):
    response_hourly = requests.get(f'https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&hourly=temperature_2m,precipitation,pressure')
    result_hourly = json.loads(response_hourly.content)
    hourly = result_hourly["hourly"]
    hourly_df = pd.DataFrame.from_dict(hourly)
    hourly_df.rename(columns={'time':'Week ahead', 'temperature_2m':'Temperature °C', 'precipitation':'Precipitation mm', 'pressure':'Pressure hPa'}, inplace=True)
    
    tz = tzwhere.tzwhere(forceTZ=True)
    timezone_str = tz.tzNameAt(lat, lng, forceTZ=True)
    timezone_loc = pytz.timezone(timezone_str)
    dt = datetime.now()
    tzoffset = timezone_loc.utcoffset(dt)
    
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    week_ahead = pd.to_datetime(hourly_df['Week ahead'], format="%Y-%m-%dT%H:%M")
    
    fig.add_trace(go.Scatter(x=week_ahead+tzoffset, y=hourly_df['Temperature °C'], name="Temperature °C"), secondary_y=False)
    fig.add_trace(go.Bar(x=week_ahead+tzoffset, y=hourly_df['Precipitation mm'], name="Precipitation mm"), secondary_y=True)
    fig.add_trace(go.Scatter(x=week_ahead+tzoffset, y=hourly_df['Pressure hPa'], name="Pressure hPa", line=dict(color='orange', width=2, dash='dash')), secondary_y=False)
    
    time_now = datetime.now(tmz.utc)+tzoffset
    fig.add_vline(x=time_now, line_color="red", opacity=0.4)
    fig.add_annotation(x=time_now, y=max(hourly_df['Temperature °C'])+5, text=time_now.strftime("%d %b %y, %H:%M"), showarrow=False, yshift=0)
    
    fig.update_yaxes(range=[min(hourly_df['Temperature °C'])-10, max(hourly_df['Temperature °C'])+10], title_text="Temperature °C", secondary_y=False, showgrid=False, zeroline=False)
    fig.update_yaxes(range=[min(hourly_df['Precipitation mm'])-2, max(hourly_df['Precipitation mm'])+8], title_text="Precipitation (rain/showers/snow) mm", secondary_y=True, showgrid=False)
    fig.update_yaxes(range=[min(hourly_df['Pressure hPa'])-10, max(hourly_df['Pressure hPa'])+10], title_text="Pressure hPa", secondary_y=False, showgrid=False)
    
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.7))
    
    m = folium.Map(location=[lat, lng], zoom_start=7)
    folium.Marker([lat, lng], popup=city+', '+country, tooltip=city+', '+country).add_to(m)
    
    make_map_responsive = """<style>[title~="st.iframe"] { width: 100%}</style>"""
    st.markdown(make_map_responsive, unsafe_allow_html=True)
    
    st.plotly_chart(fig, use_container_width=True)
    st_data = folium_static(m, height=370)

    st.write('Weather data source: [http://open-meteo.com](http://open-meteo.com) \n\n'+
             'List of 40,000+ world cities: [https://simplemaps.com/data/world-cities](https://simplemaps.com/data/world-cities) \n\n' +
             'Github repository: [streamlit-weather-app](https://github.com/ndakov/streamlit-weather-app)')
    st.write('Thanks for stopping by. Cheers!')
