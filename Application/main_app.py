#!/usr/bin/env python
# coding: utf-8

# In[23]:


from contextlib import suppress
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

import folium
import geopy
import ast
import warnings
from streamlit_folium import folium_static
from streamlit import caching
import pickle 
warnings.filterwarnings('ignore')


# In[2]:


import boto
import boto.s3.connection
from io import StringIO
import boto3
import pandas as pd
from datetime import datetime
from datetime import date
import time
import sys
import json
from opencage.geocoder import OpenCageGeocode
from math import radians


def cache_clear_dt(dummy):
	clear_dt = date.today()
	return clear_dt

if cache_clear_dt("dummy")<date.today():
	caching.clear_cache()


athena_client = boto3.client(
    "athena", 
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
    aws_secret_access_key=st.secrets["AWS_SECRET_KEY"],
    region_name=st.secrets["REGION_NAME"]
)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
    aws_secret_access_key=st.secrets["AWS_SECRET_KEY"],
    region_name=st.secrets["REGION_NAME"]
)



# In[5]:

st.set_page_config(layout="wide")


# In[6]:


row1_1, row1_4 = st.columns(2)

with row1_1:

    st.title("SafePath Los Angeles")
    
row1_2, row1_3 = st.columns((1, 0.75))

with row1_2:
    st.write(
        """
    ##
    Examining risk level of street roads in Los Angeles. 
    """
    )
    hour_selected = st.slider("Select hour of day", 0, 23, 4)
    day_selected = st.selectbox("Pick a day of the week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
                                                           'Saturday', 'Sunday'])
    address_selected = st.text_input("Enter street address here", 'FIGUEROA ST')


# In[61]:
@st.cache(suppress_st_warning=True)
def open_model():
    obj = s3_client.get_object(Bucket='sofians3', Key='k_means/kmeans.pkl')
    body = obj['Body'].read()
    return body

kmeans_model = open_model()

class project_data:
    
    def __init__(self, address_selected, day_selected, hour_selected):
        self.address = str(address_selected).upper()
        self.day = day_selected
        self.hour = hour_selected
    
    def get_risk(self, x):
        if x > 0 and x <= 10:
            return 'low'
        elif x > 11 and x <= 20:
            return 'medium'
        else:
            return 'high'

    # Determining color for marker based on risk
    def get_risk_color(self, x):
        dict_risk = {
            'low': "green",
            'medium': "orange",
            'high': "red"
        }
        return dict_risk[x]
    
    # Determining size of marker based on risk
    def get_risk_size(self, x):
        dict_risk = {
            'low': 10,
            'medium': 20,
            'high': 30
        }
        return dict_risk[x]

    # Needed for pop-up
    def get_user_location_data(self, dict_traffic, dict_accident, dict_risk, predicted_label):

        return self.address, dict_traffic[predicted_label], dict_accident[predicted_label], self.get_risk(dict_risk[predicted_label])

    #Geocoding for finding the cluster 
    def coordinates(self):
        if ', Los Angeles' not in self.address:
            full_address = self.address + ', Los Angeles, California, United States'
        key = st.secrets["COORDINATE_KEY"]
        geocoder = OpenCageGeocode(key)
        result = geocoder.geocode(full_address, no_annotations="1")
        if result and len(result):
            longitude = result[0]["geometry"]["lng"]  
            latitude = result[0]["geometry"]["lat"]
        
        else:
            return 'No location found'
        return [latitude, longitude]

    
    # Finding the closest cluster and its relevant details
    def find_cluster(self, coordinates):
        # Get the cluster
        prediction = kmeans_model.predict(np.array([coordinates[0], coordinates[1]]).reshape(1, -1))[0]
        return prediction


    # Use this for getting data for plotting graph.
    def get_traffic_accident(self, data):
        df = data[['accident_address','day_week', 'hour', 'mean_trafficvolume', 'mean_accidentvolume']][data['accident_address'].str.startswith(self.address)].fillna(0)
        df = df.groupby(by=['day_week', 'hour']).agg({'mean_trafficvolume': 'sum', 'mean_accidentvolume': 'sum'}).reset_index(level=1)
        traffic = df.loc[self.day][['hour','mean_trafficvolume']]
        accident = df.loc[self.day][['hour','mean_accidentvolume']]

        return traffic.set_index('hour'), accident.set_index('hour')

    # Use this for plotting the map
    def plot_map_data(self, data):
        df = data[['accident_address', 'accident_latitude', 'accident_longitude', 'risk_level', 'mean_trafficvolume', 'mean_accidentvolume', "color", "size"]][(data['day_week'] == self.day)&(data['hour'] == self.hour)].fillna(0)

        return df
    

# In[76]:

# Function to obtain data from S3 
def query_results_S3(data_name):

    s3_client.download_file(
        st.secrets["S3_BUCKET_NAME"],
        f'{st.secrets["OUTPUT_DIRECTORY"]}/{st.secrets[data_name]}.csv', data_name
        )

    return pd.read_csv(data_name)

 # Initializing an object of instance project data
funcs = project_data(address_selected, day_selected, hour_selected)

# Function to obtain all data 
@st.cache(suppress_st_warning=True)
def get_all_data():
    dict_traffic = query_results_S3("TRAFFIC").set_index('predicted_labels').to_dict()['mean_traffic']
    dict_accident = query_results_S3("ACCIDENT").set_index('predicted_labels').to_dict()['mean_accident']
    dict_risk = query_results_S3("CRASHRATE").set_index('predicted_labels').to_dict()['mean_cr']
    all_data = query_results_S3("ALL_DATA")
    all_data["color"] = all_data["risk_level"].apply(lambda x: funcs.get_risk_color(x))
    all_data["size"] = all_data["risk_level"].apply(lambda x: funcs.get_risk_size(x))
    return dict_traffic, dict_accident, dict_risk, all_data

# Get all data
dict_traffic, dict_accident, dict_risk, all_data = get_all_data()

# Get user coordinates
coordinates = funcs.coordinates()
# Get the predicted label for the location
predicted_label = funcs.find_cluster(coordinates)
# Get user location details
address, traffic_volume, accident_volume, risk_level = funcs.get_user_location_data(dict_traffic, dict_accident, dict_risk, predicted_label)
# Get risk factor details
color, size = funcs.get_risk_color(risk_level), funcs.get_risk_size(risk_level)

# Get data for chart plotting
traffic_by_hour, accident_by_hour = funcs.get_traffic_accident(all_data)
# Get data for plotting map based on particular day and hour
map_df = funcs.plot_map_data(all_data)

popup_map = ["accident_address","mean_trafficvolume", "mean_accidentvolume"]


# In[68]:

# This block is for plotting just the coordinate 
map_ = folium.Map(location=[coordinates[0], coordinates[1]], 
                      tiles='cartodbpositron', zoom_start=20)
folium.Marker(location=[coordinates[0], coordinates[1]],popup={
    "address": address, 
    "traffic count": int(traffic_volume), 
    "accident count": int(accident_volume)},tooltip='Click here to see Popup').add_to(map_)
folium.CircleMarker(location=[coordinates[0], coordinates[1]],color=color, fill=True,radius=size).add_to(map_)


# In[77]:


map_df.apply(lambda row: folium.CircleMarker(
           location=[row["accident_latitude"], row["accident_longitude"]], popup=row[popup_map],
           color=row["color"], fill=True,
           radius=row["size"]).add_to(map_), axis=1)


# In[82]:
with row1_3:

    st.subheader(
        f"""**Risk Level for {str(address_selected).capitalize()} at {hour_selected}:00 on {day_selected}**"""
    )
    folium_static(map_)

row2_1, row2_2 = st.columns((1, 0.75))
row3_1, row3_2 = st.columns((1,0.75))

with row2_1:

    st.subheader(
        f"""**Traffic Volume on {day_selected} for {str(address_selected).capitalize()}**"""
    )
with row2_2:

    st.subheader(
        f"""**Accident Count on {day_selected} for {str(address_selected).capitalize()}**"""
    )



with row3_1:

    st.bar_chart(traffic_by_hour, width=500, height=500, use_container_width=True)
    


# In[81]:


with row3_2:
    st.bar_chart(accident_by_hour, width=500, height=500, use_container_width=True)
    