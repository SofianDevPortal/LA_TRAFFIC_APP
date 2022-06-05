#!/usr/bin/env python
# coding: utf-8

# In[23]:


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
from sklearn import preprocessing 
import pickle 
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')


# In[2]:


import boto
import boto.s3.connection
from io import StringIO
import boto3
import pandas as pd
import sys
from opencage.geocoder import OpenCageGeocode
from sklearn.metrics.pairwise import haversine_distances
from math import radians

with open(r'api_keys.txt', 'r') as f:
    data = f.readlines()
    AWS_ACCESS_KEY = str(data[0]).strip()
    AWS_SECRET_KEY = str(data[1]).strip()

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
    user_input = st.text_input("Enter street address here", 'FIGUEROA ST')


    client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)
    csv_obj_traffic = client.get_object(Bucket='sofians3', Key='all_data/data_final.csv')
    csv_obj_crash = client.get_object(Bucket='sofians3', Key='k_means/kmeans_k=25.csv')
    # body_traffic = csv_obj_traffic['Body']
    # body_crash = csv_obj_crash['Body']
    # csv_string_traffic = body_traffic.read().decode('utf-8')
    # csv_string_crash = body_crash.read().decode('utf-8')
    # return csv_string_traffic, csv_string_crash
# In[61]:


class project_data:
    
    def __init__(self, user_input, day_selected, hour_selected):
        self.address = str(user_input).capitalize()
        self.day = day_selected
        self.hour = hour_selected

    def get_data_cloud(self):

        body_traffic = csv_obj_traffic['Body']
        body_crash = csv_obj_crash['Body']
        csv_string_traffic = body_traffic.read().decode('utf-8')
        csv_string_crash = body_crash.read().decode('utf-8')
        return csv_string_traffic, csv_string_crash

        
    def get_dataframe(self): 

        csv_file_traffic, csv_file_crash = self.get_data_cloud()
        df_traffic = pd.read_csv(StringIO(csv_file_traffic))
        df_crash = pd.read_csv(StringIO(csv_file_crash))

        return df_traffic, df_crash

    def get_risk(self, x):
        if x > 0 and x <= 10:
            return 'low'
        elif x > 11 and x <= 20:
            return 'medium'
        else:
            return 'high'
            
    def get_data(self):
        _, crash = self.get_dataframe()
        predicted_label = self.find_cluster()

        # Get mean traffic data by label
        dict_traffic = crash.groupby('label')['traffic_volume'].mean().to_dict()
        # Get mean accident data by label
        dict_accident = crash.groupby('label')['accident_count'].mean().to_dict()
        # Get risk level by label
        dict_risk = crash.groupby('label')['crash_rate'].mean().to_dict()
        risk = self.get_risk(dict_risk[predicted_label])
        return dict_traffic[predicted_label], dict_accident[predicted_label], risk

    # Get API
    # def get_api(self):
    #     geocode_api = ''
    #     with open(r'../data/geocode_api_keys.txt', 'r') as f:
    #         geocode_api += str(f.read())
        
    #     return geocode_api

    # Geocoding...
    def coordinates(self):
        if ', Los Angeles' not in self.address:
            full_address = self.address + ', Los Angeles'
        key = "4dda65f25f7049319ff93133a38c47c4"
        geocoder = OpenCageGeocode(key)
        result = geocoder.geocode(full_address, no_annotations="1")
        if result and len(result):
            longitude = result[0]["geometry"]["lng"]  
            latitude = result[0]["geometry"]["lat"]
        
        else:
            return 'No location found'
        return [latitude, longitude]
    
    def open_model(self):
        return pickle.load(open("kmeans.pkl", "rb"))
    

    # Finding the closest cluster and its relevant details
    def find_cluster(self):
        model = self.open_model()
        coordinates = self.coordinates()

        # Get the cluster
        prediction = model.predict(np.array([coordinates[0], coordinates[1]]).reshape(1, -1))[0]
        return prediction

    # Use this for getting data for plotting graph.
    def get_traffic_accident(self):
        data_,_ = self.get_dataframe()
        traffic_volume = pd.DataFrame(data_.groupby(['day_week', 'hour'])['mean_trafficVolume'].mean().unstack(level=-1).fillna(0))
        accident_volume = pd.DataFrame(data_.groupby(['day_week', 'hour'])['mean_accidentVolume'].mean().unstack(level=-1).fillna(0))
        return traffic_volume.loc[(self.day, self.hour)], accident_volume.loc[(self.day, self.hour)]
    
    # Use this for plotting stuff for all points on the maps
    # Do standard scaling for the data points, maybe save the weights for later
    def plot_maps(self):
        data_,_ = self.get_dataframe()
        scaler = StandardScaler()
        latitude = list(data_['accident_latitude'])
        longitude = list(data_['accident_longitude'])
        address = list(data_['accident_address'])
        day_week = list(data_['day_week'])
        traffic_vol = list(data_['mean_trafficVolume'])
        acc_count = list(data_['mean_accidentVolume'])
        hour = list(data_['hour'])
        crash_rate = scaler.fit_transform(np.array(data_['mean_crashRate']).reshape(-1,1))
        risk = data_['risk_level'].values.tolist()
        return [address, latitude, longitude, crash_rate, risk, day_week, traffic_vol, acc_count, hour]


# In[76]:


funcs = project_data(user_input, day_selected, hour_selected)
prediction_cluster = funcs.find_cluster()
location = funcs.coordinates()
color = "risk_level"
size = "acc_count"
popup = ["address","traffic_vol", "acc_count"]


# In[64]:


ans = funcs.plot_maps()
map_df = pd.DataFrame(ans).transpose()
map_df.columns = ['address', 'latitude', 'longitude', 'crash_rate', 'risk_level', 'day_week', 'traffic_vol', 'acc_count',  'hour']

# In[65]:

address = user_input + ', Los Angeles'
location_lat = list(map_df.loc[(map_df['address'] == address) & (map_df['day_week']==day_selected) & (map_df['hour']==hour_selected), 'latitude'])[0]
location_long = list(map_df.loc[(map_df['address'] == address) & (map_df['day_week']==day_selected) & (map_df['hour']==hour_selected), 'longitude'])[0]




copy = map_df.copy()


# In[66]:


lst_colors=["red","green","orange"]
lst_elements = sorted(list(map_df[color].unique()))
copy["color"] = copy[color].apply(lambda x: 
                lst_colors[lst_elements.index(x)])


# In[68]:


map_ = folium.Map(location=[location_lat, location_long], 
                      tiles='cartodbpositron', zoom_start=20)
folium.Marker(location=[location_lat, location_long],popup=["address","traffic_vol", "acc_count"],tooltip='Click here to see Popup').add_to(map_)


# In[71]:


required_df = copy[(copy['day_week']==day_selected) & (copy['hour']==hour_selected)]

size = []

for risk in required_df["risk_level"]:

    if risk == "low":
        size.append(10)
    elif risk == "medium":
        size.append(20)
    elif risk == "high":
        size.append(30)

required_df["size"] = size


# In[77]:


required_df.apply(lambda row: folium.CircleMarker(
           location=[row["latitude"], row["longitude"]], popup=row[popup],
           color=row["color"], fill=True,
           radius=row["size"]).add_to(map_), axis=1)


# In[10]:

# In[82]:
with row1_3:

    st.subheader(
        f"""**Risk Level for {user_input} at {hour_selected}:00 on {day_selected}**"""
    )
    folium_static(map_)

row2_1, row2_2 = st.columns((1, 0.75))
row3_1, row3_2 = st.columns((1,0.75))

with row2_1:

    st.subheader(
        f"""**Traffic Volume on {day_selected} for {user_input}**"""
    )
with row2_2:

    st.subheader(
        f"""**Accident Count on {day_selected} for {user_input}**"""
    )



with row3_1:

    chart_data = copy[(copy['address']==address)]
    chart_data= chart_data[(chart_data['day_week']==day_selected)][['hour', 'traffic_vol']]
    chart_data = pd.DataFrame(chart_data.groupby(by=['hour'])['traffic_vol'].sum(numeric_only=False))
    chart_data.reset_index(inplace=True)

    st.bar_chart(chart_data, width=500, height=500, use_container_width=True)
    


# In[81]:


with row3_2:
    chart_data_acc = copy[(copy['address']==address)]
    chart_data_acc = chart_data_acc[(chart_data_acc['day_week']==day_selected)][['hour', 'acc_count']]
    chart_data_acc = pd.DataFrame(chart_data_acc.groupby(by=['hour'])['acc_count'].sum(numeric_only=False))
    chart_data_acc.reset_index(inplace=True)
    
    st.bar_chart(chart_data_acc, width=500, height=500, use_container_width=True)