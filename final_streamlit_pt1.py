
#importing packages
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from vega_datasets import data

#reading in data
atlas_census_data = pd.read_csv("ERSAtlas_CensusData.csv")

############## MANIPULATING DATA ##############
#getting state level information into df
state_level = atlas_census_data.groupby(["State", "region", "food_desert_label"]).aggregate({"food_desert_label":"sum", "MedianIncome":"median", "Walk": "mean", "TotalPop": "sum", "ChildPoverty": "mean", "Service": "mean", "Construction":"mean", "Hispanic":"sum", "Asian":"sum", "White":"sum", "Black":"sum", "Native":"sum", "Pacific":"sum"})
state_level = state_level.rename(columns={"food_desert_label": "FoodDesert_Totals"})
state_level = state_level.reset_index()
state_level = state_level.rename(columns={"region": "Region"})

#getting vega dataset just for map element
state_pop = data.population_engineers_hurricanes()[['state', 'id', 'population']]
state_map = alt.topo_feature(data.us_10m.url, 'states')
state_pop = state_pop.rename(columns={'state':"State"})

#final state level data
final_state_level  = state_pop[["State", "id"]].merge(state_level, how="inner", on="State")

############## PAGE FEATURES ##############
#adding page features
st.markdown('<style>body{background-color: Black;color: White}</style>',unsafe_allow_html=True)

############## CREATING VISUALIZATIONS ##############
#adding click feature
click = alt.selection_multi(fields=['State'])

#combined scatter plot
scatter_plot = alt.Chart(final_state_level
).mark_point(filled=True).encode(
    x=alt.X("MedianIncome:Q", scale=alt.Scale(domain=[35000, 115000]), axis=alt.Axis(title="Median Income", gridOpacity=0.1)),
    y=alt.Y("Walk:Q", axis=alt.Axis(gridOpacity=0.1)),
    size=alt.Size("TotalPop:Q", legend=alt.Legend(title="Total Population")),
    color=alt.Color("food_desert_label:N", legend=alt.Legend(title="Food Desert Label")),
    tooltip = ["State:N", "MedianIncome:Q", "FoodDesert_Totals:Q", "Region:N"],
    opacity=alt.condition(click, alt.value(1), alt.value(0.2))
).properties(
    width=800
).add_selection(click)

#no food desert regression line
no_regline = alt.Chart(final_state_level).transform_filter(
    alt.datum.food_desert_label == 0
).transform_regression(
    "MedianIncome", "Walk"
).mark_line(opacity=0.3).encode(
    x=alt.X("MedianIncome:Q"),
    y=alt.Y("Walk:Q")
)

#yes food desert regression line
yes_regline = alt.Chart(final_state_level).transform_filter(
    alt.datum.food_desert_label == 1
).transform_regression(
    "MedianIncome", "Walk"
).mark_line(opacity=0.3, color="orange").encode(
    x=alt.X("MedianIncome:Q"),
    y=alt.Y("Walk:Q")
)

#combining reglines
reglines = no_regline+yes_regline

#combining scatter plot and reglines
final_plot = scatter_plot+reglines

#creating bar chart
mini_bar = alt.Chart(final_state_level).transform_fold(
    ["Hispanic", "White", "Black", "Native", "Asian", "Pacific"],
    as_=["Race", "values"]
).mark_bar().encode(
    y = alt.Y("Race:N"),
    x=alt.X("values:Q", axis=alt.Axis(title="Count of Population", tickCount=5)),
    color=alt.Color("food_desert_label:N")
).properties(
    height = 175
).transform_filter(click)

#creating map
mini_map = (alt.Chart(state_map).mark_geoshape().transform_lookup(
    lookup = "id",
    from_=alt.LookupData(final_state_level, "id", ["State", "Region", "TotalPop", "ChildPoverty", "FoodDesert_Totals"])
).encode(
    color=alt.Color("FoodDesert_Totals:Q", legend=alt.Legend(title="Food Desert Totals")),
    opacity = alt.condition(click, alt.value(1), alt.value(0.1)),
    tooltip = alt.Tooltip(["State:N", "Region:N", "TotalPop:Q"])
).add_selection(click
).project(type='albersUsa')).properties(
    width = 250,
    height=250
)

#combining map and bar
bar_map = mini_bar| mini_map

#combining bar and scatter
combined_visuals = alt.vconcat(bar_map, final_plot)

#adding final configurations
final_combined_visuals = combined_visuals.configure(background='Black'
).configure_axisLeft(
    labelColor='white',
    titleColor='white'
).configure_axisRight(
    labelColor='white',
    titleColor='white'
).configure_axisBottom(
    labelColor='white',
    titleColor='white'
).configure_axisTop(
    labelColor='black',
    titleColor = 'white'
).configure_legend(
    labelColor='white',
    titleColor='white'
)

############## PAGE SET UP ##############
#page title

#intro

#why food deserts matter

#visuals explanation and how to use
st.header("Overall Distributions of Food Deserts")
"""
Below is x that shows y and Overall Purpose. What each component is, why, and how to interact.
"""

#visuals
st.altair_chart(final_combined_visuals)