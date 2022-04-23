import geopandas as gpd
import os
import pandas as pd
import altair as alt
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from vega_datasets import data



ROOT_DIR = os.path.dirname(__file__)
print(ROOT_DIR)

FOOD_ATLAS_NAME = "MI_food_atlas2019.csv"
CENSUS_TRACT_NAME = "cb_2019_us_tract_500k.shx"
ERSAtlas_CensusData_FILE_NAME = "ERSAtlas_CensusData.csv"

COMBINED_DATA_PATH = os.path.join(ROOT_DIR, ERSAtlas_CensusData_FILE_NAME)
FOOD_ATLAS_PATH = os.path.join(ROOT_DIR, FOOD_ATLAS_NAME)
CENSUS_TRACT_PATH = os.path.join(ROOT_DIR, CENSUS_TRACT_NAME)


#reading in data
atlas_census_data = pd.read_csv(COMBINED_DATA_PATH)

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
).mark_point(filled=True, stroke="white", strokeWidth=0.5).encode(
    x=alt.X("MedianIncome:Q", scale=alt.Scale(domain=[35000, 115000]), axis=alt.Axis(title="Median Income", gridOpacity=0.1)),
    y=alt.Y("Walk:Q", axis=alt.Axis(gridOpacity=0.1)),
    size=alt.Size("TotalPop:Q", legend=alt.Legend(title="Total Population", symbolFillColor = "gray")),
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

d = {'Region': ['SouthWest', 'SouthEast', 'MidWest', 'Pacific', 'NonContiguous', 'DC', 'NorthEast', 'RockyMountains'], 'Percentage': [39.12, 36.21, 30.4, 28.39, 20.48, 30.16, 18.49, 29.02]}

regions_df = pd.DataFrame(data = d)

selection = alt.selection_single()
regions_chart = alt.Chart(regions_df).mark_bar().encode(
    # encode x as the percent, and hide the axis
    x = alt.X('Percentage:Q', title = "Percentage of Food Tracks"),
    y=alt.Y('Region'),
    tooltip = [alt.Tooltip('Percentage:Q'),
               alt.Tooltip('Region:N')
              ],
    color=alt.condition(selection, 'Percentage:Q', alt.value('grey'))
).add_selection(selection)



#Downloading Atlas and geodata

census_tracts2019 = gpd.read_file(CENSUS_TRACT_PATH)
census_tracts2019['STATEFP'] = pd.to_numeric(census_tracts2019['STATEFP'])
census_tracts2019['GEOID'] = pd.to_numeric(census_tracts2019['GEOID'])

MI_census_tracts2019 = census_tracts2019[census_tracts2019['STATEFP'] == 26]




def food_desert_label(row):
    '''This function lables a tract as a food desert if it falls under 1 of 4
    measures as determined by the USDA ERS Atlas'''
    if row['LILATracts_1And10'] == 1:
        return 1
    if row['LILATracts_halfAnd10'] == 1:
        return 1
    if row['LILATracts_1And20'] == 1:
        return 1
    if row['LILATracts_Vehicle'] == 1:
        return 1
    else:
        return 0



MI_food_atlas2019 = pd.read_csv(FOOD_ATLAS_PATH)

MI_food_atlas2019['food_desert_label'] = MI_food_atlas2019.apply(lambda row: food_desert_label(row), axis=1)

# Merge atlas and geodataframe
MI_censustract_df_merged_2019 = MI_census_tracts2019.merge(MI_food_atlas2019, left_on='GEOID', right_on='CensusTract', how='inner')
MI_censustract_df_merged_2019 = MI_censustract_df_merged_2019[['geometry', 'CensusTract', 'TractSNAP', 'food_desert_label', 'County', 'TractSeniors', 'PovertyRate']]



# load as a GeoJSON object.
json_features = json.loads(MI_censustract_df_merged_2019.to_json())
data_geo = alt.Data(values=json_features['features'])

points = []
for tract in json_features['features']:
    first_point = tract['geometry']['coordinates'][0][0]
    points.append(first_point)
points_df = pd.DataFrame(np.vstack(points))

chart_points = alt.Chart(points_df).mark_point(opacity = 0).encode(
    longitude='0:Q',
    latitude='1:Q'
    )


##Creating ALtair visualizations

SNAP_vis = alt.Chart(data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.TractSNAP:Q', title = 'Number on SNAP' ))

interactive_state_snap= SNAP_vis.encode(
    tooltip=[alt.Tooltip('properties.TractSNAP:N', title='Number on SNAP'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')]

)

label_vis = alt.Chart(data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.food_desert_label:N', title = 'Food Desert Label' ))

interactive_label_state = label_vis.encode(
    tooltip=[alt.Tooltip('properties.food_desert_label:N', title='Food Desert Label'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')]

)

Seniors_vis = alt.Chart(data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.TractSeniors:Q', title = 'Number of Seniors' )).encode(
    tooltip=[alt.Tooltip('properties.TractSeniors:Q', title='Number of Seniors'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')])

poverty_vis = alt.Chart(data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.PovertyRate:Q', title = 'Poverty Rate' )).encode(
    tooltip=[alt.Tooltip('properties.PovertyRate:Q', title='Poverty Rate'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')])

Wayne_County = MI_censustract_df_merged_2019[MI_censustract_df_merged_2019['County'] == 'Wayne County']

wayne_json_gdf = Wayne_County.to_json()
wayne_json_features = json.loads(wayne_json_gdf)

wayne_points = []
for tract_wayne in wayne_json_features['features']:
    first_point_wayne = tract_wayne['geometry']['coordinates'][0][0]
    wayne_points.append(first_point_wayne)
wayne_points_df = pd.DataFrame(np.vstack(wayne_points))

wayne_chart_points = alt.Chart(wayne_points_df).mark_point(opacity = 0).encode(
    longitude='0:Q',
    latitude='1:Q'
    )

# load as a GeoJSON object.

wayne_data_geo = alt.Data(values=wayne_json_features['features'])

wayne_label_vis = alt.Chart(wayne_data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.food_desert_label:N', title = 'Food Desert Label' ))

interactive_wayne_label_vis = wayne_label_vis.encode(
    tooltip=[alt.Tooltip('properties.food_desert_label:N', title='Food Desert Label'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')]

)

wayne_SNAP_vis = alt.Chart(wayne_data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.TractSNAP:Q', title = 'Number on SNAP' ))

interactive_wayne_snap_vis = wayne_SNAP_vis.encode(
    tooltip=[alt.Tooltip('properties.TractSNAP:N', title='Number on SNAP'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')]

)

wayne_seniors_vis = alt.Chart(wayne_data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.TractSeniors:Q', title = 'Number of Seniors' ))

interactive_wayne_seniors_vis = wayne_seniors_vis.encode(
    tooltip=[alt.Tooltip('properties.TractSeniors:N', title='Number of Seniors'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')]

)

poverty_vis_wayne = alt.Chart(wayne_data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.PovertyRate:Q', title = 'Poverty Rate' )).encode(
    tooltip=[alt.Tooltip('properties.PovertyRate:Q', title='Poverty Rate'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')])


Washtenaw_County = MI_censustract_df_merged_2019[MI_censustract_df_merged_2019['County'] == 'Washtenaw County']


washtenaw_json_gdf = Washtenaw_County.to_json()
washtenaw_json_features = json.loads(washtenaw_json_gdf)
washtenaw_points = []
for tract in washtenaw_json_features['features']:
    first_point_washtenaw = tract['geometry']['coordinates'][0][0]
    washtenaw_points.append(first_point_washtenaw)
washtenaw_points_df = pd.DataFrame(np.vstack(washtenaw_points))

washtenaw_chart_points = alt.Chart(washtenaw_points_df).mark_point(opacity = 0).encode(
    longitude='0:Q',
    latitude='1:Q'
    )
# load as a GeoJSON object.
washtenaw_json_features = json.loads(washtenaw_json_gdf)

washtenaw_data_geo = alt.Data(values=washtenaw_json_features['features'])

washtenawlabel_vis = alt.Chart(washtenaw_data_geo).mark_geoshape(

    stroke='white'
).properties(


).encode(
    color= alt.Color('properties.food_desert_label:N', title = 'Food Desert Label' ))

interactive_washtenawlabel_vis = washtenawlabel_vis.encode(
    tooltip=[alt.Tooltip('properties.food_desert_label:N', title='Food Desert Label'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')]

)

washtenaw_SNAP_vis = alt.Chart(washtenaw_data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.TractSNAP:Q', title = 'Number on SNAP' ))

interactive_washtenaw_SNAP_vis = washtenaw_SNAP_vis.encode(
    tooltip=[alt.Tooltip('properties.TractSNAP:N', title='Number of Seniors'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')])

washtenaw_seniors_vis = alt.Chart(washtenaw_data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.TractSeniors:Q', title = 'Number of Seniors' ))

interactive_washtenaw_seniors_vis = washtenaw_seniors_vis.encode(
    tooltip=[alt.Tooltip('properties.TractSeniors:N', title='Number of Seniors'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')])

poverty_vis_washtenaw = alt.Chart(washtenaw_data_geo).mark_geoshape(

    stroke='white'
).properties(

    width=500,
    height=500
).encode(
    color= alt.Color('properties.PovertyRate:Q', title = 'Poverty Rate' )).encode(
    tooltip=[alt.Tooltip('properties.PovertyRate:Q', title='Poverty Rate'),
            alt.Tooltip('properties.CensusTract:N', title='Census Tract Number')])




state_snap = interactive_state_snap + chart_points
state_label = interactive_label_state + chart_points
state_seniors = Seniors_vis + chart_points
state_poverty = poverty_vis + chart_points

wayne_snap = interactive_wayne_snap_vis + wayne_chart_points
wayne_label = interactive_wayne_label_vis + wayne_chart_points
wayne_seniors = interactive_wayne_seniors_vis + wayne_chart_points
wayne_poverty = poverty_vis_wayne + wayne_chart_points

washtenaw_snap = interactive_washtenaw_SNAP_vis + washtenaw_chart_points
washtenaw_label = interactive_washtenawlabel_vis + washtenaw_chart_points
washtenaw_seniors = interactive_washtenaw_seniors_vis + washtenaw_chart_points
washtenaw_poverty = poverty_vis_washtenaw + washtenaw_chart_points

#Streamlit Code

selectbox1 = st.sidebar.selectbox(label='Select Topic', options=['Home', 'Seniors', 'SNAP Benefits', 'Poverty', '' 'Conclusion'])




if selectbox1 == 'Home':

    st.title('Visualizing Food Deserts')
    st. write('by Roma Patel and Katie Henning')

    st.write("""The USDA Economic Research Service has produced three iterations of a Food Access Research Atlas which identifies food desserts at the census tract level.
    This Atlas illustrates which tracts are food deserts based on certain conditions. Demographic and economic data collected at the census tract
    level was used in a predictive analysis project to determine which features were the most likely to contribute to the label of a census tract as a food desert.
    This project improves upon the USDA ERS Food Access Research Atlas by visualizing the most important features to provide more context to the public health issue of
    inequality in food access.""")

    st.write("""This tool was designed to allow users to explore connections between food deserts and multiple demographic features. Below we focus on trends
    of food deserts for all areas within the UniteD States and on later pages refine information to the state of Michigan and specifically
    the counties of Wayne and Washtenaw.""")

    st.altair_chart(regions_chart)
    st.caption('*Percentage of census tracts that qualify for food desert status by region*')

    st.header("Importance of Understanding Food Deserts")
    """
    Food deserts have been defined by the CDC as areas that “lack access to affordable fruits, vegetables, whole grains,
    low-fat milk, and other foods that make up the full range of a healthy diet[1].” They have many underlying implications
    for the community varying from health conditions like diabetes in adults to children developing these conditions and lack
    of proper development. Several factors contribute to food deserts such as income/poverty rate of an area and accessibility
    to supermarkets. And given the socioeconomic factors and other disparities in society that contribute to the income distribution
    in the U.S. races are also disproportionately affected. According to the USDA “the percent of the population that is non-Hispanic Black
    is over twice as large in urban food deserts than in other urban areas[2].”

    """
    #visuals explanation and how to use
    st.header("Overall Distributions of Food Deserts")
    """
    The visualization below consists of a large scatter plot that shows the accessibility to supermarkets (via average walking distance) as a factor of
    median income by state. The states are split by the areas that have food deserts (orange colored points) and areas that don't (blue colored points).
    There is also a stacked racial distribution to examine the possible disparities at the state level.

    Tutorial: The visualizations are connected via States, so clicking on a point on the scatter plot that is an orange point (has a
    there is a food desert) the related blue point for the same state will also be selected. The rest of the points will become less visible.
    The race bar chart will also then filter via population make up for the selected state. Multiple points can be selected by holding
    on to the 'shift' key and click the points."""

    #visuals
    st.altair_chart(final_combined_visuals, use_container_width=True)

    st.write("""*For drilled down views of Michigan, by census tract,
    select from the drop down to the left. There are several additional factors explored such as the distribution of
    SNAP Benefits compared with the areas of Michigan labeled as a Food Desert.*""")




elif selectbox1 == 'Seniors':
    st.title('Food Deserts and Seniors')
    st.write("""This page shows the comparison between food desert label and the number
    of seniors per census tract. The results of feature importance analysis showing that the number of seniors that live
within a given tract have a large role in determining the tract’s status as a food desert stands to
reason since the elderly are more susceptible to limited transportation if they are no longer able
to drive, live on fixed incomes such as a pension, and are unlikely to leave an area even if it is
facing food insecurity due to long-term connections within their local city or neighborhoods [10].
The elderly population is also less likely to be enrolled in receiving SNAP benefits as only
35.1 percent of eligible elderly adults are enrolled compared to 75.6 percent of SNAP-eligible non-elederly
adults. Of all of these factors, access to transportation is the most inhibiting for the elderly
to achieve food security as “elderly food desert residents that do not own a vehicle were 12
percentage points more likely to report food insufficiency than otherwise similar food desert
residents who owned a vehicle”. This suggests that a key goal to strive for in decreasing the
number of census tracts classified as food deserts with a high percentage of elderly residents is to
improve the quality of transportation so this demographic has the opportunity to choose from
more diverse food retailers.


Use the following interactive maps to compare food desert labels and the number of seniors by census tract.""")
    st.altair_chart(state_label | state_seniors )
    st.altair_chart(wayne_label | wayne_seniors )
    st.altair_chart(washtenaw_label | washtenaw_seniors)

elif selectbox1 == 'SNAP Benefits':
    st.title('Food Deserts and SNAP Benefits')
    st.write("""SNAP benefits being a large indicator of food desert status has contextual support
because though the purpose of SNAP benefits is to increase access to food that is nutritionally
dense, the ability to use SNAP benefits heavily depends on the existence of food retailers nearby
where these benefits can be redeemed. Even when SNAP benefits are increased, other factors
such as transportation serve as an impediment to residents of food deserts being able to benefit
from this program.""")

    st.write("""Use the following interactive maps to compare food desert labels and the number of people enrolled in SNAP benefits by census tract. """)
    st.altair_chart(state_label | state_snap )
    st.altair_chart(wayne_label | wayne_snap )
    st.altair_chart(washtenaw_label | washtenaw_snap)


elif selectbox1 == 'Poverty':
    st.title('Food Deserts and Poverty')

    st.write("""Poverty plays a role in food desert status designation in both measures of access the term 'food desert'
    refers to:increase in poverty rate means citizens have less money to both spend on healthy, nutritious food as well
    as the transportation needed to access the locations where food is purchased. This is consitent across census tracts
    regardless of whether they are classified as urban or rural. Impoverished areas are also less likely to have access to healthcare and fitness
    education which would help to educate citizens on the prioritization of healthy food alternatives. Changes in poverty rate overtime are also
    not likely to be statistically significant as a tract with a high poverty rate at any given time is much more likely to be classified as a food desert
    dmonstrating that the effects of poverty are long lasting and difficult to reverse (USDA ERS).""")

    st.write("""Use the following interactive maps to compare food desert labels and Poverty Rate by census tract.""")

    st.altair_chart(state_label | state_poverty )
    st.altair_chart(wayne_label | wayne_poverty )
    st.altair_chart(washtenaw_label | washtenaw_poverty)

elif selectbox1 == 'Conclusion':
    st.title('Conclusion')

    st.write(""" Thank you for your interest in our project. We hope these visualizations have helped shape the
    way you think about food deserts and the factors which contribute to their existence.""")

    st.header('References')
    st.write('USDA ERS Food Access Atlas: https://www.ers.usda.gov/data-products/food-access-research-atlas/go-to-the-atlas/')
    st.write('Characteristics and Influential Factors of Food Deserts: https://www.ers.usda.gov/webdocs/publications/45014/30940_err140.pdf')
    st.write('US Census Demographic Data: https://www.kaggle.com/muonneutrino/us-census-demographic-data.')
    st.write('Chronic Disease [article from site]: https://chronicdisease.org/')
    st.write('Health and Socioeconomic Disparities of Food Deserts: https://sites.duke.edu/lit290s-1_02_s2017/2017/03/04/health-and-socioeconomic-disparities-of-food-deserts/')









#st.altair_chart(state_seniors | state_snap | state_label )



# st.write('paragraph about charts')

# st.altair_chart(wayne_seniors | wayne_snap | wayne_label)
# st.altair_chart(washtenaw_snap | washtenaw_label)


