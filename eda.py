"""
Jason Choi
CSE 163 AD
This file manipulates and visualizes the "Indoor Plant Health & Growth Dataset"
by Souvikrana176 on Kaggle, the "Annual percent of possible sunshine by US
City" dataset by thedevastator on Kaggle, and the PRISM's Weather Data by State
dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm


def find_missing_data(data: pd.DataFrame, data_name: str) -> int:
    """
    This function checks if a given DataFrame has any missing values. If it
    does, print the amount of missing values along with where they are missing.
    It returns the count of missing values.
    """
    nan_values = data[data.isna().any(axis=1)]
    if len(nan_values) > 0:
        print("In dataset: ", data_name, "There are ", len(nan_values),
              " missing values.")
        print(nan_values)
        return len(nan_values)
    print("No missing values!")
    return None


def find_unique_values(data: pd.DataFrame, data_name: str,
                       column: str) -> int:
    """
    This function prints the unique values that aren't null in a given column
    in a given dataframe. It returns the count for unique values
    """
    if (column not in data.columns):
        print("Column not in data!")
        return None
    unique_values = data[column].dropna().unique()
    print("In dataset: ", data_name, "in column",
          column, " there are ", len(unique_values), "unique values")
    print('The unique values are: ', unique_values)
    return len(unique_values)


def clean_plant_data(indoor_plant_df) -> pd.DataFrame:
    """
    This function returns a cleaned version of the indoor plant dataset to only
    include the necessary data and numerizes the sunlight descriptions from
    1 to 5, keeping ony
    """
    # Numerize sunlight descriptions into value 1 - 5
    # 3h direct morning sun' 'Indirect light all day' 'Low light corner'
    # 'Filtered sunlight through curtain' '6h full sun
    sunlight_scores = {
        'Low light corner': 1,
        'Filtered sunlight through curtain': 2,
        'Indirect light all day': 3,
        '3h direct morning sun': 4,
        '6h full sun': 5
    }
    indoor_plant_cleaned_df = indoor_plant_df.copy()
    indoor_plant_cleaned_df['Sunlight_Score'] = \
        indoor_plant_cleaned_df['Sunlight_Exposure'].map(sunlight_scores)
    # Drop unnecessary columns
    unnecessary_columns = ['Height_cm', 'Leaf_Count', 'Sunlight_Exposure',
                           'New_Growth_Count', 'Health_Notes',
                           'Watering_Amount_ml', 'Watering_Frequency_days',
                           'Fertilizer_Type', 'Fertilizer_Amount_ml',
                           'Pest_Presence', 'Pest_Severity',
                           'Soil_Moisture_%', 'Soil_Type'
                           ]
    indoor_plant_cleaned_df = indoor_plant_cleaned_df.drop(
                                columns=unnecessary_columns)
    return indoor_plant_cleaned_df


def clean_temp_data(temp_df) -> pd.DataFrame:
    """
    This function returns a cleaner version of the state weather dataset to by
    numerizing the months from 1 to 12 and drops unnecessary columns.
    """
    temp_cleaned = temp_df.copy()
    value_to_month_map = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY',
                          6: 'JUN', 7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT',
                          11: 'NOV', 12: 'DEC'
                          }
    temp_cleaned['ym'] = temp_cleaned['ym'] % 100
    temp_cleaned = temp_cleaned.rename(columns={"st_abb": "state"})
    temp_cleaned['month'] = temp_cleaned['ym'].map(value_to_month_map)
    temp_cleaned = temp_cleaned.drop(columns=['ym', 'st_code'])
    return temp_cleaned


def clean_sunshine_data(sunshine_df) -> pd.DataFrame:
    """
    This function returns a cleaner version of the city sunshine dataset to
    only include the necessary data.
    """
    sunshine_cleaned = sunshine_df.drop_duplicates().copy()

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
              'OCT', 'NOV', 'DEC']
    sunshine_cleaned = pd.melt(sunshine_cleaned, id_vars=['CITY'],
                               value_vars=months, var_name='month',
                               value_name='sunshine_score'
                               )
    sunshine_cleaned['state'] = sunshine_cleaned['CITY'].str[-2:]
    sunshine_cleaned = sunshine_cleaned.rename(columns={"CITY": "city"})
    return sunshine_cleaned


def merge_temp_and_sunshine(temp_df, sunshine_df) -> pd.DataFrame:
    """
    This function merges the state weather dataset and the city sunshine
    dataset to represent climate data for each city each month.
    """
    # merge
    merged_weather = temp_df.merge(sunshine_df, on=['state', 'month'])
    # numerize sunshine_score
    merged_weather['sunshine_score'] = \
        pd.to_numeric(merged_weather['sunshine_score'], errors="coerce")
    return merged_weather


def summarize_city_data(data: pd.DataFrame, city: str) -> None:
    """
    This function summarizes the quantitative variables in the merged city
    weather data set.
    """
    city_data = data[data['city'] == city]
    precip_summary = city_data['ppt'].describe()
    temp_summary = city_data['tavg'].describe()
    sunshine_summary = city_data['sunshine_score'].describe()
    print("Summaries for : ", city)
    print("The summary for precipitation is:", precip_summary)
    print("The summary for temperature is:", temp_summary)
    print("The summary for sunshine is:", sunshine_summary)


def summarize_plant_data(data: pd.DataFrame, plant: str) -> None:
    """
    This function summarizes the quantitative variables in the house plant
    health data set.
    """
    plant_data = data[data['Plant_ID'] == plant]
    room_temp_summary = plant_data['Room_Temperature_C'].describe()
    health_summary = plant_data['Health_Score'].describe()
    humidity_summary = plant_data['Humidity_%'].describe()
    print("Summaries for : ", plant)
    print("The summary for room temperature is:", room_temp_summary)
    print("The summary for humidity is:", humidity_summary)
    print("The summary for health is:", health_summary)


def visualize_city_data(city_data: pd.DataFrame, city: str) -> None:
    """
    This function generates two data visualizations for the merged city weather
    dataset. The first graph shows a correlation between the annual average
    precipitation and the annual average temperature for each U.S. continental
    city. The second graph shows a correlation between the monthly average
    temperature and precipitation for a specified city.
    """
    average_sunshine_and_precip_by_city = \
        city_data.groupby('city')[['sunshine_score', 'ppt']].mean()
    sns.lmplot(data=average_sunshine_and_precip_by_city, x='sunshine_score',
               y='ppt')
    plt.title('Precipitation vs sunshine annual average in each city')
    plt.xlabel('Average annual sunshine (%)')
    plt.ylabel('Average annual precipitation (mm)')
    plt.savefig('percipitation_vs_sunshine_by_city.png', bbox_inches='tight')
    plt.close()

    average_sunshine_and_temp_by_city = \
        city_data.groupby('city')[['sunshine_score', 'tavg']].mean()
    sns.lmplot(data=average_sunshine_and_temp_by_city, x='sunshine_score',
               y='tavg')
    plt.title('Temperature vs sunshine annual average in each city')
    plt.xlabel('Average annual sunshine (%)')
    plt.ylabel('Average annual temperature (C)')
    plt.savefig('temperature_vs_sunshine_by_city.png', bbox_inches='tight')
    plt.close()

    city_specific_data = city_data[city_data['city'] == city]
    sns.relplot(data=city_specific_data, x='tavg', y='ppt', kind='line')
    plt.title(f'Percipitation vs average monthly temperature for {city}')
    plt.xlabel('Average monthly temperature (C)')
    plt.ylabel('Average monthly precipitation (mm)')
    plt.savefig(f'percipitation_vs_temperature_for_{city}.png',
                bbox_inches='tight')
    plt.close()


def visualize_plant_data(plant_data: pd.DataFrame, plant: str):
    """
    This function generates two data visualizations for the plant dataset.
    The first graph shows the correlation between the health score and
    temperature for a specified plant. The second graoh shows the count of
    each plant in the dataset.
    """
    plant_specific_data = plant_data[plant_data['Plant_ID'] == plant]
    sns.relplot(data=plant_specific_data, x='Room_Temperature_C',
                y='Health_Score', hue='Sunlight_Score', kind='line',
                height=8)
    plt.title("Correlation between temperature and plant health in different"
              f"sunlight levels for {plant}")
    plt.xlabel('Temperature in C')
    plt.ylabel('Health score')
    plt.savefig(f'health_vs_temperature_graph for {plant}.png')
    plt.close()

    sns.catplot(data=plant_data, x='Plant_ID', kind='count',
                color='g', height=8)
    plt.title("Amount of each plant in the Dataset")
    plt.xticks(rotation=-90)
    plt.xlabel('Plant type')
    plt.ylabel('count')
    plt.savefig('plant_count.png', bbox_inches='tight')
    plt.close()


def prepare_city_climate_data(temp_df: pd.DataFrame,
                              sunshine_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a prepared city climate dataframe given data without unnecessary
    columns and merged data.
    """
    cleaned_temp_data = clean_temp_data(temp_df=temp_df)
    cleaned_sunshine_data = clean_sunshine_data(sunshine_df=sunshine_df)
    merged_data = merge_temp_and_sunshine(cleaned_temp_data,
                                          cleaned_sunshine_data)
    merged_data = merged_data.drop(columns=['state'])
    return merged_data


def prepare_plant_data(indoor_plant_df: pd.DataFrame) -> pd.DataFrame:
    cleaned_plant_data = clean_plant_data(indoor_plant_df=indoor_plant_df)
    return cleaned_plant_data


def explore_city_data(city_data: pd.DataFrame, city: str) -> None:
    """
    Explores the city climate dataset by showing analytics of the data along
    with visualzing trends in the data.
    """
    if city not in city_data['city'].unique():
        raise ValueError('City not found in dataset!')
    find_missing_data(city_data, 'city_data')
    find_unique_values(city_data, 'city_data', 'city')
    find_unique_values(city_data, 'city_data', 'month')
    summarize_city_data(city_data, city)
    sns.set_theme()
    visualize_city_data(city_data, city)


def explore_plant_data(plant_data: pd.DataFrame, plant: str) -> None:
    """
    Explores the plant dataset by showing analytics of the data along with
    visualzing trends in the data for the overall plant data set and a
    specified plant.
    """
    if plant not in plant_data['Plant_ID'].unique():
        raise ValueError('Plant not found in dataset!')
    find_missing_data(plant_data, "cleaned_plant_data")
    find_unique_values(plant_data, "cleaned_plant_data", 'Plant_ID')
    summarize_plant_data(plant_data, plant)

    # Visualize!!
    sns.set_theme()
    find_climate_health_impact_graph(plant_data, plant)
    visualize_plant_data(plant_data, plant)


def find_climate_health_impact_graph(plant_data: pd.DataFrame, plant: str):
    """
    This function creates a 3D visual with a plane of best fit to visualize the
    correlations between a specified plant's room temperature and humidity to
    it's plant health.
    """
    plant_specific_data = plant_data[plant_data['Plant_ID'] == plant]
    plant_features = plant_specific_data[['Room_Temperature_C',
                                          'Humidity_%']]
    X = sm.add_constant(plant_features)
    y = plant_specific_data['Health_Score']

    room_temps = plant_specific_data['Room_Temperature_C'].values
    humidity_vals = plant_specific_data['Humidity_%'].values
    health_scores = plant_specific_data['Health_Score'].values
    correlation_model = sm.OLS(y, X).fit()
    const, room_temp_coef, humid_coef = correlation_model.params

    temp_range = np.linspace(room_temps.min(), room_temps.max(), 30)
    humid_range = np.linspace(humidity_vals.min(), humidity_vals.max(),  30)
    temp_axis, humid_axis = np.meshgrid(temp_range, humid_range)

    health_predictions = \
        const + room_temp_coef * temp_axis + humid_coef * humid_axis
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': '3d'})
    ax.plot_surface(temp_axis, humid_axis, health_predictions, alpha=0.3)

    # scatter observations colored by health score
    sc = ax.scatter(room_temps, humidity_vals, health_scores, c=health_scores,
                    cmap='RdYlGn', vmin=1, vmax=5, alpha=0.85)
    plt.colorbar(sc, ax=ax, shrink=0.5, label='Health Score')
    ax.set_xlabel('Temperature (°C)', labelpad=10)
    ax.set_ylabel('Humidity (%)', labelpad=10)
    ax.set_zlabel('Health Score', labelpad=10)

    ax.set_title('Correlations between temperature and humidity on health'
                 f' for {plant}')
    plt.tight_layout()
    plt.savefig(f'health_score_impact_for_{plant.replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
