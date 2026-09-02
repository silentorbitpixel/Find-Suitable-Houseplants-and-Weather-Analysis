"""
Jason Choi
CSE 163 AD
This file utilizes the "Indoor Plant Health & Growth Dataset" by Souvikrana176
on Kaggle and a climate dataset based on "Annual percent of possible sunshine
by US City" dataset by thedevastator on Kaggle, and the PRISM's Weather Data
by State dataset to process data and train a gradient boosting model to predict
what plants would most likely survive in a given city.
"""
import pandas as pd
import statsmodels.api as sm
import eda
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

SENSITIVITY_COEFFICIENT_THRESHOLD = 0.3
TEMPERATURE_VARIANCE_TRESHOLD = 7  # in Celsius
SUNLIGHT_VARIANCE_TRESHOLD = 25  # in %


def find_climate_sensitivities(plant_data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame describing how sensitive each plant is to changes in
    room temperature and sunlight air.
    """
    updated_plant_data = plant_data.copy()
    plants = plant_data['Plant_ID'].unique()
    total_sensitivity_data = pd.DataFrame()
    for plant in plants:
        current_sensitivity_data = \
            _find_climate_sensitivies_per_plant(updated_plant_data, plant)
        total_sensitivity_data = pd.concat([total_sensitivity_data,
                                           current_sensitivity_data])
    return total_sensitivity_data


def _find_climate_sensitivies_per_plant(plant_data: pd.DataFrame,
                                        plant: str) -> pd.DataFrame:
    """
    This function returns a DataFrame describing how sensitive a specific
    plant's health is to changes in room temperature and sunlight, utilizing
    multiple linear regression to determine the correlation between the plant's
    health and room temperature and sunlight.
    """
    # standardize data and find coeficients
    plant_specific_data = plant_data[plant_data['Plant_ID'] == plant]
    plant_features = plant_specific_data[['Room_Temperature_C',
                                          'Sunlight_Score']]
    scaler = StandardScaler()
    X = sm.add_constant(scaler.fit_transform(plant_features))
    y = plant_specific_data['Health_Score']
    sensitivity_model = sm.OLS(y, X)
    sensitivity_result = sensitivity_model.fit()
    # relabel coefficients from multiple linear regression model
    sensitivity_result.model.exog_names[:] = ['const',
                                              'standardized_temp',
                                              'standardized_sunlight']
    # create sensitivity DataFrame
    sensitivity_coefficients = sensitivity_result.params
    temp_sensitive = (abs(sensitivity_coefficients['standardized_temp']) >=
                      SENSITIVITY_COEFFICIENT_THRESHOLD)
    sunlight_sensitive = (
        abs(sensitivity_coefficients['standardized_sunlight']) >=
        SENSITIVITY_COEFFICIENT_THRESHOLD)
    plant_sensitivities = pd.DataFrame([{'Plant_ID': plant,
                                         'temp_sensitive': temp_sensitive,
                                         'sunlight_sensitive':
                                        sunlight_sensitive
                                         }])
    return plant_sensitivities


def format_city_climate_profile_data(city_data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of the cities formatted similar to that of the plant
    climate using the temperature and sunshine data.
    """
    cities = city_data['city'].unique()
    total_profiles_data = pd.DataFrame()
    for city in cities:
        current_climate_profile = \
            _format_specific_city_climate_data(city_data, city)
        total_profiles_data = pd.concat([total_profiles_data,
                                         current_climate_profile])
    total_profiles_data = total_profiles_data.set_index('city')
    return total_profiles_data


def _format_specific_city_climate_data(city_data: pd.DataFrame,
                                       city: str) -> pd.DataFrame:
    """
    Return a DataFrame representing a specified city's climate in a similar
    format to the plant data, further adding information about the variance of
    the city's temperature and sunlight.
    """
    city_specific_data = city_data[city_data['city'] == city]
    city_climate_profile = pd.DataFrame([{
        'city': city,
        'tavg': city_specific_data['tavg'].mean(),
        # divide sunshine by 20 to put in scale of 0-5, same as plant data
        'sunshine_avg': city_specific_data['sunshine_score'].mean() / 20,
        'temp_variable': _is_metric_variable(city_specific_data['tavg'],
                                             TEMPERATURE_VARIANCE_TRESHOLD),
        'sunshine_variable': _is_metric_variable(
            city_specific_data['sunshine_score'], SUNLIGHT_VARIANCE_TRESHOLD)
    }])

    return city_climate_profile


def _is_metric_variable(metric: pd.Series, threshold: float) -> bool:
    """
    Given a Series of values, determine if they are highly variable if the
    standard deviation is higher than the given threshold.
    """
    return metric.std() > threshold


def train_plant_model(plant_data: pd.DataFrame) -> xgb.XGBRegressor:
    """
    This function trains an XGRegressor model based on the given plant data to
    predict health scores, analyzes, and returns it.
    """
    features = ['Plant_ID', 'Room_Temperature_C', 'Sunlight_Score']
    X = plant_data[features].copy()
    X['Plant_ID'] = X['Plant_ID'].astype('category')
    y = plant_data['Health_Score']
    model = xgb.XGBRegressor(enable_categorical=True)
    model.fit(X, y)
    return model


def predict_city_plants(plant_data: pd.DataFrame,
                        climate_profiles: pd.DataFrame,
                        plant_sensitivities: pd.DataFrame,
                        city: str,
                        model: xgb.XGBRegressor) -> list[str]:
    """
    Returns a list of plants in the given plant data that would be best suited
    for a given city using the given model, further verifying the suitability
    using the plants' sensitivities and city climate data.
    """
    plants = plant_data['Plant_ID'].unique()
    city_climate = climate_profiles.loc[city]
    best_plants = []
    for plant in plants:
        reformatted_city_climate = pd.DataFrame([{
            "Plant_ID":           plant,
            "Room_Temperature_C": city_climate['tavg'],
            "Sunlight_Score":     city_climate['sunshine_avg']
        }])
        reformatted_city_climate['Plant_ID'] = \
            reformatted_city_climate['Plant_ID'].astype('category')
        predicted_plant_score = model.predict(reformatted_city_climate)[0]
        if predicted_plant_score >= 3.5:
            best_plants.append(plant)
    best_plants = _verify_climate_tolerance(best_plants, plant_sensitivities,
                                            city_climate)
    return best_plants


def _verify_climate_tolerance(plants: list[str],
                              plant_sensitivities: pd.DataFrame,
                              climate_profile: pd.DataFrame) -> list[str]:
    """
    Returns a modified list of plants, keeping only the plants that are suited
    for the variance in the given city using it's climate profile.
    """
    city_temp_variable = climate_profile['temp_variable']
    city_sunshine_variable = climate_profile['sunshine_variable']
    for plant in plants[:]:
        plant_data = plant_sensitivities[plant_sensitivities['Plant_ID'] ==
                                         plant].iloc[0]
        is_temp_sensitive = plant_data['temp_sensitive']
        is_sunlight_sensitive = plant_data['sunlight_sensitive']
        if (city_temp_variable and is_temp_sensitive) or \
           (city_sunshine_variable and is_sunlight_sensitive):
            plants.remove(plant)
    return plants


def find_suitable_houseplants(plant_data: pd.DataFrame,
                              city_data: pd.DataFrame, city: str) -> None:
    """
    This function uses the given plant data and city data to predict the most
    suitable houseplants for a specified city by the user using a gradient
    boosting model.
    """

    # preprocess data
    plant_sensitivities = find_climate_sensitivities(plant_data)
    climate_profiles = format_city_climate_profile_data(city_data)

    # machine learning
    model = train_plant_model(plant_data)
    best_plants = predict_city_plants(plant_data, climate_profiles,
                                      plant_sensitivities, city,
                                      model)
    print(f"The best plant(s) for {city} is/are: ", best_plants)


def main():
    indoor_plant_df = \
        pd.read_csv("./data/Indoor_Plant_Health_and_Growth_Factors.csv")
    sunshine_df = pd.read_csv("./data/Average Percent of Possible"
                              " Sunshine by US City.csv", usecols=range(1, 15))
    temp_df = pd.read_csv("./data/state_month_temp_and_prec_2025.csv")
    # clean data
    plant_data = eda.prepare_plant_data(indoor_plant_df)
    city_data = eda.prepare_city_climate_data(temp_df, sunshine_df)

    print("---PLANT PREDICTIONS FOR A CITY---")
    city = input("Enter the city you'd like to see the plants for "
                 "(ex. SEATTLE,WA): ").strip().upper()
    find_suitable_houseplants(plant_data, city_data, city)

    print("---EXPLORATORY DATA ANALYSIS---")
    run_eda = \
        input("Would you like to run the EDA as well? Y/N: ").strip().upper()
    if run_eda == "Y":
        city_analytics = input("Enter the city you'd like to analyze "
                               "(ex. SEATTLE,WA): ").strip().upper()
        plant = input("Enter the plant you'd like to see the visualizations "
                      " for (ex. Aloe vera): ").strip()
        eda.explore_city_data(city_data, city_analytics)
        eda.explore_plant_data(plant_data, plant)


if __name__ == '__main__':
    main()
