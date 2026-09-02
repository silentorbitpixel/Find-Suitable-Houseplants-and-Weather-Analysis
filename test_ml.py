"""
Jason Choi
CSE 163 AD
This program tests the methods that process the houseplant and weather datasets
in the machine learning file
"""
import ml
import eda
import pandas as pd


def test_is_metric_variable() -> None:
    """
    Tests the is_metric_variable method
    """
    test_list1 = [1, 5, 9, 20, 25]
    test_series1 = pd.Series(test_list1)
    test_list2 = [1, 1, 1, 1, 2]
    test_series2 = pd.Series(test_list2)
    assert ml._is_metric_variable(test_series1, 4)
    assert ml._is_metric_variable(test_series1, 500) is False
    assert ml._is_metric_variable(test_series2, 1) is False
    assert ml._is_metric_variable(test_series2, 0)


def test_find_climate_sensitivities(test_plant_data: pd.DataFrame) -> None:
    """
    Tests the find_climate_sensitivities
    """
    prepared_plant_data = eda.prepare_plant_data(test_plant_data)
    test_sensitivities = ml.find_climate_sensitivities(prepared_plant_data)
    expected_columns = ['Plant_ID', 'temp_sensitive', 'sunlight_sensitive']
    assert expected_columns == test_sensitivities.columns.tolist()

    ficus_row = test_sensitivities[test_sensitivities['Plant_ID']
                                   == 'Ficus lyrata'].iloc[0]
    assert not ficus_row['temp_sensitive']
    assert not ficus_row['sunlight_sensitive']

    aloe_row = test_sensitivities[test_sensitivities['Plant_ID'] ==
                                  'Aloe vera'].iloc[0]
    assert aloe_row['temp_sensitive']
    assert aloe_row['sunlight_sensitive']


def test_format_city_climate_profile_data(
        test_city_data: pd.DataFrame) -> None:
    """
    Tests the format_city_climate_profile_data method
    """
    formatted_city_data = ml.format_city_climate_profile_data(test_city_data)
    expected_columns = ['tavg', 'sunshine_avg',
                        'temp_variable', 'sunshine_variable']
    assert expected_columns == formatted_city_data.columns.tolist()


def test_verify_climate_tolerance(test_city_data: pd.DataFrame,
                                  test_plant_data: pd.DataFrame):
    """
    Tests the _verify_climate_tolerance method
    """
    plants = ['Ficus lyrata', 'Aloe vera']
    cleaned_plant_data = eda.prepare_plant_data(test_plant_data)
    test_plant_sensitivities = \
        ml.find_climate_sensitivities(cleaned_plant_data)
    test_city_climates = \
        ml.format_city_climate_profile_data(test_city_data)
    test_oakland_climate = test_city_climates.loc['OAKLAND,CA']
    test_sf_climate = test_city_climates.loc['SAN FRANCISCO,CA']
    updated_plants_oak = ml._verify_climate_tolerance(plants.copy(),
                                                      test_plant_sensitivities,
                                                      test_oakland_climate)
    updated_planted_sf = ml._verify_climate_tolerance(plants.copy(),
                                                      test_plant_sensitivities,
                                                      test_sf_climate)
    assert 'Ficus lyrata' in updated_plants_oak
    assert 'Aloe vera' in updated_plants_oak
    assert 'Aloe vera' not in updated_planted_sf
    assert 'Ficus lyrata' in updated_planted_sf


def main():
    test_plant_data = pd.read_csv('./data/test_plant_data.csv')
    test_sunshine_data = \
        pd.read_csv('./data/test_sunshine_data.csv', usecols=range(1, 15))
    test_temp_data = pd.read_csv('./data/test_temp_data.csv')
    test_city_data = eda.prepare_city_climate_data(test_temp_data,
                                                   test_sunshine_data)
    test_is_metric_variable()
    test_find_climate_sensitivities(test_plant_data)
    test_format_city_climate_profile_data(test_city_data)
    test_verify_climate_tolerance(test_city_data, test_plant_data)


if __name__ == '__main__':
    main()
