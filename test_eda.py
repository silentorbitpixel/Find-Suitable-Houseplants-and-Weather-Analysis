"""
Jason Choi
CSE 163 AD
This program tests the methods that process the houseplant and weather datasets
"""
import eda
import pandas as pd


def test_find_missing_data(test_data: pd.DataFrame) -> None:
    """
    Tests the find_missing_data method
    """
    assert 1 == eda.find_missing_data(test_data, 'test data')


def test_find_unique_values(test_data: pd.DataFrame) -> None:
    """
    Tests the find_unique_values method
    """
    assert 4 == eda.find_unique_values(test_data, 'test data', 'name')
    assert 3 == eda.find_unique_values(test_data, 'test data', 'age')
    assert 4 == eda.find_unique_values(test_data, 'test data', 'favorite_food')


def test_clean_plant_data(test_plant_data: pd.DataFrame) -> None:
    """
    Tests the clean_plant_data method
    """
    cleaned_test_plant_data = eda.clean_plant_data(test_plant_data)
    expected_columns = ['Plant_ID', 'Room_Temperature_C', 'Humidity_%',
                        'Health_Score', 'Sunlight_Score']
    assert expected_columns == cleaned_test_plant_data.columns.tolist()


def test_clean_temp_data(test_temp_data: pd.DataFrame) -> None:
    """
    Tests the clean_temp_data method
    """
    cleaned_test_temp_data = eda.clean_temp_data(test_temp_data)
    expected_columns = ['state', 'ppt', 'tavg', 'month']
    assert expected_columns == cleaned_test_temp_data.columns.tolist()
    assert 'FEB' == cleaned_test_temp_data.loc[0, 'month']


def test_clean_sunshine_data(test_sunshine_data: pd.DataFrame) -> None:
    """
    Tests the clean_sunshine_data method
    """
    cleaned_test_sunshine_data = eda.clean_sunshine_data(test_sunshine_data)
    expected_columns = ['city', 'month', 'sunshine_score', 'state']
    assert expected_columns == cleaned_test_sunshine_data.columns.tolist()
    assert 'CA' == cleaned_test_sunshine_data.loc[0, 'state']
    assert 'JAN' == cleaned_test_sunshine_data.loc[0, 'month']
    assert 74 == cleaned_test_sunshine_data.loc[0, 'sunshine_score']


def test_merge_temp_and_sunshine(test_temp_data: pd.DataFrame,
                                 test_sunshine_data: pd.DataFrame):
    """
    Tests the merge_temp_and_sunshine method
    """
    cleaned_test_temp_data = eda.clean_temp_data(test_temp_data)
    cleaned_test_sunshine_data = eda.clean_sunshine_data(test_sunshine_data)
    test_merged_data = eda.merge_temp_and_sunshine(cleaned_test_temp_data,
                                                   cleaned_test_sunshine_data)
    expected_columns = ['state', 'ppt', 'tavg', 'month', 'city',
                        'sunshine_score']
    assert expected_columns == test_merged_data.columns.tolist()
    assert 8.678 == test_merged_data.loc[0, 'tavg']
    assert 141.668 == test_merged_data.loc[0, 'ppt']
    assert 'CO' not in test_merged_data['state'].unique()


def main():
    test_data = pd.read_csv('./data/test_eda.csv')
    test_plant_data = pd.read_csv('./data/test_plant_data.csv')
    test_sunshine_data = \
        pd.read_csv('./data/test_sunshine_data.csv', usecols=range(1, 15))
    test_temp_data = pd.read_csv('./data/test_temp_data.csv')
    test_find_missing_data(test_data)
    test_find_unique_values(test_data)
    test_clean_plant_data(test_plant_data)
    test_clean_temp_data(test_temp_data)
    test_clean_sunshine_data(test_sunshine_data)
    test_merge_temp_and_sunshine(test_temp_data, test_sunshine_data)


if __name__ == '__main__':
    main()
