# Find Suitable Houseplants and Weather Analysis

**Jason Choi, Elizabeth Astor**

This project analyzes U.S. city climate datasets and an indoor houseplant dataset with multiple climate factors to determine the most suitable houseplants for a given city depending on their climate.

# Files

* `eda.py` handles the data processing, cleaning, and visualizations of the datasets. 
* `ml.py` handles preprocessing necessary to train the plant prediction model, trains model used for plant prediction, and handles the plant prediction based on user input.
* `test_eda.py` tests the quantitative functions within `eda.py` to verify their functionality.
* `test_ml.py` tests the quantitative functions within `ml.py` to verify their functionality.

# Instructions

## Set-Up

* Prior to running the program, make sure that the proper libraries are downloaded. These libraries include `statsmodels`, `pandas`, `matplotlib`, `seaborn`, `numpy`, `xgboost`, and `scikit-learn`.
* To install all necessary libraries, run the command '`pip install statsmodels pandas matplotlib seaborn numpy xgboost scikit-learn` within the terminal

## Usage

* To use the program, run the main script in `ml.py`
* After running, enter your city of choice formatted as `CITY,ST`. You also have the option of running the EDA, where you will have to enter a city of choice to analyze and plant to analyze

