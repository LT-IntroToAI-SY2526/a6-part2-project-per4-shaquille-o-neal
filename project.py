"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
-Christopher Llinas-Aviles 
- WenJun Ou
- 
- 

Dataset: [Energy Consumption]
Predicting: [How much energy a Building consumes]
Features: [Building Type,Square Footage,Number of Occupants,Appliances Used,Average Temperature,Day of Week,Energy Consumption]
Residential = 0
commercial = 1
Industrial = 2
weekday = 0
weekend =1
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'train_energy_data.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    data = pd.read_csv(filename)
    
    print("Hpuse price data")
    print(f"\nFirst 7 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print(f"\nBasic statistics: ")
    print(data.describe())
    print(f"\nColumn names: {list(data.columns)}")

    return data


def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    
    # TODO: Create a figure with 2x2 subplots, size (12, 10)
    fig, axes = plt.subplots(2,3,figsize=(12, 10))
    # TODO: Add a main title: 'House Features vs Price'
    fig.suptitle('House Features vs Price', fontsize = 16, fontweight = 'bold')
   
    axes[0,0].scatter(data['Building Type'], data['Energy Consumption'], color= 'blue', alpha=0.6) 
    axes[0,0].set_xlabel('Building Type')
    axes[0,0].set_ylabel('Energy Consumption(kWh)')
    axes[0,0].set_title('Building Type vs Energy Consumption')
    axes[0,0].grid(True, alpha=0.3)
   
    axes[0,1].scatter(data['Square Footage'], data['Energy Consumption'], color= 'green', alpha=0.6) 
    axes[0,1].set_xlabel('Square Footage (sqrft)')
    axes[0,1].set_ylabel('Energy Consumption(kWh)')
    axes[0,1].set_title('SquareFootage vs Energy Consumption')
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].scatter(data['Number of Occupants'], data['Energy Consumption'], color= 'red', alpha=0.6) 
    axes[1,0].set_xlabel('Number of Occupants')
    axes[1,0].set_ylabel('Energy Consumption(kWh)')
    axes[1,0].set_title('Number of Occupants vs Energy Consumption')
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].scatter(data['Appliances Used'], data['Energy Consumption'], color= 'orange', alpha=0.6) 
    axes[1,1].set_xlabel('Appliances Used')
    axes[1,1].set_ylabel('Energy Consumption(kWh)')
    axes[1,1].set_title('Appliances Used vs Energy Consumption')
    axes[1,1].grid(True, alpha=0.3)

    axes[0,2].scatter(data['Average Temperature'], data['Energy Consumption'], color= 'yellow', alpha=0.6) 
    axes[0,2].set_xlabel('Average Temperature (C)')
    axes[0,2].set_ylabel('Energy Consumption(kWh)')
    axes[0,2].set_title('Average Temperature vs Energy Consumption')
    axes[0,2].grid(True, alpha=0.3)

    axes[1,2].scatter(data['Day of Week'], data['Energy Consumption'], color= 'black', alpha=0.6) 
    axes[1,2].set_xlabel('Day of Week')
    axes[1,2].set_ylabel('Energy Consumption(kWh)')
    axes[1,2].set_title('Day of Week vs Energy Consumption')
    axes[1,2].grid(True, alpha=0.3)
    # TODO: Use plt.tight_layout() to make plots fit nicely
    plt.tight_layout()
    # TODO: Save the figure as 'feature_plots.png' with dpi=300
    plt.savefig('energy_data.png', dpi = 300, bbox_inches = 'tight')
    print("\nFeature plots saved as 'energy_data.png'")
    # TODO: Show the plot
    plt.show()

    # Your code here
    # Hint: Use subplots like in Part 2!


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
   # TODO: Create a list of feature column names
    #       ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Age']
    feature_columns = ['Building Type','Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week']
    # TODO: Create X by selecting those columns from data
    X = data[feature_columns]
    # TODO: Create y by selecting the 'Price' column
    y = data['Energy Consumption']
    # TODO: Print the shape of X and y
    print(f"\n=== Feature Preparation ===")
    print(f"Features (x) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    # TODO: Print the feature column names
    print(f"\nFeature columns: {list(X.columns)}")
    # TODO: Return X and y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 800, random_state = 201)

    # TODO: Print how many samples are in training and testing sets
    print(f"\n=== Data Split (Matching Unplugged Activity) ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    print(f"\nNOTE: We're NOT scaling features here so coefficients are easy to interpret!")
    # TODO: Return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    
    model = LinearRegression()
   
    model.fit(X_train, y_train)
    
    print("\n✓ Model trained successfully!")
    
    # TODO: Print the equation with intercept and coefficients
    print(f"\n=== Model Equation ===")
    print(f"Energy Consumption = {model.intercept_:.2f}")
    
    feature_names = X_train.columns
    for name, coef in zip(feature_names, model.coef_):
        sign = "+" if coef >= 0 else ""
        print(f"                     {sign} {coef:.2f} × {name}")
    
    # TODO: Calculate and display feature importance
    print(f"\n=== Feature Importance (by coefficient magnitude) ===")
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print("\nRanked from most to least important:")
    for idx, row in importance.iterrows():
        print(f"{row['Feature']:25s}: {row['Coefficient']:8.2f} (|{row['Abs_Coefficient']:.2f}|)")
    
    # TODO: Return the trained model
    return model
    
    pass


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
     """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # TODO: Make predictions using the model
    predictions = model.predict(X_test)
    
    # TODO: Calculate R² score using r2_score()
    r2 = r2_score(y_test, predictions)
    # TODO: Calculate RMSE using mean_squared_error() and np.sqrt()
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    # TODO: Print the metrics with clear explanations
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → This means our model explains {r2*100:.2f}% of the variance in energy consumption")
    print(f"\nRMSE: {rmse:.2f} kWh")
    print(f"  → On average, our predictions are off by ±{rmse:.2f} kWh")
    
    # TODO: Create a comparison table showing actual vs predicted values
    print(f"\n=== Actual vs Predicted (First 10 Examples) ===")
    comparison = pd.DataFrame({
        'Actual': y_test.values[:10],
        'Predicted': predictions[:10],
        'Error': y_test.values[:10] - predictions[:10]
    })
    comparison['Error %'] = (comparison['Error'] / comparison['Actual'] * 100).abs()
    
    print(comparison.to_string(index=False))
    
    # TODO: Calculate additional error statistics
    mean_error = np.mean(np.abs(y_test - predictions))
    mean_percent_error = np.mean(np.abs((y_test - predictions) / y_test) * 100)
    
    print(f"\n=== Overall Error Statistics ===")
    print(f"Mean Absolute Error: {mean_error:.2f} kWh")
    print(f"Mean Absolute Percent Error: {mean_percent_error:.2f}%")
    
    # TODO: Return the predictions
    return predictions
    


def make_prediction(model,Building_Type,Square_Footage, Number_of_Occupants, Appliances_Used, Average_Temperature, Day_of_Week):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    building_features = pd.DataFrame([[Building_Type,Square_Footage, Number_of_Occupants, Appliances_Used, Average_Temperature, Day_of_Week]], columns = ['Building Type','Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week'])
    
    # TODO: Make a prediction using model.predict()
    predicted_energyuse = model.predict(building_features)[0]
    # TODO: Print the house specs and predicted price nicely formatted
    print(f"\n=== New Prediction ===")
    print(f"building specs: {Building_Type:.0f} building type, {Square_Footage} Square feet, {Number_of_Occupants} people, {Appliances_Used} Appliances used, {Average_Temperature} Celcius, and {Day_of_Week}.")
    print(f"Predicted energy use: kWh{predicted_energyuse:,.2f}")
    # TODO: Return the predicted price
    return predicted_energyuse

def user_make_prediction(model):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    print("Let's make our own prediction :D")
    Building_Type = input("Building type (0 = Residential, 1 = commercial, 2 = industrial.)\n")
    Square_Footage = input("Square Footage (max = 50,000)\n")
    Number_of_Occupants = input("Number of occupants(max = 100)\n")
    Appliances_Used = input("Appliances Used (Max = 50)\n")
    Average_Temperature = input("Average temperature (Max= 35C )\n")
    Day_of_Week = input("Day of week (0 = weeday, 1 = weekend.)\n")
    building_features = pd.DataFrame([[Building_Type,Square_Footage, Number_of_Occupants, Appliances_Used, Average_Temperature, Day_of_Week]], columns = ['Building Type','Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week'])
    
    # TODO: Make a prediction using model.predict()
    predicted_energyuse = model.predict(building_features)[0]
    # TODO: Print the house specs and predicted price nicely formatted
    print(f"\n=== New user Prediction ===")
    print(f"building specs: {Building_Type} building type, {Square_Footage} Square feet, {Number_of_Occupants} people, {Appliances_Used} Appliances used, {Average_Temperature} Celcius, and {Day_of_Week}.")
    print(f"Predicted energy use: kWh {predicted_energyuse:,.2f}")
    # TODO: Return the predicted price
    return predicted_energyuse


if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # # Step 4: Train
    model = train_model(X_train, y_train,data.columns)
    
    # # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # # Step 6: Make a prediction, add features as an argument
    make_prediction(model,0, 20000,40,40,30.00,0)
    user_make_prediction(model)
    # print("\n" + "=" * 70)
    # print("PROJECT COMPLETE!")
    # print("=" * 70)

