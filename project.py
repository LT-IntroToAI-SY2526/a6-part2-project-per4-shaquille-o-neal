"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
-Christopher Llinas-Aviles 
- 
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
    axes[0,2].set_xlabel('Average Temperature (F)')
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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 200, random_state = 42)

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
    # TODO: Create a LinearRegression model
    model = LinearRegression()
    # TODO: Train the model using fit()
    model.fit(X_train, y_train)
    # TODO: Print the intercept
    print(f"\n===Model Training Complete ===")
    print(f"Intercept: kWh{model.intercept_:.2f}")
    # TODO: Print each coefficient with its feature name
    #       Hint: use zip(feature_names, model.coef_)
    print(f"\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f" {name}: {coef:.2f}")
    # TODO: Print the full equation in readable format
    print(f"\nEquation:")
    equation = f"price = "
    for i, (name, coef) in enumerate(zip(feature_names, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} x {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    return model


def evaluate_model(model, X_test, y_test, feature_names):
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
     # TODO: Make predictions on X_test
    predictions = model.predict(X_test)
    # TODO: Calculate R² score
    r2 = r2_score(y_test, predictions)
    # TODO: Calculate MSE and RMSE
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    # TODO: Print R² score with interpretation
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of energy use variation")
    # TODO: Print RMSE with interpretation
    print(f"\nRoot Mean Squared Error: kWh{rmse:.2f}")
    print(f"  → On average, predictions are off by kWh{rmse:.2f}")
    # TODO: Calculate and print feature importance
    #       Hint: Use np.abs(model.coef_) and sort by importance
    #       Show which features matter most
    print(f"\n === Feature Importance ===")
    feature_importance = list(zip(feature_names, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x:x[1], reverse=True)

    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}, {name}: {importance:.2F}")
    # TODO: Return predictions
    return predictions


def make_prediction(model):
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
    
    pass


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
    # predictions = evaluate_model(model, X_test, y_test)
    
    # # Step 6: Make a prediction, add features as an argument
    # make_prediction(model)
    
    # print("\n" + "=" * 70)
    # print("PROJECT COMPLETE!")
    # print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

