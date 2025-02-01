# Labour Wage Prediction

## Overview
This project implements a machine learning model for predicting labour wages based on various factors such as capital, labour, and output. The model is built using TensorFlow's Keras library and a neural network. It also utilizes Scikit-Learn for data preprocessing and visualization with Matplotlib and Seaborn.

## Dataset
The dataset used in this project is assumed to be the **Labour.csv** file. It contains features related to economic factors and a label indicating the wage amount.

## Requirements
Ensure you have the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Implementation Details

### Steps Involved
1. **Load Dataset**: Reads the labour dataset using Pandas.
2. **Data Preprocessing**:
   - Checks for missing values.
   - Splits data into features (`X`) and labels (`Y`).
   - Splits data into training and testing sets (80-20 split).
   - Standardizes the features using `StandardScaler`.
3. **Data Visualization**:
   - Plots line graphs for capital, labour, and output using Seaborn.
4. **Model Creation**:
   - Defines a Sequential Neural Network with three layers:
     - Input layer with 10 neurons and ReLU activation.
     - Hidden layer with 5 neurons and ReLU activation.
     - Output layer with 1 neuron (for regression output).
   - Compiles the model using the Adam optimizer and mean absolute error (MAE) loss function.
5. **Training & Evaluation**:
   - Trains the model for 200 epochs.
   - Plots the loss graph over epochs.
6. **Making Predictions**:
   - Uses the trained model to predict wages on new data.
   - Example:
   
   ```python
   sample_input = np.array([[5000, 200, 1000]])  # Example input data
   sample_input = scaler.transform(sample_input)  # Apply same scaling
   prediction = model.predict(sample_input)
   print(f"Predicted Wage: {prediction[0][0]}")
   ```

## Code Structure
- **Data Preprocessing**: Loads and standardizes the dataset.
- **Data Visualization**: Plots important features for understanding trends.
- **Model Building**: Defines and compiles the neural network.
- **Training & Visualization**: Trains the model and plots the loss graph.
- **Prediction**: Uses the trained model to predict wages based on input data.

## Usage
Run the script using Python:
```bash
python labour_wage_prediction.py
```
Ensure the `Labour.csv` file is available in the working directory.

## Link to colab file

https://colab.research.google.com/drive/1pjoJcW0NkQN6b1cV4C2iJOhdsenyvbpn?usp=sharing

## Results
The trained model predicts labour wages based on input factors. The performance can be evaluated using MAE.

## Future Enhancements
- Improve model accuracy by tuning hyperparameters.
- Use different architectures (LSTMs, CNNs) for better predictions.
- Implement a web interface for easy user interaction.
