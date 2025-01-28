# Stock Market Price Prediction Using LSTM

## Project Overview

This project focuses on predicting stock market prices using Long Short-Term Memory (LSTM) networks. The objective is to forecast future stock prices based on historical data to aid investors in making informed decisions. The model leverages LSTM’s ability to capture temporal patterns in time-series data, offering a more accurate prediction compared to traditional methods. The project uses the `symbols_valid_meta.csv` dataset, which contains historical stock prices for multiple companies.

---

## Features

- **Time-Series Prediction**: Uses LSTM networks to predict future stock prices based on past performance.
- **Hyperparameter Tuning**: Incorporates hyperparameter optimization to enhance the prediction accuracy.
- **Data Visualization**: Provides visualizations of the actual vs. predicted stock prices to demonstrate the model's performance.
- **Performance Evaluation**: Evaluates the model’s performance using metrics like Mean Absolute Error (MAE) and Mean Squared Error (MSE).
- **Real-Time Deployment**: Optionally includes a Flask API for real-time stock price predictions.

---

## Objectives

1. Build a deep learning model using LSTM networks to predict stock prices based on historical data.
2. Perform hyperparameter tuning to optimize the model for better prediction accuracy.
3. Visualize the results to compare actual vs. predicted prices.
4. Evaluate the model’s performance using robust metrics like MAE and MSE.
5. Provide an option for real-time deployment via a Flask API.

---

## Technologies Used

### Frameworks and Libraries
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Hyperparameter Tuning**: Keras Tuner (Hyperband algorithm)

### Model Architecture
- **Base Model**: Long Short-Term Memory (LSTM) network for time-series forecasting.
- **Enhancements**: Hyperparameter tuning for improved model performance.
- **Training**: Adam optimizer, Min-Max scaling for data normalization, and early stopping to prevent overfitting.

### Deployment
- **Platform**: Google Colab (for model training and testing)
- **API**: Flask (for real-time predictions, if needed)

---

## Dataset and Preprocessing

- **Data Source**: The `symbols_valid_meta.csv` dataset, containing historical stock prices for multiple companies.
- **Preprocessing**:
  - Missing values handled using imputation techniques.
  - Data normalized using Min-Max scaling to improve model performance.
  - Time-series sequences created from historical data to capture temporal patterns.
- **Train-Test Split**: Data was split into an 80:20 ratio for training and testing.

---

## Training Details

- **Hyperparameters**:
  - LSTM units: 64
  - Dropout rate: 0.2
  - Batch size: 32
  - Learning rate: 0.001
  - Epochs: 50
- **Optimizer**: Adam optimizer for efficient gradient descent.
- **Loss Function**: Mean Squared Error (MSE) for regression.
- **Regularization**: Dropout layers and early stopping to prevent overfitting.
- **Evaluation Metrics**:
  - **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
  - **Mean Squared Error (MSE)**: Quantifies the average squared difference between the predicted and actual values.

---

## Deployment Instructions

1. **Google Colab Setup**:
   - Upload the trained model to Google Colab in TensorFlow’s SavedModel format.
   - Run the provided Colab notebook for real-time stock price predictions.



---

## Challenges and Solutions

### Challenges
- **Data Preprocessing**: Stock market data often contains missing values and noisy patterns.
- **Model Optimization**: Fine-tuning the LSTM model for optimal accuracy was challenging.
- **Overfitting**: Preventing the model from overfitting to historical data was a key concern.

### Solutions
- **Data Imputation**: Missing values were handled using imputation techniques to ensure the model had complete data.
- **Hyperparameter Tuning**: Keras Tuner’s Hyperband algorithm was used to find the best hyperparameters.
- **Regularization**: Dropout layers and early stopping were implemented to avoid overfitting.

---

## Future Work

1. **Data Expansion**: Include additional features such as market sentiment, trading volume, and news sentiment to improve the model’s accuracy.
2. **Cloud Deployment**: Deploy the system on cloud platforms such as AWS or Google Cloud for scalability.
3. **Extended Applications**: Adapt the model to predict stock prices for different markets or individual stocks.


---

## How to Use

1. Clone the repository containing the project.
2. Set up the environment by installing the required dependencies.
3. Run the Google Colab notebook for model execution or set up the Flask API for local predictions.
4. Input historical stock data for real-time predictions.

---

## Contact Information

- **Author**: Atharva Talegaonkar
- **Email**: atale014@ucr.edu


