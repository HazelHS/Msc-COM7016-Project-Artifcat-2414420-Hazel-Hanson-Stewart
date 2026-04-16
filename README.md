# Time series financial forecast AI modelling framework and software application

## What is this project?

This project was created to address issues widely discussed in the literature surrounding AI modelling public research, specifically to answer these two reseach questions:

"Demonstrating effectiveness in AI and deep learning for financial time series forecast AI modelling",
and
"Flexibility and interoperability of the application/framework for each stage in the AI model development cycle, and show how this could benefit education or further research in the domain."   

It includes the full pipeline for training, data colleciton, processing and evaluation steps for two AI models (using pytorch), listed below is the full functionality:

### Models included:

1. xLSTM-TS: This model was reproduced from the liturature found in the References.txt as (Gil et al., 2024) APA7th.
2. MEMD-TCN:T his model was reproduced from the liturature found in the References.txt as (Bai et al., 2018), ( Rehman & Mandic, 2009), (Yao et al., 2023)   APA7th.
3. AI model diagram generation script, including a customisable key.
4. AI model training scripts for both models with a shared utility, exposed tuning parameters to the GUI with saving the weights to files.

### Automated dataset feature collection:

These features can be collected using a combination of the yfinance and requests (for api.blockchain.info) libraries.
 
1. S2F Model - "stock to flow" bitcoin indicator approximation.
2. BTC/USD Open - bitcoin / US dollar.
3. BTC/USD High
4. BTC/USD Low
5. BTC/USD
6. BTC Volume
7. Currency US Dollar Index
8. Currency Gold Futures
9. Gold/BTC Ratio
10. Onchain Active Addresses
11. Onchain Median Confirmation Time (min)
12. Onchain Mining Difficulty
13. Onchain Hash Rate (GH/s)
14. Onchain Transaction Count
15. Onchain Transaction Fees (BTC)
16. Global averaged stocks(USD)
17. Global averaged stocks (volume)
19. Volatility_Crude Oil Volatility Index (OVX) - these are oil volatility indicators.
20. Volatility_CBOE SKEW Index
21. Volatility_CBOE Volatility Index (VIX)

### Dataset processing methods:

1. Interpolation between existing values to create rudimentary synthetic data for missing values.
2. Normalisation of values
3. De-noising of time series data using wavelet transform algorithm referenced in (Gil et al., 2024), with optional plot graph output.
4. Feature selection method: Boruta.
5. Feature selection method: LASSO.
6. Feature selection method: Random Forest.

### Dataset analysis methods:

1. Check for outliers and missing data scripts.
2. Plot graph for Correlation Matrix.
3. Plot graph for Distribution Outliers.
4. Plot graph for Missing Values Heatmap.
5. Plot graph for Pairplot.
6. Plot graph for Time series.

### AI model Evaluations:

Classification:

1. Accuracy
2. Precision
3. Recall
4. F1 Score

Error and regression:

5. MAPE (Mean Absolute Percentage Error)
6. MASE (Mean Absolute Scaled Error)
7. MSE (Mean Squared Error)
8. R2 (R-Squared)
9. MAE (Mean Absolute Error)
10. RMASE (Root Mean Squared Error)

Misc:

11. Predictions vs. Actuals (Time Series Plot, Comparison)

## Who is this project for?

The project was designed for anyone (free to use, under the MIT licence) with the technical ability and interest to use python for data science and AI modelling, with the technical proficiency expected to use this as baseline for furthering their own research into deep learning models for financial forescasting. 

If utilising the inlcuded models and data collection/processing methods are sufficeent for the users needs without the need for creating additional python scripts, then the GUI interface may still be useful to less adept users. 

## How to use this project?

Included in the root directory of the repo is a user manual (User_Manual.docx) for the most indepth details of how to use this application. 

In brief:

1. Run the command "pip install -r requirements.txt" from python terminal or IDE of choise to install all the required dependencies.

2. Then run "py ./Model_Designer.py" to open the GUI script in the repositories root directory. From there the user can collect, process, select and train one of two AI models for finanical time series forecasting and evaluate them with the included scripts.

3. Lastly, to run the unit tests, the user may use the command "py -m pytest".