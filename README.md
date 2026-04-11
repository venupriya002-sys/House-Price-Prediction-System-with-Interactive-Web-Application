# 🏠 House Price Prediction System

> **AI-powered property price prediction** using Gradient Boosting Machine Learning with an interactive Premium Web Application built in Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?logo=plotly&logoColor=white)

---

## ✨ Features

### 🏠 Dashboard
- Key metrics at a glance (total properties, average price, lot size, model accuracy)
- Interactive price distribution histogram
- Price vs Lot Size scatter plot (colored by bedrooms)
- Full correlation heatmap
- Feature importance chart

### 📊 Data Explorer
- **Multi-filter system**: filter by lot size range, bedrooms, bathrooms, and stories
- Interactive scatter plots with configurable axes and color coding
- Average price breakdown by bedrooms and stories
- Raw data table with **CSV download**
- Statistical summary of filtered data

### 🤖 AI Price Predictor
- **Gradient Boosting Regressor** trained on 11 property features
- Model performance metrics: R² Score, MAE, RMSE, Cross-Validation
- Actual vs Predicted visualization
- Intuitive input form with dropdowns and sliders
- **Animated price prediction** with confidence score and price range
- Feature contribution breakdown for each prediction

---

## 🧠 Machine Learning Model

| Metric | Value |
|--------|-------|
| **Algorithm** | Gradient Boosting Regressor |
| **Features Used** | 11 (lot size, bedrooms, bathrooms, stories, driveway, rec room, basement, gas HW, AC, garage, preferred area) |
| **Train/Test Split** | 80/20 |
| **Cross-Validation** | 5-fold |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/House-Price-Prediction-System-with-Interactive-Web-Application.git
cd House-Price-Prediction-System-with-Interactive-Web-Application

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlitapps.py
```

The application will open at `http://localhost:8501`

---

## 📁 Project Structure

```
├── .streamlit/
│   └── config.toml          # Theme configuration (dark mode)
├── streamlitapps.py          # Main application (ML + UI)
├── housing.csv               # Dataset (546 properties)
├── requirements.txt          # Python dependencies
├── Housepricepredictionimage.png
├── streamliticon.png
└── README.md
```

---

## 🎨 Tech Stack

- **Frontend**: Streamlit + Custom CSS (Glassmorphism, Gradient animations)
- **Charts**: Plotly (Interactive) + Seaborn (Correlation)
- **ML**: scikit-learn (Gradient Boosting Regressor)
- **Data**: Pandas, NumPy

---

## 📊 Dataset

The dataset contains **546 property records** with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `lotsize` | Lot area in sqft | Continuous |
| `bedrooms` | Number of bedrooms | Discrete |
| `bathrooms` | Number of bathrooms | Discrete |
| `stories` | Number of stories | Discrete |
| `driveway` | Has driveway | Binary (0/1) |
| `recroom` | Has recreation room | Binary (0/1) |
| `fullbase` | Has full basement | Binary (0/1) |
| `gashw` | Gas hot water heating | Binary (0/1) |
| `airco` | Has air conditioning | Binary (0/1) |
| `garagepl` | Garage spaces | Discrete |
| `prefarea` | In preferred area | Binary (0/1) |
| `price` | Property price ($) | Target |

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
