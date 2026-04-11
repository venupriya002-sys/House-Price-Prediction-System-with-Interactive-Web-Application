import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor | ML-Powered",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# PREMIUM CSS INJECTION
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* ── Hide default Streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 50%, #24243e 100%);
        border-right: 1px solid rgba(108, 99, 255, 0.2);
    }

    [data-testid="stSidebar"] .stRadio > label {
        font-size: 0.9rem;
        font-weight: 500;
    }

    /* ── Glass Cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(108, 99, 255, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(108, 99, 255, 0.02));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover {
        border-color: rgba(108, 99, 255, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(108, 99, 255, 0.2);
    }

    .metric-icon {
        font-size: 2rem;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 4px 0;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #8b8fa3;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        margin: 30px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(108, 99, 255, 0.3);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ── Hero Section ── */
    .hero-container {
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.15), rgba(167, 139, 250, 0.05));
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(108, 99, 255, 0.05) 0%, transparent 70%);
        animation: pulse 6s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 1; }
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #6C63FF, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
        position: relative;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #8b8fa3;
        font-weight: 400;
        position: relative;
    }

    /* ── Prediction Result ── */
    .prediction-result {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.03));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        padding: 36px;
        text-align: center;
        animation: fadeInUp 0.6s ease-out;
    }

    .prediction-price {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #22c55e, #4ade80);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 12px 0;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Confidence Badge ── */
    .confidence-badge {
        display: inline-block;
        background: rgba(108, 99, 255, 0.15);
        border: 1px solid rgba(108, 99, 255, 0.3);
        border-radius: 30px;
        padding: 6px 18px;
        font-size: 0.85rem;
        color: #a78bfa;
        font-weight: 500;
    }

    /* ── Streamlit overrides ── */
    .stSelectbox > div > div, .stSlider > div > div > div {
        border-radius: 10px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }

    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #a78bfa) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 36px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(108, 99, 255, 0.4) !important;
    }

    /* ── Dataframe styling ── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
    }

    /* ── Sidebar branding ── */
    .sidebar-brand {
        text-align: center;
        padding: 20px 0 30px 0;
        border-bottom: 1px solid rgba(108, 99, 255, 0.15);
        margin-bottom: 20px;
    }

    .sidebar-brand-title {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6C63FF, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sidebar-brand-sub {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 4px;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* ── Feature importance bars ── */
    .feature-bar {
        display: flex;
        align-items: center;
        margin: 6px 0;
        gap: 10px;
    }

    .feature-name {
        width: 100px;
        font-size: 0.8rem;
        color: #9ca3af;
        text-align: right;
    }

    .feature-fill {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(90deg, #6C63FF, #a78bfa);
        transition: width 0.8s ease;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA LOADING (CACHED)
# ──────────────────────────────────────────────
@st.cache_data
def load_data(remove_outliers=False):
    """Load and clean the housing dataset."""
    data = pd.read_csv("housing.csv", sep=";", engine="python")

    # Strip quotes and whitespace from all cells and column names
    data.columns = data.columns.str.strip().str.replace('"', '', regex=False)
    for col in data.columns:
        if data[col].dtype == object:
            data[col] = data[col].str.strip().str.replace('"', '', regex=False)

    # Use CORRECT column names from the actual CSV
    data.columns = ["lotsize", "bedrooms", "bathrooms", "stories",
                     "driveway", "recroom", "fullbase", "gashw",
                     "airco", "garagepl", "prefarea", "price"]

    # Convert all columns to numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Drop any rows with NaN values after conversion
    data = data.dropna()

    if remove_outliers:
        # Simple IQR based outlier removal for price and lotsize
        for col in ["price", "lotsize"]:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data = data[~((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR)))]

    return data


# ──────────────────────────────────────────────
# MODEL TRAINING (CACHED)
# ──────────────────────────────────────────────
@st.cache_resource
def train_model(_data):
    """Train a Gradient Boosting model on all features."""
    feature_cols = ["lotsize", "bedrooms", "bathrooms", "stories",
                    "driveway", "recroom", "fullbase", "gashw",
                    "airco", "garagepl", "prefarea"]

    X = _data[feature_cols]
    y = _data["price"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Predictions & metrics
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
        "mae": mean_absolute_error(y_test, y_pred_test),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    }

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    metrics["cv_mean"] = cv_scores.mean()
    metrics["cv_std"] = cv_scores.std()

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))

    return model, metrics, importance, X_test, y_test, y_pred_test


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def format_price(price):
    """Format price with currency symbol and commas."""
    if price >= 1000:
        return f"${price:,.0f}"
    return f"${price:,.2f}"


def render_metric_card(icon, value, label):
    """Render a glassmorphism metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ──────────────────────────────────────────────
# LOAD DATA & TRAIN MODEL
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    # Original Branding
    try:
        st.image("streamliticon.png", width=100)
    except:
        pass

    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">House Price AI</div>
        <div class="sidebar-brand-sub">ML-Powered Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "📊 Data Explorer", "🤖 Price Predictor"],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Data Settings
    st.markdown("##### ⚙️ Settings")
    remove_outliers = st.toggle("Remove Outliers", value=False, help="Filter out extreme values using IQR method to improve model stability.")

    # LOAD DATA & TRAIN MODEL
    try:
        data = load_data(remove_outliers=remove_outliers)
        model, metrics, importance, X_test, y_test, y_pred_test = train_model(data)
        data_loaded = True
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        data_loaded = False

    if not data_loaded:
        st.stop()

    st.markdown("---")

    # Dataset info
    st.markdown("##### 📁 Dataset Info")
    st.caption(f"**Rows:** {len(data)}")
    st.caption(f"**Features:** {len(data.columns) - 1}")
    st.caption(f"**Target:** Price")

    st.markdown("---")

    # Model info
    st.markdown("##### 🧠 Model Info")
    st.caption(f"**Model:** Gradient Boosting")
    st.caption(f"**R² Score:** {metrics['r2_test']:.3f}")
    st.caption(f"**CV Score:** {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f}")

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit & ML")


# ──────────────────────────────────────────────
# PAGE: DASHBOARD
# ──────────────────────────────────────────────
if page == "🏠 Dashboard":

    # Hero section
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🏠 House Price Prediction</div>
        <div class="hero-subtitle">
            AI-powered price prediction using Gradient Boosting on 11 property features
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(render_metric_card(
            "🏘️", f"{len(data)}", "Total Properties"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(render_metric_card(
            "💰", format_price(data["price"].mean()), "Avg Price"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(render_metric_card(
            "📐", f"{data['lotsize'].mean():,.0f} sqft", "Avg Lot Size"
        ), unsafe_allow_html=True)

    with col4:
        st.markdown(render_metric_card(
            "🎯", f"{metrics['r2_test']:.1%}", "Model Accuracy"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="section-header">💰 Price Distribution</div>',
                    unsafe_allow_html=True)

        fig = px.histogram(
            data, x="price", nbins=40,
            color_discrete_sequence=["#6C63FF"],
            template="plotly_dark",
            labels={"price": "Price ($)", "count": "Count"}
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
            showlegend=False,
            bargap=0.05
        )
        fig.update_traces(
            marker=dict(line=dict(width=0.5, color='rgba(108,99,255,0.5)')),
            hovertemplate="<b>Price:</b> $%{x:,.0f}<br><b>Count:</b> %{y}<extra></extra>"
        )
        st.plotly_chart(fig, width="stretch")

    with chart_col2:
        st.markdown('<div class="section-header">📐 Price vs Lot Size</div>',
                    unsafe_allow_html=True)

        fig = px.scatter(
            data, x="lotsize", y="price",
            color="bedrooms",
            color_continuous_scale=["#312e81", "#6C63FF", "#a78bfa", "#c4b5fd"],
            template="plotly_dark",
            labels={"lotsize": "Lot Size (sqft)", "price": "Price ($)", "bedrooms": "Beds"},
            hover_data={"bathrooms": True, "stories": True}
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9ca3af"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=350,
        )
        fig.update_traces(
            marker=dict(size=6, opacity=0.7,
                        line=dict(width=0.5, color='rgba(255,255,255,0.2)')),
            hovertemplate="<b>Lot:</b> %{x:,} sqft<br><b>Price:</b> $%{y:,.0f}<br>"
                          "<b>Beds:</b> %{marker.color}<extra></extra>"
        )
        st.plotly_chart(fig, width="stretch")

    # Correlation Heatmap
    st.markdown('<div class="section-header">🔗 Feature Correlation Matrix</div>',
                unsafe_allow_html=True)

    corr = data.corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=["#1a1a2e", "#312e81", "#6C63FF", "#a78bfa", "#ddd6fe"],
        template="plotly_dark",
        aspect="auto"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", size=10),
        margin=dict(l=20, r=20, t=20, b=20),
        height=500,
    )
    st.plotly_chart(fig, width="stretch")

    # Feature Importance
    st.markdown('<div class="section-header">⭐ Feature Importance (Model Weights)</div>',
                unsafe_allow_html=True)

    imp_df = pd.DataFrame({
        "Feature": list(importance.keys()),
        "Importance": list(importance.values())
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#312e81", "#6C63FF", "#a78bfa"],
        template="plotly_dark"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Importance Score"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        showlegend=False,
        coloraxis_showscale=False
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    )
    st.plotly_chart(fig, width="stretch")


# ──────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ──────────────────────────────────────────────
elif page == "📊 Data Explorer":

    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📊 Data Explorer</div>
        <div class="hero-subtitle">
            Interactively explore and filter the housing dataset
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Filters
    st.markdown('<div class="section-header">🔍 Filter Data</div>',
                unsafe_allow_html=True)

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

    with filter_col1:
        area_range = st.slider(
            "Lot Size (sqft)",
            min_value=int(data["lotsize"].min()),
            max_value=int(data["lotsize"].max()),
            value=(int(data["lotsize"].min()), int(data["lotsize"].max())),
            step=100
        )

    with filter_col2:
        bed_filter = st.multiselect(
            "Bedrooms",
            options=sorted(data["bedrooms"].unique().astype(int)),
            default=sorted(data["bedrooms"].unique().astype(int))
        )

    with filter_col3:
        bath_filter = st.multiselect(
            "Bathrooms",
            options=sorted(data["bathrooms"].unique().astype(int)),
            default=sorted(data["bathrooms"].unique().astype(int))
        )

    with filter_col4:
        stories_filter = st.multiselect(
            "Stories",
            options=sorted(data["stories"].unique().astype(int)),
            default=sorted(data["stories"].unique().astype(int))
        )

    # Apply filters
    filtered = data[
        (data["lotsize"] >= area_range[0]) &
        (data["lotsize"] <= area_range[1]) &
        (data["bedrooms"].isin(bed_filter)) &
        (data["bathrooms"].isin(bath_filter)) &
        (data["stories"].isin(stories_filter))
    ]

    # Filtered stats
    st.markdown(f"**Showing {len(filtered)} of {len(data)} properties**")

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(render_metric_card("📊", f"{len(filtered)}", "Properties"), unsafe_allow_html=True)
    with s2:
        st.markdown(render_metric_card("💰", format_price(filtered["price"].mean()) if len(filtered) > 0 else "$0", "Avg Price"), unsafe_allow_html=True)
    with s3:
        st.markdown(render_metric_card("📈", format_price(filtered["price"].max()) if len(filtered) > 0 else "$0", "Max Price"), unsafe_allow_html=True)
    with s4:
        st.markdown(render_metric_card("📉", format_price(filtered["price"].min()) if len(filtered) > 0 else "$0", "Min Price"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Interactive charts
    tab1, tab2, tab3 = st.tabs(["📈 Scatter Plot", "📊 Bar Charts", "📋 Raw Data"])

    with tab1:
        sc1, sc2 = st.columns(2)
        with sc1:
            x_axis = st.selectbox("X-Axis", ["lotsize", "bedrooms", "bathrooms", "stories", "garagepl"], index=0)
        with sc2:
            color_by = st.selectbox("Color By", ["bedrooms", "stories", "bathrooms", "airco", "prefarea"], index=0)

        if len(filtered) > 0:
            fig = px.scatter(
                filtered, x=x_axis, y="price",
                color=color_by,
                color_continuous_scale=["#312e81", "#6C63FF", "#a78bfa", "#c4b5fd"],
                template="plotly_dark",
                labels={"price": "Price ($)"},
                hover_data=["lotsize", "bedrooms", "bathrooms"]
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9ca3af"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=20, r=20, t=20, b=20),
                height=450,
            )
            fig.update_traces(marker=dict(size=7, opacity=0.75,
                                          line=dict(width=0.5, color='rgba(255,255,255,0.2)')))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No data matches your filters. Adjust the filters above.")

    with tab2:
        bar_col1, bar_col2 = st.columns(2)

        with bar_col1:
            st.markdown("**Average Price by Bedrooms**")
            if len(filtered) > 0:
                avg_by_bed = filtered.groupby("bedrooms")["price"].mean().reset_index()
                fig = px.bar(
                    avg_by_bed, x="bedrooms", y="price",
                    color="price",
                    color_continuous_scale=["#312e81", "#6C63FF", "#a78bfa"],
                    template="plotly_dark",
                    labels={"price": "Avg Price ($)", "bedrooms": "Bedrooms"}
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9ca3af"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=350,
                    showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, width="stretch")

        with bar_col2:
            st.markdown("**Average Price by Stories**")
            if len(filtered) > 0:
                avg_by_stories = filtered.groupby("stories")["price"].mean().reset_index()
                fig = px.bar(
                    avg_by_stories, x="stories", y="price",
                    color="price",
                    color_continuous_scale=["#312e81", "#6C63FF", "#a78bfa"],
                    template="plotly_dark",
                    labels={"price": "Avg Price ($)", "stories": "Stories"}
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9ca3af"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=350,
                    showlegend=False,
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig, width="stretch")

    with tab3:
        st.markdown("**Filtered Dataset**")
        st.dataframe(
            filtered.style.format({"price": "${:,.0f}", "lotsize": "{:,.0f} sqft"}),
            width="stretch",
            height=500
        )

        # Download button
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_housing_data.csv",
            mime="text/csv"
        )

    # Statistical Summary
    st.markdown('<div class="section-header">📊 Statistical Summary</div>',
                unsafe_allow_html=True)
    st.dataframe(
        filtered.describe().round(2),
        width="stretch"
    )


# ──────────────────────────────────────────────
# PAGE: PRICE PREDICTOR
# ──────────────────────────────────────────────
elif page == "🤖 Price Predictor":

    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🤖 AI Price Predictor</div>
        <div class="hero-subtitle">
            Enter property details to get an ML-powered price estimate using Gradient Boosting
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Performance section
    st.markdown('<div class="section-header">📈 Model Performance</div>',
                unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(render_metric_card(
            "🎯", f"{metrics['r2_test']:.1%}", "Test R² Score"
        ), unsafe_allow_html=True)
    with m2:
        st.markdown(render_metric_card(
            "📉", format_price(metrics['mae']), "Mean Abs Error"
        ), unsafe_allow_html=True)
    with m3:
        st.markdown(render_metric_card(
            "📊", format_price(metrics['rmse']), "RMSE"
        ), unsafe_allow_html=True)
    with m4:
        st.markdown(render_metric_card(
            "🔄", f"{metrics['cv_mean']:.1%}", "Cross-Val Score"
        ), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted chart
    st.markdown('<div class="section-header">🔍 Actual vs Predicted Prices</div>',
                unsafe_allow_html=True)

    compare_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": y_pred_test
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=compare_df["Actual"], y=compare_df["Predicted"],
        mode="markers",
        marker=dict(
            size=8, opacity=0.7,
            color="#6C63FF",
            line=dict(width=1, color='rgba(255,255,255,0.2)')
        ),
        name="Predictions",
        hovertemplate="<b>Actual:</b> $%{x:,.0f}<br><b>Predicted:</b> $%{y:,.0f}<extra></extra>"
    ))

    # Perfect prediction line
    min_val = min(compare_df["Actual"].min(), compare_df["Predicted"].min())
    max_val = max(compare_df["Actual"].max(), compare_df["Predicted"].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines",
        line=dict(color="#22c55e", dash="dash", width=2),
        name="Perfect Prediction",
        hoverinfo="skip"
    ))

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af"),
        xaxis=dict(title="Actual Price ($)", gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(title="Predicted Price ($)", gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11)
        )
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Prediction Form
    st.markdown('<div class="section-header">🏡 Property Details</div>',
                unsafe_allow_html=True)

    form_col1, form_col2, form_col3 = st.columns(3)

    with form_col1:
        input_lotsize = st.number_input(
            "📐 Lot Size (sqft)",
            min_value=int(data["lotsize"].min()),
            max_value=int(data["lotsize"].max()),
            value=int(data["lotsize"].median()),
            step=100,
            help="Total lot area in square feet"
        )

        input_bedrooms = st.selectbox(
            "🛏️ Bedrooms",
            options=[1, 2, 3, 4, 5, 6],
            index=2,
            help="Number of bedrooms"
        )

        input_bathrooms = st.selectbox(
            "🚿 Bathrooms",
            options=[1, 2, 3, 4],
            index=0,
            help="Number of bathrooms"
        )

        input_stories = st.selectbox(
            "🏢 Stories",
            options=[1, 2, 3, 4],
            index=0,
            help="Number of stories/floors"
        )

    with form_col2:
        input_driveway = st.selectbox(
            "🚗 Driveway",
            options=["No", "Yes"],
            index=1,
            help="Does the property have a driveway?"
        )

        input_recroom = st.selectbox(
            "🎮 Recreation Room",
            options=["No", "Yes"],
            index=0,
            help="Does it have a recreation room?"
        )

        input_fullbase = st.selectbox(
            "🏗️ Full Basement",
            options=["No", "Yes"],
            index=0,
            help="Does it have a full basement?"
        )

        input_gashw = st.selectbox(
            "🔥 Gas Hot Water",
            options=["No", "Yes"],
            index=0,
            help="Gas hot water heating?"
        )

    with form_col3:
        input_airco = st.selectbox(
            "❄️ Air Conditioning",
            options=["No", "Yes"],
            index=0,
            help="Does it have air conditioning?"
        )

        input_garagepl = st.selectbox(
            "🅿️ Garage Spaces",
            options=[0, 1, 2, 3],
            index=0,
            help="Number of garage parking spaces"
        )

        input_prefarea = st.selectbox(
            "⭐ Preferred Area",
            options=["No", "Yes"],
            index=0,
            help="Is the property in a preferred neighbourhood?"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict button
    if st.button("🔮 Predict House Price", width="stretch"):

        # Convert Yes/No to 1/0
        input_features = np.array([[
            input_lotsize,
            input_bedrooms,
            input_bathrooms,
            input_stories,
            1 if input_driveway == "Yes" else 0,
            1 if input_recroom == "Yes" else 0,
            1 if input_fullbase == "Yes" else 0,
            1 if input_gashw == "Yes" else 0,
            1 if input_airco == "Yes" else 0,
            input_garagepl,
            1 if input_prefarea == "Yes" else 0,
        ]])

        try:
            prediction = model.predict(input_features)[0]

            # Calculate confidence based on how close features are to training data
            mae_pct = (metrics['mae'] / data['price'].mean()) * 100
            confidence = max(0, min(100, 100 - mae_pct))

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="prediction-result">
                <div style="font-size: 1rem; color: #9ca3af; margin-bottom: 8px;">
                    Estimated Property Value
                </div>
                <div class="prediction-price">
                    {format_price(max(0, prediction))}
                </div>
                <div style="margin-top: 12px;">
                    <span class="confidence-badge">
                        🎯 Model Confidence: {confidence:.0f}%
                    </span>
                </div>
                <div style="color: #6b7280; font-size: 0.8rem; margin-top: 16px;">
                    Based on Gradient Boosting model trained on {len(data)} properties
                    <br>
                    Price range: {format_price(max(0, prediction - metrics['mae']))} 
                    — {format_price(prediction + metrics['mae'])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature contribution breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">📊 What Influences This Price?</div>',
                        unsafe_allow_html=True)

            # Show feature contributions using importance × feature value rank
            features_used = {
                "📐 Lot Size": input_lotsize,
                "🛏️ Bedrooms": input_bedrooms,
                "🚿 Bathrooms": input_bathrooms,
                "🏢 Stories": input_stories,
                "🚗 Driveway": 1 if input_driveway == "Yes" else 0,
                "🎮 Rec Room": 1 if input_recroom == "Yes" else 0,
                "🏗️ Basement": 1 if input_fullbase == "Yes" else 0,
                "🔥 Gas HW": 1 if input_gashw == "Yes" else 0,
                "❄️ AC": 1 if input_airco == "Yes" else 0,
                "🅿️ Garage": input_garagepl,
                "⭐ Pref Area": 1 if input_prefarea == "Yes" else 0,
            }

            imp_keys = list(importance.keys())
            contrib_df = pd.DataFrame({
                "Feature": list(features_used.keys()),
                "Your Value": list(features_used.values()),
                "Importance": [importance[k] for k in imp_keys]
            }).sort_values("Importance", ascending=True)

            fig = px.bar(
                contrib_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#1e3a5f", "#6C63FF", "#22c55e"],
                template="plotly_dark",
                hover_data={"Your Value": True}
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9ca3af"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=350,
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, width="stretch")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
