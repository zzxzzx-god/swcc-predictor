# app_combined_enhanced_with_VG_fitting.py - Biochar-amended soil SWCC prediction system (Enhanced, with VG model fitting)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import font_manager
import io
import warnings
from scipy.optimize import curve_fit
import json
import os

warnings.filterwarnings('ignore')

# Set matplotlib fonts (placed immediately after imports)
import matplotlib


def _detect_cjk_font():
    """Detect available CJK fonts in the current environment."""
    candidate_fonts = [
        'Noto Sans CJK JP',
        'Noto Serif CJK JP',
        'Microsoft YaHei',
        'SimHei',
        'WenQuanYi Micro Hei',
        'WenQuanYi Zen Hei',
        'PingFang SC',
        'Heiti SC',
        'Source Han Sans CN',
        'Source Han Sans SC',
        'Arial Unicode MS',
        'AR PL KaitiM GB'
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidate_fonts:
        if font_name in available_fonts:
            return font_name
    return None


CJK_FONT_NAME = _detect_cjk_font()
MATPLOTLIB_FONT_FALLBACKS = [
    font_name for font_name in [
        CJK_FONT_NAME,
        'Microsoft YaHei',
        'SimHei',
        'WenQuanYi Micro Hei',
        'Noto Sans CJK JP',
        'Arial Unicode MS',
        'DejaVu Sans'
    ] if font_name
]
HAS_CJK_FONT = CJK_FONT_NAME is not None


def configure_matplotlib_fonts():
    """Unified matplotlib font settings for better Chinese display."""
    matplotlib.rcParams['font.sans-serif'] = MATPLOTLIB_FONT_FALLBACKS
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = MATPLOTLIB_FONT_FALLBACKS
    plt.rcParams['axes.unicode_minus'] = False


configure_matplotlib_fonts()


def plot_text(chinese_text, english_text=None):
    """Prefer Chinese in plots; fallback to English if no CJK font available."""
    return chinese_text if HAS_CJK_FONT or english_text is None else english_text


# Set page configuration
st.set_page_config(
    page_title="Biochar-amended Soil SWCC Prediction System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3CB371;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4682B4;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fffacd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #32CD32;
        margin: 10px 0;
    }
    .stButton > button {
        background-color: #2E8B57;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #3CB371;
        transform: scale(1.05);
    }
    .model-selector {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #87CEEB;
        margin-bottom: 20px;
    }
    .stNumberInput input {
        font-size: 14px;
    }
    .batch-results {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
    }
    .parameter-table {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
    }
    .vg-equation {
        font-family: "Times New Roman", Times, serif;
        font-size: 1.2rem;
        text-align: center;
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border: 1px solid #ccc;
    }
</style>
""", unsafe_allow_html=True)

# Biochar type mapping (display English, internal Chinese for model compatibility)
BIOCHAR_TYPE_DISPLAY = ["Agricultural waste", "Forestry residue", "Livestock manure", "Municipal sludge", "Other"]
BIOCHAR_TYPE_INTERNAL = ["农业废弃物", "林业残余物", "畜禽粪便", "城市污泥", "其他"]
BIOCHAR_TYPE_MAP = dict(zip(BIOCHAR_TYPE_DISPLAY, BIOCHAR_TYPE_INTERNAL))
BIOCHAR_TYPE_REVERSE_MAP = dict(zip(BIOCHAR_TYPE_INTERNAL, BIOCHAR_TYPE_DISPLAY))


# VG model function definition
def vg_model(h, theta_r, theta_s, alpha, n):
    """
    van Genuchten model
    θ = θr + (θs - θr) / [1 + (α·h)^n]^m
    where m = 1 - 1/n
    """
    m = 1 - 1 / n
    return theta_r + (theta_s - theta_r) / ((1 + (alpha * h) ** n) ** m)


def fit_vg_model(suction_data, theta_data, initial_guess=None):
    """
    Fit SWCC data with VG model

    Parameters:
    - suction_data: suction data (kPa)
    - theta_data: water content data
    - initial_guess: initial guess parameters [θr, θs, α, n]

    Returns:
    - popt: optimal fitting parameters
    - pcov: covariance matrix of parameters
    - r_squared: coefficient of determination R²
    - fitted_theta: fitted values
    """
    # Default initial guess
    if initial_guess is None:
        # θr: 90% of minimum water content
        # θs: 110% of maximum water content
        # α: 1/median suction
        # n: typical value 1.5
        theta_min = np.min(theta_data)
        theta_max = np.max(theta_data)
        suction_median = np.median(suction_data[suction_data > 0])

        initial_guess = [
            max(0, theta_min * 0.9),  # θr
            min(0.5, theta_max * 1.1),  # θs
            1.0 / suction_median if suction_median > 0 else 0.01,  # α
            1.5  # n
        ]

    # Parameter bounds
    lower_bounds = [0, 0, 0.00001, 1.01]  # n must be > 1
    upper_bounds = [0.5, 0.6, 10, 10]  # Reasonable range

    try:
        # Use curve_fit for fitting
        popt, pcov = curve_fit(
            vg_model,
            suction_data,
            theta_data,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000
        )

        # Calculate fitted values
        fitted_theta = vg_model(suction_data, *popt)

        # Calculate R²
        residuals = theta_data - fitted_theta
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((theta_data - np.mean(theta_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return popt, pcov, r_squared, fitted_theta

    except Exception as e:
        st.warning(f"VG model fitting failed: {e}")
        return None, None, 0, None


def plot_swcc_with_vg_fit(suction_range, predictions, vg_params=None, current_point=None):
    """Plot SWCC curve and VG model fitting results"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ensure Chinese fonts are used
    configure_matplotlib_fonts()

    # Main plot: SWCC curve
    ax.plot(suction_range, predictions, 'b-', linewidth=2,
            label=plot_text('SWCC（XGBoost预测曲线）', 'SWCC (XGBoost prediction curve)'))

    # If VG fitting parameters are provided, plot the fitted curve
    if vg_params is not None:
        theta_r, theta_s, alpha, n = vg_params
        m = 1 - 1 / n
        fitted_curve = vg_model(suction_range, theta_r, theta_s, alpha, n)
        ax.plot(suction_range, fitted_curve, 'r--', linewidth=2, label=plot_text('VG拟合曲线', 'VG fitted curve'))

        # Add VG equation to the plot
        vg_eq = plot_text('VG：θ = θr + (θs - θr) / [1 + (αh)^n]^m', 'VG: θ = θr + (θs - θr) / [1 + (αh)^n]^m')
        ax.text(0.02, 0.98, vg_eq, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # If current point is provided, mark it on the plot
    if current_point:
        ax.plot(current_point[0], current_point[1], 'ro', markersize=10,
                label=plot_text('当前单点预测', 'Current single-point prediction'))
        ax.annotate(f'({current_point[0]:.1f} kPa, {current_point[1]:.3f})',
                    xy=current_point,
                    xytext=(current_point[0] * 1.5, current_point[1] * 0.9),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

    # Set axis labels
    ax.set_xscale('log')
    ax.set_xlabel(plot_text('吸力 (kPa)', 'Suction (kPa)'), fontsize=12)
    ax.set_ylabel(plot_text('体积含水率', 'Volumetric water content'), fontsize=12)
    ax.set_title(plot_text('SWCC 与 VG 拟合结果', 'SWCC and VG fitting results'), fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.set_facecolor('#f8f9fa')

    # Set axis ranges
    ax.set_xlim(min(suction_range), max(suction_range))
    y_min = max(0, min(predictions) - 0.05)
    y_max = min(1, max(predictions) + 0.05)
    ax.set_ylim(y_min, y_max)

    # Suction range annotation
    ax.text(0.02, 0.02, plot_text(f'吸力范围: {min(suction_range):.2f} - {max(suction_range):.0f} kPa',
                                  f'Suction range: {min(suction_range):.2f} - {max(suction_range):.0f} kPa'),
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    return fig


def display_vg_parameters(popt, r_squared, suction_range, theta_data):
    """Display VG model parameters"""
    if popt is None:
        st.warning("VG model fitting failed, cannot display parameters")
        return

    theta_r, theta_s, alpha, n = popt
    m = 1 - 1 / n

    # Create parameter table
    st.markdown('<div class="sub-header">📊 VG Model Fitting Parameters</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="parameter-table">', unsafe_allow_html=True)
        st.markdown("##### Model Parameters")

        param_data = {
            'Parameter': ['θr (residual water content)', 'θs (saturated water content)', 'α (inverse of suction)',
                          'n (shape parameter)', 'm (=1-1/n)',
                          'R² (coefficient of determination)'],
            'Value': [
                f"{theta_r:.6f}",
                f"{theta_s:.6f}",
                f"{alpha:.6f}",
                f"{n:.6f}",
                f"{m:.6f}",
                f"{r_squared:.6f}"
            ],
            'Physical Meaning': [
                'Minimum water content at high suction',
                'Maximum water content at zero suction',
                'Inverse of air entry value',
                'Pore size distribution index',
                'Curve shape parameter',
                'Goodness of fit (1 = perfect fit)'
            ]
        }

        param_df = pd.DataFrame(param_data)
        st.dataframe(param_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="parameter-table">', unsafe_allow_html=True)
        st.markdown("##### Characteristic Suction Values")

        # Calculate characteristic suction
        # Air entry value
        ha = 1 / alpha if alpha > 0 else 0

        # Suction at effective saturation of 0.5
        se = 0.5
        h50 = (1 / alpha) * ((1 / se ** (1 / m)) - 1) ** (1 / n) if alpha > 0 and m > 0 and n > 0 else 0

        feature_data = {
            'Characteristic Point': ['Air entry value ha', 'Suction at Se=0.5 h₅₀', 'Min predicted suction',
                                     'Max predicted suction', 'Number of data points'],
            'Suction (kPa)': [
                f"{ha:.3f}",
                f"{h50:.3f}",
                f"{np.min(suction_range):.3f}",
                f"{np.max(suction_range):.3f}",
                f"{len(suction_range)}"
            ],
            'Note': [
                '1/α',
                'Se = (θ-θr)/(θs-θr) = 0.5',
                'Curve start point',
                'Curve end point',
                'SWCC curve points'
            ]
        }

        feature_df = pd.DataFrame(feature_data)
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Display VG model equation
    st.markdown('<div class="vg-equation">', unsafe_allow_html=True)
    st.markdown("### van Genuchten (VG) Model Equation")
    st.latex(r'''
    \theta(h) = \theta_r + \frac{\theta_s - \theta_r}{\left[1 + (\alpha \cdot h)^n\right]^m}
    ''')
    st.markdown(f'''
    Where:
    - θ(h): Volumetric water content at suction h
    - θr = {theta_r:.4f} (residual water content)
    - θs = {theta_s:.4f} (saturated water content)
    - α = {alpha:.6f} kPa⁻¹
    - n = {n:.4f}
    - m = 1 - 1/n = {m:.4f}
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

    # Provide parameter download
    vg_params_dict = {
        'theta_r': theta_r,
        'theta_s': theta_s,
        'alpha': alpha,
        'n': n,
        'm': m,
        'R_squared': r_squared,
        'ha': ha,
        'h50': h50
    }

    vg_params_df = pd.DataFrame([vg_params_dict])

    csv_params = vg_params_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download VG Model Parameters",
        data=csv_params,
        file_name=f"VG_model_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


# Load models
@st.cache_resource
def load_models():
    """Load two trained XGBoost models"""
    models = {}

    # Try multiple encoding options for pickle files
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']

    for model_name in ['group1', 'group2']:
        model_path = f'xgboost_optimized_results/model_{model_name}.pkl'

        if not os.path.exists(model_path):
            st.sidebar.warning(f"⚠️ Model file not found: {model_path}")
            continue

        try:
            # Try loading with different encodings
            loaded = False
            for encoding in encodings_to_try:
                try:
                    with open(model_path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                    st.sidebar.success(f"✅ Group {model_name[-1]} model loaded successfully!")
                    loaded = True
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Error loading with {encoding}: {str(e)[:100]}")
                    continue

            if not loaded:
                st.sidebar.warning(f"⚠️ Could not load model {model_name} with any encoding")

        except Exception as e:
            st.sidebar.warning(f"⚠️ Group {model_name[-1]} model loading failed: {e}")

    return models


# Load feature information
@st.cache_resource
def load_feature_info():
    """Load feature information file with robust encoding handling"""
    feature_info_path = 'xgboost_optimized_results/feature_info.json'

    # Default feature info with Chinese categories
    default_info = {
        'group1': {
            'feature_names': ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature',
                              'Biochar_type_combined'],
            'feature_order': ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature',
                              'Biochar_type_combined'],
            'biochar_categories': BIOCHAR_TYPE_INTERNAL.copy()
        },
        'group2': {
            'feature_names': ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT'],
            'feature_order': ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']
        }
    }

    if not os.path.exists(feature_info_path):
        st.sidebar.info("ℹ️ feature_info.json not found, using default values")
        return default_info

    try:
        # Try UTF-8 first
        try:
            with open(feature_info_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
        except UnicodeDecodeError:
            # Try other encodings
            encodings_to_try = ['latin-1', 'cp1252', 'ISO-8859-1']
            info = None
            for encoding in encodings_to_try:
                try:
                    with open(feature_info_path, 'r', encoding=encoding) as f:
                        info = json.load(f)
                    break
                except:
                    continue

            if info is None:
                st.sidebar.warning("⚠️ Could not decode feature_info.json, using default values")
                return default_info

        # Ensure biochar categories are in Chinese (for model compatibility)
        if 'group1' in info and 'biochar_categories' in info['group1']:
            categories = info['group1']['biochar_categories']
            # If categories are in English, convert to Chinese
            if any(cat in categories for cat in BIOCHAR_TYPE_DISPLAY):
                info['group1']['biochar_categories'] = BIOCHAR_TYPE_INTERNAL.copy()
            # If categories are empty or invalid, use defaults
            elif not categories or len(categories) == 0:
                info['group1']['biochar_categories'] = BIOCHAR_TYPE_INTERNAL.copy()

        return info

    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading feature_info.json: {e}, using default values")
        return default_info


def generate_swcc_curve(model, model_type, base_input, suction_range):
    """Generate SWCC curve data"""
    predictions = []

    for suction in suction_range:
        # Copy base input data
        input_data = base_input.copy()
        input_data['suction'] = suction

        # Create feature DataFrame
        if model_type == 'group1':
            # Group 1: use categorical features
            feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature', 'Biochar_type_combined']
            biochar_categories = BIOCHAR_TYPE_INTERNAL

            features_df = pd.DataFrame([input_data])
            features_df['Biochar_type_combined'] = pd.Categorical(
                features_df['Biochar_type_combined'],
                categories=biochar_categories
            )
            features_df = features_df[feature_order]
        else:
            # Group 2: use raw values directly
            feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']
            features_df = pd.DataFrame([input_data])[feature_order]

        # Make prediction
        prediction = model.predict(features_df)[0]
        predictions.append(prediction)

    return predictions


def batch_predict_group1(model, data_df, feature_info):
    """Batch prediction - Group 1"""
    predictions = []

    # Get feature information and categories (always use Chinese internal categories)
    biochar_categories = BIOCHAR_TYPE_INTERNAL

    for idx, row in data_df.iterrows():
        try:
            # Prepare input data
            input_data = {
                'suction': float(row.get('suction', 0)),
                'clay': float(row.get('clay', 0)),
                'silt': float(row.get('silt', 0)),
                'sand': float(row.get('sand', 0)),
                'dd': float(row.get('dd', 0)),
                'BC': float(row.get('BC', 0)) / 100.0,  # Convert to decimal
                'temperature': float(row.get('temperature', 0)),
                'Biochar_type_combined': str(row.get('Biochar_type_combined', '农业废弃物'))
            }

            # Handle case where biochar type is in English display format
            if input_data['Biochar_type_combined'] in BIOCHAR_TYPE_DISPLAY:
                input_data['Biochar_type_combined'] = BIOCHAR_TYPE_MAP[input_data['Biochar_type_combined']]

            # When BC=0, adjust parameters
            if input_data['BC'] == 0:
                input_data['temperature'] = 0.0
                input_data['Biochar_type_combined'] = '农业废弃物'  # Default value

            # Create DataFrame
            features_df = pd.DataFrame([input_data])
            features_df['Biochar_type_combined'] = pd.Categorical(
                features_df['Biochar_type_combined'],
                categories=biochar_categories
            )

            # Arrange by feature order
            feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature', 'Biochar_type_combined']
            features_df = features_df[feature_order]

            # Make prediction
            prediction = model.predict(features_df)[0]
            predictions.append(prediction)

        except Exception as e:
            st.warning(f"Row {idx + 1} prediction failed: {e}")
            predictions.append(np.nan)

    return predictions


def batch_predict_group2(model, data_df, feature_info):
    """Batch prediction - Group 2"""
    predictions = []

    for idx, row in data_df.iterrows():
        try:
            # Prepare input data
            bc = float(row.get('BC', 0)) / 100.0  # Convert to decimal

            # Handle BC=0 case
            if bc == 0:
                ph = 0.0
                at = 0.0
                ct = 0.0
            else:
                ph = float(row.get('pH', 8.0))
                at = float(row.get('AT', 25.0))
                ct = float(row.get('CT', 60.0))

            # Create feature list
            features = [
                float(row.get('suction', 100.0)),
                float(row.get('clay', 0.2)),
                float(row.get('silt', 0.25)),
                float(row.get('sand', 0.55)),
                float(row.get('dd', 1.45)),
                bc,  # Decimal form
                ph,
                at,  # Percentage form
                ct  # Percentage form
            ]

            # Create DataFrame
            feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']
            features_df = pd.DataFrame([features], columns=feature_order)

            # Make prediction
            prediction = model.predict(features_df)[0]
            predictions.append(prediction)

        except Exception as e:
            st.warning(f"Row {idx + 1} prediction failed: {e}")
            predictions.append(np.nan)

    return predictions


def validate_batch_data(data_df, model_type, feature_info):
    """Validate batch data"""
    errors = []

    if model_type == 'group1':
        required_columns = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature', 'Biochar_type_combined']
        biochar_categories = BIOCHAR_TYPE_INTERNAL
    else:
        required_columns = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']

    # Check required columns
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    # Check data types
    for col in required_columns:
        if col in data_df.columns:
            # Try to convert to numeric (except biochar type)
            if col != 'Biochar_type_combined':
                try:
                    data_df[col] = pd.to_numeric(data_df[col])
                except:
                    errors.append(f"Column '{col}' contains non-numeric data")

    # Check if biochar type is valid (accept both display and internal formats)
    if model_type == 'group1' and 'Biochar_type_combined' in data_df.columns:
        valid_types = set(BIOCHAR_TYPE_INTERNAL + BIOCHAR_TYPE_DISPLAY)
        invalid_types = data_df[~data_df['Biochar_type_combined'].isin(valid_types)][
            'Biochar_type_combined'].unique()
        if len(invalid_types) > 0:
            errors.append(f"Invalid biochar types: {', '.join(invalid_types)}")

    # Check soil particle composition sum
    if all(col in data_df.columns for col in ['clay', 'silt', 'sand']):
        data_df['total_particles'] = data_df['clay'] + data_df['silt'] + data_df['sand']
        invalid_rows = data_df[(data_df['total_particles'] > 1.0) | (data_df['total_particles'] < 0)].index.tolist()
        if invalid_rows:
            errors.append(f"Rows {[i + 1 for i in invalid_rows]} have soil particle composition sum outside 0-1 range")

    return errors


def main():
    """Main application function"""
    # App title
    st.markdown('<div class="main-header">🌱 Biochar-amended Soil Water Retention Curve (SWCC) Prediction System</div>',
                unsafe_allow_html=True)

    # Load models and feature information
    models = load_models()
    feature_info = load_feature_info()

    if not models:
        st.error("❌ No available models, please check model files")
        return

    # Sidebar - Model selection and system information
    with st.sidebar:
        st.title("🔧 System Settings")

        # Prediction mode selection
        st.markdown("### 📊 Select Prediction Mode")
        prediction_mode = st.radio(
            "Select prediction mode",
            ["Single-point Prediction", "Batch Prediction"],
            index=0
        )

        # Model selection
        st.markdown("### 🤖 Select Prediction Model")
        model_options = []
        if 'group1' in models:
            model_options.append("Group 1: Biochar type and pyrolysis temperature")
        if 'group2' in models:
            model_options.append("Group 2: Biochar physicochemical indicators")

        if not model_options:
            st.error("No available models")
            st.stop()

        selected_model = st.radio(
            "Select model to use",
            model_options,
            index=0
        )

        # Determine model type
        if "Group 1" in selected_model:
            model_type = 'group1'
            model_info = feature_info.get('group1', {})
            st.info("Using Group 1: Includes biochar type and pyrolysis temperature")
        else:
            model_type = 'group2'
            model_info = feature_info.get('group2', {})
            st.info("Using Group 2: Includes biochar physicochemical indicators (pH, AT, CT)")

        st.divider()

        # SWCC curve settings (only displayed in single-point prediction mode)
        if prediction_mode == "Single-point Prediction":
            st.markdown("### 📈 SWCC Curve Settings")

            # VG model fitting options
            st.markdown("#### 🔧 VG Model Fitting Options")
            enable_vg_fitting = st.checkbox("Enable VG Model Fitting", value=True,
                                            help="Fit the generated SWCC curve with the van Genuchten model",
                                            key="enable_vg_fitting")

            curve_points = st.slider(
                "Number of curve points",
                min_value=20,
                max_value=200,
                value=100,
                help="Number of points on the SWCC curve",
                key="curve_points"
            )

            min_suction = st.number_input(
                "Minimum suction (kPa)",
                min_value=0.001,
                max_value=1000.0,
                value=0.01,
                step=0.01,
                format="%.3f",
                help="Minimum suction value for the SWCC curve",
                key="min_suction"
            )

            max_suction = st.number_input(
                "Maximum suction (kPa)",
                min_value=100.0,
                max_value=10000000.0,
                value=284804.0,
                step=100.0,
                help="Maximum suction value for the SWCC curve",
                key="max_suction"
            )

            # Check if max_suction is greater than min_suction
            if max_suction <= min_suction:
                st.warning("Maximum suction must be greater than minimum suction, auto-adjusting")
                max_suction = min_suction * 100
                st.session_state['max_suction'] = max_suction

        else:
            # Batch prediction settings
            st.markdown("### 📊 Batch Prediction Settings")
            st.info("Upload a file containing multiple parameter sets for batch prediction")

    # Main content area
    st.markdown(f"""
    <div class="info-box">
    <strong>📖 Current Mode:</strong> {prediction_mode}<br>
    <strong>🤖 Current Model:</strong> {selected_model}<br>
    <strong>🔬 System Description:</strong> This system uses XGBoost machine learning models to predict the volumetric water content of biochar-amended soils.
    </div>
    """, unsafe_allow_html=True)

    # Display different interfaces based on prediction mode
    if prediction_mode == "Single-point Prediction":
        # Single-point prediction interface
        display_single_prediction_interface(models, model_type, model_info, feature_info)
    else:
        # Batch prediction interface
        display_batch_prediction_interface(models, model_type, model_info, feature_info)


def display_single_prediction_interface(models, model_type, model_info, feature_info):
    """Display single-point prediction interface"""
    # Display different input interfaces based on selected model
    if model_type == 'group1':
        # Group 1 input interface
        st.markdown('<div class="sub-header">🔬 Input Parameters - Group 1</div>', unsafe_allow_html=True)

        # Get feature information
        feature_order = model_info.get('feature_order', [])
        biochar_categories = BIOCHAR_TYPE_INTERNAL

        # Create three-column layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 💧 Suction and Soil Parameters")

            # Suction parameter - manual input for any value
            suction = st.number_input(
                "Matric suction (kPa)",
                min_value=0.001,
                max_value=1000000.0,
                value=100.0,
                step=1.0,
                format="%.3f",
                help="Enter matric suction value in kPa (can be any value)",
                key="suction_input"
            )

            st.divider()

            # Soil particle composition - manual input
            st.markdown("**Soil particle composition (decimal)**")

            clay = st.number_input(
                "Clay content",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
                format="%.3f",
                help="Clay content, range 0-1, e.g., 0.2 means 20%",
                key="clay_input"
            )

            silt = st.number_input(
                "Silt content",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                format="%.3f",
                help="Silt content, range 0-1, e.g., 0.25 means 25%",
                key="silt_input"
            )

            sand = st.number_input(
                "Sand content",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                format="%.3f",
                help="Sand content, range 0-1, e.g., 0.55 means 55%",
                key="sand_input"
            )

            # Display particle composition sum
            total_particles = clay + silt + sand
            if abs(total_particles - 1.0) > 0.01:
                st.warning(f"Particle composition sum: {total_particles:.3f} (recommended close to 1.0)")
            else:
                st.success(f"Particle composition sum: {total_particles:.3f}")

        with col2:
            st.markdown("### 🌿 Soil and Biochar Basic Parameters")

            # Dry density - manual input
            dd = st.number_input(
                "Dry density (dd, g/cm³)",
                min_value=0.5,
                max_value=2.5,
                value=1.45,
                step=0.01,
                format="%.2f",
                help="Soil dry density, unit: g/cm³",
                key="dd_input"
            )

            st.divider()

            # Biochar content - manual input
            bc_percent = st.number_input(
                "Biochar content (BC, %)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.1,
                format="%.1f",
                help="Biochar content, unit: %",
                key="bc_percent_input"
            )

            # Convert to decimal
            bc = bc_percent / 100.0

            st.divider()

            # Pyrolysis temperature - dynamically adjusted based on BC value
            if bc == 0:
                # When BC=0, pyrolysis temperature doesn't exist
                st.markdown(
                    '<div class="warning-box">⚠️ Biochar content is 0, pyrolysis temperature does not exist</div>',
                    unsafe_allow_html=True)
                temperature = 0.0
            else:
                temperature = st.number_input(
                    "Pyrolysis temperature (°C)",
                    min_value=200.0,
                    max_value=900.0,
                    value=500.0,
                    step=10.0,
                    format="%.0f",
                    help="Biochar pyrolysis temperature, unit: °C",
                    key="temperature_input"
                )

        with col3:
            st.markdown("### 🧪 Biochar Type Parameters")

            # Biochar type selection - dynamically adjusted based on BC value
            if bc == 0:
                # When BC=0, biochar type doesn't exist
                st.markdown('<div class="warning-box">⚠️ Biochar content is 0, biochar type does not exist</div>',
                            unsafe_allow_html=True)
                biochar_type_internal = "农业废弃物"  # Default value, won't affect prediction
            else:
                # Display English options, store Chinese value internally
                biochar_type_display = st.selectbox(
                    "Biochar type",
                    options=BIOCHAR_TYPE_DISPLAY,
                    index=0,
                    help="Select the raw material type of biochar",
                    key="biochar_type_input"
                )
                # Convert to internal Chinese value for model
                biochar_type_internal = BIOCHAR_TYPE_MAP[biochar_type_display]

            st.divider()

            # Parameter summary card
            st.markdown("### 📋 Current Parameter Overview")

            # Display the English version in the summary if BC > 0
            biochar_type_display_value = BIOCHAR_TYPE_REVERSE_MAP.get(biochar_type_internal,
                                                                      biochar_type_internal) if bc > 0 else "N/A"

            param_summary = pd.DataFrame({
                'Parameter': ['Suction', 'Clay', 'Silt', 'Sand', 'Dry density', 'BC content', 'Pyrolysis temp',
                              'Biochar type'],
                'Value': [
                    f"{suction:.3f} kPa",
                    f"{clay:.3f}",
                    f"{silt:.3f}",
                    f"{sand:.3f}",
                    f"{dd:.2f} g/cm³",
                    f"{bc_percent:.1f}%",
                    f"{temperature:.0f}°C" if bc > 0 else "N/A",
                    biochar_type_display_value
                ]
            })

            st.dataframe(param_summary, use_container_width=True, hide_index=True)

    else:
        # Group 2 input interface
        st.markdown('<div class="sub-header">🔬 Input Parameters - Group 2</div>', unsafe_allow_html=True)

        # Get feature information
        feature_order = model_info.get('feature_order', [])

        # Create three-column layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 💧 Suction and Soil Parameters")

            # Suction parameter - manual input for any value
            suction = st.number_input(
                "Matric suction (kPa)",
                min_value=0.001,
                max_value=1000000.0,
                value=100.0,
                step=1.0,
                format="%.3f",
                help="Enter matric suction value in kPa (can be any value)",
                key="suction_input_group2"
            )

            st.divider()

            # Soil particle composition - manual input
            st.markdown("**Soil particle composition (decimal)**")

            clay = st.number_input(
                "Clay content",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
                format="%.3f",
                help="Clay content, range 0-1, e.g., 0.2 means 20%",
                key="clay_input_group2"
            )

            silt = st.number_input(
                "Silt content",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                format="%.3f",
                help="Silt content, range 0-1, e.g., 0.25 means 25%",
                key="silt_input_group2"
            )

            sand = st.number_input(
                "Sand content",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                format="%.3f",
                help="Sand content, range 0-1, e.g., 0.55 means 55%",
                key="sand_input_group2"
            )

            # Display particle composition sum
            total_particles = clay + silt + sand
            if abs(total_particles - 1.0) > 0.01:
                st.warning(f"Particle composition sum: {total_particles:.3f} (recommended close to 1.0)")
            else:
                st.success(f"Particle composition sum: {total_particles:.3f}")

        with col2:
            st.markdown("### 🌿 Soil and Biochar Basic Parameters")

            # Dry density - manual input
            dd = st.number_input(
                "Dry density (dd, g/cm³)",
                min_value=0.5,
                max_value=2.5,
                value=1.45,
                step=0.01,
                format="%.2f",
                help="Soil dry density, unit: g/cm³",
                key="dd_input_group2"
            )

            st.divider()

            # Biochar content - manual input
            bc_percent = st.number_input(
                "Biochar content (BC, %)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.1,
                format="%.1f",
                help="Biochar content, unit: %",
                key="bc_percent_input_group2"
            )

            # Convert to decimal
            bc = bc_percent / 100.0

            st.divider()

            # pH value - dynamically adjusted based on BC value
            if bc == 0:
                # When BC=0, pH is 0
                st.markdown('<div class="warning-box">⚠️ Biochar content is 0, pH value is 0</div>',
                            unsafe_allow_html=True)
                ph = 0.0
            else:
                ph = st.number_input(
                    "pH value",
                    min_value=0.0,
                    max_value=14.0,
                    value=8.0,
                    step=0.1,
                    format="%.1f",
                    help="Biochar pH value",
                    key="ph_input"
                )

        with col3:
            st.markdown("### 🧪 Biochar Physicochemical Parameters")

            # Dynamically adjusted based on BC value
            if bc == 0:
                st.markdown('<div class="warning-box">⚠️ Biochar content is 0, following parameters set to 0</div>',
                            unsafe_allow_html=True)
                at = 0.0
                ct = 0.0
            else:
                # Ash content (percentage) - manual input
                at = st.number_input(
                    "Ash content (AT, %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0,
                    step=0.1,
                    format="%.1f",
                    help="Biochar ash content, unit: %",
                    key="at_input"
                )

                # Carbon content (percentage) - manual input
                ct = st.number_input(
                    "Carbon content (CT, %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=60.0,
                    step=0.1,
                    format="%.1f",
                    help="Biochar carbon content, unit: %",
                    key="ct_input"
                )

            st.divider()

            # Parameter summary card
            st.markdown("### 📋 Current Parameter Overview")

            param_summary = pd.DataFrame({
                'Parameter': ['Suction', 'Clay', 'Silt', 'Sand', 'Dry density', 'BC content', 'pH', 'AT', 'CT'],
                'Value': [
                    f"{suction:.3f} kPa",
                    f"{clay:.3f}",
                    f"{silt:.3f}",
                    f"{sand:.3f}",
                    f"{dd:.2f} g/cm³",
                    f"{bc_percent:.1f}%",
                    f"{ph:.1f}" if bc > 0 else "N/A",
                    f"{at:.1f}%" if bc > 0 else "N/A",
                    f"{ct:.1f}%" if bc > 0 else "N/A"
                ]
            })

            st.dataframe(param_summary, use_container_width=True, hide_index=True)

    # Save variables to session state for prediction function
    if model_type == 'group1':
        st.session_state['suction'] = suction
        st.session_state['clay'] = clay
        st.session_state['silt'] = silt
        st.session_state['sand'] = sand
        st.session_state['dd'] = dd
        st.session_state['bc_percent'] = bc_percent
        st.session_state['bc'] = bc
        st.session_state['temperature'] = temperature if bc > 0 else 0.0
        st.session_state['biochar_type_internal'] = biochar_type_internal if bc > 0 else "农业废弃物"
    else:
        st.session_state['suction'] = suction
        st.session_state['clay'] = clay
        st.session_state['silt'] = silt
        st.session_state['sand'] = sand
        st.session_state['dd'] = dd
        st.session_state['bc_percent'] = bc_percent
        st.session_state['bc'] = bc
        st.session_state['ph'] = ph if bc > 0 else 0.0
        st.session_state['at'] = at if bc > 0 else 0.0
        st.session_state['ct'] = ct if bc > 0 else 0.0

    # Prediction button at bottom of page
    st.divider()

    # Prediction button container
    predict_container = st.container()
    with predict_container:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

        with col_btn2:
            if st.button("🚀 Start Single-point Prediction", type="primary", use_container_width=True):
                # Call single-point prediction function
                single_point_prediction(models, model_type, model_info, feature_info)


def display_batch_prediction_interface(models, model_type, model_info, feature_info):
    """Display batch prediction interface"""
    st.markdown('<div class="sub-header">📊 Batch Prediction</div>', unsafe_allow_html=True)

    # Display data format requirements
    st.markdown("""
    <div class="info-box">
    <strong>📋 Data Format Requirements:</strong>
    1. Upload CSV or Excel file containing multiple parameter sets
    2. File must contain the following columns (based on selected model):
    """, unsafe_allow_html=True)

    if model_type == 'group1':
        st.markdown("""
        - **Group 1 required columns:** suction, clay, silt, sand, dd, BC, temperature, Biochar_type_combined
        - **Notes:** BC column is in percentage (e.g., 5 means 5%), Biochar_type_combined is biochar type
        - **Biochar type options:** Agricultural waste, Forestry residue, Livestock manure, Municipal sludge, Other
        - **Note:** You can use either English display names or Chinese internal names
        """)
    else:
        st.markdown("""
        - **Group 2 required columns:** suction, clay, silt, sand, dd, BC, pH, AT, CT
        - **Notes:** BC column is in percentage (e.g., 5 means 5%), AT and CT are in percentage (e.g., 25 means 25%)
        - **Constraint:** When BC=0, pH, AT, CT are automatically set to 0
        """)

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Important Notes:</strong>
    1. Ensure file encoding is UTF-8
    2. Soil particle composition (clay, silt, sand) sum should be in 0-1 range
    3. It is recommended to download the template file first for data preparation
    </div>
    """, unsafe_allow_html=True)

    # Create two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload data file (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Select file containing batch prediction data",
            key="batch_file_uploader"
        )

    with col2:
        # Download template file
        st.markdown("### 📥 Download Template")
        if model_type == 'group1':
            # Create group 1 template with English biochar types
            template_data = {
                'suction': [100.0, 1000.0, 10000.0],
                'clay': [0.2, 0.3, 0.1],
                'silt': [0.25, 0.3, 0.2],
                'sand': [0.55, 0.4, 0.7],
                'dd': [1.45, 1.5, 1.4],
                'BC': [5.0, 10.0, 0.0],  # percentage
                'temperature': [500, 600, 0],
                'Biochar_type_combined': ['Agricultural waste', 'Forestry residue', 'Agricultural waste']
            }
            template_df = pd.DataFrame(template_data)
            csv = template_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Group 1 Template",
                data=csv,
                file_name="template_group1.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_template_group1"
            )
        else:
            # Create group 2 template
            template_data = {
                'suction': [100.0, 1000.0, 10000.0],
                'clay': [0.2, 0.3, 0.1],
                'silt': [0.25, 0.3, 0.2],
                'sand': [0.55, 0.4, 0.7],
                'dd': [1.45, 1.5, 1.4],
                'BC': [5.0, 10.0, 0.0],  # percentage
                'pH': [8.0, 7.5, 0.0],
                'AT': [25.0, 30.0, 0.0],  # percentage
                'CT': [60.0, 65.0, 0.0]  # percentage
            }
            template_df = pd.DataFrame(template_data)
            csv = template_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Group 2 Template",
                data=csv,
                file_name="template_group2.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_template_group2"
            )

    # If file is uploaded, show preview and perform prediction
    if uploaded_file is not None:
        try:
            # Read file with encoding detection
            if uploaded_file.name.endswith('.csv'):
                # Try different encodings for CSV
                encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
                data_df = None
                for encoding in encodings_to_try:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        data_df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception:
                        continue

                if data_df is None:
                    st.error("❌ Could not read CSV file with any encoding. Please ensure file is UTF-8 encoded.")
                    return
            else:
                data_df = pd.read_excel(uploaded_file)

            # Show data preview
            st.markdown("### 📋 Data Preview")
            st.write(f"File: {uploaded_file.name}")
            st.write(f"Number of rows: {len(data_df)}")
            st.write(f"Number of columns: {len(data_df.columns)}")

            # Show first few rows
            with st.expander("View Data Details"):
                st.dataframe(data_df.head(10))

            # Validate data
            st.markdown("### 🔍 Data Validation")
            validation_errors = validate_batch_data(data_df, model_type, feature_info)

            if validation_errors:
                st.error("❌ Data validation failed:")
                for error in validation_errors:
                    st.error(f"  - {error}")
                return
            else:
                st.success("✅ Data validation passed")

            # Start batch prediction
            st.markdown("### 🚀 Start Batch Prediction")
            if st.button("Start Batch Prediction", type="primary", use_container_width=True,
                         key="batch_predict_button"):
                with st.spinner("Performing batch prediction..."):
                    model = models[model_type]

                    # Select prediction function based on model type
                    if model_type == 'group1':
                        predictions = batch_predict_group1(model, data_df, feature_info)
                    else:
                        predictions = batch_predict_group2(model, data_df, feature_info)

                    # Add prediction results to dataframe
                    result_df = data_df.copy()
                    result_df['Predicted Volumetric Water Content'] = predictions

                    # Calculate prediction success rate
                    success_rate = (1 - result_df['Predicted Volumetric Water Content'].isna().sum() / len(
                        result_df)) * 100

                    # Display prediction results
                    st.markdown("### 📊 Batch Prediction Results")
                    st.success(f"✅ Batch prediction completed! Success rate: {success_rate:.1f}%")

                    # Display result statistics
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Total samples", len(result_df))
                    with col_stat2:
                        st.metric("Successful predictions",
                                  len(result_df) - result_df['Predicted Volumetric Water Content'].isna().sum())
                    with col_stat3:
                        st.metric("Failed predictions", result_df['Predicted Volumetric Water Content'].isna().sum())

                    # Show result preview
                    st.markdown("#### 🔍 Prediction Results Preview")
                    st.dataframe(result_df.head(10))

                    # Plot prediction result distribution
                    st.markdown("#### 📈 Prediction Result Distribution")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Ensure Chinese fonts are used
                    configure_matplotlib_fonts()

                    # Plot histogram
                    valid_predictions = result_df['Predicted Volumetric Water Content'].dropna()
                    if len(valid_predictions) > 0:
                        ax.hist(valid_predictions, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                        ax.axvline(valid_predictions.mean(), color='red', linestyle='--', linewidth=2,
                                   label=f'Mean: {valid_predictions.mean():.3f}')
                        ax.axvline(valid_predictions.median(), color='green', linestyle='--', linewidth=2,
                                   label=f'Median: {valid_predictions.median():.3f}')

                        ax.set_xlabel(plot_text('体积含水率', 'Volumetric water content'), fontsize=12)
                        ax.set_ylabel(plot_text('频数', 'Frequency'), fontsize=12)
                        ax.set_title(plot_text('预测结果分布直方图', 'Prediction result distribution histogram'),
                                     fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)

                        # Display statistics
                        col_stat4, col_stat5, col_stat6, col_stat7 = st.columns(4)
                        with col_stat4:
                            st.metric("Mean", f"{valid_predictions.mean():.4f}")
                        with col_stat5:
                            st.metric("Median", f"{valid_predictions.median():.4f}")
                        with col_stat6:
                            st.metric("Minimum", f"{valid_predictions.min():.4f}")
                        with col_stat7:
                            st.metric("Maximum", f"{valid_predictions.max():.4f}")

                    # Provide result download
                    st.markdown("#### 💾 Download Results")

                    # Create download button
                    csv_result = result_df.to_csv(index=False).encode('utf-8')
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_result,
                            file_name=f"batch_predictions_{model_type}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv_batch"
                        )

                    with col_dl2:
                        # Convert to Excel format
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            result_df.to_excel(writer, index=False, sheet_name='prediction results')
                        excel_buffer.seek(0)

                        st.download_button(
                            label="📥 Download Excel",
                            data=excel_buffer,
                            file_name=f"batch_predictions_{model_type}_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_excel_batch"
                        )

                    # Show detailed prediction results
                    with st.expander("📋 View Detailed Prediction Results"):
                        st.dataframe(result_df)

        except Exception as e:
            st.error(f"❌ File processing failed: {e}")
            st.error("Please check file format and content")


def single_point_prediction(models, model_type, model_info, feature_info):
    """Execute single-point prediction"""
    # Get input parameters from session state
    suction = st.session_state.get('suction', 100.0)
    clay = st.session_state.get('clay', 0.2)
    silt = st.session_state.get('silt', 0.25)
    sand = st.session_state.get('sand', 0.55)
    dd = st.session_state.get('dd', 1.45)
    bc_percent = st.session_state.get('bc_percent', 5.0)
    bc = bc_percent / 100.0

    # Validate input
    total_particles = clay + silt + sand
    if total_particles > 1.0:
        st.error("❌ Sum of clay, silt, and sand content cannot exceed 1.0!")
        return

    if model_type not in models:
        st.error(f"❌ Model {model_type} not loaded successfully")
        return

    model = models[model_type]

    # Prepare input data based on model type
    if model_type == 'group1':
        # Group 1: use categorical features
        biochar_categories = BIOCHAR_TYPE_INTERNAL
        temperature = st.session_state.get('temperature', 500.0)
        biochar_type_internal = st.session_state.get('biochar_type_internal', "农业废弃物")

        # When BC=0, set pyrolysis temperature and biochar type to default values
        if bc == 0:
            temperature = 0.0
            biochar_type_internal = "农业废弃物"  # Default value

        # Create feature DataFrame - follow training feature order
        features_dict = {
            'suction': float(suction),
            'clay': float(clay),
            'silt': float(silt),
            'sand': float(sand),
            'dd': float(dd),
            'BC': float(bc),  # Decimal form
            'temperature': float(temperature),
            'Biochar_type_combined': biochar_type_internal  # Categorical feature (Chinese)
        }

        # Create DataFrame, ensure Biochar_type_combined is category type
        features_df = pd.DataFrame([features_dict])

        # Convert Biochar_type_combined to category type
        features_df['Biochar_type_combined'] = pd.Categorical(
            features_df['Biochar_type_combined'],
            categories=biochar_categories
        )

        # Rearrange columns according to feature order
        feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature', 'Biochar_type_combined']
        features_df = features_df[feature_order]

        # Save input data for display
        input_data = {
            'suction': float(suction),
            'clay': float(clay),
            'silt': float(silt),
            'sand': float(sand),
            'dd': float(dd),
            'BC': float(bc),
            'temperature': float(temperature),
            'Biochar_type_combined': biochar_type_internal
        }

    else:
        # Group 2: use raw values directly
        ph = st.session_state.get('ph', 8.0)
        at = st.session_state.get('at', 25.0)
        ct = st.session_state.get('ct', 60.0)

        if bc == 0:
            # Ensure AT, CT, pH are 0 when BC=0
            ph = 0.0
            at = 0.0
            ct = 0.0

        # Create feature DataFrame
        feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']

        features = [
            float(suction),
            float(clay),
            float(silt),
            float(sand),
            float(dd),
            float(bc),  # Decimal form
            float(ph),
            float(at),  # Percentage form
            float(ct)  # Percentage form
        ]

        features_df = pd.DataFrame([features], columns=feature_order)

        # Save input data for display
        input_data = {
            'suction': float(suction),
            'clay': float(clay),
            'silt': float(silt),
            'sand': float(sand),
            'dd': float(dd),
            'BC': float(bc),
            'pH': float(ph),
            'AT': float(at),
            'CT': float(ct)
        }

    # Make prediction
    with st.spinner("Performing prediction calculation..."):
        try:
            # Display input features (for debugging)
            with st.expander("🔍 View Input Feature Values"):
                st.write(f"Model type: {model_type}")
                st.write("Input feature values:")
                st.dataframe(features_df)

                # If model has feature_names_in_ attribute, display expected features
                if hasattr(model, 'feature_names_in_'):
                    st.write("Model expects features:", list(model.feature_names_in_))
                    st.write("Input features:", list(features_df.columns))

            # Make prediction
            prediction = model.predict(features_df)[0]

            # Display prediction results
            st.markdown('<div class="sub-header">📊 Prediction Results</div>', unsafe_allow_html=True)

            # Create result display area
            col_a, col_b = st.columns([2, 1])

            with col_a:
                st.markdown(f"""
                <div class="success-box" style="text-align: center;">
                    <h2 style="margin: 0;">Predicted Volumetric Water Content</h2>
                    <h1 style="color: #2E8B57; font-size: 3rem; margin: 10px 0;">{prediction:.4f}</h1>
                    <p>Volume of water per unit volume of soil</p>
                </div>
                """, unsafe_allow_html=True)

                # Auxiliary metrics
                col_a1, col_a2 = st.columns(2)

                with col_a1:
                    # Calculate saturation (assuming porosity of 0.4)
                    porosity = 0.4
                    saturation = (prediction / porosity) * 100 if porosity > 0 else 0
                    st.metric("Estimated Saturation", f"{saturation:.1f}%")

                with col_a2:
                    # Provide qualitative assessment
                    if prediction > 0.35:
                        assessment = "High water retention capacity"
                        color = "#32CD32"
                        emoji = "🔵"
                    elif prediction > 0.2:
                        assessment = "Medium water retention capacity"
                        color = "#FFA500"
                        emoji = "🟡"
                    else:
                        assessment = "Low water retention capacity"
                        color = "#FF4500"
                        emoji = "🔴"

                    st.markdown(
                        f"**Assessment:** {emoji} <span style='color:{color};font-weight:bold'>{assessment}</span>",
                        unsafe_allow_html=True)

            with col_b:
                st.markdown("### 📋 Input Parameter Details")

                # Display input parameters
                detail_data = []
                for key, value in input_data.items():
                    if key == 'BC':
                        display_value = f"{value * 100:.1f}%"
                        unit = "%"
                    elif key in ['AT', 'CT']:
                        display_value = f"{value:.1f}%"
                        unit = "%"
                    elif key in ['clay', 'silt', 'sand']:
                        display_value = f"{value:.3f}"
                        unit = "decimal"
                    elif key == 'suction':
                        display_value = f"{value:.3f} kPa"
                        unit = "kPa"
                    elif key == 'dd':
                        display_value = f"{value:.2f} g/cm³"
                        unit = "g/cm³"
                    elif key == 'temperature':
                        display_value = f"{value:.0f}°C"
                        unit = "°C"
                    elif key == 'pH':
                        display_value = f"{value:.1f}"
                        unit = "-"
                    elif key == 'Biochar_type_combined':
                        # Display English version
                        display_value = BIOCHAR_TYPE_REVERSE_MAP.get(value, value)
                        unit = "type"
                    else:
                        display_value = str(value)
                        unit = ""

                    detail_data.append({
                        'Parameter': key,
                        'Value': display_value,
                        'Unit': unit
                    })

                detail_df = pd.DataFrame(detail_data)
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                # Download button
                st.download_button(
                    label="📥 Download Prediction Result",
                    data=detail_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"SWCC_prediction_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_single_result"
                )

            # Get SWCC curve settings from session state
            curve_points = st.session_state.get('curve_points', 100)
            min_suction = st.session_state.get('min_suction', 0.01)
            max_suction = st.session_state.get('max_suction', 284804.0)
            enable_vg_fitting = st.session_state.get('enable_vg_fitting', True)

            # Check if max_suction is greater than min_suction
            if max_suction <= min_suction:
                st.warning("Maximum suction must be greater than minimum suction, auto-adjusting")
                max_suction = min_suction * 100
                st.session_state['max_suction'] = max_suction

            # Generate SWCC curve
            st.markdown('<div class="sub-header">📈 SWCC Curve</div>', unsafe_allow_html=True)

            # Generate suction range (log-uniform distribution)
            suction_range = np.logspace(np.log10(min_suction), np.log10(max_suction), curve_points)

            # Generate SWCC curve data
            with st.spinner("Generating SWCC curve..."):
                predictions = generate_swcc_curve(model, model_type, input_data, suction_range)

                # VG model fitting
                vg_params = None
                r_squared = 0
                fitted_curve = None

                if enable_vg_fitting:
                    with st.spinner("Performing VG model fitting..."):
                        popt, pcov, r_squared, fitted_curve = fit_vg_model(suction_range, predictions)

                        if popt is not None:
                            vg_params = popt
                            st.success(f"✅ VG model fitting successful! R² = {r_squared:.6f}")

                # Plot SWCC curve
                current_point = (suction, prediction) if suction >= min_suction and suction <= max_suction else None

                fig = plot_swcc_with_vg_fit(suction_range, predictions, vg_params, current_point)

                st.pyplot(fig)

                # Display VG model parameters
                if enable_vg_fitting and vg_params is not None:
                    display_vg_parameters(vg_params, r_squared, suction_range, predictions)

                # Provide curve data download
                curve_data = pd.DataFrame({
                    'Suction(kPa)': suction_range,
                    'Volumetric_Water_Content': predictions
                })

                # If VG fitting results exist, add to data
                if fitted_curve is not None:
                    curve_data['VG_Fitted_Water_Content'] = fitted_curve
                    curve_data['Residual'] = predictions - fitted_curve

                csv_curve = curve_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download SWCC Curve Data",
                    data=csv_curve,
                    file_name=f"SWCC_curve_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_swcc_curve"
                )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            st.error("Please check feature order or model file")

            # Display detailed error information
            with st.expander("🔍 View Detailed Error Information"):
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()