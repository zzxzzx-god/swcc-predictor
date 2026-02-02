# app_combined_enhanced_with_VG_fitting.py - 生物炭改性土SWCC预测系统（增强版，带VG模型拟合）
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import io
import warnings
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')

# 设置matplotlib中文字体（放在导入后立即设置）
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置页面配置
st.set_page_config(
    page_title="生物炭改性土SWCC预测系统",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面样式
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


# VG模型函数定义
def vg_model(h, theta_r, theta_s, alpha, n):
    """
    van Genuchten模型
    θ = θr + (θs - θr) / [1 + (α·h)^n]^m
    其中 m = 1 - 1/n
    """
    m = 1 - 1 / n
    return theta_r + (theta_s - theta_r) / ((1 + (alpha * h) ** n) ** m)


def fit_vg_model(suction_data, theta_data, initial_guess=None):
    """
    对SWCC数据进行VG模型拟合

    参数:
    - suction_data: 吸力数据(kPa)
    - theta_data: 含水率数据
    - initial_guess: 初始猜测参数 [θr, θs, α, n]

    返回:
    - popt: 最优拟合参数
    - pcov: 参数的协方差矩阵
    - r_squared: 决定系数R²
    - fitted_theta: 拟合值
    """
    # 默认初始猜测
    if initial_guess is None:
        # θr: 最小含水率的90%
        # θs: 最大含水率的110%
        # α: 1/中值吸力
        # n: 典型值1.5
        theta_min = np.min(theta_data)
        theta_max = np.max(theta_data)
        suction_median = np.median(suction_data[suction_data > 0])

        initial_guess = [
            max(0, theta_min * 0.9),  # θr
            min(0.5, theta_max * 1.1),  # θs
            1.0 / suction_median if suction_median > 0 else 0.01,  # α
            1.5  # n
        ]

    # 参数边界条件
    lower_bounds = [0, 0, 0.00001, 1.01]  # n必须大于1
    upper_bounds = [0.5, 0.6, 10, 10]  # 合理范围

    try:
        # 使用curve_fit进行拟合
        popt, pcov = curve_fit(
            vg_model,
            suction_data,
            theta_data,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=5000
        )

        # 计算拟合值
        fitted_theta = vg_model(suction_data, *popt)

        # 计算R²
        residuals = theta_data - fitted_theta
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((theta_data - np.mean(theta_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return popt, pcov, r_squared, fitted_theta

    except Exception as e:
        st.warning(f"VG模型拟合失败: {e}")
        return None, None, 0, None


def plot_swcc_with_vg_fit(suction_range, predictions, vg_params=None, current_point=None):
    """绘制SWCC曲线和VG模型拟合结果"""
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 确保使用中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 主图：SWCC曲线
    ax.plot(suction_range, predictions, 'b-', linewidth=2, label='SWCC (XGBoost)')

    # 如果提供了VG拟合参数，绘制拟合曲线
    if vg_params is not None:
        theta_r, theta_s, alpha, n = vg_params
        m = 1 - 1 / n
        fitted_curve = vg_model(suction_range, theta_r, theta_s, alpha, n)
        ax.plot(suction_range, fitted_curve, 'r--', linewidth=2, label=' vG model fitting curve')

        # 在图中添加VG方程
        vg_eq = r'$\theta = \theta_r + \frac{\theta_s - \theta_r}{[1 + (\alpha h)^n]^m}$'
        ax.text(0.02, 0.98, vg_eq, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 如果提供了当前点，在图上标出
    if current_point:
        ax.plot(current_point[0], current_point[1], 'ro', markersize=10, label='Current prediction point')
        ax.annotate(f'({current_point[0]:.1f} kPa, {current_point[1]:.3f})',
                    xy=current_point,
                    xytext=(current_point[0] * 1.5, current_point[1] * 0.9),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')

    # 设置主图坐标轴
    ax.set_xscale('log')
    ax.set_xlabel('Suction (kPa)', fontsize=12)
    ax.set_ylabel('Volumetric water content', fontsize=12)
    ax.set_title(' SWCC and the VG model fitting curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    ax.set_facecolor('#f8f9fa')

    # 设置坐标轴范围
    ax.set_xlim(min(suction_range), max(suction_range))
    y_min = max(0, min(predictions) - 0.05)
    y_max = min(1, max(predictions) + 0.05)
    ax.set_ylim(y_min, y_max)

    # 吸力范围标记
    ax.text(0.02, 0.02, f'Suction range: {min(suction_range):.2f} - {max(suction_range):.0f} kPa',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    return fig


def display_vg_parameters(popt, r_squared, suction_range, theta_data):
    """显示VG模型参数"""
    if popt is None:
        st.warning("VG模型拟合失败，无法显示参数")
        return

    theta_r, theta_s, alpha, n = popt
    m = 1 - 1 / n

    # 创建参数表格
    st.markdown('<div class="sub-header">📊 VG模型拟合参数</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="parameter-table">', unsafe_allow_html=True)
        st.markdown("##### 模型参数")

        param_data = {
            '参数': ['θr (残余含水率)', 'θs (饱和含水率)', 'α (倒数的吸力)', 'n (形状参数)', 'm (=1-1/n)',
                     'R² (决定系数)'],
            '值': [
                f"{theta_r:.6f}",
                f"{theta_s:.6f}",
                f"{alpha:.6f}",
                f"{n:.6f}",
                f"{m:.6f}",
                f"{r_squared:.6f}"
            ],
            '物理意义': [
                '高吸力下的最小含水率',
                '零吸力下的最大含水率',
                '进气值的倒数',
                '孔径分布指数',
                '曲线形状参数',
                '拟合优度 (1为完美拟合)'
            ]
        }

        param_df = pd.DataFrame(param_data)
        st.dataframe(param_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="parameter-table">', unsafe_allow_html=True)
        st.markdown("##### 特征吸力值")

        # 计算特征吸力
        # 进气值 (air entry value)
        ha = 1 / alpha if alpha > 0 else 0

        # 有效饱和度为0.5时的吸力
        se = 0.5
        h50 = (1 / alpha) * ((1 / se ** (1 / m)) - 1) ** (1 / n) if alpha > 0 and m > 0 and n > 0 else 0

        feature_data = {
            '特征点': ['进气值 ha', 'Se=0.5时吸力 h₅₀', '预测最小吸力', '预测最大吸力', '数据点数量'],
            '吸力值 (kPa)': [
                f"{ha:.3f}",
                f"{h50:.3f}",
                f"{np.min(suction_range):.3f}",
                f"{np.max(suction_range):.3f}",
                f"{len(suction_range)}"
            ],
            '备注': [
                '1/α',
                'Se = (θ-θr)/(θs-θr) = 0.5',
                '曲线起点',
                '曲线终点',
                'SWCC曲线点数'
            ]
        }

        feature_df = pd.DataFrame(feature_data)
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 显示VG模型方程
    st.markdown('<div class="vg-equation">', unsafe_allow_html=True)
    st.markdown("### van Genuchten (VG) 模型方程")
    st.latex(r'''
    \theta(h) = \theta_r + \frac{\theta_s - \theta_r}{\left[1 + (\alpha \cdot h)^n\right]^m}
    ''')
    st.markdown(f'''
    其中:
    - θ(h): 吸力为 h 时的体积含水率
    - θr = {theta_r:.4f} (残余含水率)
    - θs = {theta_s:.4f} (饱和含水率)
    - α = {alpha:.6f} kPa⁻¹
    - n = {n:.4f}
    - m = 1 - 1/n = {m:.4f}
    ''')
    st.markdown('</div>', unsafe_allow_html=True)

    # 提供参数下载
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
        label="📥 下载VG模型参数",
        data=csv_params,
        file_name=f"VG_model_parameters_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


# 加载模型
@st.cache_resource
def load_models():
    """加载两个训练好的XGBoost模型"""
    models = {}

    try:
        # 加载变量组一模型（含生物炭类型和热解温度）
        with open('xgboost_optimized_results/model_group1.pkl', 'rb') as f:
            models['group1'] = pickle.load(f)
        st.sidebar.success("✅ 变量组一模型加载成功！")

        # 调试：显示模型的特征名称
        if hasattr(models['group1'], 'feature_names_in_'):
            st.sidebar.info(f"模型期望的特征: {list(models['group1'].feature_names_in_)}")
        else:
            st.sidebar.info("⚠️ 模型没有存储特征名称信息")

    except FileNotFoundError:
        st.sidebar.warning("⚠️ 变量组一模型文件未找到，请先训练并保存模型")
    except Exception as e:
        st.sidebar.warning(f"⚠️ 变量组一模型加载失败: {e}")

    try:
        # 加载变量组二模型（含生物炭理化指标）
        with open('xgboost_optimized_results/model_group2.pkl', 'rb') as f:
            models['group2'] = pickle.load(f)
        st.sidebar.success("✅ 变量组二模型加载成功！")
    except FileNotFoundError:
        st.sidebar.warning("⚠️ 变量组二模型文件未找到")
    except Exception as e:
        st.sidebar.warning(f"⚠️ 变量组二模型加载失败: {e}")

    return models


# 加载特征信息
@st.cache_resource
def load_feature_info():
    """加载特征信息文件"""
    try:
        import json
        with open('xgboost_optimized_results/feature_info.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        # 如果文件不存在，返回默认值
        return {
            'group1': {
                'feature_names': ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature',
                                  'Biochar_type_combined'],
                'feature_order': ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature',
                                  'Biochar_type_combined'],
                'biochar_categories': ["农业废弃物", "林业残余物", "畜禽粪便", "城市污泥", "其他"]
            },
            'group2': {
                'feature_names': ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT'],
                'feature_order': ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']
            }
        }


def generate_swcc_curve(model, model_type, base_input, suction_range):
    """生成SWCC曲线数据"""
    predictions = []

    for suction in suction_range:
        # 复制基础输入数据
        input_data = base_input.copy()
        input_data['suction'] = suction

        # 创建特征DataFrame
        if model_type == 'group1':
            # 变量组一：使用分类特征
            feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature', 'Biochar_type_combined']
            biochar_categories = ["农业废弃物", "林业残余物", "畜禽粪便", "城市污泥", "其他"]

            features_df = pd.DataFrame([input_data])
            features_df['Biochar_type_combined'] = pd.Categorical(
                features_df['Biochar_type_combined'],
                categories=biochar_categories
            )
            features_df = features_df[feature_order]
        else:
            # 变量组二：直接使用原始值
            feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']
            features_df = pd.DataFrame([input_data])[feature_order]

        # 进行预测
        prediction = model.predict(features_df)[0]
        predictions.append(prediction)

    return predictions


def batch_predict_group1(model, data_df, feature_info):
    """批量预测 - 变量组一"""
    predictions = []

    # 获取特征信息和类别
    biochar_categories = feature_info.get('biochar_categories',
                                          ["农业废弃物", "林业残余物", "畜禽粪便", "城市污泥", "其他"])

    for idx, row in data_df.iterrows():
        try:
            # 准备输入数据
            input_data = {
                'suction': float(row.get('suction', 0)),
                'clay': float(row.get('clay', 0)),
                'silt': float(row.get('silt', 0)),
                'sand': float(row.get('sand', 0)),
                'dd': float(row.get('dd', 0)),
                'BC': float(row.get('BC', 0)) / 100.0,  # 转换为小数
                'temperature': float(row.get('temperature', 0)),
                'Biochar_type_combined': str(row.get('Biochar_type_combined', '农业废弃物'))
            }

            # 当BC=0时，调整参数
            if input_data['BC'] == 0:
                input_data['temperature'] = 0.0
                input_data['Biochar_type_combined'] = '农业废弃物'  # 默认值

            # 创建DataFrame
            features_df = pd.DataFrame([input_data])
            features_df['Biochar_type_combined'] = pd.Categorical(
                features_df['Biochar_type_combined'],
                categories=biochar_categories
            )

            # 按照特征顺序排列
            feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature', 'Biochar_type_combined']
            features_df = features_df[feature_order]

            # 进行预测
            prediction = model.predict(features_df)[0]
            predictions.append(prediction)

        except Exception as e:
            st.warning(f"第 {idx + 1} 行数据预测失败: {e}")
            predictions.append(np.nan)

    return predictions


def batch_predict_group2(model, data_df, feature_info):
    """批量预测 - 变量组二"""
    predictions = []

    for idx, row in data_df.iterrows():
        try:
            # 准备输入数据
            bc = float(row.get('BC', 0)) / 100.0  # 转换为小数

            # 处理BC=0的情况
            if bc == 0:
                ph = 0.0
                at = 0.0
                ct = 0.0
            else:
                ph = float(row.get('pH', 8.0))
                at = float(row.get('AT', 25.0))
                ct = float(row.get('CT', 60.0))

            # 创建特征列表
            features = [
                float(row.get('suction', 100.0)),
                float(row.get('clay', 0.2)),
                float(row.get('silt', 0.25)),
                float(row.get('sand', 0.55)),
                float(row.get('dd', 1.45)),
                bc,  # 小数形式
                ph,
                at,  # 百分数形式
                ct  # 百分数形式
            ]

            # 创建DataFrame
            feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']
            features_df = pd.DataFrame([features], columns=feature_order)

            # 进行预测
            prediction = model.predict(features_df)[0]
            predictions.append(prediction)

        except Exception as e:
            st.warning(f"第 {idx + 1} 行数据预测失败: {e}")
            predictions.append(np.nan)

    return predictions


def validate_batch_data(data_df, model_type, feature_info):
    """验证批量数据"""
    errors = []

    if model_type == 'group1':
        required_columns = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature', 'Biochar_type_combined']
        biochar_categories = feature_info.get('biochar_categories',
                                              ["农业废弃物", "林业残余物", "畜禽粪便", "城市污泥", "其他"])
    else:
        required_columns = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']

    # 检查必需列
    missing_columns = [col for col in required_columns if col not in data_df.columns]
    if missing_columns:
        errors.append(f"缺少必需列: {', '.join(missing_columns)}")

    # 检查数据类型
    for col in required_columns:
        if col in data_df.columns:
            # 尝试转换为数值型（除生物炭类型外）
            if col != 'Biochar_type_combined':
                try:
                    data_df[col] = pd.to_numeric(data_df[col])
                except:
                    errors.append(f"列 '{col}' 包含非数值数据")

    # 检查生物炭类型是否有效
    if model_type == 'group1' and 'Biochar_type_combined' in data_df.columns:
        invalid_types = data_df[~data_df['Biochar_type_combined'].isin(biochar_categories)][
            'Biochar_type_combined'].unique()
        if len(invalid_types) > 0:
            errors.append(f"无效的生物炭类型: {', '.join(invalid_types)}")

    # 检查土壤颗粒组成之和
    if all(col in data_df.columns for col in ['clay', 'silt', 'sand']):
        data_df['total_particles'] = data_df['clay'] + data_df['silt'] + data_df['sand']
        invalid_rows = data_df[(data_df['total_particles'] > 1.0) | (data_df['total_particles'] < 0)].index.tolist()
        if invalid_rows:
            errors.append(f"行 {[i + 1 for i in invalid_rows]} 的土壤颗粒组成之和不在0-1范围内")

    return errors


def main():
    """主应用函数"""
    # 应用标题
    st.markdown('<div class="main-header">🌱 生物炭改性土持水特征曲线(SWCC)预测系统</div>', unsafe_allow_html=True)

    # 加载模型和特征信息
    models = load_models()
    feature_info = load_feature_info()

    if not models:
        st.error("❌ 没有可用的模型，请检查模型文件")
        return

    # 侧边栏 - 模型选择和系统信息
    with st.sidebar:
        st.title("🔧 系统设置")

        # 预测模式选择
        st.markdown("### 📊 选择预测模式")
        prediction_mode = st.radio(
            "选择预测模式",
            ["单点预测", "批量预测"],
            index=0
        )

        # 模型选择
        st.markdown("### 🤖 选择预测模型")
        model_options = []
        if 'group1' in models:
            model_options.append("变量组一：含生物炭类型和热解温度")
        if 'group2' in models:
            model_options.append("变量组二：含生物炭理化指标")

        if not model_options:
            st.error("没有可用的模型")
            st.stop()

        selected_model = st.radio(
            "选择要使用的模型",
            model_options,
            index=0
        )

        # 确定模型类型
        if "变量组一" in selected_model:
            model_type = 'group1'
            model_info = feature_info.get('group1', {})
            st.info("使用变量组一：包含生物炭类型和热解温度")
        else:
            model_type = 'group2'
            model_info = feature_info.get('group2', {})
            st.info("使用变量组二：包含生物炭理化指标(pH, AT, CT)")

        st.divider()

        # SWCC曲线设置（仅单点预测时显示）
        if prediction_mode == "单点预测":
            st.markdown("### 📈 SWCC曲线设置")

            # VG模型拟合选项
            st.markdown("#### 🔧 VG模型拟合选项")
            enable_vg_fitting = st.checkbox("启用VG模型拟合", value=True,
                                            help="对生成的SWCC曲线进行van Genuchten模型拟合",
                                            key="enable_vg_fitting")

            curve_points = st.slider(
                "曲线点数",
                min_value=20,
                max_value=200,
                value=100,
                help="SWCC曲线上的点数",
                key="curve_points"
            )

            min_suction = st.number_input(
                "最小吸力 (kPa)",
                min_value=0.001,
                max_value=1000.0,
                value=0.01,
                step=0.01,
                format="%.3f",
                help="SWCC曲线的最小吸力值",
                key="min_suction"
            )

            max_suction = st.number_input(
                "最大吸力 (kPa)",
                min_value=100.0,
                max_value=10000000.0,  # 增加最大值范围
                value=284804.0,
                step=100.0,
                help="SWCC曲线的最大吸力值",
                key="max_suction"
            )

            # 检查max_suction是否大于min_suction
            if max_suction <= min_suction:
                st.warning("最大吸力必须大于最小吸力，已自动调整")
                max_suction = min_suction * 100
                st.session_state['max_suction'] = max_suction

        else:
            # 批量预测时的设置
            st.markdown("### 📊 批量预测设置")
            st.info("上传包含多组参数的文件进行批量预测")

    # 主内容区域
    st.markdown(f"""
    <div class="info-box">
    <strong>📖 当前模式:</strong> {prediction_mode}<br>
    <strong>🤖 当前模型:</strong> {selected_model}<br>
    <strong>🔬 系统简介:</strong> 本系统基于XGBoost机器学习模型，预测生物炭改性土的体积含水率。
    </div>
    """, unsafe_allow_html=True)

    # 根据预测模式显示不同的界面
    if prediction_mode == "单点预测":
        # 单点预测界面
        display_single_prediction_interface(models, model_type, model_info, feature_info)
    else:
        # 批量预测界面
        display_batch_prediction_interface(models, model_type, model_info, feature_info)


def display_single_prediction_interface(models, model_type, model_info, feature_info):
    """显示单点预测界面"""
    # 根据选择的模型显示不同的输入界面
    if model_type == 'group1':
        # 变量组一输入界面
        st.markdown('<div class="sub-header">🔬 输入参数 - 变量组一</div>', unsafe_allow_html=True)

        # 获取特征信息
        feature_order = model_info.get('feature_order', [])
        biochar_categories = model_info.get('biochar_categories',
                                            ["农业废弃物", "林业残余物", "畜禽粪便", "城市污泥", "其他"])

        # 创建三列布局
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 💧 吸力与土体参数")

            # 吸力参数 - 手动输入任意值
            suction = st.number_input(
                "基质吸力 (kPa)",
                min_value=0.001,
                max_value=1000000.0,
                value=100.0,
                step=1.0,
                format="%.3f",
                help="输入基质吸力值，单位kPa（可以是任意值）",
                key="suction_input"
            )

            st.divider()

            # 土壤颗粒组成 - 手动输入
            st.markdown("**土壤颗粒组成（小数形式）**")

            clay = st.number_input(
                "黏粒含量 (clay)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
                format="%.3f",
                help="黏粒含量，范围0-1，如0.2表示20%",
                key="clay_input"
            )

            silt = st.number_input(
                "粉粒含量 (silt)",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                format="%.3f",
                help="粉粒含量，范围0-1，如0.25表示25%",
                key="silt_input"
            )

            sand = st.number_input(
                "砂粒含量 (sand)",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                format="%.3f",
                help="砂粒含量，范围0-1，如0.55表示55%",
                key="sand_input"
            )

            # 显示颗粒组成之和
            total_particles = clay + silt + sand
            if abs(total_particles - 1.0) > 0.01:
                st.warning(f"颗粒组成之和: {total_particles:.3f} (建议接近1.0)")
            else:
                st.success(f"颗粒组成之和: {total_particles:.3f}")

        with col2:
            st.markdown("### 🌿 土体与生物炭基本参数")

            # 干密度 - 手动输入
            dd = st.number_input(
                "干密度 (dd, g/cm³)",
                min_value=0.5,
                max_value=2.5,
                value=1.45,
                step=0.01,
                format="%.2f",
                help="土体干密度，单位：g/cm³",
                key="dd_input"
            )

            st.divider()

            # 生物炭掺量 - 手动输入
            bc_percent = st.number_input(
                "生物炭掺量 (BC, %)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.1,
                format="%.1f",
                help="生物炭掺量，单位：%",
                key="bc_percent_input"
            )

            # 转换为小数形式
            bc = bc_percent / 100.0

            st.divider()

            # 热解温度 - 根据BC值动态调整
            if bc == 0:
                # 当BC=0时，热解温度不存在
                st.markdown('<div class="warning-box">⚠️ 生物炭掺量为0，热解温度不存在</div>', unsafe_allow_html=True)
                temperature = 0.0
            else:
                temperature = st.number_input(
                    "热解温度 (temperature, °C)",
                    min_value=200.0,
                    max_value=900.0,
                    value=500.0,
                    step=10.0,
                    format="%.0f",
                    help="生物炭热解温度，单位：°C",
                    key="temperature_input"
                )

        with col3:
            st.markdown("### 🧪 生物炭类型参数")

            # 生物炭类型选择 - 根据BC值动态调整
            if bc == 0:
                # 当BC=0时，生物炭类型不存在
                st.markdown('<div class="warning-box">⚠️ 生物炭掺量为0，生物炭类型不存在</div>', unsafe_allow_html=True)
                biochar_type = "农业废弃物"  # 默认值，但不会影响预测
            else:
                biochar_type = st.selectbox(
                    "生物炭类型 (Biochar_type_combined)",
                    options=biochar_categories,
                    index=0,
                    help="选择生物炭的原材料类型",
                    key="biochar_type_input"
                )

            st.divider()

            # 参数汇总卡片
            st.markdown("### 📋 当前参数概览")

            param_summary = pd.DataFrame({
                '参数': ['吸力', '黏粒', '粉粒', '砂粒', '干密度', 'BC掺量', '热解温度', '生物炭类型'],
                '值': [
                    f"{suction:.3f} kPa",
                    f"{clay:.3f}",
                    f"{silt:.3f}",
                    f"{sand:.3f}",
                    f"{dd:.2f} g/cm³",
                    f"{bc_percent:.1f}%",
                    f"{temperature:.0f}°C",
                    biochar_type
                ]
            })

            st.dataframe(param_summary, use_container_width=True, hide_index=True)

    else:
        # 变量组二输入界面
        st.markdown('<div class="sub-header">🔬 输入参数 - 变量组二</div>', unsafe_allow_html=True)

        # 获取特征信息
        feature_order = model_info.get('feature_order', [])

        # 创建三列布局
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### 💧 吸力与土体参数")

            # 吸力参数 - 手动输入任意值
            suction = st.number_input(
                "基质吸力 (kPa)",
                min_value=0.001,
                max_value=1000000.0,
                value=100.0,
                step=1.0,
                format="%.3f",
                help="输入基质吸力值，单位kPa（可以是任意值）",
                key="suction_input_group2"
            )

            st.divider()

            # 土壤颗粒组成 - 手动输入
            st.markdown("**土壤颗粒组成（小数形式）**")

            clay = st.number_input(
                "黏粒含量 (clay)",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.01,
                format="%.3f",
                help="黏粒含量，范围0-1，如0.2表示20%",
                key="clay_input_group2"
            )

            silt = st.number_input(
                "粉粒含量 (silt)",
                min_value=0.0,
                max_value=1.0,
                value=0.25,
                step=0.01,
                format="%.3f",
                help="粉粒含量，范围0-1，如0.25表示25%",
                key="silt_input_group2"
            )

            sand = st.number_input(
                "砂粒含量 (sand)",
                min_value=0.0,
                max_value=1.0,
                value=0.55,
                step=0.01,
                format="%.3f",
                help="砂粒含量，范围0-1，如0.55表示55%",
                key="sand_input_group2"
            )

            # 显示颗粒组成之和
            total_particles = clay + silt + sand
            if abs(total_particles - 1.0) > 0.01:
                st.warning(f"颗粒组成之和: {total_particles:.3f} (建议接近1.0)")
            else:
                st.success(f"颗粒组成之和: {total_particles:.3f}")

        with col2:
            st.markdown("### 🌿 土体与生物炭基本参数")

            # 干密度 - 手动输入
            dd = st.number_input(
                "干密度 (dd, g/cm³)",
                min_value=0.5,
                max_value=2.5,
                value=1.45,
                step=0.01,
                format="%.2f",
                help="土体干密度，单位：g/cm³",
                key="dd_input_group2"
            )

            st.divider()

            # 生物炭掺量 - 手动输入
            bc_percent = st.number_input(
                "生物炭掺量 (BC, %)",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=0.1,
                format="%.1f",
                help="生物炭掺量，单位：%",
                key="bc_percent_input_group2"
            )

            # 转换为小数形式
            bc = bc_percent / 100.0

            st.divider()

            # pH值 - 根据BC值动态调整
            if bc == 0:
                # 当BC=0时，pH为0
                st.markdown('<div class="warning-box">⚠️ 生物炭掺量为0，pH值为0</div>', unsafe_allow_html=True)
                ph = 0.0
            else:
                ph = st.number_input(
                    "pH值 (pH)",
                    min_value=0.0,
                    max_value=14.0,
                    value=8.0,
                    step=0.1,
                    format="%.1f",
                    help="生物炭pH值",
                    key="ph_input"
                )

        with col3:
            st.markdown("### 🧪 生物炭理化参数")

            # 根据BC值动态调整
            if bc == 0:
                st.markdown('<div class="warning-box">⚠️ 生物炭掺量为0，以下参数自动设为0</div>', unsafe_allow_html=True)
                at = 0.0
                ct = 0.0
            else:
                # 灰分含量（百分数形式）- 手动输入
                at = st.number_input(
                    "灰分含量 (AT, %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=25.0,
                    step=0.1,
                    format="%.1f",
                    help="生物炭灰分含量，单位：%",
                    key="at_input"
                )

                # 碳含量（百分数形式）- 手动输入
                ct = st.number_input(
                    "碳含量 (CT, %)",
                    min_value=0.0,
                    max_value=100.0,
                    value=60.0,
                    step=0.1,
                    format="%.1f",
                    help="生物炭碳含量，单位：%",
                    key="ct_input"
                )

            st.divider()

            # 参数汇总卡片
            st.markdown("### 📋 当前参数概览")

            param_summary = pd.DataFrame({
                '参数': ['吸力', '黏粒', '粉粒', '砂粒', '干密度', 'BC掺量', 'pH', 'AT', 'CT'],
                '值': [
                    f"{suction:.3f} kPa",
                    f"{clay:.3f}",
                    f"{silt:.3f}",
                    f"{sand:.3f}",
                    f"{dd:.2f} g/cm³",
                    f"{bc_percent:.1f}%",
                    f"{ph:.1f}",
                    f"{at:.1f}%",
                    f"{ct:.1f}%"
                ]
            })

            st.dataframe(param_summary, use_container_width=True, hide_index=True)

    # 页面底部的预测按钮
    st.divider()

    # 预测按钮容器
    predict_container = st.container()
    with predict_container:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

        with col_btn2:
            if st.button("🚀 开始单点预测", type="primary", use_container_width=True):
                # 调用单点预测函数
                single_point_prediction(models, model_type, model_info, feature_info, locals())


def display_batch_prediction_interface(models, model_type, model_info, feature_info):
    """显示批量预测界面"""
    st.markdown('<div class="sub-header">📊 批量预测</div>', unsafe_allow_html=True)

    # 显示数据格式要求
    st.markdown("""
    <div class="info-box">
    <strong>📋 数据格式要求：</strong>
    1. 上传CSV或Excel文件，包含多组参数
    2. 文件必须包含以下列（根据所选模型）：
    """, unsafe_allow_html=True)

    if model_type == 'group1':
        st.markdown("""
        - **变量组一必需列：** suction, clay, silt, sand, dd, BC, temperature, Biochar_type_combined
        - **注意事项：** BC列为百分数（如5表示5%），Biochar_type_combined为生物炭类型
        - **生物炭类型选项：** 农业废弃物, 林业残余物, 畜禽粪便, 城市污泥, 其他
        """)
    else:
        st.markdown("""
        - **变量组二必需列：** suction, clay, silt, sand, dd, BC, pH, AT, CT
        - **注意事项：** BC列为百分数（如5表示5%），AT和CT为百分数（如25表示25%）
        - **约束条件：** 当BC=0时，pH、AT、CT自动设为0
        """)

    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ 重要提示：</strong>
    1. 确保文件编码为UTF-8
    2. 土壤颗粒组成(clay, silt, sand)之和应在0-1范围内
    3. 建议先下载模板文件进行数据准备
    </div>
    """, unsafe_allow_html=True)

    # 创建两列布局
    col1, col2 = st.columns([2, 1])

    with col1:
        # 文件上传
        uploaded_file = st.file_uploader(
            "上传数据文件 (CSV或Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="选择包含批量预测数据的文件",
            key="batch_file_uploader"
        )

    with col2:
        # 下载模板文件
        st.markdown("### 📥 下载模板")
        if model_type == 'group1':
            # 创建变量组一模板
            template_data = {
                'suction': [100.0, 1000.0, 10000.0],
                'clay': [0.2, 0.3, 0.1],
                'silt': [0.25, 0.3, 0.2],
                'sand': [0.55, 0.4, 0.7],
                'dd': [1.45, 1.5, 1.4],
                'BC': [5.0, 10.0, 0.0],  # 百分数
                'temperature': [500, 600, 0],
                'Biochar_type_combined': ['农业废弃物', '林业残余物', '农业废弃物']
            }
            template_df = pd.DataFrame(template_data)
            csv = template_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="下载变量组一模板",
                data=csv,
                file_name="template_group1.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_template_group1"
            )
        else:
            # 创建变量组二模板
            template_data = {
                'suction': [100.0, 1000.0, 10000.0],
                'clay': [0.2, 0.3, 0.1],
                'silt': [0.25, 0.3, 0.2],
                'sand': [0.55, 0.4, 0.7],
                'dd': [1.45, 1.5, 1.4],
                'BC': [5.0, 10.0, 0.0],  # 百分数
                'pH': [8.0, 7.5, 0.0],
                'AT': [25.0, 30.0, 0.0],  # 百分数
                'CT': [60.0, 65.0, 0.0]  # 百分数
            }
            template_df = pd.DataFrame(template_data)
            csv = template_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="下载变量组二模板",
                data=csv,
                file_name="template_group2.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_template_group2"
            )

    # 如果有文件上传，显示预览和进行预测
    if uploaded_file is not None:
        try:
            # 读取文件
            if uploaded_file.name.endswith('.csv'):
                data_df = pd.read_csv(uploaded_file)
            else:
                data_df = pd.read_excel(uploaded_file)

            # 显示数据预览
            st.markdown("### 📋 数据预览")
            st.write(f"文件: {uploaded_file.name}")
            st.write(f"数据行数: {len(data_df)}")
            st.write(f"数据列数: {len(data_df.columns)}")

            # 显示前几行数据
            with st.expander("查看数据详情"):
                st.dataframe(data_df.head(10))

            # 验证数据
            st.markdown("### 🔍 数据验证")
            validation_errors = validate_batch_data(data_df, model_type, feature_info)

            if validation_errors:
                st.error("❌ 数据验证失败：")
                for error in validation_errors:
                    st.error(f"  - {error}")
                return
            else:
                st.success("✅ 数据验证通过")

            # 开始批量预测
            st.markdown("### 🚀 开始批量预测")
            if st.button("开始批量预测", type="primary", use_container_width=True, key="batch_predict_button"):
                with st.spinner("正在进行批量预测..."):
                    model = models[model_type]

                    # 根据模型类型选择预测函数
                    if model_type == 'group1':
                        predictions = batch_predict_group1(model, data_df, feature_info)
                    else:
                        predictions = batch_predict_group2(model, data_df, feature_info)

                    # 添加预测结果到数据框
                    result_df = data_df.copy()
                    result_df['预测体积含水率'] = predictions

                    # 计算预测成功率
                    success_rate = (1 - result_df['预测体积含水率'].isna().sum() / len(result_df)) * 100

                    # 显示预测结果
                    st.markdown("### 📊 批量预测结果")
                    st.success(f"✅ 批量预测完成！预测成功率: {success_rate:.1f}%")

                    # 显示结果统计
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("总样本数", len(result_df))
                    with col_stat2:
                        st.metric("预测成功数", len(result_df) - result_df['预测体积含水率'].isna().sum())
                    with col_stat3:
                        st.metric("预测失败数", result_df['预测体积含水率'].isna().sum())

                    # 显示结果预览
                    st.markdown("#### 🔍 预测结果预览")
                    st.dataframe(result_df.head(10))

                    # 绘制预测结果分布
                    st.markdown("#### 📈 预测结果分布")
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # 确保使用中文字体
                    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
                    plt.rcParams['axes.unicode_minus'] = False

                    # 绘制直方图
                    valid_predictions = result_df['Prediction of volumetric moisture content'].dropna()
                    if len(valid_predictions) > 0:
                        ax.hist(valid_predictions, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                        ax.axvline(valid_predictions.mean(), color='red', linestyle='--', linewidth=2,
                                   label=f'mean value: {valid_predictions.mean():.3f}')
                        ax.axvline(valid_predictions.median(), color='green', linestyle='--', linewidth=2,
                                   label=f'median: {valid_predictions.median():.3f}')

                        ax.set_xlabel('Volumetric water content', fontsize=12)
                        ax.set_ylabel('frequency', fontsize=12)
                        ax.set_title('Distribution histogram of prediction results', fontsize=14, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        st.pyplot(fig)

                        # 显示统计信息
                        col_stat4, col_stat5, col_stat6, col_stat7 = st.columns(4)
                        with col_stat4:
                            st.metric("平均值", f"{valid_predictions.mean():.4f}")
                        with col_stat5:
                            st.metric("中位数", f"{valid_predictions.median():.4f}")
                        with col_stat6:
                            st.metric("最小值", f"{valid_predictions.min():.4f}")
                        with col_stat7:
                            st.metric("最大值", f"{valid_predictions.max():.4f}")

                    # 提供结果下载
                    st.markdown("#### 💾 下载预测结果")

                    # 创建下载按钮
                    csv_result = result_df.to_csv(index=False).encode('utf-8')
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="📥 下载CSV格式",
                            data=csv_result,
                            file_name=f"batch_predictions_{model_type}_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key="download_csv_batch"
                        )

                    with col_dl2:
                        # 转换为Excel格式
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            result_df.to_excel(writer, index=False, sheet_name='prediction results')
                        excel_buffer.seek(0)

                        st.download_button(
                            label="📥 下载Excel格式",
                            data=excel_buffer,
                            file_name=f"batch_predictions_{model_type}_{timestamp}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_excel_batch"
                        )

                    # 显示预测结果详情
                    with st.expander("📋 查看详细预测结果"):
                        st.dataframe(result_df)

        except Exception as e:
            st.error(f"❌ 文件处理失败: {e}")
            st.error("请检查文件格式和内容是否正确")


def single_point_prediction(models, model_type, model_info, feature_info, local_vars):
    """执行单点预测"""
    # 从局部变量获取输入参数
    suction = local_vars.get('suction', 100.0)
    clay = local_vars.get('clay', 0.2)
    silt = local_vars.get('silt', 0.25)
    sand = local_vars.get('sand', 0.55)
    dd = local_vars.get('dd', 1.45)
    bc_percent = local_vars.get('bc_percent', 5.0)
    bc = bc_percent / 100.0

    # 验证输入
    total_particles = clay + silt + sand
    if total_particles > 1.0:
        st.error("❌ 黏粒、粉粒、砂粒含量之和不能超过1.0！")
        return

    if model_type not in models:
        st.error(f"❌ 模型 {model_type} 未加载成功")
        return

    model = models[model_type]

    # 根据模型类型准备输入数据
    if model_type == 'group1':
        # 变量组一：使用分类特征
        biochar_categories = model_info.get('biochar_categories',
                                            ["农业废弃物", "林业残余物", "畜禽粪便", "城市污泥", "其他"])
        temperature = local_vars.get('temperature', 500.0)
        biochar_type = local_vars.get('biochar_type', "农业废弃物")

        # 当BC=0时，热解温度和生物炭类型设为默认值
        if bc == 0:
            temperature = 0.0
            biochar_type = "农业废弃物"  # 默认值

        # 创建特征DataFrame - 按照训练时的特征顺序
        features_dict = {
            'suction': float(suction),
            'clay': float(clay),
            'silt': float(silt),
            'sand': float(sand),
            'dd': float(dd),
            'BC': float(bc),  # 小数形式
            'temperature': float(temperature),  # 使用 temperature 而不是 Temp
            'Biochar_type_combined': biochar_type  # 分类特征
        }

        # 创建DataFrame，确保Biochar_type_combined是category类型
        features_df = pd.DataFrame([features_dict])

        # 将Biochar_type_combined转换为category类型
        features_df['Biochar_type_combined'] = pd.Categorical(
            features_df['Biochar_type_combined'],
            categories=biochar_categories
        )

        # 按照特征顺序重新排列列
        feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'temperature', 'Biochar_type_combined']
        features_df = features_df[feature_order]

        # 保存输入数据用于显示
        input_data = {
            'suction': float(suction),
            'clay': float(clay),
            'silt': float(silt),
            'sand': float(sand),
            'dd': float(dd),
            'BC': float(bc),
            'temperature': float(temperature),
            'Biochar_type_combined': biochar_type
        }

    else:
        # 变量组二：直接使用原始值
        ph = local_vars.get('ph', 8.0)
        at = local_vars.get('at', 25.0)
        ct = local_vars.get('ct', 60.0)

        if bc == 0:
            # 确保当BC=0时，AT、CT、pH为0
            ph = 0.0
            at = 0.0
            ct = 0.0

        # 创建特征DataFrame
        feature_order = ['suction', 'clay', 'silt', 'sand', 'dd', 'BC', 'pH', 'AT', 'CT']

        features = [
            float(suction),
            float(clay),
            float(silt),
            float(sand),
            float(dd),
            float(bc),  # 小数形式
            float(ph),
            float(at),  # 百分数形式
            float(ct)  # 百分数形式
        ]

        features_df = pd.DataFrame([features], columns=feature_order)

        # 保存输入数据用于显示
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

    # 进行预测
    with st.spinner("正在进行预测计算..."):
        try:
            # 显示输入特征（用于调试）
            with st.expander("🔍 查看输入特征值"):
                st.write(f"模型类型: {model_type}")
                st.write("输入特征值:")
                st.dataframe(features_df)

                # 如果模型有feature_names_in_属性，显示模型期望的特征
                if hasattr(model, 'feature_names_in_'):
                    st.write("模型期望的特征:", list(model.feature_names_in_))
                    st.write("输入的特征:", list(features_df.columns))

            # 进行预测
            prediction = model.predict(features_df)[0]

            # 显示预测结果
            st.markdown('<div class="sub-header">📊 预测结果</div>', unsafe_allow_html=True)

            # 创建结果展示区域
            col_a, col_b = st.columns([2, 1])

            with col_a:
                st.markdown(f"""
                <div class="success-box" style="text-align: center;">
                    <h2 style="margin: 0;">预测体积含水率</h2>
                    <h1 style="color: #2E8B57; font-size: 3rem; margin: 10px 0;">{prediction:.4f}</h1>
                    <p>单位体积土壤中水的体积</p>
                </div>
                """, unsafe_allow_html=True)

                # 辅助指标
                col_a1, col_a2 = st.columns(2)

                with col_a1:
                    # 计算饱和度（假设孔隙率为0.4）
                    porosity = 0.4
                    saturation = (prediction / porosity) * 100 if porosity > 0 else 0
                    st.metric("估算饱和度", f"{saturation:.1f}%")

                with col_a2:
                    # 提供定性评估
                    if prediction > 0.35:
                        assessment = "高持水能力"
                        color = "#32CD32"
                        emoji = "🔵"
                    elif prediction > 0.2:
                        assessment = "中等持水能力"
                        color = "#FFA500"
                        emoji = "🟡"
                    else:
                        assessment = "低持水能力"
                        color = "#FF4500"
                        emoji = "🔴"

                    st.markdown(f"**评估:** {emoji} <span style='color:{color};font-weight:bold'>{assessment}</span>",
                                unsafe_allow_html=True)

            with col_b:
                st.markdown("### 📋 输入参数详情")

                # 显示输入参数
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
                        unit = "小数"
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
                        display_value = str(value)
                        unit = "类型"
                    else:
                        display_value = str(value)
                        unit = ""

                    detail_data.append({
                        '参数': key,
                        '值': display_value,
                        '单位': unit
                    })

                detail_df = pd.DataFrame(detail_data)
                st.dataframe(detail_df, use_container_width=True, hide_index=True)

                # 下载按钮
                st.download_button(
                    label="📥 下载预测结果",
                    data=detail_df.to_csv(index=False).encode('utf-8'),
                    file_name=f"SWCC预测_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_single_result"
                )

            # 从session state获取SWCC曲线设置
            curve_points = st.session_state.get('curve_points', 100)
            min_suction = st.session_state.get('min_suction', 0.01)
            max_suction = st.session_state.get('max_suction', 284804.0)
            enable_vg_fitting = st.session_state.get('enable_vg_fitting', True)

            # 检查max_suction是否大于min_suction
            if max_suction <= min_suction:
                st.warning("最大吸力必须大于最小吸力，已自动调整")
                max_suction = min_suction * 100
                st.session_state['max_suction'] = max_suction

            # 生成SWCC曲线
            st.markdown('<div class="sub-header">📈 SWCC曲线</div>', unsafe_allow_html=True)

            # 生成吸力范围（对数均匀分布）
            suction_range = np.logspace(np.log10(min_suction), np.log10(max_suction), curve_points)

            # 生成SWCC曲线数据
            with st.spinner("正在生成SWCC曲线..."):
                predictions = generate_swcc_curve(model, model_type, input_data, suction_range)

                # VG模型拟合
                vg_params = None
                r_squared = 0
                fitted_curve = None

                if enable_vg_fitting:
                    with st.spinner("正在进行VG模型拟合..."):
                        popt, pcov, r_squared, fitted_curve = fit_vg_model(suction_range, predictions)

                        if popt is not None:
                            vg_params = popt
                            st.success(f"✅ VG模型拟合成功！R² = {r_squared:.6f}")

                # 绘制SWCC曲线
                current_point = (suction, prediction) if suction >= min_suction and suction <= max_suction else None

                fig = plot_swcc_with_vg_fit(suction_range, predictions, vg_params, current_point)

                st.pyplot(fig)

                # 显示VG模型参数
                if enable_vg_fitting and vg_params is not None:
                    display_vg_parameters(vg_params, r_squared, suction_range, predictions)

                # 提供曲线数据下载
                curve_data = pd.DataFrame({
                    'Suction(kPa)': suction_range,
                    'Volumetric_Water_Content': predictions
                })

                # 如果有VG拟合结果，添加到数据中
                if fitted_curve is not None:
                    curve_data['VG_Fitted_Water_Content'] = fitted_curve
                    curve_data['Residual'] = predictions - fitted_curve

                csv_curve = curve_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 下载SWCC曲线数据",
                    data=csv_curve,
                    file_name=f"SWCC曲线_{model_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="download_swcc_curve"
                )

        except Exception as e:
            st.error(f"❌ 预测失败: {e}")
            st.error("请检查特征顺序或模型文件")

            # 显示详细错误信息
            with st.expander("🔍 查看详细错误信息"):
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()