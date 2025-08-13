#region --- LIBRARIES ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from helpers.models import *
import warnings
import math
import streamlit as st
import time
import random

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
#endregion

# region SIDEBAR
models = [
    'Kinetic Model', 'Gompertz Model', 'Generalized Gompertz Model', 'Modified Gompertz Model',
    'Re-modified Gompertz Model', 'Logistic Model', 'Generalized Logistic Model', 'Modified Logistic Model',
    'Re-modified Logistic Model', 'Richards Model', 'Generalized Richards Model', 'Modified Richards Model',
    'Re-modified Richards Model', 'Stannard Model', 'Weibull Model', 'Morgen-Mercer-Flodin Model', 'Baranyi Model',
    'Huang Model', 'Fitzhugh Model', 'Cone Model', 'Asymmetric Model'
]

# Set up the sidebar
st.sidebar.title(f"💾 Data 💾")

# Add components to the sidebar
with st.sidebar.expander(f"**Load Data**"):
    uploaded_file_1 = st.file_uploader('Upload a Data File', type=['xlsx', 'csv', 'txt'])

    # st.subheader(f"**Load Cost Data**")
    # uploaded_file_2 = st.file_uploader('Upload a Cost File', type=['xlsx', 'csv', 'txt'], key='unique_key_for_file_uploader')

    # Upload default datasets
    df = pd.read_excel(r"datasets\suspended_cells.xlsx") if uploaded_file_1 is None else pd.read_excel(uploaded_file_1)

selected_models = st.sidebar.multiselect('**Select models**', sorted(models), default=sorted(models))

# endregion

# region DATA PREPROCESSING

model_functions = {
    "Kinetic Model": kinetic_modeling,
    "Gompertz Model": gompertz_model,
    "Modified Gompertz Model": modified_gompertz_model,
    "Logistic Model": logistic_model,
    "Modified Logistic Model": modified_logistic_model,
    "Generalized Gompertz Model": generalized_gompertz_model,
    "Re-Modified Gompertz Model": re_modified_gompertz_model,
    "Generalized Logistic Model": generalized_logistic_model,
    "Re-Modified Logistic Model": re_modified_logistic_model,
    "Richards Model": richards_model,
    "Modified Richards Model": modified_richards_model,
    "Stannard Model": stannard_model,
    "Weibull Model": weibull_model,
    "Morgen-Mercer-Flodin Model": morgen_mercer_flodin_model,
    "Baranyi Model": baranyi_model,
    "Huang Model": huang_model,
    "Fitzhugh Model": fitzhugh_model,
    "Cone Model": cone_model,
    "Asymmetric Model": asymmetric_model,
    "Generalized Richards Model": generalized_richards_model,
    "Re-Modified Richards Model": re_modified_richards_model,
}

def ensure_dict(val, key_name="value"):
    if isinstance(val, dict):
        return val
    elif hasattr(val, "to_dict"):
        return val.to_dict()
    else:
        return {key_name: val}

model_params_dict = {}
kinetic_params_dict = {}
statistics_dict = {}
sensitivity_dict = {}

for model_name in selected_models:
    model_func = model_functions.get(model_name)
    if model_func is None:
        continue

    try:
        result = model_func(df)
        if len(result) == 5:
            dataframe, popt, _, _, model_parameters = result
        elif len(result) == 4:
            dataframe, popt, _, model_parameters = result
        elif len(result) == 3:
            dataframe, popt, model_parameters = result
        else:
            continue
    except Exception:
        continue

    errors, df_statistics = errors_evaluation(dataframe)
    kinetic_params = kinetic_parameters(dataframe)
    sensitivity_results = sensitivity_analysis(dataframe)

    model_params_dict[model_name] = ensure_dict(model_parameters, "model_parameter")
    kinetic_params_dict[model_name] = ensure_dict(kinetic_params, "kinetic_parameter")
    statistics_dict[model_name] = ensure_dict(df_statistics, "statistic")
    sensitivity_dict[model_name] = ensure_dict(sensitivity_results, "sensitivity")

#endregion

#region --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Multi-Mathematical Modeling",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "This app is designed for multi-mathematical modeling of fermentations related to value-added products production. It allows users to visualize experimental data, model predictions, and perform statistical analyses."
    }
)
#endregion

# İki eşit genişlikte kolon oluşturun
st.title("Multi-Mathematical Modeling of Fermentations Related to Value-Added Products Production")

col1, col2 = st.columns(2)

with col1:
    #region --- MATHEMATICAL MODELS ---
    st.markdown("### Mathematical Models for Fermentation")
    st.markdown("This section provides the mathematical models used for the fermentation processes related to value-added products production. Each model is represented with its respective equations.")

    # Display each model with its equations
    st.subheader("Asymmetric Model")
    st.latex(r'''
            \begin{align*}
            X(t) &= X_{\text{min}} + (X_{\text{max}} - X_{\text{min}}) \cdot \left(1 - \frac{1}{\left(1 + \left(\frac{t}{T_x}\right)^{\gamma_x}\right)}\right) \\
            P(t) &= P_{\text{min}} + (P_{\text{max}} - P_{\text{min}}) \cdot \left(1 - \frac{1}{\left(1 + \left(\frac{t}{T_p}\right)^{\gamma_p}\right)}\right) \\
            S(t) &= S_{\text{max}} - (S_{\text{max}} - S_{\text{min}}) \cdot \left(1 - \frac{1}{\left(1 + \left(\frac{t}{T_s}\right)^{\gamma_s}\right)}\right)
            \end{align*}
            '''
    )

    st.subheader("Baranyi Model")
    st.latex(r"""
            h_t = Q_x \cdot \lambda_x \\
            b_t = t + \frac{1}{Q_x} \cdot \log\left(
                \exp(-Q_x \cdot t) +
                \exp(-h_t) -
                \exp(-Q_x \cdot t - h_t)
            \right) \\
            X(t) = X_{\min} + Q_x \cdot b_t - \log\left(
                1 + \frac{\exp(Q_x \cdot b_t) - 1}{\exp(X_{\max} - X_{\min})}
            \right)
            """)
    st.latex(r"""
    h_t = Q_p \cdot \lambda_p \\
    b_t = t + \frac{1}{Q_p} \cdot \log\left(
        \exp(-Q_p \cdot t) +
        \exp(-h_t) -
        \exp(-Q_p \cdot t - h_t)
    \right) \\
    P(t) = P_{\min} + Q_p \cdot b_t - \log\left(
        1 + \frac{\exp(Q_p \cdot b_t) - 1}{\exp(P_{\max} - P_{\min})}
    \right)
    """)
    st.latex(r"""
    h_t = Q_s \cdot \lambda_s \\
    b_t = t + \frac{1}{Q_s} \cdot \log\left(
        \exp(-Q_s \cdot t) +
        \exp(-h_t) -
        \exp(-Q_s \cdot t - h_t)
    \right) \\
    S(t) = S_{\min} - Q_s \cdot b_t + \log\left(
        1 + \frac{\exp(Q_s \cdot b_t) - 1}{\exp(S_{\max} - S_{\min})}
    \right)
    """)

    st.subheader("Cone Model")
    st.latex(r'''
            \begin{align*}
            X(t) &= X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{1 + \left(\frac{1}{{\lambda_x \cdot t}}\right)^{\sigma_x}}} \\
            P(t) &= P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{1 + \left(\frac{1}{{\lambda_p \cdot t}}\right)^{\sigma_p}}} \\
            S(t) &= S_{\text{max}} - \frac{{S_{\text{max}} - S_{\text{min}}}}{{1 + \left(\frac{1}{{\lambda_s \cdot t}}\right)^{\sigma_s}}}
            \end{align*}
            ''')

    st.subheader("Fitzhugh Model")
    st.latex(r'''
            \begin{align*}
            X(t) & = X_{\text{min}} + (X_{\text{max}} - X_{\text{min}}) \cdot (1 - \exp(-\lambda_x \cdot t))^{\theta_x} \\
            P(t) & = P_{\text{min}} + (P_{\text{max}} - P_{\text{min}}) \cdot (1 - \exp(-\lambda_p \cdot t))^{\theta_p} \\
            S(t) & = S_{\text{max}} - (S_{\text{max}} - S_{\text{min}}) \cdot (1 - \exp(-\lambda_s \cdot t))^{\theta_s} \\
            \end{align*}
            ''')

    st.subheader("Generalized Gompertz Model")
    st.latex(r'X(t) = X_{\text{min}} + (X_{\text{max}} - X_{\text{min}}) \cdot \exp\left(-\exp(Q_x \cdot (I_x - t))\right)')
    st.latex(r'P(t) = P_{\text{min}} + (P_{\text{max}} - P_{\text{min}}) \cdot \exp\left(-\exp(Q_p \cdot (I_p - t))\right)')
    st.latex(r'S(t) = S_{\text{min}} + (S_{\text{max}} - S_{\text{min}}) \cdot \exp\left(-\exp(-Q_s \cdot (I_s - t))\right)')

    st.subheader("Generalized Logistic Model")
    st.latex(r'''
    \begin{align*}
    X(t) = X_{\text{min}} + \frac{X_{\text{max}} - X_{\text{min}}}{1 + \exp(Q_x \cdot (I_x - t))} \\
    P(t) = P_{\text{min}} + \frac{P_{\text{max}} - P_{\text{min}}}{1 + \exp(Q_p \cdot (I_p - t))} \\
    S(t) = S_{\text{min}} + \frac{S_{\text{max}} - S_{\text{min}}}{1 + \exp(-Q_s \cdot (I_s - t))}
    \end{align*}
    ''')

    st.subheader("Generalized Richards Model")
    st.latex(r'''
            \begin{align*}
            X(t) & = X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{(1 + v_x \exp(Q_x (I_x - t)))^{1 / v_x}}} \\
            P(t) & = P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{(1 + v_p \exp(Q_p (I_p - t)))^{1 / v_p}}} \\
            S(t) & = S_{\text{min}} + \frac{{S_{\text{max}} - S_{\text{min}}}}{{(1 + v_s \exp(Q_s (t - I_s)))^{1 / v_s}}}
            \end{align*}
            ''')

    st.subheader("Gompertz Model")
    st.latex(r"X(t) = X_{\text{max}} \cdot \exp\left(-\exp(Q_x \cdot (I_x - t))\right)")
    st.latex(r"P(t) = P_{\text{max}} \cdot \exp\left(-\exp(Q_p \cdot (I_p - t))\right)")
    st.latex(r"S(t) = S_{\text{max}} \cdot \exp\left(-\exp(-Q_s \cdot (I_s - t))\right)")

    st.subheader("Huang Model")
    st.latex(r'''
            \begin{align*}
            h_t & = t + \frac{1}{4} \cdot \log\left(\frac{1 + \exp(-4 \cdot (t - \lambda_x))}{1 + \exp(4 \cdot \lambda_x)}\right) \\
            X(t) & = X_{\text{min}} + X_{\text{max}} - \log\left(\exp(X_{\text{min}}) + (\exp(X_{\text{max}}) - \exp(X_{\text{min}})) \cdot \exp(-Q_x \cdot h_t)\right) \\
            h_t & = t + \frac{1}{4} \cdot \log\left(\frac{1 + \exp(-4 \cdot (t - \lambda_p))}{1 + \exp(4 \cdot \lambda_p)}\right) \\
            P(t) & = P_{\text{min}} + P_{\text{max}} - \log\left(\exp(P_{\text{min}}) + (\exp(P_{\text{max}}) - \exp(P_{\text{min}})) \cdot \exp(-Q_p \cdot h_t)\right) \\
            h_t & = t + \frac{1}{4} \cdot \log\left(\frac{1 + \exp(-4 \cdot (t - \lambda_s))}{1 + \exp(4 \cdot \lambda_s)}\right) \\
            S(t) & = S_{\text{min}} + S_{\text{max}} + \log\left(\exp(S_{\text{min}}) + (\exp(S_{\text{max}}) - \exp(S_{\text{min}})) \cdot \exp(-Q_s \cdot h_t)\right) \\
            \end{align*}
            ''')

    st.subheader("Kinetic Model")
    st.latex(r"X(t) = \frac{{{X_0} \cdot {X_{\max}}}}{{{X_0} + ({X_{\max}} - {X_0}) \cdot \exp(-{\mu} \cdot t)}}")
    st.latex(r"\frac{{dP}}{{dt}} = {P_0} + {a} \cdot X({t}) + {β} \cdot {X_0} \cdot \left(\frac{{{X_{\max}}}}{{\mu}}\right) \cdot \left(e^{{\mu {t}}} - 1\right)")
    st.latex(r"- \frac{dS}{dt} = S_0 - Y_{xs} \cdot (X(t) - X_0) - γ \cdot X(t) - m \cdot X_0 \cdot \left(\frac{X_{\max}}{\mu}\right) \cdot \left(e^{\mu t} - 1\right)")

    st.subheader("Logistic Model")
    st.latex(r'''
    \begin{align*}
    X(t) = \frac{X_{\text{max}}}{1 + \exp(Q_x \cdot (I_x - t))} \\
    P(t) = \frac{P_{\text{max}}}{1 + \exp(Q_p \cdot (I_p - t))} \\
    S(t) = \frac{S_{\text{max}}}{1 + \exp(-Q_s \cdot (I_s - t))}
    \end{align*}
    ''')

    st.subheader("Modified Gompertz Model")
    st.latex(r'''
    \begin{align*}
    X(t) &= X_{\text{max}} \cdot \exp\left(-\exp\left(\frac{Q_x \cdot \text{e} \cdot (\lambda_x - t) + X_{\text{max}}}{X_{\text{max}}}\right)\right) \\
    P(t) &= P_{\text{max}} \cdot \exp\left(-\exp\left(\frac{Q_p \cdot \text{e} \cdot (\lambda_p - t) + P_{\text{max}}}{P_{\text{max}}}\right)\right) \\
    S(t) &= S_{\text{max}} \cdot \exp\left(-\exp\left(-\frac{Q_s \cdot \text{e} \cdot (\lambda_s - t) + S_{\text{max}}}{S_{\text{max}}}\right)\right)
    \end{align*}
    ''')

    st.subheader("Modified Logistic Model")
    st.latex(r'''
    \begin{align*} 
    X(t) = \frac{X_{\text{max}}}{1 + \exp\left(\frac{4 Q_x (t - \lambda_x)}{X_{\text{max}}} + 2\right)} \\
    P(t) = \frac{P_{\text{max}}}{1 + \exp\left(\frac{4 Q_p (\lambda_p - t)}{P_{\text{max}}} + 2\right)} \\
    S(t) = \frac{S_{\text{max}}}{1 + \exp\left(\frac{4 (-Q_s) (\lambda_s - t)}{S_{\text{max}}} + 2\right)}
    \end{align*}''')

    st.subheader("Modified Richards Model")
    st.latex(r'''
    \begin{align*}
    X(t) = \frac{X_{\text{max}}}{\left(1 + v_x \cdot \exp(1 + v_x) \cdot \exp\left(Q_x \cdot (1 + v_x)^{\left(1 + \frac{1}{v_x}\right)} \cdot \frac{(\lambda_x - t)}{X_{\text{max}}}\right)\right)^{\frac{1}{v_x}}} \\
    P(t) = \frac{P_{\text{max}}}{\left(1 + v_p \cdot \exp(1 + v_p) \cdot \exp\left(Q_p \cdot (1 + v_p)^{\left(1 + \frac{1}{v_p}\right)} \cdot \frac{(\lambda_p - t)}{P_{\text{max}}}\right)\right)^{\frac{1}{v_p}}} \\
    S(t) = \frac{S_{\text{max}}}{\left(1 + v_s \cdot \exp(1 + v_s) \cdot \exp\left(-Q_s \cdot (1 + v_s)^{\left(1 + \frac{1}{v_s}\right)} \cdot \frac{(\lambda_s - t)}{S_{\text{max}}}\right)\right)^{\frac{1}{v_s}}}
    \end{align*}
    ''')

    st.subheader("Morgen-Mercer-Flodin Model")
    st.latex(r'''
    \begin{align*}
    X(t) &= X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{1 + (\lambda_x t)^{\gamma_x}}} \\
    P(t) &= P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{1 + (\lambda_p t)^{\gamma_p}}} \\
    S(t) &= S_{\text{min}} + \frac{{S_{\text{max}} - S_{\text{min}}}}{{1 + (\lambda_s t)^{\gamma_s}}}
    \end{align*}
    ''')

    st.subheader("Re-Modified Gompertz Model")
    st.latex(r'''
    \begin{align*}
    X(t) = X_{\text{min}} + (X_{\text{max}} - X_{\text{min}}) \cdot \exp\left(-\exp\left(\frac{Q_x \cdot e \cdot (\lambda_x - t) + X_{\text{max}}}{X_{\text{max}}}\right)\right) \\
    P(t) = P_{\text{min}} + (P_{\text{max}} - P_{\text{min}}) \cdot \exp\left(-\exp\left(\frac{Q_p \cdot e \cdot (\lambda_p - t) + P_{\text{max}}}{P_{\text{max}}}\right)\right) \\
    S(t) = S_{\text{min}} + (S_{\text{max}} - S_{\text{min}}) \cdot \exp\left(-\exp\left(-\frac{Q_s \cdot e \cdot (\lambda_s - t) + S_{\text{max}}}{S_{\text{max}}}\right)\right)
    \end{align*}
    ''')

    st.subheader("Re-Modified Logistic Model")
    st.latex(r'''
    \begin{align*}
    X(t) & = X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{1 + \exp\left(\frac{{4 \cdot Q_x \cdot (\lambda_x - t)}}{{X_{\text{max}}}} + 2\right)}} \\
    P(t) & = P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{1 + \exp\left(\frac{{4 \cdot Q_p \cdot (\lambda_p - t)}}{{P_{\text{max}}}} + 2\right)}} \\
    S(t) & = S_{\text{min}} + \frac{{S_{\text{max}} - S_{\text{min}}}}{{1 + \exp\left(\frac{{4 \cdot Q_s \cdot (t - \lambda_s)}}{{S_{\text{max}}}} + 2\right)}}
    \end{align*}
    ''')

    st.subheader("Re-Modified Richards Model")
    st.latex(r"""
    \begin{align*}
    X(t) &= X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{(1 + v_x \exp(1 + v_x) \exp(Q_x (1 + v_x)^{\left(1 + \frac{1}{v_x}\right)} (\lambda_x - t) / X_{\text{max}}))^{\frac{1}{v_x}}}} \\
    P(t) &= P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{(1 + v_p \exp(1 + v_p) \exp(Q_p (1 + v_p)^{\left(1 + \frac{1}{v_p}\right)} (\lambda_p - t) / P_{\text{max}}))^{\frac{1}{v_p}}}} \\
    S(t) &= S_{\text{min}} + \frac{{S_{\text{max}} - S_{\text{min}}}}{{(1 + v_s \exp(1 + v_s) \exp(-Q_s (1 + v_s)^{\left(1 + \frac{1}{v_s}\right)} (\lambda_s - t) / S_{\text{max}}))^{\frac{1}{v_s}}}} \\
    \end{align*}
    """)

    st.subheader("Richards Model")
    st.latex(r'''
    \begin{align*}
    X(t) = \frac{X_{\text{max}}}{\left(1 + v_x \exp(Q_x (I_x - t))\right)^{\frac{1}{v_x}}} \\
    P(t) = \frac{P_{\text{max}}}{\left(1 + v_p \exp(Q_p (I_p - t))\right)^{\frac{1}{v_p}}} \\
    S(t) = \frac{S_{\text{max}}}{\left(1 + v_s \exp(Q_s (t - I_s))\right)^{\frac{1}{v_s}}}
    \end{align*}
    ''')

    st.subheader("Stannard Model")
    st.latex(r"""
    \begin{align*}
    X(t) &= \frac{X_{\text{max}}}{(1 + \exp(-\beta_x Q_x t / k_x))^{k_x}} \\
    P(t) &= \frac{P_{\text{max}}}{(1 + \exp(-\beta_p Q_p t / k_p))^{k_p}} \\
    S(t) &= \frac{S_{\text{max}}}{(1 + \exp(\beta_s Q_s t / k_s))^{k_s}}
    \end{align*}
    """)

    st.subheader("Weibull Model")
    st.latex(r'''
    \begin{align*}
    X(t) &= X_{\text{max}} + (X_{\text{min}} - X_{\text{max}}) \cdot \exp\left(-\left(\lambda_x \cdot t\right)^{\sigma_x}\right) \\
    P(t) &= P_{\text{max}} + (P_{\text{min}} - P_{\text{max}}) \cdot \exp\left(-\left(\lambda_p \cdot t\right)^{\sigma_p}\right) \\
    S(t) &= S_{\text{min}} + (S_{\text{max}} - S_{\text{min}}) \cdot \exp\left(-\left(\frac{t}{\lambda_s}\right)^{\sigma_s}\right)
    \end{align*}
    ''')

    #endregion

    #region --- CALCULATION OF MODEL PARAMETERS ---
    st.markdown("### Model Parameters")
    st.markdown("This section displays the parameters of the selected mathematical models. Each model's parameters are shown in a table format for easy comparison.")

    # Yeni DataFrame'i oluşturmak için boş bir liste
    new_data_rows = []

    # Model adı ve parametrelerini al
    for model_name, model_params in model_params_dict.items():
        # Modelin içindeki 'Unit' ve 'Values' sözlüklerini al
        units = model_params['Unit']
        values = model_params['Values']

        # Her bir parametre için döngü
        for param_name in values.keys():
            # Birim (unit) bilgisini al, yoksa boş bırak
            unit = units.get(param_name, '')
            
            # Değer bilgisini al
            value = values.get(param_name, None)
            
            # Yeni DataFrame için bir satır sözlüğü oluştur
            new_data_rows.append({
                "Model": model_name,
                "Model parameter": param_name,
                "Unit": unit,
                "Values": value
            })

    # Listeden DataFrame oluştur
    df_final = pd.DataFrame(new_data_rows)

    # DataFrame'i Streamlit'te göster
    st.dataframe(df_final)

    #endregion

    #region --- EXPERIMENTAL & MODEL PREDICTION TABLE (MultiIndex) ---
    st.markdown("### Experimental & Model Prediction Table")
    st.markdown("This section provides a table that combines experimental data with model predictions. The table is structured to show time, experimental values, and predictions from each selected model.")

    # Zaman verisini DataFrame'den çekin
    time = df["time"].values if "time" in df.columns else np.arange(len(df))
    biomass = df["biomass"].values if "biomass" in df.columns else np.nan
    product = df["product"].values if "product" in df.columns else np.nan
    sugar = df["sugar"].values if "sugar" in df.columns else np.nan

    exp_columns = ["biomass", "product", "sugar"]
    model_pred_columns = ["biomass_pred", "product_pred", "sugar_pred"]

    # Tablo başlıklarını oluştur
    main_columns = (
        ["time"] +
        ["Experimental data"] * len(exp_columns) +
        sum([[f"Prediction with {model}"] * len(model_pred_columns) for model in selected_models], [])
    )
    sub_columns = (
        [""] +
        exp_columns +
        sum([model_pred_columns for _ in selected_models], [])
    )
    multi_columns = pd.MultiIndex.from_arrays([main_columns, sub_columns])

    # Model tahminlerini bir kez hesapla
    model_predictions = {}
    valid_models = []
    for model in selected_models:
        model_func = model_functions.get(model)
        try:
            result = model_func(df)
            if len(result) >= 1 and hasattr(result[0], "columns"):
                model_predictions[model] = result[0].copy()
                valid_models.append(model)
            else:
                model_predictions[model] = None
        except Exception:
            model_predictions[model] = None

    # Tablo başlıklarını oluştur (sadece valid_models ile)
    main_columns = (
        ["time"] +
        ["Experimental data"] * len(exp_columns) +
        sum([[f"Prediction with {model}"] * len(model_pred_columns) for model in valid_models], [])
    )
    sub_columns = (
        [""] +
        exp_columns +
        sum([model_pred_columns for _ in valid_models], [])
    )
    multi_columns = pd.MultiIndex.from_arrays([main_columns, sub_columns])

    # Tabloyu doldur
    table_data = []
    for i in range(len(time)):
        row = [time[i]]
        row += [biomass[i], product[i], sugar[i]]
        for model in valid_models:
            pred_df = model_predictions.get(model)
            value_biomass = value_product = value_sugar = np.nan
            if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
                if "time" in pred_df.columns and time[i] in pred_df["time"].values:
                    match = pred_df[pred_df["time"] == time[i]]
                    if not match.empty:
                        value_biomass = match["biomass_pred"].values[0] if "biomass_pred" in match.columns and not match["biomass_pred"].isnull().all() else np.nan
                        value_product = match["product_pred"].values[0] if "product_pred" in match.columns and not match["product_pred"].isnull().all() else np.nan
                        value_sugar = match["sugar_pred"].values[0] if "sugar_pred" in match.columns and not match["sugar_pred"].isnull().all() else np.nan
                elif i < len(pred_df):
                    value_biomass = pred_df.iloc[i]["biomass_pred"] if "biomass_pred" in pred_df.columns and not pd.isnull(pred_df.iloc[i]["biomass_pred"]) else np.nan
                    value_product = pred_df.iloc[i]["product_pred"] if "product_pred" in pred_df.columns and not pd.isnull(pred_df.iloc[i]["product_pred"]) else np.nan
                    value_sugar = pred_df.iloc[i]["sugar_pred"] if "sugar_pred" in pred_df.columns and not pd.isnull(pred_df.iloc[i]["sugar_pred"]) else np.nan
            row += [value_biomass, value_product, value_sugar]
        table_data.append(row)

    prediction_df = pd.DataFrame(table_data, columns=multi_columns)

    st.dataframe(prediction_df)
    #endregion

    #region --- DESCRIPTIVE STATISTICS ---
    st.markdown("### Descriptive Statistics")
    st.markdown("This section provides descriptive statistics for the experimental and model prediction data. It includes measures such as mean, standard deviation, variance, skewness, kurtosis, and more.")

    # "time" sütununu kaldır
    prediction_df.drop(columns=[("time", "")], inplace=True, errors='ignore')

    # Temel istatistikler
    descriptive_stats = prediction_df.describe().T

    # Ek istatistikler
    descriptive_stats["Variance"] = prediction_df.var()
    descriptive_stats["Standard Error"] = prediction_df.sem()
    descriptive_stats["Skewness"] = prediction_df.skew()
    descriptive_stats["Kurtosis"] = prediction_df.kurtosis()

    # MAD (manuel hesaplama)
    descriptive_stats["Mean Absolute Deviation"] = prediction_df.apply(lambda x: (x - x.mean()).abs().mean())

    # IQR
    descriptive_stats["Interquartile Range"] = descriptive_stats["75%"] - descriptive_stats["25%"]

    # CV
    descriptive_stats["Coefficient of Variation"] = descriptive_stats["std"] / descriptive_stats["mean"]

    # Range
    descriptive_stats["Range"] = descriptive_stats["max"] - descriptive_stats["min"]

    # İsimleri düzenle
    descriptive_stats = descriptive_stats.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std Dev",
            "Variance": "Variance",
            "Standard Error": "Standard Error",
            "min": "Min",
            "25%": "25th Percentile",
            "50%": "Median",
            "75%": "75th Percentile",
            "max": "Max",
            "Skewness": "Skewness",
            "Kurtosis": "Kurtosis",
            "Mean Absolute Deviation": "Mean Absolute Deviation",
            "Interquartile Range": "Interquartile Range",
            "Coefficient of Variation": "Coefficient of Variation",
            "Range": "Range"
        },
        index={
            "biomass": "Biomass",
            "product": "Product",
            "sugar": "Sugar",
            "biomass_pred": "Biomass Prediction",
            "product_pred": "Product Prediction",
            "sugar_pred": "Sugar Prediction"
        }
    )

    st.dataframe(descriptive_stats)

    #endregion

with col2:
    #region --- VISUALIZATION OF EXPERIMENTAL AND PREDICTED DATA (MultiIndex) ---
    st.markdown("### Visualization of Experimental and Predicted Data")
    st.markdown("This section provides visualizations of the experimental data and model predictions. Each model's predictions are compared against the experimental data for biomass, product, and sugar.")

    # Veri hazırlığı
    visualization_df = pd.DataFrame({
        "time": df["time"].values if "time" in df.columns else np.arange(len(df)),
        "biomass": df["biomass"].values if "biomass" in df.columns else np.nan,
        "product": df["product"].values if "product" in df.columns else np.nan,
        "sugar": df["sugar"].values if "sugar" in df.columns else np.nan
    })

    for model in valid_models:
        pred_df = model_predictions.get(model)
        if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
            visualization_df[f"{model} - Biomass"] = pred_df["biomass_pred"] if "biomass_pred" in pred_df.columns else np.nan
            visualization_df[f"{model} - Product"] = pred_df["product_pred"] if "product_pred" in pred_df.columns else np.nan
            visualization_df[f"{model} - Sugar"] = pred_df["sugar_pred"] if "sugar_pred" in pred_df.columns else np.nan
        else:
            visualization_df[f"{model} - Biomass"] = np.nan
            visualization_df[f"{model} - Product"] = np.nan
            visualization_df[f"{model} - Sugar"] = np.nan

    # Grafik düzeni
    rows, cols = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 25))
    axes = axes.flatten()

    for idx, model in enumerate(valid_models):
        ax1 = axes[idx]
        ax2 = ax1.twinx()  # sağ y ekseni

        # Sugar (yeşil)
        ax1.plot(visualization_df["time"], visualization_df["sugar"], color="tab:green", label="Experimental Sugar")
        ax1.plot(visualization_df["time"], visualization_df[f"{model} - Sugar"], color="tab:green", linestyle="--", alpha=0.7)

        # Product (kırmızı, sağ eksen)
        ax1.plot(visualization_df["time"], visualization_df["product"], color="tab:red", label="Experimental Product")
        ax1.plot(visualization_df["time"], visualization_df[f"{model} - Product"], color="tab:red", linestyle="--", alpha=0.7)

        # Biomass (mavi)
        ax2.plot(visualization_df["time"], visualization_df["biomass"], color="tab:blue", label="Experimental Biomass")
        ax2.plot(visualization_df["time"], visualization_df[f"{model} - Biomass"], color="tab:blue", linestyle="--", alpha=0.7)

        ax1.set_title(model, fontsize=10)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Product / Sugar")
        ax2.set_ylabel("Biomass")

    # Boş kalan subplot'ları kapat
    for j in range(len(valid_models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Experimental and Predicted Data per Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    st.pyplot(fig)

    #endregion

    #region --- SCATTER PLOT VISUALIZATION (MultiIndex) ---
    st.markdown("### Scatter Plot Visualization")
    st.markdown("This section provides scatter plots comparing experimental data with model predictions for biomass, product, and sugar. Each model's predictions are visualized against the experimental data.")

    rows, cols = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 25))
    axes = axes.flatten()

    for idx, model in enumerate(valid_models):
        ax = axes[idx]
        
        # Harf etiketi (A, B, C, ...)
        letter = chr(65 + idx)  # 65 = 'A'
        
        y_true = df["biomass"].values
        y_pred = model_predictions[model]["biomass_pred"].values
        
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) > 1:
            reg = LinearRegression()
            reg.fit(y_true.reshape(-1, 1), y_pred)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r2 = r2_score(y_true, y_pred)
            
            ax.scatter(y_true, y_pred, color="blue", alpha=0.6, label="Data")
            
            x_line = np.linspace(min(y_true), max(y_true), 100)
            y_line = reg.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_line, color="red", label="Fit")
            
            eq_text = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}"
            ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
            
            # Başlığa harfi ekle
            ax.set_title(f"{letter}) {model}")
            
            ax.set_xlabel("Experimental Biomass")
            ax.set_ylabel("Predicted Biomass")
            ax.legend()
        else:
            ax.set_visible(False)

    # Boş kalan subplot'ları kapat
    for j in range(len(valid_models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Experimental vs Predicted Biomass per Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    st.pyplot(fig)

    rows, cols = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 25))
    axes = axes.flatten()

    for idx, model in enumerate(valid_models):
        ax = axes[idx]

        # Harf etiketi (A, B, C, ...)
        letter = chr(65 + idx)  # 65 = 'A'
        
        # Deneysel ve tahmin edilen verileri al
        y_true = df["product"].values
        y_pred = model_predictions[model]["product_pred"].values
        
        # Geçerli veri filtresi (NaN hariç)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) > 1:
            # Regresyon modeli
            reg = LinearRegression()
            reg.fit(y_true.reshape(-1, 1), y_pred)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r2 = r2_score(y_true, y_pred)
            
            # Scatter plot
            ax.scatter(y_true, y_pred, color="blue", alpha=0.6, label="Data")
            
            # Regresyon doğrusu
            x_line = np.linspace(min(y_true), max(y_true), 100)
            y_line = reg.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_line, color="red", label="Fit")
            
            # R² ve denklem metni
            eq_text = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}"
            ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
            
            # Başlığa harfi ekle
            ax.set_title(f"{letter}) {model}")

            ax.set_xlabel("Experimental Product")
            ax.set_ylabel("Predicted Product")
            ax.legend()
        else:
            ax.set_visible(False)

    # Boş kalan subplot'ları kapat
    for j in range(len(valid_models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Experimental vs Predicted Product per Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    st.pyplot(fig)

    rows, cols = 6, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 25))
    axes = axes.flatten()

    for idx, model in enumerate(valid_models):
        ax = axes[idx]

        # Harf etiketi (A, B, C, ...)
        letter = chr(65 + idx)  # 65 = 'A'
        
        # Deneysel ve tahmin edilen verileri al
        y_true = df["sugar"].values
        y_pred = model_predictions[model]["sugar_pred"].values
        
        # Geçerli veri filtresi (NaN hariç)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) > 1:
            # Regresyon modeli
            reg = LinearRegression()
            reg.fit(y_true.reshape(-1, 1), y_pred)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r2 = r2_score(y_true, y_pred)
            
            # Scatter plot
            ax.scatter(y_true, y_pred, color="blue", alpha=0.6, label="Data")
            
            # Regresyon doğrusu
            x_line = np.linspace(min(y_true), max(y_true), 100)
            y_line = reg.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_line, color="red", label="Fit")
            
            # R² ve denklem metni
            eq_text = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {r2:.3f}"
            ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
            
            # Başlığa harfi ekle
            ax.set_title(f"{letter}) {model}")

            ax.set_xlabel("Experimental Sugar")
            ax.set_ylabel("Predicted Sugar")
            ax.legend()
        else:
            ax.set_visible(False)

    # Boş kalan subplot'ları kapat
    for j in range(len(valid_models), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Experimental vs Predicted Sugar per Model", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    st.pyplot(fig)
    #endregion

    #region --- BOX PLOT VISUALIZATION (MultiIndex) ---
    st.markdown("### Box Plot Visualization")
    st.markdown("This section provides box plots for the predictions of biomass, product, and sugar across different models. It allows for a visual comparison of the distributions of predictions from each model.")

    def multi_box_plot_visualization(dataframe, columns):
        rows = 3
        cols = 1  # 2x2 grid
        fig, axes = plt.subplots(rows, cols, figsize=(10, 15), sharex=True)
        axes = axes.flatten()  # 2D -> 1D

        for i, col in enumerate(columns):
            if col in dataframe.columns:
                sns.boxplot(data=dataframe, x='Model', y=col, ax=axes[i])
                axes[i].set_title(f"{col} Box Plot")
                axes[i].set_xlabel("Model")
                axes[i].set_ylabel(col)
                axes[i].tick_params(axis='x', rotation=45)
            else:
                axes[i].set_visible(False)

        # Kalan boş subplot varsa kapat
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig

    box_plot_list = []

    for model in valid_models:
        pred_df = model_predictions.get(model)
        if isinstance(pred_df, pd.DataFrame) and len(pred_df) > 0:
            df_copy = pred_df.copy()
            df_copy["Model"] = model
            box_plot_list.append(df_copy)

    if box_plot_list:
        box_plot_df = pd.concat(box_plot_list, ignore_index=True)

        needed_cols = [
            'biomass_pred',
            'product_pred',
            'sugar_pred',
            'Model'
        ]
        box_plot_df = box_plot_df[[c for c in needed_cols if c in box_plot_df.columns]]
        box_plot_df = box_plot_df.rename(columns={
            'biomass_pred': 'Biomass Prediction',
            'product_pred': 'Product Prediction',
            'sugar_pred': 'Sugar Prediction'
        })

        fig = multi_box_plot_visualization(box_plot_df, ['Biomass Prediction', 'Product Prediction', 'Sugar Prediction'])
        st.pyplot(fig)

    #endregion

    #region --- REGRESSION STATISTICS TABLE (MultiIndex) ---
    st.markdown("### Regression Statistics Table")
    st.markdown("This section provides a table of regression statistics for each model. It includes metrics such as RMSE, R², P-value (F-statistic), MAE, F-statistic, BIC, AIC, and Adjusted R².")

    stats_rows = [
        "RMSE", "R²", "P-value (F-statistic)", "MAE", "F-statistic", "BIC", "AIC", "Adjusted R²"
    ]
    param_types = ["Values for biomass", "Values for product", "Values for substrate"]

    stats_data = {}
    for model_name in statistics_dict:
        stats = statistics_dict[model_name]
        model_stats = []
        for stat in stats_rows:
            row = []
            for param in param_types:
                try:
                    value = stats[param][stat]
                except Exception:
                    value = stats.get(f"{param}_{stat}", None)
                row.append(value)
            model_stats.append(row)
        stats_data[model_name] = model_stats

    columns = pd.MultiIndex.from_product(
        [[model for model in statistics_dict.keys()], param_types],
        names=["Model", "Parameter"]
    )
    data = []
    for i, stat in enumerate(stats_rows):
        row = []
        for model in statistics_dict.keys():
            row.extend(stats_data[model][i])
        data.append(row)

    regression_df = pd.DataFrame(data, index=stats_rows, columns=columns)
    st.dataframe(regression_df)
    #endregion

    #region --- KINETIC PARAMETERS TABLE (MultiIndex) ---
    st.markdown("### Kinetic Parameters Table")
    st.markdown("This section provides a table of kinetic parameters for each model. It includes experimental values, predicted values, and variations (5% increase and decrease) for various kinetic parameters.")

    # Kinetic parameters tablosu için veri hazırlama (düzgün anahtar yapısı ile)
    kinetic_param_rows = [
        ("ΔX", "g/L"), ("ΔP", "g/L"), ("ΔS", "g/L"),
        ("Yx/s", "gX/gS"), ("Yp/s", "gP/gS"), ("Yp/x", "gP/gX"), ("Ys/x", "gS/gX"),
        ("Qx", "g/L/h"), ("Qp", "g/L/h"), ("Qs", "g/L/h"),
        ("µmax", "1/h"), ("td", "h"), ("SUY", "%")
    ]
    kinetic_param_types = ["Experimental", "Predicted", "5% increase", "5% decrease"]

    kinetic_table_data = []
    for param, unit in kinetic_param_rows:
        row = []
        for model in kinetic_params_dict.keys():
            params = kinetic_params_dict[model]
            # Her tip için değeri bul
            for typ in kinetic_param_types:
                value = np.nan
                # params[typ][param] şeklinde ise
                if typ in params and param in params[typ]:
                    value = params[typ][param]
                # params[param][typ] şeklinde ise
                elif param in params and isinstance(params[param], dict) and typ in params[param]:
                    value = params[param][typ]
                # params[param] doğrudan değer ise ve tip Experimental ise
                elif param in params and typ == "Experimental" and not isinstance(params[param], dict):
                    value = params[param]
                # params[f"{param}_{typ}"] şeklinde ise
                elif f"{param}_{typ}" in params:
                    value = params[f"{param}_{typ}"]
                row.append(value)
        kinetic_table_data.append(row)

    # MultiIndex ile sütunları oluştur
    kinetic_columns = pd.MultiIndex.from_product(
        [[model for model in kinetic_params_dict.keys()], kinetic_param_types],
        names=["Model", "Type"]
    )
    kinetic_index = pd.MultiIndex.from_tuples(kinetic_param_rows, names=["Kinetic parameters", "Unit"])

    kinetic_params_df = pd.DataFrame(kinetic_table_data, index=kinetic_index, columns=kinetic_columns)

    st.dataframe(kinetic_params_df)
    #endregion

    #region --- SENSITIVITY ANALYSIS TABLE (MultiIndex) ---
    st.markdown("### Sensitivity Analysis Table")
    st.markdown("This section provides a table of sensitivity analysis results for each model. It includes metrics such as ED, ID, and IAD for different types (5% increase and decrease) and outputs (biomass production, product production, substrate consumption).")

    sensitivity_metrics = ["ED", "ID", "IAD"]
    sensitivity_types = ["5% increase", "5% decrease"]
    sensitivity_outputs = ["Biomass production", "Product production", "Substrate consumption"]

    # Satır indexlerini oluştur
    sensitivity_index = []
    for metric in sensitivity_metrics:
        for typ in sensitivity_types:
            sensitivity_index.append((metric, typ))
    sensitivity_index = pd.MultiIndex.from_tuples(sensitivity_index, names=["Sensitivity metrics", "Type"])

    # Tablo verisini hazırla
    sensitivity_table_data = []
    for metric, typ in sensitivity_index:
        row = []
        for model in sensitivity_dict.keys():
            params = sensitivity_dict[model]
            # Doğru indexleri bulmak için
            idx_list = [k for k in params["Sensitivity metrics"].keys()
                        if params["Sensitivity metrics"][k] == metric and params["Type"][k] == typ]
            # Her output için değeri ekle
            for output in sensitivity_outputs:
                value = np.nan
                if idx_list and output in params and idx_list[0] in params[output]:
                    value = params[output][idx_list[0]]
                row.append(value)
        sensitivity_table_data.append(row)

    # Sütunları oluştur
    sensitivity_columns = pd.MultiIndex.from_product(
        [[model for model in sensitivity_dict.keys()], sensitivity_outputs],
        names=["Model", "Output"]
    )

    sensitivity_df = pd.DataFrame(sensitivity_table_data, index=sensitivity_index, columns=sensitivity_columns)
    st.dataframe(sensitivity_df)
    #endregion
