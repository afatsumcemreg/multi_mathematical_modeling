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
    df = pd.read_excel("datasets/suspended_cells.xlsx")

selected_models = st.sidebar.multiselect('**Select models**', sorted(models), default=sorted(models))

# endregion

# region FREQUENTLY USED WORDS
given_synonyms = random.choice(["specified", "particular", "specific", "designated", "stated", "provided", "given", "fixed", "established", "defined", "supplied", "presented", "offered", "set forth"])
equations = random.choice(["formulas", "expressions", "mathematical statements", "mathematical equations", "mathematical formulas"])
represent = random.choice(["represent", "depict", "portray", "illustrate", "express", "render", "manifest", "mirror", "symbolize", "embody", "exhibit"])
calculated_synonyms = random.choice(["calculated", "computed", "reckoned", "estimated"])
based_on = random.choice(["based on", "according to", "in accordance with", "as per", "in conformity with", "following", "pursuant to", "in line with", "in light of", "in consideration of", "taking into account", ])
follows = random.choice(["follows", "follows up", "pursues", "chases", "trails", "traces", "tracks"])
similar_synonyms = random.choice(["similar", "similary", "homologous", "nearly the same", "comparable", "analogous", "analogic", "alike", "akin", "resembling","almost identical", "equivalent"])
minimum_synonyms = random.choice(["minimum", "smallest", "lowest", "at least", "minimal", "tiniest"])
maximum_synonyms = random.choice(["maximum", "highest", "greatest", "utmost", "peak", "supreme", "ultimate", "maximal", "uppermost"])
values = random.choice(["values", "data", "datum", "observations"])
steepness = random.choice(["steepness", "slope", "inclination", "gradient", "grade"])
sigmoidal = random.choice(["sigmoidal", "S-shaped", "curvilinear", "sigmoid"])
constrain = random.choice(["constrain", "restrict", "limit", "restrain", "curb", "constrict"])
output = random.choice(["result", "finding", "outcome", "conclusion", "output", "consequence", "accomplishment", "achievement", "observation"])
describe = random.choice(["characterize", "define", "describe", "depict", "identify", "outline"])
variables = random.choice(["variables", "factors", "parameters", "elements", "attributes", "features"])
include = random.choice(["include", "involve", "engage", "participate", "contain", "entail", "incorporate"])
impact_synonym = random.choice(["impact", "affect", "influence"])
determining = random.choice(["determining", "ascertaining", "identifying", "detecting", "pinpointing", "finding out", "figuring out"])
exhibit = random.choice(["exhibit", "display", "show", "present", "demonstrate", "manifest", "reveal", "expose", "feature"])
significant = random.choice(["significant", "pivotal", "central", "critical", "crucial", "essential", "key", "vital", "fundamental", "important", "paramount"])
changes_synonyms = random.choice(["changes", "modifications", "alterations", "variations"])
production_synonyms = random.choice(["production", "manufacturing", "fabrication", "creation", "formation", "generation"])
consumption_synonyms = random.choice(["consumption", "usage", "utilization", "use", "utilisation"])
constrained = random.choice(["constrained", "restricted", "limited", "restrained", "curbed", "constricted"])
influenced = random.choice(["influenced", "affected", "impacted"])
response_synonym = random.choice(["response", "answer", "reply"])
controlled = random.choice(["controlled", "regulated", "governed", "managed", "directed", "supervised", "restrained", "checked", "inspected"])
similarly_synonyms = random.choice(["Similarly", "Likewise", "In the same vein", "In a similar manner", "In like fashion", "Correspondingly", "Equally", "In a parallel fashion", "In a similar way", "In the same way"])
represents = random.choice(["represents", "depicts", "portrays", "illustrates", "expresses", "renders", "manifests", "mirrors", "symbolizes", "embodies", "exhibits"])
commonly = random.choice(["generally", "extensively", "widely ", "broadly", "broad-mindedly", "far and wide", "universally", "open-mindedly", "commonly"])
used = random.choice(["used", "utilized", "employed", "applied", "exploited", "deployed", "exercised", "implemented", "operated", "practiced"])
various_synonyms = random.choice(["various", "diverse", "assorted", "different", "numerous", "several", "varied", "many", "multifarious", "myriad", "sundry"])
different_synonyms = random.choice(["different", "various", "diverse", "assorted", "numerous", "several", "varied", "many", "multifarious", "myriad", "sundry"])
process = random.choice(['method', 'procedure', 'technique', 'operation', 'approach'])
biomass_production = random.choice(["biomass production", "biomass cultivation", "biomass generation", "biomass synthesis", "biomass formation", "biomass manufacturing", "biomass creation", "biomass growth", "biomass development", "cell growth", "cell development", "cell formation", "cell production", "cell generation"])
initial = random.choice(["initial", "starting", "first", "commencing", "beginning", "preliminary"])
specific_synonym = random.choice(["specific", "certain", "particular", "definite", "determinate", "specificial"])
substrate_consumption = random.choice(["substrate consumption", "substrate utilization", "substrate expenditure", "substrate depletion", "substrate usage", "substrate utilizing", "sugar consumption", "sugar utilization", "sugar expenditure", "sugar depletion", "sugar usage", "sugar utilizing"])
characterized_synonym = random.choice(["characterized", "defined", "described", "depicted", "identified", "outlined"])
influencing = random.choice(["influencing", "affecting", "shaping", "effecting", "impacting"])
vary = random.choice(["vary", "change", "alter", "modify", "shift", "diversify", "switch", "alternate"])
concentration = random.choice(["concentration", "amount", "level"])
additionally_synonyms = random.choice(["additionally", "furthermore", "moreover", "in addition", "besides", "also", "likewise", "plus", "extra"])
substrate_source_ = random.choice(["substrate source", "nutrient source", "feedstock", "substrate", "nutrient supply", "raw material", "nutrient reservoir", "nutrient origin"])
respective = random.choice(["respective", "individual", "particular", "specific", "corresponding", "separate", "distinct", "appropriate"])
ensure_synonyms = random.choice(["ensure", "allow", "permit", "enable", "facilitate", "let", "make possible", "consent", "authorize", "authorise", "empower"])
synonyms_technique = random.choice(["technique", "method", "approach", "strategy", "tactic", "procedure"])
influences = random.choice(["influences", "affects", "impacts"])
insights_synonyms = random.choice(["insights", "understandings", "perceptions", "realizations", "comprehensions", "discernments", "interpretations"])
determine_synonyms = random.choice(["determine", "ascertain", "identify", "detect", "pinpoint", "find out", "figure out"])
by_using_synonyms = random.choice(["by using", "using", "through the use of", "via", "by means of", "with the help of", "by utilizing", "through", "employing", "with", "utilizing"])

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

# İki eşit genişlikte kolon oluşturma
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

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $T_x$, and $\gamma_x$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $T_p$, and $\gamma_p$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $T_s$, and $\gamma_s$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($\gamma_x$, $\gamma_p$, $\gamma_s$), and the points of inflection ($T_x$, $T_p$, $T_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1016/j.ijfoodmicro.2011.02.022).""")

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

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{\min}}$, $X_{{\max}}$, $Q_x$, $\lambda_x$, and $t$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{\min}}$, $P_{{\max}}$, $Q_p$, $\lambda_p$, and $t$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{\min}}$, $S_{{\max}}$, $Q_s$, $\lambda_s$, and $t$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($Q_x$, $Q_p$, $Q_s$), and the points of inflection ($\lambda_x$, $\lambda_p$, $\lambda_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1016/0168-1605(94)90157-0).""")

    st.subheader("Cone Model")
    st.latex(r'''
            \begin{align*}
            X(t) &= X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{1 + \left(\frac{1}{{\lambda_x \cdot t}}\right)^{\sigma_x}}} \\
            P(t) &= P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{1 + \left(\frac{1}{{\lambda_p \cdot t}}\right)^{\sigma_p}}} \\
            S(t) &= S_{\text{max}} - \frac{{S_{\text{max}} - S_{\text{min}}}}{{1 + \left(\frac{1}{{\lambda_s \cdot t}}\right)^{\sigma_s}}}
            \end{align*}
            ''')
    
    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $\lambda_x$, and $\sigma_x$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $\lambda_p$, and $\sigma_p$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $\lambda_s$, and $\sigma_s$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($\sigma_x$, $\sigma_p$, $\sigma_s$), 
        and the points of inflection ($\lambda_x$, $\lambda_p$, $\lambda_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1016/0377-8401(96)00950-9).""")

    st.subheader("Fitzhugh Model")
    st.latex(r'''
            \begin{align*}
            X(t) & = X_{\text{min}} + (X_{\text{max}} - X_{\text{min}}) \cdot (1 - \exp(-\lambda_x \cdot t))^{\theta_x} \\
            P(t) & = P_{\text{min}} + (P_{\text{max}} - P_{\text{min}}) \cdot (1 - \exp(-\lambda_p \cdot t))^{\theta_p} \\
            S(t) & = S_{\text{max}} - (S_{\text{max}} - S_{\text{min}}) \cdot (1 - \exp(-\lambda_s \cdot t))^{\theta_s} \\
            \end{align*}
            ''')
    
    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $\lambda_x$, and $ϑ_x$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $ϑ_p$, and $ϑ_p$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $ϑ_s$, and $ϑ_s$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions and the points of inflection ($ϑ_x$, $ϑ_p$, $ϑ_s$). The exponents $ϑ_x$, $ϑ_p$, and $ϑ_s$ control the {steepness} of the {sigmoidal.lower()} curves [[Reference]](https://doi.org/10.2527/jas1976.4241036x).""")

    st.subheader("Generalized Gompertz Model")
    st.latex(r'X(t) = X_{\text{min}} + (X_{\text{max}} - X_{\text{min}}) \cdot \exp\left(-\exp(Q_x \cdot (I_x - t))\right)')
    st.latex(r'P(t) = P_{\text{min}} + (P_{\text{max}} - P_{\text{min}}) \cdot \exp\left(-\exp(Q_p \cdot (I_p - t))\right)')
    st.latex(r'S(t) = S_{\text{min}} + (S_{\text{max}} - S_{\text{min}}) \cdot \exp\left(-\exp(-Q_s \cdot (I_s - t))\right)')

    st.markdown(f"""The {equations.lower()} {describe} dynamic processes where $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} {variables.lower()} evolving over time $t$. The parameters {include.lower()} the {minimum_synonyms.lower()} $X_{{min}}$, $P_{{min}}$, $S_{{min}}$ and {maximum_synonyms.lower()} $X_{{max}}$, $P_{{max}}$, $S_{{max}}$ {values} of each variable. The coefficients $Q_x$, $Q_p$, and $Q_s$ {impact_synonym.lower()} the rate of change. The terms $I_x$, $I_p$, and $I_s$ are inflection points, {determining.lower()} the time at which the {variables.lower()} {exhibit.lower()} {significant.lower()} {changes_synonyms.lower()} in their rates of {production_synonyms.lower()}, {production_synonyms.lower()}, or {consumption_synonyms.lower()} [[Reference]](https://www.pisces-conservation.com/pdf/growthiihelp.pdf).""")

    st.subheader("Generalized Logistic Model")
    st.latex(r'''
    \begin{align*}
    X(t) = X_{\text{min}} + \frac{X_{\text{max}} - X_{\text{min}}}{1 + \exp(Q_x \cdot (I_x - t))} \\
    P(t) = P_{\text{min}} + \frac{P_{\text{max}} - P_{\text{min}}}{1 + \exp(Q_p \cdot (I_p - t))} \\
    S(t) = S_{\text{min}} + \frac{S_{\text{max}} - S_{\text{min}}}{1 + \exp(-Q_s \cdot (I_s - t))}
    \end{align*}
    ''')

    st.markdown(f"""The {equations.lower()} {represent.lower()} dynamic systems where $X(t)$, $P(t)$, and $S(t)$ are functions of time $t$. $X(t)$ denotes a variable {constrained} between $X_{{min}}$ and $X_{{max}}$, {influenced} by an input $I_x$ with a {sigmoidal.lower()} {response_synonym.lower()} {controlled.lower()} by the parameter $Q_x$. {similarly_synonyms}, $P(t)$ {represents.lower()} another variable {constrained} between $P_{{min}}$ and $P_{{max}}$, {influenced} by an input $I_p$ with a {sigmoidal.lower()} {response_synonym.lower()} {controlled.lower()} by $Q_p$. Lastly, $S(t)$ is a variable {constrained} between $S_{{min}}$ and $S_{{max}}$, {influenced} by an input $I_s$ with a {sigmoidal.lower()} {response_synonym.lower()} {controlled.lower()} by the parameter $Q_s$. These {equations.lower()} model the dynamic behavior of the {variables.lower()} $X$, $P$, and $S$ over time [[Reference]](https://www.pisces-conservation.com/pdf/growthiihelp.pdf).""")

    st.subheader("Generalized Richards Model")
    st.latex(r'''
            \begin{align*}
            X(t) & = X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{(1 + v_x \exp(Q_x (I_x - t)))^{1 / v_x}}} \\
            P(t) & = P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{(1 + v_p \exp(Q_p (I_p - t)))^{1 / v_p}}} \\
            S(t) & = S_{\text{min}} + \frac{{S_{\text{max}} - S_{\text{min}}}}{{(1 + v_s \exp(Q_s (t - I_s)))^{1 / v_s}}}
            \end{align*}
            ''')
    
    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $Q_x$, $v_x$, and $t$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $Q_p$, $v_p$, and $t$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $Q_s$, $v_s$, and $t$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($Q_x$, $Q_p$, $Q_s$), and the dimensionless shape parameter of the curves ($v_x$, $v_p$, $v_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://www.pisces-conservation.com/pdf/growthiihelp.pdf).""")

    st.subheader("Gompertz Model")
    st.latex(r"X(t) = X_{\text{max}} \cdot \exp\left(-\exp(Q_x \cdot (I_x - t))\right)")
    st.latex(r"P(t) = P_{\text{max}} \cdot \exp\left(-\exp(Q_p \cdot (I_p - t))\right)")
    st.latex(r"S(t) = S_{\text{max}} \cdot \exp\left(-\exp(-Q_s \cdot (I_s - t))\right)")

    st.markdown(f"""These {equations.lower()} {describe} a {sigmoidal.lower()} model {commonly.lower()} {used.lower()} in {various_synonyms.lower()} fields, where the functions $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} {different_synonyms.lower()} biological or physical processes over time $t$. The inflection points $I_x$, $I_p$, $I_s$ are the time at the inflection point, and the constants $Q_x$, $Q_p$, $Q_s$ control the rate of {production_synonyms.lower()}, {production_synonyms.lower()}, or {consumption_synonyms.lower()}. The {maximum_synonyms.lower()} {values} $X_{{max}}$, $P_{{max}}$, $S_{{max}}$ {represent.lower()} the upper bounds of each {process} [[Reference]](https://doi.org/10.1098/rstl.1825.0026).""")

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
    
    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{\min}}$, $X_{{\max}}$, $Q_x$, $\lambda_x$, and $t$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{\min}}$, $P_{{\max}}$, $Q_p$, $\lambda_p$, and $t$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{\min}}$, $S_{{\max}}$, $Q_s$, $\lambda_s$, and $t$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($Q_x$, $Q_p$, $Q_s$), and the points of inflection ($\lambda_x$, $\lambda_p$, $\lambda_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1016/j.foodcont.2012.11.019).""")

    st.subheader("Kinetic Model")
    st.latex(r"X(t) = \frac{{{X_0} \cdot {X_{\max}}}}{{{X_0} + ({X_{\max}} - {X_0}) \cdot \exp(-{\mu} \cdot t)}}")
    st.latex(r"\frac{{dP}}{{dt}} = {P_0} + {a} \cdot X({t}) + {β} \cdot {X_0} \cdot \left(\frac{{{X_{\max}}}}{{\mu}}\right) \cdot \left(e^{{\mu {t}}} - 1\right)")
    st.latex(r"- \frac{dS}{dt} = S_0 - Y_{xs} \cdot (X(t) - X_0) - γ \cdot X(t) - m \cdot X_0 \cdot \left(\frac{X_{\max}}{\mu}\right) \cdot \left(e^{\mu t} - 1\right)")

    st.markdown(f"Where, $X(t)$ {represents.lower()} the {biomass_production.lower()} at time $t$, where $X_0$ is the {initial} {biomass_production.lower()}, and $X_{{\max}}$ is the {maximum_synonyms.lower()} achievable {biomass_production.lower()}. The parameter ${{\mu}}$ denotes the {specific_synonym.lower()} {production_synonyms.lower()} rate. The {production_synonyms.lower()} of product $P$ is {controlled.lower()} by a {different_synonyms.lower()} equation where $P_0$ is the {initial} product {concentration.lower()}, and $a$ and $β$ are the product {production_synonyms.lower()} constants that may {vary.lower()} with the fermentation condition. {additionally_synonyms.capitalize()}, the {substrate_consumption.lower()} $S$ is {characterized_synonym} by another {different_synonyms.lower()} equation, where $S_0$ is the {initial} substrate {concentration.lower()}, $Y_{{xs}}$ is the yield coefficient, ${{\gamma}}$ is a coefficient {influencing} {substrate_consumption.lower()}, and $m$ is the {substrate_source_} {used.lower()} to promote cell maintenance, and $t$ is the time variable in the {equations.lower()} [[Reference](https://doi.org/10.1073/pnas.6.6.275), [Reference](https://doi.org/10.1002/jbmte.390010406), [Reference](https://doi.org/10.1016/0961-9534(95)00092-L)].")

    st.subheader("Logistic Model")
    st.latex(r'''
    \begin{align*}
    X(t) = \frac{X_{\text{max}}}{1 + \exp(Q_x \cdot (I_x - t))} \\
    P(t) = \frac{P_{\text{max}}}{1 + \exp(Q_p \cdot (I_p - t))} \\
    S(t) = \frac{S_{\text{max}}}{1 + \exp(-Q_s \cdot (I_s - t))}
    \end{align*}
    ''')

    st.markdown(f"""The {equations.lower()} {represent.lower()} dynamic processes where $X(t)$, $P(t)$, and $S(t)$ denote the states of {different_synonyms.lower()} {variables.lower()} over time. $X_{{max}}$, $P_{{max}}$, and $S_{{max}}$ are the {maximum_synonyms.lower()} {values} that these {variables.lower()} can attain. $Q_x$, $Q_p$, and $Q_s$ are constants {determining.lower()} the rate of change. $I_x$, $I_p$, and $I_s$ are inflection points or thresholds that {impact_synonym.lower()} the behavior of the {respective} {variables.lower()}. The {sigmoidal.lower()} functions in the denominators {ensure_synonyms.lower()} that the {variables.lower()} {synonyms_technique} their {maximum_synonyms.lower()} {values} asymptotically [[Reference]](https://doi.org/10.1073/pnas.6.6.275).""")

    st.subheader("Modified Gompertz Model")
    st.latex(r'''
    \begin{align*}
    X(t) &= X_{\text{max}} \cdot \exp\left(-\exp\left(\frac{Q_x \cdot \text{e} \cdot (\lambda_x - t) + X_{\text{max}}}{X_{\text{max}}}\right)\right) \\
    P(t) &= P_{\text{max}} \cdot \exp\left(-\exp\left(\frac{Q_p \cdot \text{e} \cdot (\lambda_p - t) + P_{\text{max}}}{P_{\text{max}}}\right)\right) \\
    S(t) &= S_{\text{max}} \cdot \exp\left(-\exp\left(-\frac{Q_s \cdot \text{e} \cdot (\lambda_s - t) + S_{\text{max}}}{S_{\text{max}}}\right)\right)
    \end{align*}
    ''')

    st.markdown(f"""The {equations.lower()} {represent.lower()} dynamic processes over time. In the {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ denote the {values} of {different_synonyms.lower()} {variables.lower()} at time $t$. The parameters $X_{{max}}$, $P_{{max}}$, and $S_{{max}}$ are the {maximum_synonyms.lower()} {values} that $X(t)$, $P(t)$, and $S(t)$ can reach, respectively. The parameters $Q_x$, $Q_p$, and $Q_s$ {impact_synonym.lower()} the {steepness} of the decline in the {variables.lower()}. The parameters $\lambda_x$, $\lambda_p$, and $\lambda_s$ {represent.lower()} time-shifts or delays in the processes. The constant $e$ is the base of the natural logarithm [[Reference]](https://doi.org/10.1128/aem.56.6.1875-1881.1990).""")

    st.subheader("Modified Logistic Model")
    st.latex(r'''
    \begin{align*} 
    X(t) = \frac{X_{\text{max}}}{1 + \exp\left(\frac{4 Q_x (t - \lambda_x)}{X_{\text{max}}} + 2\right)} \\
    P(t) = \frac{P_{\text{max}}}{1 + \exp\left(\frac{4 Q_p (\lambda_p - t)}{P_{\text{max}}} + 2\right)} \\
    S(t) = \frac{S_{\text{max}}}{1 + \exp\left(\frac{4 (-Q_s) (\lambda_s - t)}{S_{\text{max}}} + 2\right)}
    \end{align*}''')

    st.markdown(f"""The {equations.lower()} {represent.lower()} {sigmoidal.lower()} functions {commonly.lower()} {used.lower()} in mathematical modeling. In the first equation, $X(t)$ {represents.lower()} the {sigmoidal.lower()} function with parameters $X_{{max}}$, $Q_x$, $\lambda_x$, and $t$. $X_{{max}}$ is the {maximum_synonyms.lower()} value, $Q_x$ {influences} the {steepness} of the curve, $\lambda_x$ is the inflection point, and $t$ is the variable. {similar_synonyms.capitalize()} {insights_synonyms.lower()} apply to the second and third {equations.lower()} for $P(t)$ and $S(t)$ with parameters $P_{{max}}$, $Q_p$, $\lambda_p$, $S_{{max}}$, $Q_s$, and $\lambda_s$ [[Reference]](https://doi.org/10.1128/aem.56.6.1875-1881.1990).""")

    st.subheader("Modified Richards Model")
    st.latex(r'''
    \begin{align*}
    X(t) = \frac{X_{\text{max}}}{\left(1 + v_x \cdot \exp(1 + v_x) \cdot \exp\left(Q_x \cdot (1 + v_x)^{\left(1 + \frac{1}{v_x}\right)} \cdot \frac{(\lambda_x - t)}{X_{\text{max}}}\right)\right)^{\frac{1}{v_x}}} \\
    P(t) = \frac{P_{\text{max}}}{\left(1 + v_p \cdot \exp(1 + v_p) \cdot \exp\left(Q_p \cdot (1 + v_p)^{\left(1 + \frac{1}{v_p}\right)} \cdot \frac{(\lambda_p - t)}{P_{\text{max}}}\right)\right)^{\frac{1}{v_p}}} \\
    S(t) = \frac{S_{\text{max}}}{\left(1 + v_s \cdot \exp(1 + v_s) \cdot \exp\left(-Q_s \cdot (1 + v_s)^{\left(1 + \frac{1}{v_s}\right)} \cdot \frac{(\lambda_s - t)}{S_{\text{max}}}\right)\right)^{\frac{1}{v_s}}}
    \end{align*}
    ''')

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $Q_x$, $\lambda_x$, and $t$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $Q_p$, $\lambda_p$, and $t$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $Q_s$, $\lambda_s$, and $t$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($Q_x$, $Q_p$, $Q_s$), the points of inflection ($\lambda_x$, $\lambda_p$, $\lambda_s$), and the dimensionless shape parameter of the curves ($v_x$, $v_p$, $v_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1128/aem.56.6.1875-1881.1990).""")

    st.subheader("Morgen-Mercer-Flodin Model")
    st.latex(r'''
    \begin{align*}
    X(t) &= X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{1 + (\lambda_x t)^{\gamma_x}}} \\
    P(t) &= P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{1 + (\lambda_p t)^{\gamma_p}}} \\
    S(t) &= S_{\text{min}} + \frac{{S_{\text{max}} - S_{\text{min}}}}{{1 + (\lambda_s t)^{\gamma_s}}}
    \end{align*}
    ''')

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X_{{min}}$, $X_{{max}}$, $P_{{min}}$, $P_{{max}}$, $S_{{min}}$, and $S_{{max}}$ are the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} for the functions. $\lambda_x$, $\lambda_p$, and $\lambda_s$ control the points of inflection. $\gamma_x$, $\gamma_p$, and $\gamma_s$ {determine_synonyms} the {steepness} of the {sigmoidal.lower()} curves. The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1073/pnas.72.11.4327).""")

    st.subheader("Re-Modified Gompertz Model")
    st.latex(r'''
    \begin{align*}
    X(t) = X_{\text{min}} + (X_{\text{max}} - X_{\text{min}}) \cdot \exp\left(-\exp\left(\frac{Q_x \cdot e \cdot (\lambda_x - t) + X_{\text{max}}}{X_{\text{max}}}\right)\right) \\
    P(t) = P_{\text{min}} + (P_{\text{max}} - P_{\text{min}}) \cdot \exp\left(-\exp\left(\frac{Q_p \cdot e \cdot (\lambda_p - t) + P_{\text{max}}}{P_{\text{max}}}\right)\right) \\
    S(t) = S_{\text{min}} + (S_{\text{max}} - S_{\text{min}}) \cdot \exp\left(-\exp\left(-\frac{Q_s \cdot e \cdot (\lambda_s - t) + S_{\text{max}}}{S_{\text{max}}}\right)\right)
    \end{align*}
    ''')

    st.markdown(f"""In the {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions over time. For $X(t)$, it models a {process} {by_using_synonyms.lower()} parameters such as $X_{{min}}$, $X_{{max}}$, $Q_x$, $\lambda_x$, and $t$. {similarly_synonyms}, $P(t)$ and $S(t)$ have {respective}parameters $P_{{min}}$, $P_{{max}}$, $Q_p$, $\lambda_p$ and $S_{{min}}$, $S_{{max}}$, $Q_s$, and $\lambda_s$, respectively. These parameters govern the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values}, decay rates, and time-dependent factors of the functions, {influencing} their behavior over time. The constant $e$ is the base of the natural logarithm [[Reference]](https://doi.org/10.1016/j.bcab.2018.03.018).""")

    st.subheader("Re-Modified Logistic Model")
    st.latex(r'''
    \begin{align*}
    X(t) & = X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{1 + \exp\left(\frac{{4 \cdot Q_x \cdot (\lambda_x - t)}}{{X_{\text{max}}}} + 2\right)}} \\
    P(t) & = P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{1 + \exp\left(\frac{{4 \cdot Q_p \cdot (\lambda_p - t)}}{{P_{\text{max}}}} + 2\right)}} \\
    S(t) & = S_{\text{min}} + \frac{{S_{\text{max}} - S_{\text{min}}}}{{1 + \exp\left(\frac{{4 \cdot Q_s \cdot (t - \lambda_s)}}{{S_{\text{max}}}} + 2\right)}}
    \end{align*}
    ''')

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $Q_x$, $\lambda_x$, and $t$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $Q_p$, $\lambda_p$, and $t$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $Q_s$, $\lambda_s$, and $t$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($Q_x$, $Q_p$, $Q_s$), and the points of inflection ($\lambda_x$, $\lambda_p$, $\lambda_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1016/j.bcab.2018.03.018).""")

    st.subheader("Re-Modified Richards Model")
    st.latex(r"""
    \begin{align*}
    X(t) &= X_{\text{min}} + \frac{{X_{\text{max}} - X_{\text{min}}}}{{(1 + v_x \exp(1 + v_x) \exp(Q_x (1 + v_x)^{\left(1 + \frac{1}{v_x}\right)} (\lambda_x - t) / X_{\text{max}}))^{\frac{1}{v_x}}}} \\
    P(t) &= P_{\text{min}} + \frac{{P_{\text{max}} - P_{\text{min}}}}{{(1 + v_p \exp(1 + v_p) \exp(Q_p (1 + v_p)^{\left(1 + \frac{1}{v_p}\right)} (\lambda_p - t) / P_{\text{max}}))^{\frac{1}{v_p}}}} \\
    S(t) &= S_{\text{min}} + \frac{{S_{\text{max}} - S_{\text{min}}}}{{(1 + v_s \exp(1 + v_s) \exp(-Q_s (1 + v_s)^{\left(1 + \frac{1}{v_s}\right)} (\lambda_s - t) / S_{\text{max}}))^{\frac{1}{v_s}}}} \\
    \end{align*}
    """)

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $Q_x$, $\lambda_x$, and $t$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $Q_p$, $\lambda_p$, and $t$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $Q_s$, $\lambda_s$, and $t$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($Q_x$, $Q_p$, $Q_s$), the points of inflection ($\lambda_x$, $\lambda_p$, $\lambda_s$), and the dimensionless shape parameter of the curves ($v_x$, $v_p$, $v_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1016/j.bcab.2018.03.018).""")

    st.subheader("Richards Model")
    st.latex(r'''
    \begin{align*}
    X(t) = \frac{X_{\text{max}}}{\left(1 + v_x \exp(Q_x (I_x - t))\right)^{\frac{1}{v_x}}} \\
    P(t) = \frac{P_{\text{max}}}{\left(1 + v_p \exp(Q_p (I_p - t))\right)^{\frac{1}{v_p}}} \\
    S(t) = \frac{S_{\text{max}}}{\left(1 + v_s \exp(Q_s (t - I_s))\right)^{\frac{1}{v_s}}}
    \end{align*}
    ''')

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $Q_x$, $v_x$, $I_x$, and $t$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $Q_p$, $v_p$, $I_p$, and $t$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $Q_s$, $v_s$, $I_s$, and $t$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($Q_x$, $Q_p$, $Q_s$), the dimensionless shape parameter of the curves ($v_x$, $v_p$, $v_s$), and the points of inflection ($I_x$, $I_p$, $I_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {minimum_synonyms.lower()} and {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1093/jxb/10.2.290).""")

    st.subheader("Stannard Model")
    st.latex(r"""
    \begin{align*}
    X(t) &= \frac{X_{\text{max}}}{(1 + \exp(-\beta_x Q_x t / k_x))^{k_x}} \\
    P(t) &= \frac{P_{\text{max}}}{(1 + \exp(-\beta_p Q_p t / k_p))^{k_p}} \\
    S(t) &= \frac{S_{\text{max}}}{(1 + \exp(\beta_s Q_s t / k_s))^{k_s}}
    \end{align*}
    """)

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{max}}$, $Q_x$, $β_x$, $k_x$, and $t$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{max}}$, $Q_p$, $β_p$, $k_p$, and $t$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{max}}$, $Q_s$, $β_s$, $k_s$, and $t$. These parameters control the {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves ($Q_x$, $Q_p$, $Q_s$), the rate of change ($β_x$, $β_p$, $β_s$), and the points of inflection ($k_x$, $k_p$, $k_s$). The functions {constrain} the {output} {values} between the {given_synonyms} {maximum_synonyms.lower()} limits [[Reference]](https://doi.org/10.1016/S0740-0020(85)80004-6).""")

    st.subheader("Weibull Model")
    st.latex(r'''
    \begin{align*}
    X(t) &= X_{\text{max}} + (X_{\text{min}} - X_{\text{max}}) \cdot \exp\left(-\left(\lambda_x \cdot t\right)^{\sigma_x}\right) \\
    P(t) &= P_{\text{max}} + (P_{\text{min}} - P_{\text{max}}) \cdot \exp\left(-\left(\lambda_p \cdot t\right)^{\sigma_p}\right) \\
    S(t) &= S_{\text{min}} + (S_{\text{max}} - S_{\text{min}}) \cdot \exp\left(-\left(\frac{t}{\lambda_s}\right)^{\sigma_s}\right)
    \end{align*}
    ''')

    st.markdown(f"""In the {given_synonyms} {equations.lower()}, $X(t)$, $P(t)$, and $S(t)$ {represent.lower()} functions of time $t$. $X(t)$ is {calculated_synonyms.lower()} {based_on.lower()} the function with parameters $X_{{min}}$, $X_{{max}}$, $\lambda_x$, and $\sigma_x$. $P(t)$ {follows.lower()} a {similar_synonyms.lower()} structure with parameters $P_{{min}}$, $P_{{max}}$, $\lambda_p$, and $\sigma_p$. $S(t)$ is {calculated_synonyms.lower()} with parameters $S_{{min}}$, $S_{{max}}$, $\lambda_s$, and $\sigma_s$. These parameters control the {minimum_synonyms.lower()} and {maximum_synonyms.lower()} {values} of the functions, the {steepness} of the {sigmoidal.lower()} curves, and the points of inflection [[Reference]](https://doi.org/10.1115/1.4010337).""")

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

with col2:
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

        # Harf etiketi (A, B, C, ...)
        letter = chr(65 + idx)  # 65 = 'A'

        # Sugar (yeşil)
        ax1.plot(visualization_df["time"], visualization_df["sugar"], color="tab:green", label="Exp_Sugar")
        ax1.plot(visualization_df["time"], visualization_df[f"{model} - Sugar"], color="tab:green", linestyle="--", alpha=0.7, label="Pred_Sugar")

        # Product (kırmızı, sağ eksen)
        ax1.plot(visualization_df["time"], visualization_df["product"], color="tab:red", label="Exp_Product")
        ax1.plot(visualization_df["time"], visualization_df[f"{model} - Product"], color="tab:red", linestyle="--", alpha=0.7, label="Pred_Product")

        # Biomass (mavi)
        ax2.plot(visualization_df["time"], visualization_df["biomass"], color="tab:blue", label="Exp_Biomass")
        ax2.plot(visualization_df["time"], visualization_df[f"{model} - Biomass"], color="tab:blue", linestyle="--", alpha=0.7, label="Pred_Product")

        ax1.set_xlabel("Time")
        ax1.set_ylabel("Product / Sugar")
        ax2.set_ylabel("Biomass")
        ax1.legend()
        ax2.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)
  
        # Başlığa harfi ekle
        ax1.set_title(f"{letter}) {model}")

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
        cols = 1
        fig, axes = plt.subplots(rows, cols, figsize=(10, 12), sharex=True)
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
            'biomass_pred': 'A) Biomass Prediction',
            'product_pred': 'B) Product Prediction',
            'sugar_pred': 'C) Sugar Prediction'
        })

        fig = multi_box_plot_visualization(box_plot_df, ['A) Biomass Prediction', 'B) Product Prediction', 'C) Sugar Prediction'])
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
