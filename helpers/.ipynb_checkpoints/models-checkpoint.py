# region Libraries
import math
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_ind
import statsmodels.api as sm
# endregion

# region Pre-defined functions
def get_optimized_params(model, x, y, p0):
    popt, _ = curve_fit(model, x, y, p0=p0, maxfev=1000000)
    popt_increase = np.round(popt + 0.05 * popt, 3)
    popt_decrease = np.round(popt - 0.05 * popt, 3)
    return popt, popt_increase, popt_decrease
def predict_concentrations(dataframe, time_column, model_biomass, model_product, model_substrate, popt_x, popt_increas_x, popt_decrease_x, popt_p, popt_increase_p, popt_decrease_p, popt_s, popt_increase_s, popt_decrease_s):
    # Predict concentrations using the fitted models
    dataframe[f"biomass_pred"] = model_biomass(dataframe[time_column], *popt_x)
    dataframe[f"biomass_5%_increase"] = model_biomass(dataframe[time_column], *popt_increas_x)
    dataframe[f"biomass_5%_decrease"] = model_biomass(dataframe[time_column], *popt_decrease_x)
    dataframe[f"product_pred"] = model_product(dataframe[time_column], *popt_p)
    dataframe[f"product_5%_increase"] = model_product(dataframe[time_column], *popt_increase_p)
    dataframe[f"product_5%_decrease"] = model_product(dataframe[time_column], *popt_decrease_p)
    dataframe[f"sugar_pred"] = model_substrate(dataframe[time_column], *popt_s)
    dataframe[f"sugar_5%_increase"] = model_substrate(dataframe[time_column], *popt_increase_s)
    dataframe[f"sugar_5%_decrease"] = model_substrate(dataframe[time_column], *popt_decrease_s)
def calculate_parameter_for_kinetic_modeling(popt, percent_change=0.05):
    values = [
        round(popt[0], 3),
        round(popt[1], 3),
        round(popt[2], 3),
        round(math.log(2) / popt[2], 3),
    ]

    increase_values = [round(value + percent_change * value, 3) for value in values]
    decrease_values = [round(value - percent_change * value, 3) for value in values]

    return values, increase_values, decrease_values
def calculate_parameter_for_three_parameter_models(popt, percent_change=0.05):
    values = [
        round(popt[0], 3),
        round(popt[1], 3),
        round(popt[2], 3)
    ]

    increase_values = [round(value + percent_change * value, 3) for value in values]
    decrease_values = [round(value - percent_change * value, 3) for value in values]

    return values, increase_values, decrease_values
def calculate_parameter_for_four_parameter_models(popt, percent_change=0.05):
    values = [
        round(popt[0], 3),
        round(popt[1], 3),
        round(popt[2], 3),
        round(popt[3], 3)
    ]

    increase_values = [round(value + percent_change * value, 3) for value in values]
    decrease_values = [round(value - percent_change * value, 3) for value in values]

    return values, increase_values, decrease_values
def calculate_parameter_for_five_parameter_models(popt, percent_change=0.05):
    values = [
        round(popt[0], 3),
        round(popt[1], 3),
        round(popt[2], 3),
        round(popt[3], 3),
        round(popt[4], 3)
    ]

    increase_values = [round(value + percent_change * value, 3) for value in values]
    decrease_values = [round(value - percent_change * value, 3) for value in values]

    return values, increase_values, decrease_values
# endregion

# region Modeling steps
def kinetic_modeling(dataframe):
    def biomass(t, X0, Xmax, mu):
        return X0 * Xmax / (X0 + (Xmax - X0) * np.exp(-mu * t))

    def product(t, P0, a, b, X0, Xmax, mu):
        X = biomass(t, X0, Xmax, mu)
        return P0 + a * X + b * X0 * (Xmax / mu) * (np.exp(mu * t) - 1)

    def substrate(t, S0, Yxs, a, b, X0, Xmax, mu):
        X = biomass(t, X0, Xmax, mu)
        return S0 - Yxs * (X - X0) - a * X - b * X0 * (Xmax / mu) * (np.exp(mu * t) - 1)

    X0, Xmax, mu = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.1
    P0, a, b = dataframe["product"].min(), 0.1, 0.1
    S0, Yxs = dataframe["sugar"].max(), 0.5

    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                                                            dataframe["time"],
                                                                            dataframe["biomass"],
                                                                            [X0, Xmax, mu])

    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                                                            dataframe["time"],
                                                                            dataframe["product"],
                                                                            [P0, a, b, X0, Xmax, mu])

    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                                                      dataframe["time"],
                                                                      dataframe["sugar"],
                                                                      [S0, Yxs, a, b, X0, Xmax, mu])

    # Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['X0', 'Xmax', 'µmax', 'td', 'P0', 'α', 'β', 'S0', 'Yx/s', 'γ', 'm'],
        'Unit': ['g/L', 'g/L', '1/h', 'h', 'g/L', 'gP/gX', 'gP/gX.h', 'g/L', '%', 'gS/gX', 'gS/gX.h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_kinetic_modeling(popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_kinetic_modeling(popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_kinetic_modeling(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)
    model_parameters = model_parameters.loc['X0':'m']

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def gompertz_model(dataframe):

    # Defining gompertz model for biomass and product production and substrate consumption
    def biomass(t, Xmax, Qx, inflection_point_x):
        return Xmax * np.exp(-np.exp(Qx * (inflection_point_x - t)))

    def product(t, Pmax, Qp, inflection_point_p):
        return Pmax * np.exp(-np.exp(Qp * (inflection_point_p - t)))

    def substrate(t, Smax, Qs, inflection_point_s):
        return Smax * np.exp(-np.exp(-Qs * (inflection_point_s - t)))

    # Initial guesses for the parameters
    Xmax, Qx, inflection_point_x = dataframe["biomass"].max(), 0.1, 1
    Pmax, Qp, inflection_point_p = dataframe["product"].max(), 0.2, 2
    Smax, Qs, inflection_point_s = dataframe["sugar"].max(), 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                                                            dataframe["time"],
                                                                            dataframe["biomass"],
                                                                            [Xmax, Qx, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                                                            dataframe["time"],
                                                                            dataframe["product"],
                                                                            [Pmax, Qp, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                                                      dataframe["time"],
                                                                      dataframe["sugar"],
                                                                      [Smax, Qs, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmax', 'Qx', 'Ix', 'Pmax', 'Qp', 'Ip', 'Smax', 'Qs', 'Is'],
        'Unit': ['g/L', 'g/L/h', 'h', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_three_parameter_models(popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_three_parameter_models(popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_three_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def generalized_gompertz_model(dataframe):

    # Defining generalized gompertz model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, Qx, inflection_point_x):
        return Xmin + (Xmax - Xmin) * np.exp(-np.exp(Qx * (inflection_point_x - t)))

    def product(t, Pmin, Pmax, Qp, inflection_point_p):
        return Pmin + (Pmax - Pmin) * np.exp(-np.exp(Qp * (inflection_point_p - t)))

    def substrate(t, Smin, Smax, Qs, inflection_point_s):
        return Smin + (Smax - Smin) * np.exp(-np.exp(-Qs * (inflection_point_s - t)))

    # Initial guesses for the parameters
    Xmin, Xmax, Qx, inflection_point_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.1, 1
    Pmin, Pmax, Qp, inflection_point_p = dataframe["product"].min(), dataframe["product"].max(), 0.2, 2
    Smin, Smax, Qs, inflection_point_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, Qx, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, Qp, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, Qs, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'Qx', 'Ix', 'Pmin', 'Pmax', 'Qp', 'Ip', 'Smin', 'Smax', 'Qs', 'Is'],
        'Unit': ['g/L', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def modified_gompertz_model(dataframe):

    # Defining modified gompertz model for biomass and product production and substrate consumption
    def biomass(t, Xmax, Qx, lambda_x):
        return Xmax * np.exp(-np.exp((Qx * np.e * (lambda_x - t) + Xmax) / Xmax))

    def product(t, Pmax, Qp, lambda_p):
        return Pmax * np.exp(-np.exp((Qp * np.e * (lambda_p - t) + Pmax) / Pmax))

    def substrate(t, Smax, Qs, lambda_s):
        return Smax * np.exp(-np.exp((-Qs * np.e * (lambda_s - t) + Smax) / Smax))

    # Initial guesses for the parameters
    Xmax, Qx, lambda_x = dataframe["biomass"].max(), 0.1, 1
    Pmax, Qp, lambda_p = dataframe["product"].max(), 0.2, 2
    Smax, Qs, lambda_s = dataframe["sugar"].max(), 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmax, Qx, lambda_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmax, Qp, lambda_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smax, Qs, lambda_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmax', 'Qx', 'λx', 'Pmax', 'Qp', 'λp', 'Smax', 'Qs', 'λs'],
        'Unit': ['g/L', 'g/L/h', 'h', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_three_parameter_models(popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_three_parameter_models(popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_three_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def re_modified_gompertz_model(dataframe):

    # Defining re-modified gompertz model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, Qx, lambda_x):
        return Xmin + (Xmax - Xmin) * np.exp(-np.exp((Qx * np.e * (lambda_x - t) + Xmax) / Xmax))

    def product(t, Pmin, Pmax, Qp, lambda_p):
        return Pmin + (Pmax - Pmin) * np.exp(-np.exp((Qp * np.e * (lambda_p - t) + Pmax) / Pmax))

    def substrate(t, Smin, Smax, Qs, lambda_s):
        return Smin + (Smax - Smin) * np.exp(-np.exp((-Qs * np.e * (lambda_s - t) + Smax) / Smax))

    # Initial guesses for the parameters
    Xmin, Xmax, Qx, lambda_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.1, 1
    Pmin, Pmax, Qp, lambda_p = dataframe["product"].min(), dataframe["product"].max(), 0.2, 2
    Smin, Smax, Qs, lambda_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, Qx, lambda_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, Qp, lambda_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, Qs, lambda_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmax', 'Xmin', 'Qx', 'ƛx', 'Pmin', 'Pmax', 'Qp', 'ƛp', 'Smin', 'Smax', 'Qs', 'ƛs'],
        'Unit': ['g/L', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_three_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_three_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_three_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def logistic_model(dataframe):
    # Defining logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmax, Qx, inflection_point_x):
        return Xmax / (1 + np.exp(Qx * (inflection_point_x - t)))

    def product(t, Pmax, Qp, inflection_point_p):
        return Pmax / (1 + np.exp(Qp * (inflection_point_p - t)))

    def substrate(t, Smax, Qs, inflection_point_s):
        return Smax / (1 + np.exp(-Qs * (inflection_point_s - t)))

    # Initial guesses for the parameters
    Xmax, Qx, inflection_point_x = dataframe["biomass"].max(), 0.1, 1
    Pmax, Qp, inflection_point_p = dataframe["product"].max(), 0.2, 2
    Smax, Qs, inflection_point_s = dataframe["sugar"].max(), 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                                                            dataframe["time"],
                                                                            dataframe["biomass"],
                                                                            p0=[Xmax, Qx, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                                                            dataframe["time"],
                                                                            dataframe["product"],
                                                                            p0=[Pmax, Qp, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                                                      dataframe["time"],
                                                                      dataframe["sugar"],
                                                                      p0=[Smax, Qs, inflection_point_s])

    # Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc,
                           popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar,
                           popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmax', 'Qx', 'Ix', 'Pmax', 'Qp', 'Ip', 'Smax', 'Qs', 'Is'],
        'Unit': ['g/L', 'g/L/h', 'h', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_three_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_three_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_three_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def generalized_logistic_model(dataframe):
    # Defining generalized logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, Qx, inflection_point_x):
        return Xmin + (Xmax - Xmin) / (1 + np.exp(Qx * (inflection_point_x - t)))

    def product(t, Pmin, Pmax, Qp, inflection_point_p):
        return Pmin + (Pmax - Pmin) / (1 + np.exp(Qp * (inflection_point_p - t)))

    def substrate(t, Smin, Smax, Qs, inflection_point_s):
        return Smin + (Smax - Smin) / (1 + np.exp(-Qs * (inflection_point_s - t)))

    # Initial guesses for the parameters
    Xmin, Xmax, Qx, inflection_point_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.1, 1
    Pmin, Pmax, Qp, inflection_point_p = dataframe["product"].min(), dataframe["product"].max(), 0.2, 2
    Smin, Smax, Qs, inflection_point_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, Qx, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, Qp, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, Qs, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'Qx', 'Ix', 'Pmin', 'Pmax', 'Qp', 'Ip', 'Smin', 'Smax', 'Qs', 'Is'],
        'Unit': ['g/L', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def modified_logistic_model(dataframe):
    # Defining modified logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmax, Qx, lambda_x):
        return Xmax / (1 + np.exp((4 * Qx * (t - lambda_x) / Xmax) + 2))

    def product(t, Pmax, Qp, lambda_p):
        return Pmax / (1 + np.exp((4 * Qp * (lambda_p - t) / Pmax) + 2))

    def substrate(t, Smax, Qs, lambda_s):
        return Smax / (1 + np.exp((4 * -Qs * (lambda_s - t) / Smax) + 2))

    # Initial guesses for the parameters
    Xmax, Qx, lambda_x = dataframe["biomass"].max(), 0.1, 1
    Pmax, Qp, lambda_p = dataframe["product"].max(), 0.2, 2
    Smax, Qs, lambda_s = dataframe["sugar"].max(), 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmax, Qx, lambda_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmax, Qp, lambda_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smax, Qs, lambda_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmax', 'Qx', 'λx', 'Pmax', 'Qp', 'λp', 'Smax', 'Qs', 'λs'],
        'Unit': ['g/L', 'g/L/h', 'h', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_three_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_three_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_three_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def re_modified_logistic_model(dataframe):
    # Defining generalized logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, Qx, lambda_x):
        return Xmin + (Xmax - Xmin) / (1 + np.exp((4 * Qx * (lambda_x - t) / Xmax) + 2))

    def product(t, Pmin, Pmax, Qp, lambda_p):
        return Pmin + (Pmax - Pmin) / (1 + np.exp((4 * Qp * (lambda_p - t) / Pmax) + 2))

    if dataframe["sugar"].max() < 15:
        def substrate(t, Smin, Smax, Qs, lambda_s):
            return Smin + (Smax - Smin) / (1 + np.exp((4 * Qs * (lambda_s - t) / Smax) + 2))
    else:
        def substrate(t, Smin, Smax, Qs, lambda_s):
            return Smin + (Smax - Smin) / (1 + np.exp((4 * Qs * (t - lambda_s) / Smax) + 2))

    # Initial guesses for the parameters
    Xmin, Xmax, Qx, lambda_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.1, 1
    Pmin, Pmax, Qp, lambda_p = dataframe["product"].min(), dataframe["product"].max(), 0.2, 2
    Smin, Smax, Qs, lambda_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, Qx, lambda_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, Qp, lambda_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, Qs, lambda_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'Qx', 'ƛx', 'Pmin', 'Pmax', 'Qp', 'ƛp', 'Smin', 'Smax', 'Qs', 'ƛs'],
        'Unit': ['g/L', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L', 'g/L/h', 'h', 'g/L', 'g/L', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def richards_model(dataframe):
    # Defining generalized logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmax, shape_parameter_x, Qx, inflection_point_x):
        return Xmax / np.power(1 + shape_parameter_x * np.exp(Qx * (inflection_point_x - t)),
                               1 / shape_parameter_x)

    def product(t, Pmax, shape_parameter_p, Qp, inflection_point_p):
        return Pmax / np.power(1 + shape_parameter_p * np.exp(Qp * (inflection_point_p - t)),
                               1 / shape_parameter_p)

    def substrate(t, Smax, shape_parameter_s, Qs, inflection_point_s):
        return Smax / np.power(1 + shape_parameter_s * np.exp(Qs * (t - inflection_point_s)),
                               1 / shape_parameter_s)

    # Initial guesses for the parameters
    Xmax, shape_parameter_x, Qx, inflection_point_x = dataframe["biomass"].max(), 0.00001, 0.1, 1
    Pmax, shape_parameter_p, Qp, inflection_point_p = dataframe["product"].max(), 0.00001, 0.2, 2
    Smax, shape_parameter_s, Qs, inflection_point_s = dataframe["sugar"].max(), 0.00001, 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmax, shape_parameter_x, Qx, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmax, shape_parameter_p, Qp, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smax, shape_parameter_s, Qs, inflection_point_s])

    # Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmax', 'Vx', 'Qx', 'Ix', 'Pmax', 'Vp', 'Qp', 'Ip', 'Smax', 'Vs', 'Qs', 'Is'],
        'Unit': ['g/L', '', 'g/L/h', 'h', 'g/L', '', 'g/L/h', 'h', 'g/L', '', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def modified_richards_model(dataframe):
    # Defining generalized logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmax, shape_parameter_x, Qx, lambda_x):
        return Xmax / np.power(1 + shape_parameter_x * np.exp(1 + shape_parameter_x) * np.exp(
            Qx * np.power(1 + shape_parameter_x, (1 + 1 / shape_parameter_x)) * (lambda_x - t) / Xmax),
                               1 / shape_parameter_x)

    def product(t, Pmax, shape_parameter_p, Qp, lambda_p):
        return Pmax / np.power(1 + shape_parameter_p * np.exp(1 + shape_parameter_p) * np.exp(
            Qp * np.power(1 + shape_parameter_p, (1 + 1 / shape_parameter_p)) * (lambda_p - t) / Pmax),
                               1 / shape_parameter_p)

    def substrate(t, Smax, shape_parameter_s, Qs, lambda_s):
        return Smax / np.power(1 + shape_parameter_s * np.exp(1 + shape_parameter_s) * np.exp(
            -Qs * np.power(1 + shape_parameter_s, (1 + 1 / shape_parameter_s)) * (lambda_s - t) / Smax),
                               1 / shape_parameter_s)

    # Initial guesses for the parameters
    Xmax, shape_parameter_x, Qx, lambda_x = dataframe["biomass"].max(), 0.00015, 0.1, 1
    Pmax, shape_parameter_p, Qp, lambda_p = dataframe["product"].max(), 0.00001, 0.5, 1
    Smax, shape_parameter_s, Qs, lambda_s = dataframe["sugar"].max(), 0.00001, 0.1, 1

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmax, shape_parameter_x, Qx, lambda_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmax, shape_parameter_p, Qp, lambda_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smax, shape_parameter_s, Qs, lambda_s])

    # Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmax', 'Vx', 'Qx', 'ƛx', 'Pmax', 'Vp', 'Qp', 'ƛp', 'Smax', 'Vs', 'Qs', 'ƛs'],
        'Unit': ['g/L', '', 'g/L/h', 'h', 'g/L', '', 'g/L/h', 'h', 'g/L', '', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def stannard_model(dataframe):
    # Defining generalized logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmax, beta_x, Qx, inflection_point_x):
        return Xmax / np.power(1 + np.exp(-beta_x * Qx * t / inflection_point_x), inflection_point_x)

    def product(t, Pmax, beta_p, Qp, inflection_point_p):
        return Pmax / np.power(1 + np.exp(-beta_p * Qp * t / inflection_point_p), inflection_point_p)

    def substrate(t, Smax, beta_s, Qs, inflection_point_s):
        return Smax / np.power(1 + np.exp(beta_s * Qs * t / inflection_point_s), inflection_point_s)

    # Initial guesses for the parameters
    Xmax, beta_x, Qx, inflection_point_x = dataframe["biomass"].max(), 0.01, 0.1, 1
    Pmax, beta_p, Qp, inflection_point_p = dataframe["product"].max(), 0.01, 0.1, 1
    Smax, beta_s, Qs, inflection_point_s = dataframe["sugar"].max(), 0.01, 0.1, 1

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmax, beta_x, Qx, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmax, beta_p, Qp, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smax, beta_s, Qs, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmax', 'βx', 'Qx', 'Kx', 'Pmax', 'βp', 'Qp', 'Kp', 'Smax', 'βs', 'Qs', 'Ks'],
        'Unit': ['g/L', '', 'g/L/h', 'h', 'g/L', '', 'g/L/h', 'h', 'g/L', '', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def weibull_model(dataframe):

    # Defining logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, lambda_x, inflection_point_x):
        return Xmax + (Xmin - Xmax) * np.exp(-np.power(lambda_x * t, inflection_point_x))

    def product(t, Pmin, Pmax, lambda_p, inflection_point_p):
        return Pmax + (Pmin - Pmax) * np.exp(-np.power(lambda_p * t, inflection_point_p))

    def substrate(t, Smin, Smax, lambda_s, inflection_point_s):
        return Smin + (Smax - Smin) * np.exp(-np.power(t / lambda_s, inflection_point_s))

    # Initial guesses for the parameters
    Xmin, Xmax, lambda_x, inflection_point_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.15, 2
    Pmin, Pmax, lambda_p, inflection_point_p = dataframe["product"].min(), dataframe["product"].max(), 0.2, 2
    Smin, Smax, lambda_s, inflection_point_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 1, 1

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, lambda_x, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, lambda_p, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, lambda_s, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'ƛx', 'σx', 'Pmin', 'Pmax', 'ƛp', 'σp', 'Smin', 'Smax', 'ƛs', 'σs'],
        'Unit': ['g/L', 'g/L', '', '', 'g/L', 'g/L', '', '', 'g/L', 'g/L', '', '']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def morgen_mercer_flodin_model(dataframe):

    # Defining Morgen-Mercer-Flodin model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, lambda_x, inflection_point_x):
        return Xmin + (Xmax - Xmin) / (1 + np.power(lambda_x * t, inflection_point_x))

    def product(t, Pmin, Pmax, lambda_p, inflection_point_p):
        return Pmin + (Pmax - Pmin) / (1 + np.power(lambda_p * t, inflection_point_p))

    def substrate(t, Smin, Smax, lambda_s, inflection_point_s):
        return Smin + (Smax - Smin) / (1 + np.power(lambda_s * t, inflection_point_s))

    # Initial guesses for the parameters
    Xmin, Xmax, lambda_x, inflection_point_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.15, 2
    Pmin, Pmax, lambda_p, inflection_point_p = dataframe["product"].min(), dataframe["product"].max(), 0.2, 2
    Smin, Smax, lambda_s, inflection_point_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 1, 1

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, lambda_x, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, lambda_p, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, lambda_s, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'ƛx', 'Ɣx', 'Pmin', 'Pmax', 'ƛp', 'Ɣp', 'Smin', 'Smax', 'ƛs', 'Ɣs'],
        'Unit': ['g/L', 'g/L', '', '', 'g/L', 'g/L', '', '', 'g/L', 'g/L', '', '']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def baranyi_model(dataframe):

    # Defining Huang model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, lambda_x, Qx):
        h_t = Qx * lambda_x
        b_t = t + (1 / Qx) * np.log(
            np.exp(-Qx * t) +
            np.exp(-h_t) -
            np.exp(-Qx * t - h_t)
        )
        return Xmin + Qx * b_t - np.log(
            1 + (np.exp(Qx * b_t) - 1) / (np.exp(Xmax - Xmin))
        )

    def product(t, Pmin, Pmax, lambda_p, Qp):
        h_t = Qp * lambda_p
        b_t = t + (1 / Qp) * np.log(
            np.exp(-Qp * t) +
            np.exp(-h_t) -
            np.exp(-Qp * t - h_t)
        )
        return Pmin + Qp * b_t - np.log(
            1 + (np.exp(Qp * b_t) - 1) / (np.exp(Pmax - Pmin))
        )

    def substrate(t, Smin, Smax, lambda_s, Qs):
        h_t = Qs * lambda_s
        b_t = t + (1 / Qs) * np.log(
            np.exp(-Qs * t) +
            np.exp(-h_t) -
            np.exp(-Qs * t - h_t)
        )
        return Smin - Qs * b_t + np.log(
            1 + (np.exp(Qs * b_t) - 1) / (np.exp(Smax - Smin))
        )

    # Initial guesses for the parameters
    Xmin, Xmax, lambda_x, Qx = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.01, 0.1
    Pmin, Pmax, lambda_p, Qp = dataframe["product"].min(), dataframe["product"].max(), 0.01, 0.3
    Smin, Smax, lambda_s, Qs = dataframe["sugar"].min(), dataframe["sugar"].max(), 0.01, 0.3

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, lambda_x, Qx])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, lambda_p, Qp])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, lambda_s, Qs])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'ƛx', 'Qx', 'Pmin', 'Pmax', 'ƛp', 'Qp', 'Smin', 'Smax', 'ƛs', 'Qs'],
        'Unit': ['g/L', 'g/L', 'h', 'g/L/h', 'g/L', 'g/L', 'h', 'g/L/h', 'g/L', 'g/L', 'h', 'g/L/h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def huang_model(dataframe):

    # Defining Huang model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, lambda_x, Qx):
        h_t = t + (1 / 4) * np.log((1 + np.exp(-4 * (t - lambda_x))) / (1 + np.exp(4 * lambda_x)))
        return Xmin + Xmax - np.log(np.exp(Xmin) + (np.exp(Xmax) - np.exp(Xmin)) * np.exp(-Qx * h_t))

    def product(t, Pmin, Pmax, lambda_p, Qp):
        h_t = t + (1 / 4) * np.log((1 + np.exp(-4 * (t - lambda_p))) / (1 + np.exp(4 * lambda_p)))
        return Pmin + Pmax - np.log(np.exp(Pmin) + (np.exp(Pmax) - np.exp(Pmin)) * np.exp(-Qp * h_t))

    def substrate(t, Smin, Smax, lambda_s, Qs):
        h_t = t + (1 / 4) * np.log((1 + np.exp(-4 * (t - lambda_s))) / (1 + np.exp(4 * lambda_s)))
        return Smin + Smax + np.log(np.exp(Smin) + (np.exp(Smax) - np.exp(Smin)) * np.exp(-Qs * h_t))

    # Initial guesses for the parameters
    Xmin, Xmax, lambda_x, Qx = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.1, 2
    Pmin, Pmax, lambda_p, Qp = dataframe["product"].min(), dataframe["product"].max(), 0.01, 3
    Smin, Smax, lambda_s, Qs = dataframe["sugar"].min(), dataframe["sugar"].max(), 0.01, 3

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, lambda_x, Qx])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, lambda_p, Qp])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, lambda_s, Qs])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'ƛx', 'Qx', 'Pmin', 'Pmax', 'ƛp', 'Qp', 'Smin', 'Smax', 'ƛs', 'Qs'],
        'Unit': ['g/L', 'g/L', 'h', 'g/L/h', 'g/L', 'g/L', 'h', 'g/L/h', 'g/L', 'g/L', 'h', 'g/L/h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def fitzhugh_model(dataframe):

    # Defining logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, lambda_x, inflection_point_x):
        return Xmin + (Xmax - Xmin) * np.power(1 - np.exp(-lambda_x * t), inflection_point_x)

    def product(t, Pmin, Pmax, lambda_p, inflection_point_p):
        return Pmin + (Pmax - Pmin) * np.power(1 - np.exp(-lambda_p * t), inflection_point_p)

    def substrate(t, Smin, Smax, lambda_s, inflection_point_s):
        return Smax - (Smax - Smin) * np.power(1 - np.exp(-lambda_s * t), inflection_point_s)

    # Initial guesses for the parameters
    Xmin, Xmax, lambda_x, inflection_point_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.15, 2
    Pmin, Pmax, lambda_p, inflection_point_p = dataframe["product"].min(), dataframe["product"].max(), 0.2, 2
    Smin, Smax, lambda_s, inflection_point_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 1, 1

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, lambda_x, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, lambda_p, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, lambda_s, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'ƛ', 'θ', 'Pmin', 'Pmax', 'ƛ', 'θ', 'Smin', 'Smax', 'ƛ', 'θ'],
        'Unit': ['g/L', 'g/L', '', '', 'g/L', 'g/L', '', '', 'g/L', 'g/L', '', '']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def cone_model(dataframe):

    # Defining cone model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, lambda_x, inflection_point_x):
        return Xmin + (Xmax - Xmin) / (1 + (1 / (lambda_x * t)) ** inflection_point_x)

    def product(t, Pmin, Pmax, lambda_p, inflection_point_p):
        return Pmin + (Pmax - Pmin) / (1 + (1 / (lambda_p * t)) ** inflection_point_p)

    def substrate(t, Smin, Smax, lambda_s, inflection_point_s):
        return Smax - (Smax - Smin) / (1 + (1 / (lambda_s * t)) ** inflection_point_s)

    # Initial guesses for the parameters
    Xmin, Xmax, lambda_x, inflection_point_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.1, 2
    Pmin, Pmax, lambda_p, inflection_point_p = dataframe["product"].min(), dataframe["product"].max(), 0.1, 2
    Smin, Smax, lambda_s, inflection_point_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 0.1, 1

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, lambda_x, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, lambda_p, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, lambda_s, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'ƛx', 'σx', 'Pmin', 'Pmax', 'ƛp', 'σp', 'Smin', 'Smax', 'ƛs', 'σs'],
        'Unit': ['g/L', 'g/L', '', '', 'g/L', 'g/L', '', '', 'g/L', 'g/L', '', '']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def asymmetric_model(dataframe):

    # Defining cone model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, lambda_x, inflection_point_x):
        return Xmin + (Xmax - Xmin) * (1 - (1 / ((1 + (t / lambda_x) ** inflection_point_x))))

    def product(t, Pmin, Pmax, lambda_p, inflection_point_p):
        return Pmin + (Pmax - Pmin) * (1 - (1 / ((1 + (t / lambda_p) ** inflection_point_p))))

    def substrate(t, Smin, Smax, lambda_s, inflection_point_s):
        return Smax - (Smax - Smin) * (1 - (1 / ((1 + (t / lambda_s) ** inflection_point_s))))

    # Initial guesses for the parameters
    Xmin, Xmax, lambda_x, inflection_point_x = dataframe["biomass"].min(), dataframe["biomass"].max(), 0.1, 0.1
    Pmin, Pmax, lambda_p, inflection_point_p = dataframe["product"].min(), dataframe["product"].max(), 0.1, 0.1
    Smin, Smax, lambda_s, inflection_point_s = dataframe["sugar"].min(), dataframe["sugar"].max(), 0.1, 0.1

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, lambda_x, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, lambda_p, inflection_point_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, lambda_s, inflection_point_s])

    ## Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'Tx', 'Ɣx', 'Pmin', 'Pmax', 'Tp', 'Ɣp', 'Smin', 'Smax', 'Ts', 'Ɣs'],
        'Unit': ['g/L', 'g/L', '', '', 'g/L', 'g/L', '', '', 'g/L', 'g/L', '', '']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_four_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_four_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_four_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def generalized_richards_model(dataframe):
    # Defining generalized logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, shape_parameter_x, Qx, inflection_point_x):
        return Xmin + (Xmax - Xmin) / np.power(1 + shape_parameter_x * np.exp(Qx * (inflection_point_x - t)),
                                               1 / shape_parameter_x)

    def product(t, Pmin, Pmax, shape_parameter_p, Qp, inflection_point_p):
        return Pmin + (Pmax - Pmin) / np.power(1 + shape_parameter_p * np.exp(Qp * (inflection_point_p - t)),
                                               1 / shape_parameter_p)

    def substrate(t, Smin, Smax, shape_parameter_s, Qs, inflection_point_s):
        return Smin + (Smax - Smin) / np.power(1 + shape_parameter_s * np.exp(Qs * (t - inflection_point_s)),
                                               1 / shape_parameter_s)

    # Initial guesses for the parameters
    Xmin, Xmax, shape_parameter_x, Qx, inflection_point_x = dataframe["biomass"].min(), dataframe[
        "biomass"].max(), 0.00001, 0.1, 1
    Pmin, Pmax, shape_parameter_p, Qp, inflection_point_p = dataframe["product"].min(), dataframe[
        "product"].max(), 0.00001, 0.2, 2
    Smin, Smax, shape_parameter_s, Qs, inflection_point_s = dataframe["sugar"].min(), dataframe[
        "sugar"].max(), 0.00001, 0.1, 2

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, shape_parameter_x, Qx, inflection_point_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, shape_parameter_p, Qp, inflection_point_p])
    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, shape_parameter_s, Qs, inflection_point_s])

    # Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'Vx', 'Qx', 'Ix', 'Pmin', 'Pmax', 'Vp', 'Qp', 'Ip', 'Smin', 'Smax', 'Vs', 'Qs', 'Is'],
        'Unit': ['g/L', 'g/L', '', 'g/L/h', 'h', 'g/L', 'g/L', '', 'g/L/h', 'h', 'g/L', 'g/L', '', 'g/L/h', 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_five_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_five_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_five_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
def re_modified_richards_model(dataframe):
    # Defining generalized logistic model for biomass and product production and substrate consumption
    def biomass(t, Xmin, Xmax, shape_parameter_x, Qx, lambda_x):
        return Xmin + (Xmax - Xmin) / np.power(1 + shape_parameter_x * np.exp(1 + shape_parameter_x) * np.exp(
            Qx * np.power(1 + shape_parameter_x, (1 + 1 / shape_parameter_x)) * (lambda_x - t) / Xmax),
                                               1 / shape_parameter_x)

    def product(t, Pmin, Pmax, shape_parameter_p, Qp, lambda_p):
        return Pmin + (Pmax - Pmin) / np.power(1 + shape_parameter_p * np.exp(1 + shape_parameter_p) * np.exp(
            Qp * np.power(1 + shape_parameter_p, (1 + 1 / shape_parameter_p)) * (lambda_p - t) / Pmax),
                                               1 / shape_parameter_p)

    def substrate(t, Smin, Smax, shape_parameter_s, Qs, lambda_s):
        return Smin + (Smax - Smin) / np.power(1 + shape_parameter_s * np.exp(1 + shape_parameter_s) * np.exp(
            -Qs * np.power(1 + shape_parameter_s, (1 + 1 / shape_parameter_s)) * (lambda_s - t) / Smax),
                                               1 / shape_parameter_s)

    Xmin, Xmax, shape_parameter_x, Qx, lambda_x = dataframe["biomass"].min(), dataframe[
        "biomass"].max(), 0.00015, 0.1, 1
    Pmin, Pmax, shape_parameter_p, Qp, lambda_p = dataframe["product"].min(), dataframe[
        "product"].max(), 0.00001, 0.5, 1
    Smin, Smax, shape_parameter_s, Qs, lambda_s = dataframe["sugar"].min(), dataframe[
        "sugar"].max(), 0.0001, 0.1, 1

    # Biomass model
    popt_biomass, popt_biomass_inc, popt_biomass_dec = get_optimized_params(biomass,
                                           dataframe["time"],
                                           dataframe["biomass"],
                                           p0=[Xmin, Xmax, shape_parameter_x, Qx, lambda_x])

    # product model
    popt_product, popt_product_inc, popt_product_dec = get_optimized_params(product,
                                           dataframe["time"],
                                           dataframe["product"],
                                           p0=[Pmin, Pmax, shape_parameter_p, Qp, lambda_p])

    # Sugar model
    popt_sugar, popt_sugar_inc, popt_sugar_dec = get_optimized_params(substrate,
                                       dataframe["time"],
                                       dataframe["sugar"],
                                       p0=[Smin, Smax, shape_parameter_s, Qs, lambda_s])


    # Predicting the concentrations using the fitted models
    predict_concentrations(dataframe, "time", biomass, product, substrate, popt_biomass, popt_biomass_inc, popt_biomass_dec, popt_product, popt_product_inc, popt_product_dec, popt_sugar, popt_sugar_inc, popt_sugar_dec)

    # Filling the NAN values with zero
    dataframe = dataframe.fillna(0)

    # Calculated model parameters
    parameters = pd.DataFrame({
        'Model parameters': ['Xmin', 'Xmax', 'Vx', 'Qx', 'ƛx', 'Pmin', 'Pmax', 'Vp', 'Qp', 'ƛp', 'Smin', 'Smax', 'Vs',
                             'Qs', 'ƛs'],
        'Unit': ['g/L', 'g/L', '', 'g/L/h', 'h', 'g/L', 'g/L', '', 'g/L/h', 'h', 'g/L', 'g/L', '', 'g/L/h',
                 'h']
    })

    # Calculate values, 5% increase, and 5% decrease for each parameter set
    values_biomass, inc_biomass, dec_biomass = calculate_parameter_for_five_parameter_models(
        popt_biomass)
    values_product, inc_product, dec_product = calculate_parameter_for_five_parameter_models(
        popt_product)
    values_sugar, inc_sugar, dec_sugar = calculate_parameter_for_five_parameter_models(popt_sugar)

    values_data = pd.DataFrame({
        'Values': values_biomass + values_product + values_sugar,
        '5%_increase': inc_biomass + inc_product + inc_sugar,
        '5%_decrease': dec_biomass + dec_product + dec_sugar,
    })

    model_parameters = pd.concat([parameters, values_data], axis=1)
    model_parameters.set_index("Model parameters", inplace=True)

    return dataframe, popt_biomass, popt_product, popt_sugar, model_parameters
# endregion

# region Post-modeling steps
def errors_evaluation(dataframe):
    # Calculating the RMSE, MAE, R2 and t-test values for each model
    # For biomass
    rmse_biomass = np.sqrt(mean_squared_error(dataframe["biomass"], dataframe["biomass_pred"]))
    mae_biomass = mean_absolute_error(dataframe["biomass"], dataframe["biomass_pred"])
    r2_biomass = r2_score(dataframe["biomass"], dataframe["biomass_pred"])
    t_test_biomass = ttest_ind(dataframe["biomass"], dataframe["biomass_pred"])

    rmse_biomass_5_percent_increase = np.sqrt(mean_squared_error(dataframe["biomass"], dataframe["biomass_5%_increase"]))
    mae_biomass_5_percent_increase = mean_absolute_error(dataframe["biomass"], dataframe["biomass_5%_increase"])
    r2_biomass_5_percent_increase = r2_score(dataframe["biomass"], dataframe["biomass_5%_increase"])
    t_test_biomass_5_percent_increase = ttest_ind(dataframe["biomass"], dataframe["biomass_5%_increase"])

    rmse_biomass_5_percent_decrease = np.sqrt(mean_squared_error(dataframe["biomass"], dataframe["biomass_5%_decrease"]))
    mae_biomass_5_percent_decrease = mean_absolute_error(dataframe["biomass"], dataframe["biomass_5%_decrease"])
    r2_biomass_5_percent_decrease = r2_score(dataframe["biomass"], dataframe["biomass_5%_decrease"])
    t_test_biomass_5_percent_decrease = ttest_ind(dataframe["biomass"], dataframe["biomass_5%_decrease"])

    # For product
    rmse_product = np.sqrt(mean_squared_error(dataframe["product"], dataframe["product_pred"]))
    mae_product = mean_absolute_error(dataframe["product"], dataframe["product_pred"])
    r2_product = r2_score(dataframe["product"], dataframe["product_pred"])
    t_test_product = ttest_ind(dataframe["product"], dataframe["product_pred"])

    rmse_product_5_percent_increase = np.sqrt(mean_squared_error(dataframe["product"], dataframe["product_5%_increase"]))
    mae_product_5_percent_increase = mean_absolute_error(dataframe["product"], dataframe["product_5%_increase"])
    r2_product_5_percent_increase = r2_score(dataframe["product"], dataframe["product_5%_increase"])
    t_test_product_5_percent_increase = ttest_ind(dataframe["product"], dataframe["product_5%_increase"])

    rmse_product_5_percent_decrease = np.sqrt(mean_squared_error(dataframe["product"], dataframe["product_5%_decrease"]))
    mae_product_5_percent_decrease = mean_absolute_error(dataframe["product"], dataframe["product_5%_decrease"])
    r2_product_5_percent_decrease = r2_score(dataframe["product"], dataframe["product_5%_decrease"])
    t_test_product_5_percent_decrease = ttest_ind(dataframe["product"], dataframe["product_5%_decrease"])

    # For substrate
    rmse_sugar = np.sqrt(mean_squared_error(dataframe["sugar"], dataframe["sugar_pred"]))
    mae_sugar = mean_absolute_error(dataframe["sugar"], dataframe["sugar_pred"])
    r2_sugar = r2_score(dataframe["sugar"], dataframe["sugar_pred"])
    t_test_sugar = ttest_ind(dataframe["sugar"], dataframe["sugar_pred"])

    rmse_sugar_5_percent_increase = np.sqrt(mean_squared_error(dataframe["sugar"], dataframe["sugar_5%_increase"]))
    mae_sugar_5_percent_increase = mean_absolute_error(dataframe["sugar"], dataframe["sugar_5%_increase"])
    r2_sugar_5_percent_increase = r2_score(dataframe["sugar"], dataframe["sugar_5%_increase"])
    t_test_sugar_5_percent_increase = ttest_ind(dataframe["sugar"], dataframe["sugar_5%_increase"])

    rmse_sugar_5_percent_decrease = np.sqrt(mean_squared_error(dataframe["sugar"], dataframe["sugar_5%_decrease"]))
    mae_sugar_5_percent_decrease = mean_absolute_error(dataframe["sugar"], dataframe["sugar_5%_decrease"])
    r2_sugar_5_percent_decrease = r2_score(dataframe["sugar"], dataframe["sugar_5%_decrease"])
    t_test_sugar_5_percent_decrease = ttest_ind(dataframe["sugar"], dataframe["sugar_5%_decrease"])

    # Printing the results

    error_metrics = pd.DataFrame({
        'Evaluation metrics': ['RMSE', 'RMSE', 'RMSE', 'MAE', 'MAE', 'MAE', 'R\u00B2', 'R\u00B2', 'R\u00B2', 't-test', 't-test', 't-test'],
        'Type': ['Prediction', '5% increase', '5% decrease', 'Prediction', '5% increase', '5% decrease', 'Prediction', '5% increase', '5% decrease', 'Prediction', '5% increase', '5% decrease']
    })

    unit_type = pd.DataFrame({'Unit': ['g/L', 'g/L', 'g/L', 'g/L', 'g/L', 'g/L', '', '', '', '', '', '']})

    error_values = pd.DataFrame({
        'Biomass': [rmse_biomass, rmse_biomass_5_percent_increase, rmse_biomass_5_percent_decrease,
                    mae_biomass, mae_biomass_5_percent_increase, mae_biomass_5_percent_decrease,
                    r2_biomass, r2_biomass_5_percent_increase, r2_biomass_5_percent_decrease,
                    t_test_biomass[1], t_test_biomass_5_percent_increase[1],
                    t_test_biomass_5_percent_decrease[1]],

        f'Product': [rmse_product, rmse_product_5_percent_increase,
                                         rmse_product_5_percent_decrease,
                                         mae_product, mae_product_5_percent_increase,
                                         mae_product_5_percent_decrease,
                                         r2_product, r2_product_5_percent_increase,
                                         r2_product_5_percent_decrease,
                                         t_test_product[1], t_test_product_5_percent_increase[1],
                                         t_test_product_5_percent_decrease[1]],

        'Sugar': [rmse_sugar, rmse_sugar_5_percent_increase, rmse_sugar_5_percent_decrease,
                  mae_sugar, mae_sugar_5_percent_increase, mae_sugar_5_percent_decrease,
                  r2_sugar, r2_sugar_5_percent_increase, r2_sugar_5_percent_decrease,
                  t_test_sugar[1], t_test_sugar_5_percent_increase[1],
                  t_test_sugar_5_percent_decrease[1]]
    })

    errors = pd.concat([error_metrics, unit_type, error_values], axis=1)

    # Regression statistics
    biomass_x = sm.add_constant(dataframe['biomass'])
    product_x = sm.add_constant(dataframe['product'])
    sugar_x = sm.add_constant(dataframe['sugar'])

    model_biomass = sm.OLS(dataframe['biomass_pred'], biomass_x)
    model_product = sm.OLS(dataframe['product_pred'], product_x)
    model_sugar = sm.OLS(dataframe['sugar_pred'], sugar_x)

    results_biomass = model_biomass.fit()
    results_product = model_product.fit()
    results_sugar = model_sugar.fit()

    statistics = {
        "Regression statistics": [
            "R\u00B2", "Adjusted R\u00B2", "F-statistic", "P-value (F-statistic)", "AIC", "BIC", "RMSE", "MAE"
        ],
        "Values for biomass": [
            round(results_biomass.rsquared, 4), round(results_biomass.rsquared_adj, 4), round(results_biomass.fvalue, 2), round(results_biomass.f_pvalue, 4),
            round(results_biomass.aic, 2), round(results_biomass.bic, 2), round(rmse_biomass, 2), round(mae_biomass, 2)
        ],
        "Values for product": [
            round(results_product.rsquared, 4), round(results_product.rsquared_adj, 4), round(results_product.fvalue, 2), round(results_product.f_pvalue, 4),
            round(results_product.aic, 2), round(results_product.bic, 2), round(rmse_product, 2), round(mae_product, 2)
        ],
        "Values for substrate": [
            round(results_sugar.rsquared, 4), round(results_sugar.rsquared_adj, 4), round(results_sugar.fvalue, 2), round(results_sugar.f_pvalue, 4),
            round(results_sugar.aic, 2), round(results_sugar.bic, 2), round(rmse_sugar, 2), round(mae_sugar, 2)
        ]
    }

    # Create a DataFrame from the dictionary
    df_statistics = pd.DataFrame(statistics)
    df_statistics.set_index('Regression statistics', inplace=True)

    return errors, df_statistics
def visualization(dataframe):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 17))

    # Plot the biomass data and the prediction
    axs[0, 0].plot(dataframe["time"], dataframe["biomass"], "x", label="Biomass Observed")
    axs[0, 0].plot(dataframe["time"], dataframe["biomass_pred"], "--", marker='s', label="Biomass Prediction", alpha=0.5)
    axs[0, 0].plot(dataframe["time"], dataframe["biomass_5%_increase"], "--", marker='s', label="Biomass 5% Increase", alpha=0.5)
    axs[0, 0].plot(dataframe["time"], dataframe["biomass_5%_decrease"], "--", marker='s', label="Biomass 5% Decrease", alpha=0.5)
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Biomass concentration")
    axs[0, 0].set_title('A')
    axs[0, 0].legend()

    # Calculate R2 value
    X = dataframe[['biomass']]
    Y_pred = dataframe['biomass_pred']
    Y_pos = dataframe['biomass_5%_increase']
    Y_neg = dataframe['biomass_5%_decrease']
    reg_model = LinearRegression().fit(X, Y_pred)
    reg_model_pos = LinearRegression().fit(X, Y_pos)
    reg_model_neg = LinearRegression().fit(X, Y_neg)
    r2 = r2_score(Y_pred, reg_model.predict(X))
    r2_pos = r2_score(Y_pos, reg_model_pos.predict(X))
    r2_neg = r2_score(Y_neg, reg_model_neg.predict(X))
    sns.regplot(x=dataframe["biomass"], y=dataframe["biomass_pred"], ax=axs[0, 1], line_kws={'linestyle': '--'}, label=f"Biomass for Prediction and R\u00B2 = {r2:.4f}")
    sns.regplot(x=dataframe["biomass"], y=dataframe["biomass_5%_increase"], ax=axs[0, 1], line_kws={'linestyle': '--'},  label=f"Biomass for 5% Increase and R\u00B2 = {r2_pos:.4f}")
    sns.regplot(x=dataframe["biomass"], y=dataframe["biomass_5%_decrease"], ax=axs[0, 1], line_kws={'linestyle': '--'}, label=f"Biomass for 5% Decrease and R\u00B2 = {r2_neg:.4f}")
    axs[0, 1].set_title('B')
    axs[0, 1].set_xlabel("Observed Biomass")
    axs[0, 1].set_ylabel("Predicted Biomass")
    axs[0, 1].legend()

    # Plot the product data and the prediction
    axs[1, 0].plot(dataframe["time"], dataframe["product"], "x", label=f"Product Observed")
    axs[1, 0].plot(dataframe["time"], dataframe["product_pred"], "--", marker='s', label=f"Product Prediction", alpha=0.5)
    axs[1, 0].plot(dataframe["time"], dataframe["product_5%_increase"], "--", marker='s', label=f"Product 5% Increase", alpha=0.5)
    axs[1, 0].plot(dataframe["time"], dataframe["product_5%_decrease"], "--", marker='s', label=f"Product 5% Decrease", alpha=0.5)
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel(f"Product concentration")
    axs[1, 0].set_title('C')
    axs[1, 0].legend()

    # Calculate R2 value
    X = dataframe[['product']]
    Y_pred = dataframe['product_pred']
    Y_pos = dataframe['product_5%_increase']
    Y_neg = dataframe['product_5%_decrease']
    reg_model = LinearRegression().fit(X, Y_pred)
    reg_model_pos = LinearRegression().fit(X, Y_pos)
    reg_model_neg = LinearRegression().fit(X, Y_neg)
    r2 = r2_score(Y_pred, reg_model.predict(X))
    r2_pos = r2_score(Y_pos, reg_model_pos.predict(X))
    r2_neg = r2_score(Y_neg, reg_model_neg.predict(X))
    sns.regplot(x=dataframe["product"], y=dataframe["product_pred"], ax=axs[1, 1], line_kws={'linestyle': '--'}, label=f"Product for Prediction and R\u00B2 = {r2:.4f}")
    sns.regplot(x=dataframe["product"], y=dataframe["product_5%_increase"], ax=axs[1, 1], line_kws={'linestyle': '--'}, label=f"Product for 5% Increase and R\u00B2 = {r2_pos:.4f}")
    sns.regplot(x=dataframe["product"], y=dataframe["product_5%_decrease"], ax=axs[1, 1], line_kws={'linestyle': '--'}, label=f"Product for 5% Decrease and R\u00B2 = {r2_neg:.4f}")
    axs[1, 1].set_title('D')
    axs[1, 1].set_xlabel(f"Observed Product")
    axs[1, 1].set_ylabel(f"Predicted Product")
    axs[1, 1].legend()

    # Plot the sugar data and the prediction
    axs[2, 0].plot(dataframe["time"], dataframe["sugar"], "x", label="Sugar Observed")
    axs[2, 0].plot(dataframe["time"], dataframe["sugar_pred"], "--", marker='s', label="Sugar Prediction", alpha=0.5)
    axs[2, 0].plot(dataframe["time"], dataframe["sugar_5%_increase"], "--", marker='s', label="Sugar 5% Increase", alpha=0.5)
    axs[2, 0].plot(dataframe["time"], dataframe["sugar_5%_decrease"], "--", marker='s', label="Sugar 5% Decrease", alpha=0.5)
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].set_ylabel("Sugar concentration")
    axs[2, 0].set_title('E')
    axs[2, 0].legend()

    # Calculate R2 value
    X = dataframe[['sugar']]
    Y_pred = dataframe['sugar_pred']
    Y_pos = dataframe['sugar_5%_increase']
    Y_neg = dataframe['sugar_5%_decrease']
    reg_model = LinearRegression().fit(X, Y_pred)
    reg_model_pos = LinearRegression().fit(X, Y_pos)
    reg_model_neg = LinearRegression().fit(X, Y_neg)
    r2 = r2_score(Y_pred, reg_model.predict(X))
    r2_pos = r2_score(Y_pos, reg_model_pos.predict(X))
    r2_neg = r2_score(Y_neg, reg_model_neg.predict(X))
    sns.regplot(x=dataframe["sugar"], y=dataframe["sugar_pred"], ax=axs[2, 1], line_kws={'linestyle': '--'}, label=f"Sugar for Prediction and R\u00B2 = {r2:.4f}")
    sns.regplot(x=dataframe["sugar"], y=dataframe["sugar_5%_increase"], ax=axs[2, 1], line_kws={'linestyle': '--'}, label=f"Sugar for 5% Increase and R\u00B2 = {r2_pos:.4f}")
    sns.regplot(x=dataframe["sugar"], y=dataframe["sugar_5%_decrease"], ax=axs[2, 1], line_kws={'linestyle': '--'}, label=f"Sugar for 5% Decrease and R\u00B2 = {r2_neg:.4f}")
    axs[2, 1].set_title('F')
    axs[2, 1].set_xlabel("Observed Sugar")
    axs[2, 1].set_ylabel("Predicted Sugar")
    axs[2, 1].legend()

    # Adjust the spacing between the subplots
    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    st.pyplot(fig)

def box_plot_visualization(dataframe):
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    # Create box plots

    sns.boxplot(data=dataframe[['biomass', 'biomass_pred', 'biomass_5%_increase', 'biomass_5%_decrease']], ax=axs[0], showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"})
    axs[0].set_title('A) Box Plots for Biomass Production')
    axs[0].set_xlabel('Variable')
    axs[0].set_ylabel('Biomass concentration')

    sns.boxplot(data=dataframe[['product', 'product_pred', 'product_5%_increase', 'product_5%_decrease']], ax=axs[1], showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"})
    axs[1].set_title(f'B) Box Plots for Product Production')
    axs[1].set_xlabel('Variable')
    axs[1].set_ylabel(f'Product concentration')

    sns.boxplot(data=dataframe[['sugar', 'sugar_pred', 'sugar_5%_increase', 'sugar_5%_decrease']], ax=axs[2], showmeans=True, meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black", "markersize":"10"})
    axs[2].set_title('C) Box Plots for Sugar Consumption')
    axs[2].set_xlabel('Variable')
    axs[2].set_ylabel('Sugar concentration')

    plt.subplots_adjust(hspace=0.3)

    st.pyplot(fig)
def kinetic_parameters(dataframe):
    # Define the data
    time = dataframe['time']
    biomass = dataframe['biomass']
    biomass_pred = dataframe['biomass_pred']
    biomass_5_percent_increase = dataframe['biomass_5%_increase']
    biomass_5_percent_decrease = dataframe['biomass_5%_decrease']

    product = dataframe['product']
    product_pred = dataframe['product_pred']
    product_5_percent_increase = dataframe['product_5%_increase']
    product_5_percent_decrease = dataframe['product_5%_decrease']

    substrate = dataframe['sugar']
    substrate_pred = dataframe['sugar_pred']
    substrate_5_percent_increase = dataframe['sugar_5%_increase']
    substrate_5_percent_decrease = dataframe['sugar_5%_decrease']

    # biomass production
    X_delta = round(biomass.max() - biomass.min(), 2)
    X_delta_1 = round(biomass_pred.max() - biomass_pred.min(), 2)
    X_delta_2 = round(biomass_5_percent_increase.max() - biomass_5_percent_increase.min(), 2)
    X_delta_3 = round(biomass_5_percent_decrease.max() - biomass_5_percent_decrease.min(), 2)

    # product production
    P_delta = round(product.max() - product.min(), 2)
    P_delta_1 = round(product_pred.max() - product_pred.min(), 2)
    P_delta_2 = round(product_5_percent_increase.max() - product_5_percent_increase.min(), 2)
    P_delta_3 = round(product_5_percent_decrease.max() - product_5_percent_decrease.min(), 2)

    # sugar consumption
    S_delta = round(substrate.max() - substrate.min(), 2)
    S_delta_1 = round(substrate_pred.max() - substrate_pred.min(), 2)
    S_delta_2 = round(substrate_5_percent_increase.max() - substrate_5_percent_increase.min(), 2)
    S_delta_3 = round(substrate_5_percent_decrease.max() - substrate_5_percent_decrease.min(), 2)

    # biomass yield
    Yx_s = round(X_delta / S_delta, 2)
    Yx_s_1 = round(X_delta_1 / S_delta_1, 2)
    Yx_s_2 = round(X_delta_2 / S_delta_2, 2)
    Yx_s_3 = round(X_delta_3 / S_delta_3, 2)

    # product yield
    Yp_s = round(P_delta / S_delta, 2)
    Yp_s_1 = round(P_delta_1 / S_delta_1, 2)
    Yp_s_2 = round(P_delta_2 / S_delta_2, 2)
    Yp_s_3 = round(P_delta_3 / S_delta_3, 2)

    # product yield per biomass
    Yp_x = round(P_delta / X_delta, 2)
    Yp_x_1 = round(P_delta_1 / X_delta_1, 2)
    Yp_x_2 = round(P_delta_2 / X_delta_2, 2)
    Yp_x_3 = round(P_delta_3 / X_delta_3, 2)

    # substrate consumption per biomass
    Ys_x = round(S_delta / X_delta, 2)
    Ys_x_1 = round(S_delta_1 / X_delta_1, 2)
    Ys_x_2 = round(S_delta_2 / X_delta_2, 2)
    Ys_x_3 = round(S_delta_3 / X_delta_3, 2)

    # maximum biomass growth rate
    Qx = round(max(np.diff(biomass) / np.diff(time)), 2)
    Qx_1 = round(max(np.diff(biomass_pred) / np.diff(time)), 2)
    Qx_2 = round(max(np.diff(biomass_5_percent_increase) / np.diff(time)), 2)
    Qx_3 = round(max(np.diff(biomass_5_percent_decrease) / np.diff(time)), 2)

    # maximum product production rate
    Qp = round(max(np.diff(product) / np.diff(time)), 2)
    Qp_1 = round(max(np.diff(product_pred) / np.diff(time)), 2)
    Qp_2 = round(max(np.diff(product_5_percent_increase) / np.diff(time)), 2)
    Qp_3 = round(max(np.diff(product_5_percent_decrease) / np.diff(time)), 2)

    # maximum substrate consumption rate
    Qs = round(max(-np.diff(substrate) / np.diff(time)), 2)
    Qs_1 = round(max(-np.diff(substrate_pred) / np.diff(time)), 2)
    Qs_2 = round(max(-np.diff(substrate_5_percent_increase) / np.diff(time)), 2)
    Qs_3 = round(max(-np.diff(substrate_5_percent_decrease) / np.diff(time)), 2)

    # maximum specific biomass growth rate
    mu_max = round(max(np.diff(np.log(biomass)) / np.diff(time)), 2)
    mu_max_1 = round(max(np.diff(np.log(biomass_pred)) / np.diff(time)), 2)
    mu_max_2 = round(max(np.diff(np.log(biomass_5_percent_increase)) / np.diff(time)), 2)
    mu_max_3 = round(max(np.diff(np.log(biomass_5_percent_decrease)) / np.diff(time)), 2)

    # doubling time
    td = round(math.log(2) / mu_max, 2)
    td_1 = round(math.log(2) / mu_max_1, 2)
    td_2 = round(math.log(2) / mu_max_2, 2)
    td_3 = round(math.log(2) / mu_max_3, 2)

    # substrate utilization rate
    suy = round(S_delta / max(substrate) * 100, 2)
    suy_1 = round(S_delta_1 / max(substrate_pred) * 100, 2)
    suy_2 = round(S_delta_2 / max(substrate_5_percent_increase) * 100, 2)
    suy_3 = round(S_delta_3 / max(substrate_5_percent_decrease) * 100, 2)

    kinetic_parameters_name = pd.DataFrame({
        'Kinetic parameters': ['ΔX', 'ΔP', 'ΔS', 'Yx/s', 'Yp/s', 'Yp/x', 'Ys/x', 'Qx', 'Qp', 'Qs', 'µmax', 'td', 'SUY'],
        'Unit': ['g/L', 'g/L', 'g/L', 'gX/gS', 'gP/gS', 'gP/gX', 'gS/gX', 'g/L/h', 'g/L/h', 'g/L/h', '1/h', 'h', '%']
    })

    values = pd.DataFrame({
        'Experimental': [X_delta, P_delta, S_delta, Yx_s, Yp_s, Yp_x, Ys_x, Qx, Qp, Qs, mu_max, td, suy],
        'Predicted': [X_delta_1, P_delta_1, S_delta_1, Yx_s_1, Yp_s_1, Yp_x_1, Ys_x_1, Qx_1, Qp_1, Qs_1, mu_max_1, td_1, suy_1],
        '5% increase': [X_delta_2, P_delta_2, S_delta_2, Yx_s_2, Yp_s_2, Yp_x_2, Ys_x_2, Qx_2, Qp_2, Qs_2, mu_max_2, td_2, suy_2],
        '5% decrease': [X_delta_3, P_delta_3, S_delta_3, Yx_s_3, Yp_s_3, Yp_x_3, Ys_x_3, Qx_3, Qp_3, Qs_3, mu_max_3, td_3, suy_3]
    })
    kinetic_parameters = pd.concat([kinetic_parameters_name, values], axis=1)
    kinetic_parameters.set_index("Kinetic parameters", inplace=True)

    return kinetic_parameters
def sensitivity_analysis(dataframe):
    # For sensitivity analysis, calculatiing endpoint deviation (ED), integral deviation (ID), and integral absolute deviation (IAD)
    # For ED
    ed_biomass_5_percent_increase = 100 * (dataframe['biomass_5%_increase'].max() / dataframe['biomass'].max() - 1)
    ed_biomass_5_percent_decrease = 100 * (dataframe['biomass_5%_decrease'].max() / dataframe['biomass'].max() - 1)
    ed_product_5_percent_increase = 100 * (dataframe['product_5%_increase'].max() / dataframe['product'].max() - 1)
    ed_product_5_percent_decrease = 100 * (dataframe['product_5%_decrease'].max() / dataframe['product'].max() - 1)
    ed_sugar_5_percent_increase = 100 * (dataframe['sugar_5%_increase'].max() / dataframe['sugar'].max() - 1)
    ed_sugar_5_percent_decrease = 100 * (dataframe['sugar_5%_decrease'].max() / dataframe['sugar'].max() - 1)

    # MFT values to calculate ID values
    dataframe['MFT_biomass_pred'] = ((dataframe['biomass_pred'] + dataframe['biomass_pred'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_biomass_pred'].fillna(dataframe['biomass_pred'][0], inplace=True)
    dataframe['MFT_biomass_pred'] = dataframe['MFT_biomass_pred'].round(2)
    dataframe['MFT_biomass_increase'] = ((dataframe['biomass_5%_increase'] + dataframe['biomass_5%_increase'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_biomass_increase'].fillna(dataframe['biomass_5%_increase'][0], inplace=True)
    dataframe['MFT_biomass_increase'] = dataframe['MFT_biomass_increase'].round(2)
    dataframe['MFT_biomass_decrease'] = ((dataframe['biomass_5%_decrease'] + dataframe['biomass_5%_decrease'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_biomass_decrease'].fillna(dataframe['biomass_5%_decrease'][0], inplace=True)
    dataframe['MFT_biomass_decrease'] = dataframe['MFT_biomass_decrease'].round(2)

    dataframe['MFT_product_pred'] = ((dataframe['product_pred'] + dataframe['product_pred'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_product_pred'].fillna(dataframe['product_pred'][0], inplace=True)
    dataframe['MFT_product_pred'] = dataframe['MFT_product_pred'].round(2)
    dataframe['MFT_product_increase'] = ((dataframe['product_5%_increase'] + dataframe['product_5%_increase'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_product_increase'].fillna(dataframe['product_5%_increase'][0], inplace=True)
    dataframe['MFT_product_increase'] = dataframe['MFT_product_increase'].round(2)
    dataframe['MFT_product_decrease'] = ((dataframe['product_5%_decrease'] + dataframe['product_5%_decrease'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_product_decrease'].fillna(dataframe['product_5%_decrease'][0], inplace=True)
    dataframe['MFT_product_decrease'] = dataframe['MFT_product_decrease'].round(2)

    dataframe['MFT_sugar_pred'] = ((dataframe['sugar_pred'] + dataframe['sugar_pred'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_sugar_pred'].fillna(dataframe['sugar_pred'][0], inplace=True)
    dataframe['MFT_sugar_pred'] = dataframe['MFT_sugar_pred'].round(2)
    dataframe['MFT_sugar_increase'] = ((dataframe['sugar_5%_increase'] + dataframe['sugar_5%_increase'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_sugar_increase'].fillna(dataframe['sugar_5%_increase'][0], inplace=True)
    dataframe['MFT_sugar_increase'] = dataframe['MFT_sugar_increase'].round(2)
    dataframe['MFT_sugar_decrease'] = ((dataframe['sugar_5%_decrease'] + dataframe['sugar_5%_decrease'].shift()) * dataframe['time'].diff()) / 2
    dataframe['MFT_sugar_decrease'].fillna(dataframe['sugar_5%_decrease'][0], inplace=True)
    dataframe['MFT_sugar_decrease'] = dataframe['MFT_sugar_decrease'].round(2)

    # For ID values
    ID_biomass_increase = np.sum(dataframe['MFT_biomass_increase'] / dataframe['MFT_biomass_pred'] - 1) * 100
    ID_biomass_decrease = np.sum(dataframe['MFT_biomass_decrease'] / dataframe['MFT_biomass_pred'] - 1) * 100
    ID_product_increase = np.sum(dataframe['MFT_product_increase'] / dataframe['MFT_product_pred'] - 1) * 100
    ID_product_decrease = np.sum(dataframe['MFT_product_decrease'] / dataframe['MFT_product_pred'] - 1) * 100
    ID_sugar_increase = np.sum(dataframe['MFT_sugar_increase'] / dataframe['MFT_sugar_pred'] - 1) * 100
    ID_sugar_decrease = np.sum(dataframe['MFT_sugar_decrease'] / dataframe['MFT_sugar_pred'] - 1) * 100

    # For IAD values
    IAD_biomass_increase = np.abs(np.sum(dataframe['MFT_biomass_increase'] / dataframe['MFT_biomass_pred'] - 1) * 100)
    IAD_biomass_decrease = np.abs(np.sum(dataframe['MFT_biomass_decrease'] / dataframe['MFT_biomass_pred'] - 1) * 100)
    IAD_product_increase = np.abs(np.sum(dataframe['MFT_product_increase'] / dataframe['MFT_product_pred'] - 1) * 100)
    IAD_product_decrease = np.abs(np.sum(dataframe['MFT_product_decrease'] / dataframe['MFT_product_pred'] - 1) * 100)
    IAD_sugar_increase = np.abs(np.sum(dataframe['MFT_sugar_increase'] / dataframe['MFT_sugar_pred'] - 1) * 100)
    IAD_sugar_decrease = np.abs(np.sum(dataframe['MFT_sugar_decrease'] / dataframe['MFT_sugar_pred'] - 1) * 100)

    sensitivity_analysis_results = pd.DataFrame({
        'Sensitivity metrics': ['ED', 'ED', 'ID', 'ID', 'IAD', 'IAD'],
        'Type': ['5% increase', '5% decrease', '5% increase', '5% decrease', '5% increase', '5% decrease'],
        'Biomass production': [ed_biomass_5_percent_increase, ed_biomass_5_percent_decrease, ID_biomass_increase, ID_biomass_decrease, IAD_biomass_increase, IAD_biomass_decrease],
        'Product production': [ed_product_5_percent_increase, ed_product_5_percent_decrease, ID_product_increase, ID_product_decrease, IAD_product_increase, IAD_product_decrease],
        'Substrate consumption': [ed_sugar_5_percent_increase, ed_sugar_5_percent_decrease, ID_product_increase, ID_product_decrease, IAD_product_increase, IAD_product_decrease]
    })

    return sensitivity_analysis_results
def sensitivity_visualization(dataframe):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Plot the biomass data and the prediction
    axs[0].plot(dataframe["time"], dataframe["biomass_pred"], "--", marker='s', label="Biomass Prediction", alpha=0.5)
    axs[0].plot(dataframe["time"], dataframe["biomass_5%_increase"], "--", marker='s', label="Biomass 5% increase", alpha=0.5)
    axs[0].plot(dataframe["time"], dataframe["biomass_5%_decrease"], "--", marker='s', label="Biomass 5% decrease", alpha=0.5)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Biomass concentration")
    axs[0].set_title('A')
    axs[0].legend()

    # Plot the sugar data and the prediction
    axs[2].plot(dataframe["time"], dataframe["sugar_pred"], "--", marker='s', label="Sugar Prediction", alpha=0.5)
    axs[2].plot(dataframe["time"], dataframe["sugar_5%_increase"], "--", marker='s', label="Sugar 5% increase", alpha=0.5)
    axs[2].plot(dataframe["time"], dataframe["sugar_5%_decrease"], "--", marker='s', label="Sugar 5% decrease", alpha=0.5)
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Sugar concentration")
    axs[2].set_title('C')
    axs[2].legend()

    # Plot the product data and the prediction
    axs[1].plot(dataframe["time"], dataframe["product_pred"], "--", marker='s', label=f"Product Prediction", alpha=0.5)
    axs[1].plot(dataframe["time"], dataframe["product_5%_increase"], "--", marker='s', label=f"Product 5% increase", alpha=0.5)
    axs[1].plot(dataframe["time"], dataframe["product_5%_decrease"], "--", marker='s', label=f"Product 5% decrease", alpha=0.5)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel(f"Product concentration")
    axs[1].set_title('B')
    axs[1].legend()

    # Adjust the spacing between the subplots
    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    st.pyplot(fig)
def economic_analysis(dataframe_cost, dataframe, product_density, mililiter, product_price_per_liter):
    # Calculate product selling price
    try:
        product_selling_price = (product_price_per_liter / product_density) / mililiter
    except ZeroDivisionError:
        print("Error: Division by zero. Please make sure product_density and mililiter are not zero.")
        product_selling_price = None  # or set a default value or handle it according to your specific use case

    # Calculate total cost
    dataframe_cost['CN'] = dataframe_cost['cost_per_gram_in_euro'] * dataframe_cost['concentration_gram_per_liter']
    total_cn = dataframe_cost['CN'].sum()

    # Economic yield
    dataframe['economic_yield_for_experimental_data'] = (dataframe['product'] * product_selling_price) / total_cn
    dataframe['economic_yield_for_predicted_data'] = (dataframe['product_pred'] * product_selling_price) / total_cn

    # Economic productivity
    dataframe['economic_productivity_for_experimental_data'] = ((dataframe['product'] * product_selling_price) / (dataframe['time'] * total_cn))
    dataframe['economic_productivity_for_predicted_data'] = (dataframe['product_pred'] * product_selling_price) / (dataframe['time'] * total_cn)

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))

    axs[0].plot(dataframe["time"], dataframe["economic_yield_for_experimental_data"], "o", label="Observed economic yield", alpha=0.5)
    axs[0].plot(dataframe["time"], dataframe["economic_yield_for_predicted_data"], "--", marker='s', label="Predicted economic yield", alpha=0.5)
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel(f"Economic yield (€ of product/€ of nutrients)")
    axs[0].set_title('A')
    axs[0].legend()

    axs[1].plot(dataframe["time"], dataframe["economic_productivity_for_experimental_data"], "o", label="Observed economic productivity", alpha=0.5)
    axs[1].plot(dataframe["time"], dataframe["economic_productivity_for_predicted_data"], "--", marker='s', label="Predicted economic productivity", alpha=0.5)
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel(f"Economic productivity (€ of product/€ of nutrients.h)")
    axs[1].set_title('B')
    axs[1].legend()
    
    plt.subplots_adjust(hspace=0.5)
    st.pyplot(fig)

    return dataframe
# endregion