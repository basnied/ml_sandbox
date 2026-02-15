import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning Template
    """)
    return


@app.cell
def _():
    import datetime
    import glob
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt


    from src.ml_template import utils

    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform


    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.feature_selection import RFECV
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score
    from sklearn.inspection import permutation_importance
    from sklearn.neural_network import MLPRegressor
    import shap

    import xgboost as xgb

    import seaborn as sns

    return (
        MLPRegressor,
        Pipeline,
        RFECV,
        SimpleImputer,
        StandardScaler,
        datetime,
        glob,
        np,
        pd,
        plt,
        r2_score,
        root_mean_squared_error,
        shap,
        sns,
        train_test_split,
        utils,
        xgb,
    )


@app.cell
def _(glob, pd):
    features = pd.read_csv(r"\\umweltbundesamt.at\Projekte\3000\3135_IntMon\Intern\ARCHneu\UBA_IM\Themen_nach_Projektbereichen\PB2\2025_26_Auswertung_C_H2O_Duerre\Ergebnisse\predictors_daily.txt", sep=';', parse_dates=True)
    rproc = pd.read_csv(r"\\umweltbundesamt.at\Projekte\3000\3135_IntMon\Intern\ARCHneu\UBA_IM\Themen_nach_Projektbereichen\PB2\0000_EddyKovarianz\NEE_Partitioning\nee-partitioning\results\EddyProc\NEEpart.csv", parse_dates=True)
    fsv_partitioned = glob.glob(r"\\umweltbundesamt.at\Projekte\3000\3135_IntMon\Intern\ARCHneu\UBA_IM\Themen_nach_Projektbereichen\PB2\0000_EddyKovarianz\NEE_Partitioning\nee-partitioning\results\fluxpart\fvs_partitioned_30min_*.txt")
    fvs_partition_df = pd.concat([pd.read_csv(file, sep=';', parse_dates=['datetime']) for file in fsv_partitioned])
    return features, fvs_partition_df, rproc


@app.cell
def _(pd, rproc):
    rproc_1 = rproc[['GPP_DT', 'Reco_DT']].groupby(pd.to_datetime(rproc.datetime).dt.date).mean()
    return (rproc_1,)


@app.cell
def _(fvs_partition_df, rproc_1):
    fluxpart = fvs_partition_df[['Fq_mmol', 'Fqt_mmol', 'Fqe_mmol', 'Fc_umol', 'Fcp_umol', 'Fcr_umol']].groupby(fvs_partition_df.datetime.dt.date).mean()
    targets = rproc_1.merge(fluxpart, how='outer', left_index=True, right_index=True)
    return (targets,)


@app.cell
def _(features, pd):
    features["date"] = pd.to_datetime(features.day)
    features.drop(columns='day', inplace=True)
    features.set_index('date', inplace=True)
    return


@app.cell
def _(targets):
    targets['NEE'] = targets.Reco_DT - targets.GPP_DT
    targets['GPP/Trans'] = targets.GPP_DT / targets.Fqt_mmol
    return


@app.cell
def _(features, targets):
    df = features.merge(targets, how = 'right', left_index=True, right_index=True)
    return (df,)


@app.cell
def _(df, np):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['WaPo.day'] = df['WaPo.day'] * -1
    return


@app.cell
def _(datetime, df):
    df_1 = df[df.index.date >= datetime.date(2018, 1, 1)]
    return (df_1,)


@app.cell
def _(df_1):
    df_1.describe()
    return


@app.cell
def _(pd):
    # df.to_csv(r"P:\3000\3135_IntMon\Intern\ARCHneu\UBA_IM\Themen_nach_Projektbereichen\PB2\2025_26_Auswertung_C_H2O_Duerre\Ergebnisse\ml_df.txt", index=True, index_label='datetime', sep=';')
    df_2 = pd.read_csv('ml_df.txt', sep=';', parse_dates=True, date_format='%Y-%m-%d')
    df_2['datetime'] = pd.to_datetime(df_2['datetime'])
    df_2.set_index('datetime', inplace=True)
    return (df_2,)


@app.cell
def _(df_2, utils):
    utils.dispaly_cluster_dendrogram(df_2)
    return


@app.cell
def _(df_2, root_mean_squared_error, train_test_split, xgb):
    target = 'GPP_DT'
    targets_1 = df_2[['GPP_DT', 'Reco_DT', 'Fq_mmol', 'Fqt_mmol', 'Fqe_mmol', 'Fc_umol', 'Fcp_umol', 'Fcr_umol', 'NEE', 'GPP/Trans']]
    features_1 = df_2.drop(columns=targets_1.columns.tolist())
    g = df_2.copy()
    _ = df_2.dropna(subset=target)
    y = _[target]
    X = _.drop(columns=targets_1.columns.to_list())
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)
    model = xgb.XGBRegressor(n_estimators=2500, max_depth=4, learning_rate=0.03, gamma=14, early_stopping_rounds=50, min_child_weight=3, colsample_bytree=1, subsample=0.5, eval_metric=root_mean_squared_error)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    return (
        X_test,
        X_train,
        features_1,
        model,
        target,
        targets_1,
        y_test,
        y_train,
    )


@app.cell
def _(
    MLPRegressor,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    X_test,
    X_train,
    r2_score,
    shap,
    y_test,
    y_train,
):
    preprocessor = Pipeline([
        ('impute', SimpleImputer()),
        ('scaler', StandardScaler())
    ])
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', MLPRegressor(hidden_layer_sizes=(10, 20, )))
        ])
    pipe.fit(X_train, y_train)
    print(r2_score(y_test, pipe.predict(X_test)))
    explainer = shap.KernelExplainer(
        pipe['model'].predict,
        shap.kmeans(X_train, 10)
    )
    shap_values = explainer.shap_values(preprocessor.transform(X_train))
    return


@app.cell
def _(X_test, model, r2_score, y_test):
    r2_score(y_test, model.predict(X_test))
    return


@app.cell
def _(X_train, model, shap):
    # explainer = shap.KernelExplainer(
    #     model.predict,
    #     shap.kmeans(X_train, 10)
    #     )
    explainer_1 = shap.TreeExplainer(model)
    shap_values_1 = explainer_1.shap_values(X_train)
    return (shap_values_1,)


@app.cell
def _(X_train, shap, shap_values_1):
    shap.summary_plot(shap_values_1, X_train, plot_type='dot')
    return


@app.cell
def _(X_train, features_1, pd, shap_values_1, targets_1):
    shap_values_doy = pd.DataFrame(shap_values_1).groupby(X_train.index.day_of_year).mean().to_numpy()
    features_doy = features_1.groupby(features_1.index.day_of_year).mean()
    targets_doy = targets_1.groupby(pd.to_datetime(targets_1.index).day_of_year).mean()
    return features_doy, shap_values_doy, targets_doy


@app.cell
def _(features_doy, np, pd, shap_values_doy):
    sorted = pd.DataFrame(np.abs(shap_values_doy), columns=features_doy.columns).mean().sort_values(ascending=False).index.to_list()
    return (sorted,)


@app.cell
def _(sorted):
    sorted
    return


@app.cell
def _(features_doy, pd, shap_values_doy):
    shap_melt = pd.DataFrame(shap_values_doy, columns=features_doy.columns).melt(ignore_index=False)
    return (shap_melt,)


@app.cell
def _(np, plt, shap_melt, sns, sorted, target, targets_doy):
    fig, ax = plt.subplots(1,1, figsize=(15,8), dpi=500)
    sns.barplot(data=shap_melt, x=shap_melt.index, y='value', hue='variable', dodge=False, palette='magma_r', fill=True, gap=0, width=.95, hue_order=sorted, alpha=1.0, edgecolor="none", saturation=1)
    plt.plot(targets_doy[target], label=target, color='black', linestyle='--')
    ticks = ax.set_xticks(list(np.arange(0,375,25)))
    ticks_lab = ax.set_xticklabels(list(np.arange(0,375,25)))
    plt.margins(x=0)
    sns.set_style('dark')
    plt.tight_layout()
    plt.legend()
    return


@app.cell
def _(features_doy, plt, shap_melt, sns, sorted, target, targets_doy):
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(15, 8), dpi=500)
    i = 0
    for variable in sorted:
        plt.bar(features_doy.index, shap_melt[shap_melt['variable'] == variable].value, label=variable, color=sns.color_palette('magma_r')[i], width=1, alpha=0.5)
        i = i + 1
    plt.plot(targets_doy[target], label=target, color='black', linestyle='--')
    plt.margins(x=0)
    plt.tight_layout()
    plt.legend()
    return


app._unparsable_cell(
    r"""
    t shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_doy[365, :],
            base_values=explainer.expected_value,
            data=features_doy.iloc[365, :],
            feature_names=features_doy.columns
        )
    )
    """,
    name="_"
)


@app.cell
def _(X_train, plt, shap_values_1):
    plt.barh(list(X_train.columns), shap_values_1.mean(axis=0))
    return


@app.cell
def _(RFECV, X_train, model, y_train):
    rfecv = RFECV(model, min_features_to_select=1, cv=5)
    rfecv.fit(X_train, y_train)
    return (rfecv,)


@app.cell
def _(pd, plt, rfecv):
    data = {
        key: value
        for key, value in rfecv.cv_results_.items()
        if key in ["n_features", "mean_test_score", "std_test_score"]
    }
    cv_results = pd.DataFrame(data)
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Mean test accuracy")
    plt.errorbar(
        x=cv_results["n_features"],
        y=cv_results["mean_test_score"],
        yerr=cv_results["std_test_score"],
    )
    plt.title("Recursive Feature Elimination \nwith correlated features")
    plt.show()
    return


@app.cell
def _(X_train, rfecv):
    X_train.columns[rfecv.support_]
    return


@app.cell
def _(X_train, model, utils, y_train):
    utils.plot_gini_permutation_importance(model, X_train, y_train)
    return


if __name__ == "__main__":
    app.run()
