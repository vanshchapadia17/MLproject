import sys
import os
import base64
import io
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.utils import load_object


def _build_shap_chart(feature_names, shap_vals):
    """Render a horizontal SHAP bar chart and return base64 PNG string."""
    pairs = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:8]
    names = [p[0] for p in pairs][::-1]
    vals  = [p[1] for p in pairs][::-1]
    colors = ["#3b82f6" if v >= 0 else "#f87171" for v in vals]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(names, vals, color=colors, height=0.55)
    ax.axvline(0, color="#475569", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on score)", color="#94a3b8", fontsize=8)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.grid(axis="x", color="#334155", linewidth=0.5, linestyle="--")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path       = os.path.join("artifects", "model.pkl")
            preprocessor_path = os.path.join("artifects", "preprocessor.pkl")

            model       = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds       = model.predict(data_scaled)

            # --- SHAP ---
            shap_img = None
            try:
                import shap
                from sklearn.linear_model import LinearRegression
                from sklearn.ensemble import AdaBoostRegressor

                # Load training data as background — statistically correct baseline
                # and avoids numerical explosion from StandardScaler(with_mean=False)
                # applied to OneHotEncoded features.
                train_df   = pd.read_csv(os.path.join("artifects", "train.csv"))
                X_train    = train_df.drop(columns=["math_score"])
                background = preprocessor.transform(X_train)

                if isinstance(model, LinearRegression):
                    explainer   = shap.LinearExplainer(model, background)
                    shap_values = explainer.shap_values(data_scaled)
                elif isinstance(model, AdaBoostRegressor):
                    explainer   = shap.Explainer(model.predict, background)
                    shap_values = explainer(data_scaled).values
                else:
                    explainer   = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(data_scaled, check_additivity=False)

                # shap_values may be 2-D (n_samples, n_features) or 1-D
                sv = np.array(shap_values)
                if sv.ndim == 3:
                    sv = sv[0][0]
                elif sv.ndim == 2:
                    sv = sv[0]

                # Clean feature names from ColumnTransformer
                raw_names   = preprocessor.get_feature_names_out()
                clean_names = [n.split("__")[-1].replace("_", " ").title()
                               for n in raw_names]

                shap_img = _build_shap_chart(clean_names, sv)

            except Exception as shap_err:
                print(f"[SHAP] Error: {shap_err}")

            return preds, shap_img

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):

        self.gender                    = gender
        self.race_ethnicity            = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch                     = lunch
        self.test_preparation_course   = test_preparation_course
        self.reading_score             = reading_score
        self.writing_score             = writing_score

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame({
                "gender":                      [self.gender],
                "race_ethnicity":              [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch":                       [self.lunch],
                "test_preparation_course":     [self.test_preparation_course],
                "reading_score":               [self.reading_score],
                "writing_score":               [self.writing_score],
            })
        except Exception as e:
            raise CustomException(e, sys)
