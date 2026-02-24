# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end ML project that predicts student math scores based on demographic and academic features (gender, race/ethnicity, parental education, lunch type, test prep course, reading score, writing score). Built with scikit-learn, CatBoost, XGBoost, and served via Flask.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install as a package (required for `src.*` imports to resolve)
```bash
pip install -e .
```

### Run the full training pipeline (data ingestion → transformation → model training)
```bash
python src/components/data_ingestion.py
```

### Start the Flask web app
```bash
python app.py
```
The app runs on `http://0.0.0.0:5000`. The home page is at `/`, the prediction form at `/predictdata`.

## Architecture

The project follows a modular ML pipeline pattern. All source code lives in `src/`, which is installed as the `mlproject` package via `setup.py` and `requirements.txt` (`-e .`).

### Training Pipeline (offline)

```
notebook/data/stud.csv
        ↓
src/components/data_ingestion.py     → splits raw data → artifects/{data,train,test}.csv
        ↓
src/components/data_transformation.py → fits sklearn ColumnTransformer (StandardScaler + OneHotEncoder)
                                       → saves artifects/preprocessor.pkl
        ↓
src/components/model_trainer.py      → GridSearchCV across 7 regressors, picks best R²>0.6
                                       → saves artifects/model.pkl
```

The full pipeline is triggered by running `data_ingestion.py` as `__main__`, which chains all three components.

### Inference Pipeline (online)

```
app.py (Flask)
  └─ POST /predictdata
       └─ src/pipeline/predict_pipeline.py
            ├─ CustomData      – maps HTML form fields to a DataFrame
            └─ PredictPipeline – loads artifects/preprocessor.pkl + model.pkl, returns math_score prediction
```

### Supporting modules

- `src/exception.py` — `CustomException` wraps all errors with filename + line number context. Usage: `raise CustomException(e, sys)`
- `src/logger.py` — configures `logging` to write timestamped log files under `logs/`
- `src/utils.py` — `save_object`/`load_object` (via `dill`), `evaluate_models` (GridSearchCV + R² scoring)

### Config pattern

Each component uses a `@dataclass` config (e.g., `DataIngestionConfig`, `ModelTrainerConfig`) to hold file paths, keeping all path constants in one place per component.

### Artifacts

Serialized objects are stored in `artifects/` (note the spelling):
- `artifects/model.pkl` — best trained regressor
- `artifects/preprocessor.pkl` — fitted sklearn preprocessing pipeline

Both are loaded at inference time by `PredictPipeline`.

### Notebooks

`notebook/` contains two Jupyter notebooks used for exploration:
- `1 . EDA STUDENT PERFORMANCE .ipynb` — exploratory data analysis
- `2. MODEL TRAINING.ipynb` — model prototyping

The raw dataset is at `notebook/data/stud.csv` and is read directly by `DataIngestion`.
