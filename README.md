# AI-Driven Explainable Hospital Readmission Risk Prediction System

Capstone project repository for a multimodal machine-learning system that predicts 30-day hospital readmission risk and provides patient-level explanations.

## Team

See [TEAM.md](TEAM.md).

## Repository Structure

```text
ai-hospital-readmission-risk/
├── data/
│   ├── raw/
│   └── processed/
├── preprocessing/
├── features/
├── models/
├── explainability/
├── api/
├── dashboard/
├── tests/
├── notebooks/
├── requirements.txt
├── TEAM.md
└── PROJECT_STATUS.md
```

## Central GitHub Repository Workflow

### 1. One person creates the repository

Create a GitHub repository named:

`ai-hospital-readmission-risk`

Add the other four members under:

**Settings → Collaborators → Add people**

### 2. Everyone clones the same repository

```powershell
git clone https://github.com/YOUR-USERNAME/ai-hospital-readmission-risk.git
cd ai-hospital-readmission-risk
```

### 3. Create a personal feature branch

Alex:

```powershell
git checkout -b feature/data-preprocessing
```

Allen:

```powershell
git checkout -b feature/feature-engineering
```

Athul:

```powershell
git checkout -b feature/ml-model
```

Azharuddin:

```powershell
git checkout -b feature/explainability
```

Lutfi:

```powershell
git checkout -b feature/api-dashboard
```

### 4. Commit and push your own work

```powershell
git add .
git commit -m "feat: describe the change"
git push -u origin YOUR-BRANCH-NAME
```

### 5. Pull Request

On GitHub create:

`YOUR-BRANCH → main`

Have at least one teammate review it, then merge.

### 6. Keep your branch updated

Before starting new work:

```powershell
git checkout main
git pull origin main
git checkout YOUR-BRANCH-NAME
git merge main
```

## Person 1: Data & Preprocessing

Alexes Roy Thomas owns the `preprocessing/` module.

Put the actual dataset in:

`data/raw/`

Do not commit real patient/EHR datasets to GitHub.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run preprocessing from the repository root:

```powershell
python -m preprocessing.run_preprocessing
```

Before running, update `INPUT_FILE` and `TARGET_COLUMN` in `preprocessing/run_preprocessing.py` to match the actual dataset.

The pipeline:

1. Loads CSV/Excel data.
2. Standardizes column names.
3. Validates the target.
4. Handles missing numeric/categorical values.
5. One-hot encodes categorical features.
6. Scales numeric features.
7. Creates train/validation/test splits.
8. Saves the fitted preprocessing pipeline.

## Important

The current preprocessing configuration uses placeholder dataset/target names until the actual dataset is selected and inspected. Do not invent the target column name; update it after the team confirms the dataset.
