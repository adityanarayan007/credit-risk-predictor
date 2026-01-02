# 📊 Dataset Profile & Features

The model was trained on the **UCI Credit Risk Dataset**, consisting of **~32,500 records**. Below is the schema and distribution the model was built to handle:

## Feature Schema

| Feature Name | Type | Description | Range / Categories |
| :--- | :--- | :--- | :--- |
| `person_age` | Numerical | Age of the applicant | 18 — 100+ |
| `person_income` | Numerical | Annual personal income | $4,000 — $2,000,000 |
| `person_home_ownership` | Categorical | Home status | `RENT`, `MORTGAGE`, `OWN`, `OTHER` |
| `person_emp_length` | Numerical | Years of employment | 0 — 60 |
| `loan_intent` | Categorical | Purpose of the loan | `EDUCATION`, `MEDICAL`, `VENTURE`, etc. |
| `loan_grade` | Categorical | Risk grade (A to G) | `A` (Best) — `G` (Riskier) |
| `loan_amnt` | Numerical | Requested amount | $500 — $35,000 |
| `loan_int_rate` | Numerical | Interest rate | 5.4% — 23.2% |
| `cb_person_default_on_file` | Categorical | Prior default history | `Y`, `N` |

## Target Variable

- **Variable:** `loan_status`
- **Values:** `0` (Non-Default), `1` (Default)
- **Class Balance:** ~22% Default / 78% Non-Default

## 🛠️ Data Governance & Preprocessing

To ensure model reliability, the following automated checks were performed:

- **Outlier Handling:** Strategic removal of unrealistic data (e.g., employment length > 60 years).
- **Missing Value Imputation:** Median imputation for `loan_int_rate` and `person_emp_length` to maintain data integrity.
- **Feature Engineering:** Derivation of the **Loan-to-Income Ratio**, a critical predictor for creditworthiness.
- **Schema Enforcement:** Pydantic validation on the Inference API prevents "garbage-in" data from hitting the model.