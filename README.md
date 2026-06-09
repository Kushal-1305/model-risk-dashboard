# Model Risk Dashboard

Banks use machine learning models to decide who gets a loan. Before any such model goes live, regulators require it to be independently tested — not just for accuracy, but for stability, fairness, and reliability under stress. This process is called **Model Risk Management**, governed by the Federal Reserve's SR 11-7 guidance.

This project builds that entire validation process from scratch for a credit scoring model, and packages the results into an interactive dashboard and a regulatory-grade validation report.

---

## What the app does

A credit scoring model (XGBoost, trained on 1,000 loan applicants) is put through four types of tests:

**Performance** — Does the model actually work on data it has never seen? The model scores 0.77 AUC on the out-of-time holdout set, meaning it correctly ranks a risky applicant above a safe one 77% of the time. It also flags that the model is overfit — it scores 0.99 on training data but drops to 0.77 on new data, a gap large enough to be a compliance concern.

**Data Drift** — Has the population of applicants changed since the model was trained? Each of the 20 input features is tested statistically. All features are stable except `credit_amount`, which shows a mild shift. The default rate (30%) is consistent across all time periods.

**Stress Testing** — What happens when the model is pushed to its limits? Every borderline applicant (those near the approval/rejection threshold) flips their decision when a single feature is nudged by 5% — a 100% instability rate, flagged as a major finding. Under a simulated recession (higher loan amounts, worse employment, depleted savings), average portfolio risk increases by 17%.

**Explainability** — Can we explain why the model makes each decision? The top driver is `checking_status` (whether the applicant has an active account). All five key features move predictions in the economically expected direction. Feature importance rankings stay perfectly consistent across 30 random subsamples — the model's logic is stable.

---

## The output

Everything above feeds into two outputs:

- An **interactive dashboard** (5 tabs) where you can explore drift charts, stress test results, SHAP importance plots, and individual prediction explanations
- An **SR 11-7 validation report** (auto-generated HTML) structured exactly as a bank's Model Risk team would submit to regulators — with a findings register, risk rating, and sign-off section

The model receives an overall risk rating of **HIGH**, driven by 4 red findings: miscalibrated probabilities, score distribution shift, overfitting, and decision boundary instability.

---

