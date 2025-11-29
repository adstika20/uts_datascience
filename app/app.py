
import gradio as gr
import pandas as pd
import joblib
import numpy as np

# =========================
# LOAD ARTIFACTS
# =========================
models = {
    "SVC": joblib.load("svm_model.pkl"),
    "KNN": joblib.load("knn_model.pkl"),
    "RFC": joblib.load("rf_model.pkl"),
    "GBC": joblib.load("gbc_model.pkl")
}

scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

categorical_cols = ['job', 'marital', 'education', 'default',
                    'housing', 'loan', 'contact', 'month', 'poutcome']

# =========================
# PREDICT FUNCTION
# =========================
def predict(model_choice, job, marital, education, default, housing, loan, contact,
            month, poutcome, age, balance, day, duration, campaign, pdays, previous):
    
    model = models[model_choice]
    
    input_dict = {
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "poutcome": poutcome,
        "age": age,
        "balance": balance,
        "day": day,
        "duration": duration,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous
    }
    
    df = pd.DataFrame([input_dict])
    
    # encode kategori dengan fallback "unknown"
    for col in categorical_cols:
        le = label_encoders[col]
        val = df.at[0, col]
        if val not in le.classes_:
            df.at[0, col] = "unknown"
        df[col] = le.transform([df.at[0, col]])
    
    df = df[feature_columns]
    
    X_scaled = scaler.transform(df)
    
    pred = model.predict(X_scaled)[0]
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)[0][1]
        proba_text = f"{proba:.2f}"
    else:
        proba_text = "N/A"
    
    target_le = label_encoders["y"]
    final_label = target_le.inverse_transform([pred])[0]
    
    
    if final_label.lower() == "yes":
        explanation = "Model memprediksi nasabah kemungkinan besar akan membuka deposito berjangka."
    else:
        explanation = "Model memprediksi nasabah kemungkinan besar tidak akan membuka deposito berjangka."

    return f"Prediction: {final_label.upper()} (Probability: {proba_text})\n{explanation}"

    
    print("Input df:\n", df)
    for col in categorical_cols:
        print(f"{col} value: {df.at[0, col]}, classes: {label_encoders[col].classes_}")


# =========================
# GRADIO APP
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 🏦 Bank Campaign Term Deposit Prediction")
    
    with gr.Row():
        with gr.Column():
            model_choice = gr.Dropdown(["SVC", "KNN", "RFC", "GBC"], label="Choose Model")
            job = gr.Dropdown([
                "blue-collar","management","technician","admin","services","retired",
                "self-employed","entrepreneur","unemployed","housemaid","student","unknown"
            ], label="Job")
            marital = gr.Dropdown(["married","single","divorced"], label="Marital Status")
            education = gr.Dropdown(["secondary","tertiary","primary","unknown"], label="Education")
            default = gr.Dropdown(["no","yes"], label="Default Credit?")
            housing = gr.Dropdown(["yes","no"], label="Housing Loan?")
            loan = gr.Dropdown(["no","yes"], label="Personal Loan?")
            contact = gr.Dropdown(["cellular","unknown","telephone"], label="Contact Type")
            month = gr.Dropdown([
                "may","jul","aug","jun","nov","apr","feb","jan","oct","sep","mar","dec"
            ], label="Month")
            poutcome = gr.Dropdown(["unknown","failure","other","success"], label="Previous Outcome")
        
        with gr.Column():
            age = gr.Slider(18, 95, step=1, label="Age")
            balance = gr.Number(label="Balance (-8019 to 102127)")
            day = gr.Slider(1, 31, step=1, label="Day of month")
            duration = gr.Slider(0, 4918, step=1, label="Duration")
            campaign = gr.Slider(1, 63, step=1, label="Campaign")
            pdays = gr.Number(label="pdays (-1 to 871)")
            previous = gr.Slider(0, 275, step=1, label="Previous contacts")
            
            btn = gr.Button("Predict Term Deposit Subscription")
            output = gr.Textbox(label="🔍 Result",lines=5)
    
    btn.click(
        predict,
        inputs=[model_choice, job, marital, education, default, housing, loan, contact,
                month, poutcome, age, balance, day, duration, campaign, pdays, previous],
        outputs=output
    )

demo.launch()

