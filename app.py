import gradio as gr
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

print("Loading dataset and training model on startup...")

# 1. Load UCI Default dataset directly via public URL
dataset_url = "[suspicious link removed]"
df = pd.read_excel(dataset_url, header=1, engine="xlrd")
df.columns = df.columns.str.strip()
if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

X = df.drop('default payment next month', axis=1)
y = df['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=20, stratify=y
)

# 2. Setup preprocessing
num_cols = [
    'LIMIT_BAL', 'AGE',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'EDUCATION', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'
]
cat_cols = ['SEX', 'MARRIAGE']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols)
    ]
)

# 3. Build & fit end-to-end Pipeline
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=64,
    learning_rate_init=0.001,
    max_iter=100,
    early_stopping=True,
    n_iter_no_change=10,
    random_state=20
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

pipeline.fit(X_train, y_train)
print("Model training complete! Launching Gradio UI...")

# 4. Define prediction function
def evaluate_credit_risk(limit_bal, sex, education, marriage, age,
                         pay_0, pay_2, pay_3, bill_amt1, pay_amt1, threshold):
    input_data = pd.DataFrame([{
        'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education,
        'MARRIAGE': marriage, 'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2, 'PAY_3': pay_3,
        'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt1, 'BILL_AMT3': bill_amt1,
        'BILL_AMT4': 0, 'BILL_AMT5': 0, 'BILL_AMT6': 0,
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt1, 'PAY_AMT3': pay_amt1,
        'PAY_AMT4': 0, 'PAY_AMT5': 0, 'PAY_AMT6': 0
    }])
    
    prob_default = pipeline.predict_proba(input_data)[0][1]
    decision = "High Risk (Default)" if prob_default >= threshold else "Low Risk (Approved)"
    
    return {
        "Assessment": decision,
        "Default Probability": f"{prob_default:.2%}",
        "Operating Threshold": f"{threshold:.2f}"
    }

# 5. Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💳 Credit Risk Assessment Application")
    gr.Markdown("Enter borrower attributes to evaluate credit card default risk using a Multi-Layer Perceptron (MLP).")
    
    with gr.Row():
        with gr.Column():
            limit_bal = gr.Number(label="Credit Limit (LIMIT_BAL)", value=50000)
            sex = gr.Dropdown(choices=[1, 2], label="Sex (1=Male, 2=Female)", value=2)
            education = gr.Dropdown(choices=[1, 2, 3, 4], label="Education (1=Grad, 2=Univ, 3=HighSchool, 4=Other)", value=2)
            marriage = gr.Dropdown(choices=[1, 2, 3], label="Marriage (1=Married, 2=Single, 3=Other)", value=1)
            age = gr.Slider(minimum=18, maximum=80, value=35, step=1, label="Age")
        with gr.Column():
            pay_0 = gr.Slider(minimum=-2, maximum=8, value=0, step=1, label="Repayment Status Month 0 (PAY_0)")
            pay_2 = gr.Slider(minimum=-2, maximum=8, value=0, step=1, label="Repayment Status Month 2 (PAY_2)")
            pay_3 = gr.Slider(minimum=-2, maximum=8, value=0, step=1, label="Repayment Status Month 3 (PAY_3)")
            bill_amt1 = gr.Number(label="Latest Bill Amount (BILL_AMT1)", value=12000)
            pay_amt1 = gr.Number(label="Latest Payment Amount (PAY_AMT1)", value=3000)
            threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Risk Threshold")
            
    submit_btn = gr.Button("Evaluate Applicant Risk", variant="primary")
    output = gr.JSON(label="Assessment Output")
    
    submit_btn.click(
        fn=evaluate_credit_risk,
        inputs=[limit_bal, sex, education, marriage, age, pay_0, pay_2, pay_3, bill_amt1, pay_amt1, threshold],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
