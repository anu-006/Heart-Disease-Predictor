import pickle
import pandas as pd
import streamlit as st

st.title('Heart Disease Prediction')

# load model, bail out early if it isn’t there
try:
    with open('pipe.pkl', 'rb') as f:
        pipe = pickle.load(f)
except FileNotFoundError:
    st.error('pipe.pkl not found – run `python train.py` first')
    st.stop()

# you don’t actually need the dataset in the app
# data = pickle.load(open('data.pkl', 'rb'))

age = st.number_input('Age in years', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox(
    'Chest Pain Type',
    ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic']
)
trestbps = st.number_input('Resting Blood Pressure (mm Hg)')
chol = st.number_input('Serum Cholesterol in mg/dl')
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
restecg = st.selectbox(
    'Resting ECG results',
    ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy']
)
thalach = st.number_input('Maximum Heart Rate Achieved')
exang = st.selectbox('Exercise‑induced angina', ['Yes', 'No'])
oldpeak = st.number_input('ST depression induced by exercise relative to rest')

# extra inputs required by the pipeline trained with slope/ca/thal
slope = st.selectbox('Slope of the peak exercise ST segment',
                     ['Upsloping', 'Flat', 'Downsloping'])
ca = st.selectbox('Number of major vessels coloured by fluoroscopy',
                  ['0', '1', '2', '3', '4'])
thal = st.selectbox('Thalassemia',
                    ['Normal', 'Fixed defect', 'Reversible defect'])

if st.button('Predict'):
    sex_val = 1 if sex == 'Male' else 0
    cp_map = {'Typical Angina': 0, 'Atypical Angina': 1,
              'Non-Anginal Pain': 2, 'Asymptomatic': 3}
    fbs_val = 1 if fbs == 'True' else 0
    restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1,
                   'Left ventricular hypertrophy': 2}
    exang_val = 1 if exang == 'Yes' else 0
    slope_map = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
    ca_val = int(ca)
    thal_map = {'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3}

    query_df = pd.DataFrame({
        'age': [age],
        'sex': [sex_val],
        'cp': [cp_map[cp]],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs_val],
        'restecg': [restecg_map[restecg]],
        'thalach': [thalach],
        'exang': [exang_val],
        'oldpeak': [oldpeak],
        'slope': [slope_map[slope]],
        'ca': [ca_val],
        'thal': [thal_map[thal]],
    })

    prediction = pipe.predict(query_df)[0]
    label = 'disease' if prediction == 1 else 'no disease'
    st.write('Prediction:', prediction, f'({label})')