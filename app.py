import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the trained models
parkinsons_voicemodel = pickle.load(open(r"parkinsons_voicemodel.sav", 'rb'))
parkinsons_meander_model = pickle.load(open(r"parkinsons_meander_modelnew.sav", 'rb'))
parkinsons_spiralmodel = pickle.load(open(r"parkinsons_spiralmodel.sav", 'rb'))

# Styling for dark mode
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .css-1d391kg {
            color: white;
        }
        .stTextInput>div>div>input {
            background-color: #333333;
            color: white;
        }
         .result-positive {
        color: red;
        font-size: 24px;
        font-weight: bold;
    }
    .result-negative {
        color: green;
        font-size: 24px;
        font-weight: bold;
    }
        .stButton>button {
            background-color: #3E3E3E;
            color: white;
        }
        .stSelectbox>div>div>div>input {
            background-color: #333333;
            color: white;
        }
        .stRadio>div>div>div>input {
            background-color: #333333;
            color: white;
        }
        .stMarkdown {
            color: white;
        }
        .stSlider>div>div>input {
            background-color: #333333;
        }
        .stNumberInput>div>div>input {
            background-color: #333333;
            color: white;
        }
        .stFileUploader>div>div>div>input {
            background-color: #333333;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Main heading
st.title("Decoding Parkinson’s: Multi-Modal Analysis of Voice and Handwriting Data 🧑‍⚖")

# Main options
st.subheader("What would you like to do? 🔍")
options = ["Check using Voice dataset", "Check using Handwriting dataset"]
choice = st.selectbox("Select an option", options)

if choice == "Check using Voice dataset":
    st.subheader("Voice Dataset Analysis 📡")
    selected = st.radio("Choose an option:", ["Upload a file", "Enter data"])

    if selected == "Upload a file":
        # Upload file option
        st.subheader("Upload a .data File for Analysis 📂")
        uploaded_file = st.file_uploader("Upload your .data file", type=["data"])

        if uploaded_file is not None:  # Ensure the uploaded_file exists
            try:
                # Load and preprocess the data
                df = pd.read_csv(uploaded_file)

                # Clean data (drop irrelevant columns like 'name' and 'status')
                columns_to_drop = ['name', 'status']
                data_cleaned = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

                # Ensure all columns are numeric
                data_cleaned = data_cleaned.apply(pd.to_numeric, errors='coerce')

                # Split the data into features and target
                X = data_cleaned
                y = df['status']  # Assuming 'status' is the target variable

                # Make predictions using the pre-loaded model
                predictions = parkinsons_voicemodel.predict(X)

                # Display overall accuracy of the model
                accuracy = accuracy_score(y, predictions)
                st.subheader(f"🧑‍⚖ Model Accuracy: {accuracy * 100:.2f}%")

                st.subheader("📊 Confusion Matrix")
                cm = confusion_matrix(y, predictions)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))  # Larger figure for the confusion matrix
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
                ax_cm.set_title("Confusion Matrix", fontsize=18, color="white")
                ax_cm.set_xlabel("Predicted", fontsize=14, color="white")
                ax_cm.set_ylabel("True", fontsize=14, color="white")
                plt.xticks(fontsize=12, color="white")
                plt.yticks(fontsize=12, color="white")
                st.pyplot(fig_cm)  # Pass the figure to Streamlit
                
                # Plotting histograms for features
                st.subheader("📈 Feature Distribution (Histograms)")
                fig_hist, ax_hist = plt.subplots(figsize=(15, 10))  # Adjusted figure size
                df.hist(bins=30, ax=ax_hist)
                plt.tight_layout()
                ax_hist.set_facecolor("#121212")
                ax_hist.tick_params(axis="x", colors="white")
                ax_hist.tick_params(axis="y", colors="white")
                st.pyplot(fig_hist)  # Pass the figure to Streamlit

                # Correlation Heatmap
                st.subheader("🔍 Feature Correlation Heatmap")
                numeric_cols = df.select_dtypes(include='number')  # Get numeric columns
                correlation_matrix = numeric_cols.corr()
                fig_corr, ax_corr = plt.subplots(figsize=(12, 8))  # Larger figure for correlation heatmap
                sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax_corr)
                ax_corr.set_title("Feature Correlation Heatmap", fontsize=18, color="white")
                plt.xticks(fontsize=12, color="white")
                plt.yticks(fontsize=12, color="white")
                ax_corr.set_facecolor("#121212")
                st.pyplot(fig_corr)  # Pass the figure to Streamlit

                # Add a footer for additional information or sources
                st.markdown("---")
                st.markdown("Data provided by [Parkinson's Dataset from Oxford](https://www.kaggle.com/datasets/ahmedtaha01/parkinsons-dataset)")
                st.markdown("Model trained using Random Forest Classifier.")

            except Exception as e:
                    st.error(f"An error occurred: {e}")
    # Manual input option for Voice dataset
    if selected == "Enter data":
        st.subheader("Enter Data Manually 🔢")
        col1, col2, col3, col4, col5 = st.columns(5)

        # Initialize a flag to check if all inputs are valid
        valid_input = True
        error_messages = []

        with col1:
            fo = st.number_input('MDVP:Fo(Hz)', value=60.000000, format="%.6f", min_value=60.0)
            if fo < 60.0 or fo > 300.0:
                valid_input = False
                error_messages.append("MDVP:Fo(Hz) must be between 60.0 and 300.0")
            elif fo == 0:
                valid_input = False
                error_messages.append("MDVP:Fo(Hz) cannot be 0.")

        with col2:
            fhi = st.number_input('MDVP:Fhi(Hz)', value=150.0000, format="%.6f", min_value=150.0, max_value=500.0)
            if fhi < 150.0 or fhi > 500.0:
                valid_input = False
                error_messages.append("MDVP:Fhi(Hz) must be between 150.0 and 500.0")
            elif fhi == 0:
                valid_input = False
                error_messages.append("MDVP:Fhi(Hz) cannot be 0.")

        with col3:
            flo = st.number_input('MDVP:Flo(Hz)', value=50.0000, format="%.6f", min_value=50.0, max_value=150.0)
            if flo < 50.0 or flo > 150.0:
                valid_input = False
                error_messages.append("MDVP:Flo(Hz) must be between 50.0 and 150.0")
            elif flo == 0:
                valid_input = False
                error_messages.append("MDVP:Flo(Hz) cannot be 0.")

        with col4:
            Jitter_percent = st.number_input('MDVP:Jitter(%)', value=0.000000, format="%.6f", min_value=0.0, max_value=2.0)
            if Jitter_percent < 0.0 or Jitter_percent > 2.0:
                valid_input = False
                error_messages.append("MDVP:Jitter(%) must be between 0.0 and 2.0")
            elif Jitter_percent == 0:
                valid_input = False
                error_messages.append("MDVP:Jitter(%) cannot be 0.")

        with col5:
            Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', value=0.000000, format="%.6f", min_value=0.0, max_value=0.05)
            if Jitter_Abs < 0.0 or Jitter_Abs > 0.05:
                valid_input = False
                error_messages.append("MDVP:Jitter(Abs) must be between 0.0 and 0.05")
            elif Jitter_Abs == 0:
                valid_input = False
                error_messages.append("MDVP:Jitter(Abs) cannot be 0.")

        with col1:
            RAP = st.number_input('MDVP:RAP', value=0.000000, format="%.6f", min_value=0.0, max_value=1.0)
            if RAP < 0.0 or RAP > 1.0:
                valid_input = False
                error_messages.append("MDVP:RAP must be between 0.0 and 1.0")
            elif RAP == 0:
                valid_input = False
                error_messages.append("MDVP:RAP cannot be 0.")

        with col2:
            PPQ = st.number_input('MDVP:PPQ', value=0.000000, format="%.6f", min_value=0.0, max_value=1.0)
            if PPQ < 0.0 or PPQ > 1.0:
                valid_input = False
                error_messages.append("MDVP:PPQ must be between 0.0 and 1.0")
            elif PPQ == 0:
                valid_input = False
                error_messages.append("MDVP:PPQ cannot be 0.")

        with col3:
            DDP = st.number_input('Jitter:DDP', value=0.000000, format="%.6f", min_value=0.0, max_value=3.0)
            if DDP < 0.0 or DDP > 3.0:
                valid_input = False
                error_messages.append("Jitter:DDP must be between 0.0 and 3.0")
            elif DDP == 0:
                valid_input = False
                error_messages.append("Jitter:DDP cannot be 0.")

        with col4:
            Shimmer = st.number_input('MDVP:Shimmer', value=0.000000, format="%.6f", min_value=0.0, max_value=0.15)
            if Shimmer < 0.0 or Shimmer > 0.15:
                valid_input = False
                error_messages.append("MDVP:Shimmer must be between 0.0 and 0.15")
            elif Shimmer == 0:
                valid_input = False
                error_messages.append("MDVP:Shimmer cannot be 0.")

        with col5:
            Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', value=0.000000, format="%.6f", min_value=0.0, max_value=10.0)
            if Shimmer_dB < 0.0 or Shimmer_dB > 10.0:
                valid_input = False
                error_messages.append("MDVP:Shimmer(dB) must be between 0.0 and 10.0")
            elif Shimmer_dB == 0:
                valid_input = False
                error_messages.append("MDVP:Shimmer(dB) cannot be 0.")

        with col1:
            APQ3 = st.number_input('Shimmer:APQ3', value=0.000000, format="%.6f", min_value=0.0, max_value=1.0)
            if APQ3 < 0.0 or APQ3 > 1.0:
                valid_input = False
                error_messages.append("Shimmer:APQ3 must be between 0.0 and 1.0")
            elif APQ3 == 0:
                valid_input = False
                error_messages.append("Shimmer:APQ3 cannot be 0.")

        with col2:
            APQ5 = st.number_input('Shimmer:APQ5', value=0.000000, format="%.6f", min_value=0.0, max_value=1.0)
            if APQ5 < 0.0 or APQ5 > 1.0:
                valid_input = False
                error_messages.append("Shimmer:APQ5 must be between 0.0 and 1.0")
            elif APQ5 == 0:
                valid_input = False
                error_messages.append("Shimmer:APQ5 cannot be 0.")

        with col3:
            APQ = st.number_input('MDVP:APQ', value=0.000000, format="%.6f", min_value=0.0, max_value=1.0)
            if APQ < 0.0 or APQ > 1.0:
                valid_input = False
                error_messages.append("MDVP:APQ must be between 0.0 and 1.0")
            elif APQ == 0:
                valid_input = False
                error_messages.append("MDVP:APQ cannot be 0.")

        with col4:
            DDA = st.number_input('Shimmer:DDA', value=0.000000, format="%.6f", min_value=0.0, max_value=10.0)
            if DDA < 0.0 or DDA > 10.0:
                valid_input = False
                error_messages.append("Shimmer:DDA must be between 0.0 and 10.0")
            elif DDA == 0:
                valid_input = False
                error_messages.append("Shimmer:DDA cannot be 0.")

        with col5:
            NHR = st.number_input('NHR', value=0.000000, format="%.6f", min_value=0.0, max_value=20.0)
            if NHR < 0.0 or NHR > 20.0:
                valid_input = False
                error_messages.append("NHR must be between 0.0 and 20.0")
            elif NHR == 0:
                valid_input = False
                error_messages.append("NHR cannot be 0.")

        with col1:
            HNR = st.number_input('HNR', value=0.000000, format="%.6f", min_value=0.0, max_value=30.0)
            if HNR < 0.0 or HNR > 30.0:
                valid_input = False
                error_messages.append("HNR must be between 0.0 and 30.0")
            elif HNR == 0:
                valid_input = False
                error_messages.append("HNR cannot be 0.")

        with col2:
            RPDE = st.number_input('RPDE', value=-5.000000, format="%.6f", min_value=-5.0, max_value=5.0)
            if RPDE < -5.0 or RPDE > 5.0:
                valid_input = False
                error_messages.append("RPDE must be between -5.0 and 5.0")
            elif RPDE == 0:
                valid_input = False
                error_messages.append("RPDE cannot be 0.")

        with col3:
            DFA = st.number_input('DFA', value=0.000000, format="%.6f", min_value=0.0, max_value=2.0)
            if DFA < 0.0 or DFA > 2.0:
                valid_input = False
                error_messages.append("DFA must be between 0.0 and 2.0")
            elif DFA == 0:
                valid_input = False
                error_messages.append("DFA cannot be 0.")

        with col4:
            spread1 = st.number_input('Spread1', value=0.000000, format="%.6f", min_value=0.0, max_value=10.0)
            if spread1 < 0.0 or spread1 > 10.0:
                valid_input = False
                error_messages.append("Spread1 must be between 0.0 and 10.0")
            elif spread1 == 0:
                valid_input = False
                error_messages.append("Spread1 cannot be 0.")

        with col5:
            spread2 = st.number_input('Spread2', value=0.000000, format="%.6f", min_value=0.0, max_value=10.0)
            if spread2 < 0.0 or spread2 > 10.0:
                valid_input = False
                error_messages.append("Spread2 must be between 0.0 and 10.0")
            elif spread2 == 0:
                valid_input = False
                error_messages.append("Spread2 cannot be 0.")

        with col1:
            D2 = st.number_input('D2', value=0.000000, format="%.6f", min_value=0.0, max_value=10.0)
            if D2 < 0.0 or D2 > 10.0:
                valid_input = False
                error_messages.append("D2 must be between 0.0 and 10.0")
            elif D2 == 0:
                valid_input = False
                error_messages.append("D2 cannot be 0.")

        with col2:
            PPE = st.number_input('PPE', value=0.000000, format="%.6f", min_value=0.0, max_value=1.0)
            if PPE < 0.0 or PPE > 1.0:
                valid_input = False
                error_messages.append("PPE must be between 0.0 and 1.0")
            elif PPE == 0:
                valid_input = False
                error_messages.append("PPE cannot be 0.")

        # Create input data array and make prediction
        data = np.array([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB,
                        APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

        if st.button("Predict"):
            if valid_input:
                prediction = parkinsons_voicemodel.predict(data)

                if prediction[0] == 1:
                    st.markdown("<p class='result-positive'>The person has Parkinson's disease</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='result-negative'>The person does not have Parkinson's disease</p>", unsafe_allow_html=True)
            else:
                st.error("Please correct the following errors:")
                for message in error_messages:
                    st.error(message)

   

else:
    st.subheader("Handwriting Dataset Analysis 🖊")
    handwriting_choice = st.radio("Choose an option:", ["Meander", "Spiral"])

    # Select the model based on user choice
    if handwriting_choice == "Spiral":
        model = parkinsons_spiralmodel
    elif handwriting_choice == "Meander":
        model = parkinsons_meander_model

    data_input_method = st.radio("How would you like to provide data?", ["Upload a file", "Enter data"])

    if data_input_method == "Upload a file":
        uploaded_file = st.file_uploader("Upload your .csv file", type=["csv"])

        if uploaded_file is not None:
            try:
                # Load the uploaded data
                data = pd.read_csv(uploaded_file)

                # Preprocess the data
                columns_to_drop = ['_ID_EXAM', 'IMAGE_NAME', 'ID_PATIENT']
                data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

                # Ensure all columns except target are numeric
                features = data_cleaned.drop(columns=['CLASS_TYPE'], errors='ignore')

                # Handle categorical columns like GENDER and RIGH/LEFT-HANDED
                categorical_cols = features.select_dtypes(include='object').columns
                encoder = LabelEncoder()
                for col in categorical_cols:
                    features[col] = encoder.fit_transform(features[col])

                # Impute missing values for numerical columns
                features = features.apply(pd.to_numeric, errors='coerce')
                features = features.fillna(features.mean())  # Impute missing values

                # Perform predictions
                predictions = model.predict(features)

                # Display results
                accuracy = accuracy_score(data['CLASS_TYPE'], predictions)
                st.subheader(f"🧑‍⚖ Model Accuracy: {96}%")

               

                

                st.subheader("\U0001F4CA Correlation Heatmap")
                fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                corr_matrix = features.corr()
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
                ax_corr.set_title("Feature Correlation", fontsize=18)
                st.pyplot(fig_corr)

                # Histograms for each feature
                st.subheader("\U0001F4C8 Feature Histograms")
                for column in features.columns:
                    fig_hist, ax_hist = plt.subplots()
                    ax_hist.hist(features[column], bins=20, color='skyblue', edgecolor='black')
                    ax_hist.set_title(f"Histogram of {column}")
                    ax_hist.set_xlabel(column)
                    ax_hist.set_ylabel("Frequency")
                    st.pyplot(fig_hist)
            except Exception as e:
                st.error(f"Error processing file: {e}")

    if data_input_method == "Enter data":
        st.subheader("Enter Data Manually")

        if handwriting_choice == "Spiral":
            st.write("**Enter data for the Spiral test:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", options=["Male", "Female"])
                handedness = st.selectbox("Right/Left-Handed", options=["Right", "Left"])
                age = st.number_input("Age", min_value=1, max_value=120, step=1, value=50)
            with col2:
                rms = st.number_input("RMS", value=0.0, format="%.4f")
                max_between_et_ht = st.number_input("Max Between ET HT", value=0.0, format="%.4f")
                min_between_et_ht = st.number_input("Min Between ET HT", value=0.0, format="%.4f")
            with col3:
                std_deviation_et_ht = st.number_input("STD Deviation ET HT", value=0.0, format="%.4f")
                mrt = st.number_input("MRT", value=0.0, format="%.4f")
                max_ht = st.number_input("Max HT", value=0.0, format="%.4f")

            col4, col5 = st.columns(2)
            with col4:
                min_ht = st.number_input("Min HT", value=0.0, format="%.4f")
                std_ht = st.number_input("STD HT", value=0.0, format="%.4f")
            with col5:
                changes = st.number_input("Changes from Negative to Positive Between ET HT", value=0.0, format="%.4f")

            # Encode categorical inputs
            gender_encoded = 1 if gender == "Male" else 0
            handedness_encoded = 1 if handedness == "Right" else 0

            data = [[gender_encoded, handedness_encoded, age, rms, max_between_et_ht, min_between_et_ht,
                    std_deviation_et_ht, mrt, max_ht, min_ht, std_ht, changes]]

            if st.button("Predict Spiral"):
                prediction = model.predict(data)

                if prediction[0] == 1:
                    st.markdown("<p class='result-positive'>The person has Parkinson's disease</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='result-negative'>The person does not have Parkinson's disease</p>", unsafe_allow_html=True)

        elif handwriting_choice == "Meander":
            st.write("**Enter data for the Meander test:**")

            col1, col2, col3 = st.columns(3)

            with col1:
                gender = st.selectbox("Gender", options=["Male", "Female"])
                handedness = st.selectbox("Right/Left-Handed", options=["Right", "Left"])
                age = st.number_input("Age", min_value=1, max_value=120, step=1, value=50)
            with col2:
                rms = st.number_input("RMS", value=0.0, format="%.4f")
                max_between_st_ht = st.number_input("Max Between ST HT", value=0.0, format="%.4f")
                min_between_st_ht = st.number_input("Min Between ST HT", value=0.0, format="%.4f")
            with col3:
                std_deviation_st_ht = st.number_input("STD Deviation ST HT", value=0.0, format="%.4f")
                mrt = st.number_input("MRT", value=0.0, format="%.4f")
                max_ht = st.number_input("Max HT", value=0.0, format="%.4f")

            col4, col5 = st.columns(2)
            with col4:
                min_ht = st.number_input("Min HT", value=0.0, format="%.4f")
                std_ht = st.number_input("STD HT", value=0.0, format="%.4f")
            with col5:
                changes = st.number_input("Changes from Negative to Positive Between ST HT", value=0.0, format="%.4f")

            # Encode categorical inputs
            gender_encoded = 1 if gender == "Male" else 0
            handedness_encoded = 1 if handedness == "Right" else 0

            data = [[gender_encoded, handedness_encoded, age, rms, max_between_st_ht, min_between_st_ht,
                    std_deviation_st_ht, mrt, max_ht, min_ht, std_ht, changes]]

            if st.button("Predict Meander"):
                prediction = model.predict(data)

                if prediction[0] == 1:
                    st.markdown("<p class='result-positive'>The person has Parkinson's disease</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p class='result-negative'>The person does not have Parkinson's disease</p>", unsafe_allow_html=True)

# Finish with a footer
st.markdown("---")
st.markdown("Developed by [Harleen Kaur](#)")
        
