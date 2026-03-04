import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Iris Species Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .sidebar .sidebar-content { background-color: #2c3e50; color: white; }
    .stButton > button { 
        background-color: #dc2626; 
        color: white; 
        border-radius: 8px; 
        padding: 10px 24px; 
        font-weight: bold; 
        width: 100%;
    }
    h1, h2, h3 { color: #1e3a8a; }
    .stNumberInput > div > div > input { text-align: center; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ─── Sidebar = Model Information ───────────────────────
# ─── Sidebar = Model Information ───────────────────────
with st.sidebar:
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #87CEEB;  /* سماوي فاتح */
            color: #ffffff !important;  /* أبيض للنصوص */
        }
        .sidebar .sidebar-content h2 {
            color: #ffffff !important;  /* العنوان الأبيض */
        }
        .sidebar .sidebar-content p, 
        .sidebar .sidebar-content strong {
            color: #f0f8ff !important;  /* أبيض فاتح جدًا للنصوص العادية */
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2>Model Information</h2>", unsafe_allow_html=True)
    st.markdown("**Model Name:** Iris Species Predictor")
    st.markdown("**Algorithm:** Gaussian Naive Bayes")
    st.markdown("**Dataset:** Iris (classic classification dataset)")
    st.markdown("**Built by:** Safa")
    st.markdown("---")
# ─── Main Page ─────────────────────────────────────────
st.title("Iris Flower Species Prediction")
st.write("This simple app uses a saved **Naive Bayes** model to predict Iris flower species")

st.success("Model loaded successfully! (GaussianNB)")

# ─── Tabs ──────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["🌼 Single Prediction", "📁 Batch Prediction"])

species_labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
required_columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# ─── Single Prediction ─────────────────────────────────────────
with tab_single:
    st.subheader("Input Flower Features")

    col_label, col_input = st.columns([2, 5])

    with col_label:
        st.markdown(
            """
            <div style="height:56px; display:flex; align-items:center; font-weight:bold;">Sepal Length (cm)</div>
            <div style="height:56px; display:flex; align-items:center; font-weight:bold;">Sepal Width (cm)</div>
            <div style="height:56px; display:flex; align-items:center; font-weight:bold;">Petal Length (cm)</div>
            <div style="height:56px; display:flex; align-items:center; font-weight:bold;">Petal Width (cm)</div>
            """,
            unsafe_allow_html=True
        )

    with col_input:
        iris_df = pd.read_csv("iris.csv")

        sepal_length = st.number_input("Sepal Length hidden", min_value=float(iris_df['sepal length (cm)'].min()),
                                       max_value=float(iris_df['sepal length (cm)'].max()),
                                       value=float(iris_df['sepal length (cm)'].mean()), step=0.1, label_visibility="collapsed")

        sepal_width = st.number_input("Sepal Width hidden", min_value=float(iris_df['sepal width (cm)'].min()),
                                      max_value=float(iris_df['sepal width (cm)'].max()),
                                      value=float(iris_df['sepal width (cm)'].mean()), step=0.1, label_visibility="collapsed")

        petal_length = st.number_input("Petal Length hidden", min_value=float(iris_df['petal length (cm)'].min()),
                                       max_value=float(iris_df['petal length (cm)'].max()),
                                       value=float(iris_df['petal length (cm)'].mean()), step=0.1, label_visibility="collapsed")

        petal_width = st.number_input("Petal Width hidden", min_value=float(iris_df['petal width (cm)'].min()),
                                      max_value=float(iris_df['petal width (cm)'].max()),
                                      value=float(iris_df['petal width (cm)'].mean()), step=0.1, label_visibility="collapsed")

    if st.button("Predict Species", type="primary"):
        model = joblib.load("model_Gaussian.pkl")
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.markdown("### Prediction Result")
        st.markdown(f"**Predicted Species:** <span style='color:#2563eb; font-weight:bold; font-size:1.3rem;'>{species_labels[pred]}</span>", unsafe_allow_html=True)

        st.write("Probabilities:")
        for i, label in species_labels.items():
            st.write(f"{label}: **{proba[i]:.2%}**")

        # ─── Scatter Plot ────────────────────────────────
        st.markdown("### Where your flower fits in the dataset")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            data=iris_df,
            x='petal length (cm)',
            y='petal width (cm)',
            hue='species',
            palette='deep',
            s=80,
            alpha=0.7,
            ax=ax
        )

        ax.scatter(
            petal_length, petal_width,
            s=300, c='red', marker='*',
            edgecolor='black', label='Your Flower',
            zorder=10
        )
        ax.set_title("Petal Length vs Petal Width")
        ax.legend()
        st.pyplot(fig)

# ─── Batch Prediction ──────────────────────────────────────────
with tab_batch:
    st.subheader("Upload CSV for Batch Prediction")
    
    st.markdown("Upload a CSV file containing at least these columns: " + ", ".join(required_columns))

    uploaded_file = st.file_uploader(" ", type="csv", label_visibility="collapsed")

    st.caption("Limit 200MB per file • CSV")

    if uploaded_file is not None:
        with st.spinner("Processing your file..."):
            try:
                df = pd.read_csv(uploaded_file)

                missing = [col for col in required_columns if col not in df.columns]

                if missing:
                    st.error(
                        f"The uploaded file is missing some required columns:\n"
                        f"**Missing:** {', '.join(missing)}\n\n"
                        f"**Required columns:** {', '.join(required_columns)}"
                    )
                else:
                    model = joblib.load("model_Gaussian.pkl")
                    X = df[required_columns]
                    predictions = model.predict(X)
                    probabilities = model.predict_proba(X)

                    df['Predicted Species'] = [species_labels.get(p, 'Unknown') for p in predictions]
                    df['Confidence'] = [f"{max(prob)*100:.1f}%" for prob in probabilities]

                    if 'species' in df.columns:
                        if df['species'].dtype == 'object':
                            true_labels = df['species'].map({v: k for k, v in species_labels.items()})
                        else:
                            true_labels = df['species'].astype(int)

                        df['Correct'] = true_labels == predictions
                        accuracy = df['Correct'].mean() * 100
                        correct_count = df['Correct'].sum()
                        st.success(f"Batch prediction completed! Accuracy: **{accuracy:.2f}%** ({correct_count} / {len(df)} correct)")

                        # ─── Confusion Matrix ────────────────────────────────
                        cm = confusion_matrix(true_labels, predictions)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                    xticklabels=list(species_labels.values()),
                                    yticklabels=list(species_labels.values()), ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title('Confusion Matrix')
                        st.pyplot(fig)

                    else:
                        st.success("Batch prediction completed! (No true labels found to calculate accuracy)")

                    st.dataframe(df.head(10))

                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download predictions as CSV",
                        data=csv_data,
                        file_name="iris_predictions_with_results.csv",
                        mime="text/csv"
                    )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

