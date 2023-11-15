# Code adapted from Code Institute's Malaria walkthrough project

import streamlit as st
import pandas as pd
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    st.image("outputs/v1/labels_distribution.png",
             caption='Labels Distribution on Train, Validation and Test Sets')

    st.write("---")

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1:
        st.image("outputs/v4/model_training_acc-softmax.png",
                 caption='Model Training Accuracy')
    with col2:
        st.image("outputs/v4/model_training_losses-softmax.png",
                 caption='Model Training Losses')

    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation("v4"),
                              index=['Loss', 'Accuracy']))
