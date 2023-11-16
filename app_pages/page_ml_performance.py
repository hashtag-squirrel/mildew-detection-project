# Code adapted from Code Institute's Malaria walkthrough project

import streamlit as st
import pandas as pd
from src.machine_learning.evaluate_clf import load_test_evaluation


def page_ml_performance_metrics():

    st.write("### Train, Validation and Test Set: Labels Frequencies")

    st.image("outputs/v1/labels_distribution.png",
             caption='Labels Distribution on Train, Validation and Test Sets')

    st.info(
        "The images provided were split into train, test and validation sets, "
        "where the train set contains 70% of all images, the test set "
        "contains 20% and the validation set contains 10%.\n"
        "The plot above visualizes the split and shows that both labels "
        "are distributed evenly across the sets."
    )

    st.write("---")

    st.write("### Model History")
    col1, col2 = st.beta_columns(2)
    with col1:
        st.image("outputs/v4/model_training_acc-softmax.png",
                 caption='Model Training Accuracy')
    with col2:
        st.image("outputs/v4/model_training_losses-softmax.png",
                 caption='Model Training Losses')

    st.info(
        "The plots above show the accuracy and losses plots for the model "
        "trained with the Softmax activation function. We can see that there "
        "are some spikes in the validation, but other than that the curves "
        "follow each other closely and show a good progression as expected."
    )

    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(load_test_evaluation("v4"),
                              index=['Loss', 'Accuracy']))

    st.info(
        "We can see that the accuracy is at 98.1%, therefore fitting the "
        "expectation of the client of a minimum accuracy of 97%.\n"
        "Therefore we can say that the ML model created has successfully "
        "answered the business requirement set by the client."
    )
