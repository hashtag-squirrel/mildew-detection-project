import streamlit as st


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        "**General Information**\n"
        "* Powdery mildew is a fungal disease that affects a wide range of "
        "plants. Powdery mildew diseases are caused by many different species "
        "of ascomycete fungi in the order Erysiphales.Powdery mildew is one of"
        "of the easier plant diseases to identify, as its symptoms are quite "
        "distinctive.\n"
        "* The fictitious company Farmy & Foods has large cherry plantations "
        "and have lately encountered powdery mildew on their cherry trees. "
        "Currently, the process to identify the affected trees is manual and "
        "quite time-consuming. The company wants to make use of an ML model "
        "to predict whether or not a tree is affected on images of leaves "
        "their staff supplies.\n"

        "**Project Dataset**\n"
        "* The available dataset contains 4208 images of cherry leaves "
        "supplied by Farmy & Foods, taken from their crops.\n"
        )

    st.write(
        "* For additional information, please visit and **read** the "
        "[Project README file](https://github.com/hashtag-squirrel/mildew-detection-project/blob/main/README.md).")  # noqa

    st.success(
        "The project has 2 business requirements:\n"
        "* 1 - The client is interested in conducting a study to visually "
        "differentiate a healthy cherry leaf from one with powdery mildew.\n"
        "* 2 - The client is interested in predicting if a cherry leaf is "
        "healthy or contains powdery mildew.\n"
        )
