# Code adapted from Code Institute's Malaria walkthrough project

import streamlit as st


def page_project_hypotheses_body():
    st.write("### Project Hypotheses and Validation")

    st.info(
        "## Hypothesis 1\n"
        "* We hypothesize that infected leaves have clear white marks with "
        "which they can be distinguished from healthy leaves.\n\n"
        "### Validation Method\n\n"
        "* This can be validated by doing an average image study.\n"
        "### Outcome\n\n"
        "* **Invalidated**\n"
        "* The image study we have done on average and variability images of "
        "healthy and powdery-mildew-affected leaves has shown no significant "
        "difference in the average images.\n"
        "* The average image of leaves affected by powdery mildew shows a "
        "slightly lighter coloring on the leaf, or even some white-ish marks, "
        "but they are not significant."
    )

    st.image(
        "outputs/v1/avg_diff.png",
        caption='Average images of cherry leaves'
    )

    st.write("---")

    st.info(
        "## Hypothesis 2\n"
        "* We hypothesize that resizing the images to 100 x 100 pixels does "
        "not affect the model performance.\n\n"
        "### Validation Method\n\n"
        "* This can be validated by training and fitting two models, once "
        "using the original image size and once using the resized images and "
        "then comparing the performance metrics of both models.\n"
        "### Outcome\n\n"
        "* **Invalidated**\n"
        "* While the model was trained in a fraction of the time than the "
        "original model at 7 minutes for all 25 epochs and the training and "
        "loss plots do not look bad at first glance, we see that clearly, the "
        "model does not perform as well as the model with the original size "
        "of 256 x 256 pixels when we do the evaluation on the test set. "
        "The accuracy only comes up to 93.6%.\n"
        "* We can also see that on prediction of random samples from the test "
        "set, the probability of the predictions drops as low as 53%."
    )

    st.write(
        'Accuracy plots of base model vs model using resized images'
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(
            "outputs/v1/model_training_acc.png",
            caption='Base model'
        )
    with col2:
        st.image(
            "outputs/v2/model_training_acc-resized.png",
            caption='Model using images of 100 x 100 pixels'
        )

    st.write(
        'Loss plots of base model vs model using resized images'
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(
            "outputs/v1/model_training_losses.png",
            caption='Base model'
        )
    with col2:
        st.image(
            "outputs/v2/model_training_losses-resized.png",
            caption='Model using images of 100 x 100 pixels'
        )

    st.warning(
        "We can see the accuracy and loss plots of the resized model having "
        "the correct shape, but the curves of the validation set and training "
        "set have a bigger difference than the ones on the base model.\n"
        "The difference for the loss plots on the resized model also seems to "
        "get bigger towards later epochs."
    )

    st.write("---")

    st.info(
        "## Hypothesis 3\n"
        "* We hypothesize that training and fitting a model on grayscale "
        "images does not affect the model performance.\n\n"
        "### Validation Method\n\n"
        "* This can be validated by training and fitting two models, once "
        "using the original RGB images and once using grayscale images and "
        "then comparing the performance metrics of both models.\n"
        "### Outcome\n\n"
        "* **Invalidated**\n"
        "* The model took about half the time to train compared to the base "
        "model. It took all 25 epochs and the loss and accuracy plots look "
        "like it trained well. However, when evaluating on the test set, the "
        "accuracy is only at 94.55%, which is not meeting the business "
        "requirement of 97% accuracy.\n"
        "* When trying the predictions on random images from the test set, "
        "the prediction probability also ranges from around 60% up to 99%, "
        "depending on the image.\n"
    )

    st.write(
        'Accuracy plots of base model vs model using grayscale images'
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(
            "outputs/v1/model_training_acc.png",
            caption='Base model'
        )
    with col2:
        st.image(
            "outputs/v3/model_training_acc-grayscale.png",
            caption='Model using grayscale images'
        )

    st.write(
        'Loss plots of base model vs model using grayscale images'
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(
            "outputs/v1/model_training_losses.png",
            caption='Base model'
        )
    with col2:
        st.image(
            "outputs/v3/model_training_losses-grayscale.png",
            caption='Model using grayscale images'
        )

    st.warning(
        "We can see the accuracy and loss plots of the grayscale model "
        "actually look like the model trained more consistently, since there "
        "are fewer spikes and a more consistent progression, however, the "
        "overall accuracy in the evaluation was not high enough to justify "
        "a switch to a model using grayscale images."
    )

    st.write("---")

    st.info(
        "## Hypothesis 4\n"
        "* We hypothesize that a model using the softmax activation function "
        "for the output layer performs better than the sigmoid activation "
        "function, which is usually chosen for binary classification.\n\n"
        "### Validation Method\n\n"
        "* This can be validated by training and fitting two models, once "
        "using the original sigmoig and once using the softmax activation "
        "function for the output layer and then comparing the performance "
        "metrics of both models.\n"
        "### Outcome\n\n"
        "* **Validated**\n"
        "* In comparison to the base model, the model using the Softmax "
        "activation function needed 5 epochs less to train and finished "
        "training after about 14 minutes. The loss and accuracy plots look a "
        "bit better than those of the base model. The accuracy of the model "
        "lands at 98.1%, which is clearly fitting our business requirement "
        "and is exactly the same as the base model.\n"
        "* When trying the predictions on random images from the test set, "
        "the prediction probability seems to be consistently high.\n"
        "* While the evaluation accuracy is the same on both models, the "
        "model using the Softmax activation function did train quicker and "
        "has fewer spikes in both loss and accuracy plots, and thus is deemed "
        "the better model."
    )

    st.write(
        'Accuracy plots of base model vs model using Softmax activation '
        'function'
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(
            "outputs/v1/model_training_acc.png",
            caption='Base model'
        )
    with col2:
        st.image(
            "outputs/v4/model_training_acc-softmax.png",
            caption='Model using Softmax activation function'
        )

    st.write(
        'Loss plots of base model vs model using Softmax activation function'
    )

    col1, col2 = st.beta_columns(2)
    with col1:
        st.image(
            "outputs/v1/model_training_losses.png",
            caption='Base model'
        )
    with col2:
        st.image(
            "outputs/v4/model_training_losses-softmax.png",
            caption='Model using Softmax activation function'
        )

    st.warning(
        "We can see the loss and accuracy curves of both models look quite "
        "similar overall, however, the model using the Softmax activation "
        "function has smaller spikes in the validation set curves and is "
        "therefore deemed slightly better than the base model."
    )
