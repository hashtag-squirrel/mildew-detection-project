# Mildew Detection in Cherry Leaves

This repo contains a machine learning project aiming to solve a fictional business case where a cherry plantation is affected by a powdery mildew infestation. 

The machine learning model is trained on healthy and diseased images of cherry leaves and the main goal is to predict wether any given new image of a cherry leaf is healthy or diseased. 

Here is a link to the dashboard:

## Table of Contents - [Mildew Detection in Cherry Leaves](#mildew-detection-in-cherry-leaves)
- [Mildew Detection in Cherry Leaves](#mildew-detection-in-cherry-leaves)
  - [Table of Contents - Mildew Detection in Cherry Leaves](#table-of-contents---mildew-detection-in-cherry-leaves)
  - [CRISP-DM](#crisp-dm)
  - [Business Understanding](#business-understanding)
  - [Dataset Content](#dataset-content)
  - [Hypothesis and how to validate?](#hypothesis-and-how-to-validate)
  - [The rationale to map the business requirements to the Data Visualisations and ML tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
  - [ML Business Case](#ml-business-case)
  - [Dashboard Design](#dashboard-design)
  - [Unfixed Bugs](#unfixed-bugs)
  - [Deployment](#deployment)
    - [Heroku](#heroku)
  - [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
  - [Credits](#credits)
    - [Code](#code)
    - [Content](#content)
    - [Media](#media)
    - [Content](#content-1)
    - [Media](#media-1)
  - [Acknowledgements (optional)](#acknowledgements-optional)

## CRISP-DM 

Doing the project, I made use of the CRISP-DM (Cross Industry Standard Process for Data Mining) workflow commonly used for data science projects. 

The workflow consists of six phases.

1. Business Understanding
   1. What are the business requirements?
   2. What problem does the business hope to solve with the model?
   3. What kind of output does the business require (Dashboard or API)?
2. Data Understanding
   1. What kind of data is available?
   2. Is there sufficient data or do we need to collect more?
3. Data Preparation
   1. Do we need to clean the data?
   2. Do we need to prepare the data in other ways?
4. Modelling
   1. What kind of ML model best suits the task?
   2. Are there other ML models that need to be tried out?
5. Evaluation
   1. If more than one model was trained: which model performs the best?
   2. Does the model fit the requirements?
   3. Can the model be improved?
6. Deployment
   1. Deploy the model in the way that was decided in the Business Understanding phase. 

Each phase may be repeated if new insight is gathered, meaning the phases are not part a straight line, but rather a cycle.  

## Business Understanding

In order to gain proper understanding of the business case presented, the case was assessed by answering 10 questions.

1. What are the business requirements?

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute.  The company has thousands of cherry trees, located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

* 1 - The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
* 2 - The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

2. Is there any business requirement that can be answered with conventional data analysis?

Yes, conventional data analysis can be used to conduct a study to visually differentiate a healthy leaf from one that contains powdery mildew.

3. Does the client need a dashboard or an API endpoint?

The client requires a dashboard which their employees can access on the plantation to assess the status of any given tree. 

4. What does the client consider as a successful project outcome?

The project is deemed successful when both business requirements are met: 

* A study showing the visual differences between healthy and diseased cherry leaves
* The capability to predict whether a leaf is healthy or contains powdery mildew

5. Can you break down the project into Epics and User Stories?

The project can be broken down into the following Epics:

* Information gathering and data collection.
* Data visualization, cleaning, and preparation.
* Model training, optimization and validation.
* Dashboard planning, designing, and development.
* Dashboard deployment and release.

As you can see, these are almost equivalent to 5 of the 6 phases of the CRISP-DM workflow. 

6. Are there any ethical or privacy concerns?

The client provided the data under an NDA (non-disclosure agreement), therefore the data should only be shared with professionals that are officially involved in the project.

7. Does the data suggest a particular model?

The data suggests a binary classifier, indicating whether a particular cherry leaf is healthy or contains powdery mildew.

8. What are the model's inputs and intended outputs?

The input is an image of a cherry leaf and the output is a prediction of whether the leaf is healthy or contains powdery mildew. 

9. What are the criteria for the performance goal of the predictions?

We agreed with the client a degree of 97% accuracy.

10. How will the client benefit?

The client will be able to tell if a tree is diseased in a much shorter time span that by manually inspecting each tree and can therefore make sure not to supply a product of compromised quality. 

## Dataset Content
* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Hypothesis and how to validate?
1. Hypothesis: We hypothesize that infected leaves have clear white marks with which they can be distinguished from healthy leaves
    * This can be validated by doing an average image study

2. Hypothesis: We hypothesize that resizing the images to 100 x 100 pixels does not affect the model performance
   * This can be validated by training and fitting two models, once using the original image size and once using the resized images and then comparing the performance metrics of both models

3. Hypothesis: We hypothesize that training and fitting a model on grayscale images does not affect the model performance
   * This can be validated by training and fitting two models, once using the original RGB images and once using grayscale images and then comparing the performance metrics of both models

4. Hypothesis: We hypothesize that a model using the softmax activation function for the output layer performs better than the sigmoid activation function, which is usually chosen for binary classification
   * This can be validated by training and fitting two models, once using the original sigmoig and once using the softmax activation function for the output layer and then comparing the performance metrics of both models

## The rationale to map the business requirements to the Data Visualisations and ML tasks
* List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.


## ML Business Case
* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports.
* Finally, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project, you were confident you would use a given plot to display an insight, but later, you chose another plot type).


## Unfixed Bugs
* You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file. 


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries used in the project and provide an example(s) of how you used these libraries.


## Credits 

### Code

### Content

* [What is CRISP-DM](https://www.datascience-pm.com/crisp-dm-2/) - Further reading on CRISP-DM, referenced to understand the concept and to write the Readme section

### Media



* In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.



## Acknowledgements (optional)
* Thank the people that provided support throughout this project.
