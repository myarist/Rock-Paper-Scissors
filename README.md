<h1 align="center">Rock ✊🏼 Paper ✋🏼 Scissors ✌🏼</h1>

<p align="center">
    <img src="RPS.png" >
</p>

<p align="center">
    In this project, I explore how we can use CNN and transfer learning to build an image classifier. <br>The dataset consists of 2188 images that classified by <strong>Rock<strong>, <strong>Paper<strong>, and <strong>Scissors<strong>.
</p>

<p align="center">
   The full dataset can be downloaded <a href="https://dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip">here</a>
</p>


## File Descriptions

There are only 3 important files in this repository.
- `modelling.ipynb` is a jupyter notebook which can be run on Google Colab (with GPU for faster training. It contains step-by-step on how to create the image classifier and export the model. 
- `model_inception_weights.h5` is the trained weights of our deep learning model's layers. This is used to load the model in our web app.
- `apps.py` is the python file to deploy our web app in Streamlit.



## How to Use

To run the project locally, you can download this repo and type 

```
streamlit run apps.py
```

To view the project as a deployed online web app hosted by Heroku, you can check out with [this link](https://rps-myarist.herokuapp.com/)

![heroku gif](heroku.gif)



## Model Description

The foundational model that we use is Inception-V3 from Keras' pretrained models. However, we cut it off until 'mixed7' layer, and then add our own layers.

Read more about this model at:
- https://keras.io/api/applications/inceptionv3/
- https://arxiv.org/abs/1512.00567



## Model Evaluation

We achieved 99% accuracy on training set and 97,5% accuracy on validation set.

<details>
<summary>Classification Report</summary>

<br>

```
              precision    recall  f1-score   support

        Rock       1.00      0.98      0.99       285
       Paper       0.99      1.00      0.99       291
    Scissors       0.99      1.00      0.99       300

    accuracy                           0.99       876
   macro avg       0.99      0.99      0.99       876
weighted avg       0.99      0.99      0.99       876
```

</details>

<details>
<summary>Training and Validation Accuracy and Loss</summary>

<br>

<img src='acc.png' align="left" height="50%" />

<img src='loss.png' align="left" height="50%" />

</details>