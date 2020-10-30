# Ethnicity, Gender and Age prediction with Deep Learning
### Tensorflow, Multi Task Learning and Transfer Learning


## Description
The EGA model identifies ethnicity, gender and age of people from images of cropped faces. It aims to be used on top of face detection and alignment tools such as [OpenCV, Dlib](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) or [MTCNN](https://towardsdatascience.com/face-detection-using-mtcnn-a-guide-for-face-extraction-with-a-focus-on-speed-c6d59f82d49) in a 2 stage pipeline for "in the wild" applications. 

# Content
- [Model Prediction Examples](##Model-Prediction-Examples)
- [Motivation](##Motivation)
- [Model use case](##Model-use-case)
- [Model Performance Preview](##Model-Performance-Preview)
   - [Gender model](###Gender-model)
   - [Age model](###Age-model)
   - [Ethnicity model](###Ethnicity-model)
- [Why Multi Task Learning?](##Why-Multi-Task-Learning?)
- [Why Transfer Learning?](##Why-Transfer-Learning?)
- [Interesting Facts](##Interesting-Facts)
- [Future Ideas and Improvements](##Future-Ideas-and-Improvements)
- [Resources and Inspiration](##Resources-and-Inspiration)
- [Model architecture](##Model-architecture)


## Model Prediction Examples

<table style="border-collapse: collapse">
   <tr> 
      <td style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_1.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_2.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_3.png"  width="300" height="300"> 
      </td>
   </tr>
   <tr>
      <td style="min-width: 300px; padding: 0px;">
         <img src="model_showcase/showcase_4.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_5.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_6.png"  width="300" height="300"> 
      </td>
   </tr>
   <tr>
      <td style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_7.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_8.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_9.png"  width="300" height="300"> 
      </td>
   </tr>
   <tr>
      <td style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_10.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_11.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_12.png"  width="300" height="300"> 
      </td>
   </tr>
   <tr>
      <td style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_13.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_14.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_15.png"  width="300" height="300"> 
      </td>
   </tr>
   <tr>
      <td style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_16.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_17.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_18.png"  width="300" height="300"> 
      </td>
   </tr>
   <tr>
      <td style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_19.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_20.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_21.png"  width="300" height="300"> 
      </td>
   </tr>
   <tr>
      <td style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_22.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_23.png"  width="300" height="300"> 
      </td>
      <td  style="min-width: 300px; padding:0px">
         <img src="model_showcase/showcase_24.png"  width="300" height="300"> 
      </td>
   </tr>
</table>

see more in `/model_showcase`


## Motivation
From the perspective of someone interested in society and culture, it would be incredibly useful to have a tool that easily outputs live demographic trends and changes in a population (social network, city, country, group, etc).


## Model use case

An age, gender and ethnicity prediction model such as this one can have numerous applications, many of them arguably unethical. However, hoping that such a model would be used for good rather than for bad, most of its applications would be related to demographic analysis. For instance, given a company or social network that say uses face registration, EGA model can be used to periodically scan users' faces and output demographics. 
With careful consideration to ethics and face anonymization, EGA can also be used to uncover a city's or country's demography, migration trends and patterns by scanning video recordings of street surveillance cameras.

Note: The model was trained on cropped faces. To use the EGA “in the wild” correctly, it has to be used on top of face detection and alignment tools, as a 2 stage pipeline. Some face detection solutions are: OpenCV, Dlib or MTCNN.


## Model Performance Preview
For a detailed performance report see `ega_model_notebook.ipynb`

### Gender model

<img src="saved_models/plots/gender_model_confusion.jpeg" style="min-width: 600px;">

Although the model generally does a great job predicting gender, it still has trouble predicting some of the newborns' gender. This is especially true for newborns of Asian and Other ethnicity. However, this is understandable as most newborn are yet to develop gender specific facial features.
<img src="saved_models/plots/gender_model_violin_aFpM.jpeg" style="min-width: 1000px; min-height: 300px;">

<img src="saved_models/plots/gender_model_violin_aMpF.jpeg" style="min-width: 1000px; min-height: 300px;">

<img src="saved_models/plots/gender_model_aFpM_sampl.jpeg"  style="min-width: 1200px; min-height: 600px;">


```

              precision    recall  f1-score   support

      Female       0.90      0.93      0.92      2261
        Male       0.94      0.90      0.92      2481

    accuracy                           0.92      4742
   macro avg       0.92      0.92      0.92      4742
weighted avg       0.92      0.92      0.92      4742
```



### Age model

For some samples the model can show a high discrepancy between actual and predicted age. However, this is mostly true for ages over 60, which are under represented in the training set.

<img src="saved_models/plots/age_model_error_dist.jpeg"  style="min-width:1000px; min-height:300px;">


The predicted age distribution hugs the actual age distribution quite well. There is however an overall underestimation for ages between 25-40 and less so for ages over 60.

<img src="saved_models/plots/age_model_AvsP_dist.jpeg"  style="min-width:1000px; min-height:300px;">

As faces get older, model underestimation increases.
<img src="saved_models/plots/age_model_errAge_reg.jpeg"  style="min-width:1000px; min-height:300px;">

While there is no significant age misestimation across ethnicity and gender, there is however age misestimation within age categories. The older the faces the more the model underestimates age.

<img src="saved_models/plots/age_model_error_ethnicity_violin.jpeg"  style="min-width:1000px; min-height:300px;">
<img src="saved_models/plots/age_model_error_age_violin.jpeg"  style="min-width:1000px; min-height:300px;">



<img src="saved_models/plots/age_categ_model_confusion.jpeg" style="min-width: 600px;">

```
                precision    recall  f1-score   support

           20s       0.64      0.82      0.72      1470
           30s       0.42      0.38      0.40       907
           40s       0.29      0.33      0.31       449
           50s       0.39      0.34      0.36       459
child below 10       0.93      0.81      0.87       613
       over 60       0.92      0.47      0.63       537
      teenager       0.49      0.49      0.49       307

      accuracy                           0.58      4742
     macro avg       0.58      0.52      0.54      4742
  weighted avg       0.60      0.58      0.58      4742
```

<img src="saved_models/plots/age_model_PvsA_age_dist.jpeg" style="min-width: 1000px; min-height: 500px;">


### Ethnicity model

<img src="saved_models/plots/ethnicity_model_confusion.jpeg" style="min-width: 700px;">

Note how the model has not labeled any faces as "Other". This is because it has found close similarities in facial features to other defined categories (Asian, Black, Indian and White) from the dataset. Hence the model has distributed the Other group accordingly.

Lots of the "Other" faces were predicted as "White". Given that no other choice is available than the defined ethnicities, this would be in fact the best label for most of the faces below.

<img src="saved_models/plots/ethnicity_model_aOpW.jpeg"  style="min-width: 1200px; min-height: 600px;">


```
              precision    recall  f1-score   support

       Asian       0.86      0.87      0.87       686
       Black       0.84      0.88      0.86       905
      Indian       0.72      0.75      0.74       796
       Other       0.00      0.00      0.00       339
       White       0.82      0.92      0.87      2016

    accuracy                           0.81      4742
   macro avg       0.65      0.69      0.67      4742
weighted avg       0.75      0.81      0.78      4742
```


## Why Multi Task Learning?
Well, why learn one task at a time when you can learn multiple tasks at a time?
Tasks such as ethnicity, age and gender prediction, share lots of facial features. As a matter of fact, ethnicity, age and gender go so much hand in hand, there’s a special term for that, [demography](https://en.wikipedia.org/wiki/Demography).
Hence it makes sense to bring the 3 models together as one. In addition, Multi Task Learning comes with a regularization bonus, preventing overfitting or at least overfitting easily. Nevertheless, Multi Task Learning increases performance. Having one model for 3 outputs is much faster than having 3 models.


## Why Transfer Learning?
More data is always better, however in a data scarcity scenario Transfer Learning substitutes the lack of data with experience from a transferee model.
The [UTK Face](https://susanqq.github.io/UTKFace/) dataset used for this model does not have enough data to match EGA model complexity. Therefore I have used pretrained weights from the VGG Face model on top of which I have built the 3 models with Dense layers. The VGG Face is an even deeper model of DeepFace trained on millions of images and originally used for face detection.
See model architecture.

## Interesting Facts 
While trying out multiple image augmentation parameters in [ImageGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator), I have noticed that when setting `[sheer_range=45]`, the model misclassified all Asian ethnicity labels as White. In the same time, when `[channel_shift_range=150]` or `[brightness_range=(0.1,0.9)]` the model misclassified all Indian and Black ethnicity labels as White.


## Future Ideas and Improvements
From the 3 EGA model predictions, age is the least accurate and toughest to predict. Perhaps a better approach for age prediction would be to predict a range, rather than the actual age. Such model could be done using a [Tensorflow Probabilistic Layer](https://blog.tensorflow.org/2019/03/regression-with-probabilistic-layers-in.html) as the last layer in the age model. This way, the model will output a mean and standard deviation for each prediction. Using the 2 variables, a 95% confidence interval can be calculated as a final prediction. I have actually tried this option, however in most cases the confidence interval was too great, 30-50 years. Perhaps a different approach in the model architecture might work out better. 


## Resources and Inspiration
* [Deep Age](https://www.researchgate.net/publication/335065216_DeepAge_Deep_Learning_of_face-based_age_estimation)
* [Deep Face](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)
* [VGG-Face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) 
* [Deep Face Recognition: Survey](https://arxiv.org/pdf/1804.06655.pdf)


## Model architecture
<img src="saved_models/plots/ega_model.png"  style="min-width: 700px;">
