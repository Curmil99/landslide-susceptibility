# Documentation



## 1. Overview

This project develops a Random Forest model for modelling spatial landslide susceptibility. The technical workflow consists of two main files.
In *prepare_data.py*, the spatial input data is preprocessed and merged into a dataset, which is then used to train the machine learning model.
In *landslide_workflow.ipynb*, the actual machine learning workflow is implemented. This includes further data preparation, training and optimisation of the Random Forest model, as well as its evaluation and interpretation. Finally, the trained model is applied to the raster data for the entire study area to generate a landslide susceptibility map.


## 2. Data Preparation
The data provided by Samodra et al. (2024) were used as input data. These comprise eleven raster datasets containing various landslide-controlling factors, as well as 743 landslide and 743 non-landslide points.

Before the data could be used for the machine learning model, however, they first had to be pre-processed. This step was carried out in *prepare_data.py*. First, it was checked that the raster datasets used the same coordinate reference system, raster dimensions and raster grid. This ensured that the raster values were spatially comparable.

The landslide points were labelled with a value of 1, while the non-landslide points were labelled with a value of 0. These were then merged into a single dataset. Additionally, the X and Y coordinates of each point were stored as separate variables. These coordinates are required later for forming spatial blocks during the spatial split evaluation.

The values of the eleven rasters were extracted at each point's location and stored as additional variables. NoData values from the rasters were treated as missing values. The output of *prepare_data.py* is the table *landslide_ml.csv*, which contains the class label (landslide or non-landslide), spatial coordinates, and eleven landslide-controlling factors for each study site. This table forms the basis of the subsequent machine learning workflow.


## 3. Machine Learning Workflow

### 3.1 Data loading and feature preparation
Once the prepared dataset had been loaded, the features were first grouped according to their data type. In some cases, they were also transformed. This was necessary because the dataset contained categorical and numerical features, as well as one cyclical feature.

*lithology* and *land_use* were defined as categorical features. The remaining landslide-controlling factors, with the exception of *aspect*, were treated as numerical features.

*aspect* describes the orientation of a slope and has a cyclical structure. For example, the directions 1° and 359° represent very similar directions, even though their numerical values are far apart. To correctly represent this property in the model, *aspect* was transformed into the two features *aspect_sin* and *aspect_cos*. It is only by combining both values that the respective slope direction can be uniquely described. This ensures that the similarity between neighbouring directions is preserved, even at the transition between 359° and 0°.

The variables not used as predictor features for the model were then removed. These include the class label (*landslide*), the X and Y coordinates, and *aspect*, as the information it contains is now represented by *aspect_sin* and *aspect_cos*.


### 3.2 Preprocessing pipeline

First, the dataset was split into training and test data. In doing so, 80% of the data was used for training and 20% for subsequent evaluation. A stratified split was used to ensure that the ratio of landslide and non-landslide points remained consistent in both datasets. The test dataset was not used during training or model optimisation.

As the model is trained several times in the subsequent workflow, a *scikit-learn* pipeline was created. This ensures that the same data preprocessing and model training steps are carried out in the same order during each training run.

The categorical features are processed first within the pipeline. The values for *lithology* and *land_use* represent different categories. Although these categories are encoded as numbers, there is no inherent order. They are therefore converted into individual binary features using one-hot encoding. For instance, the original *land_use* feature, which has 13 categories, is represented by 13 separate features.

Missing values in the categorical features are replaced by the most frequent category in the training data. The numerical features are passed to the model without any further transformation. The complete pipeline combines these preprocessing steps with a Random Forest classifier.



### 3.3 Baseline Random Forest

First, a Random Forest model was trained without hyperparameter optimisation to establish a baseline. This model is used as a reference against which the performance of the subsequently optimised model can be compared.

The model's performance was evaluated using the accuracy, precision, recall, F1 score, ROC-AUC and PR-AUC metrics, and the results were stored in *baseline_results*. Additionally, a confusion matrix and a classification report were generated to enable a more detailed examination of the classification of landslide and non-landslide points.

For subsequent hyperparameter optimisation, ROC-AUC was used as the key optimisation metric.

### 3.4 Hyperparameter tuning

The parameters *n_estimators*, *max_depth*, *min_samples_split*, *min_samples_leaf* and *max_features* were initially selected for hyperparameter optimisation. A broad search space with various possible values was first defined for these parameters. Then, using *RandomizedSearchCV*, 30 randomly selected combinations of these parameters were tested. Each combination was evaluated using stratified 5-fold cross-validation, with ROC-AUC used as the optimisation metric.

Based on the best combination of parameters found, a narrower search space was then defined around the best values identified by *RandomizedSearchCV*. This was investigated using *GridSearchCV*. Unlike the previous random selection, all the defined parameter combinations were tested using the same 5-fold cross-validation to determine the final parameter combination.

The model was then retrained on the training data using the selected hyperparameters and evaluated on the previously held-out test set. The same evaluation metrics as for the baseline model were used, and the results were stored in *final_results*. Additionally, a confusion matrix and a classification report were generated.


### 3.5 Random and spatial evaluation

To investigate how well the model generalises to spatially separated test data, a spatial split was performed in addition to the random split. For this purpose, the study area was divided into 5 km × 5 km spatial blocks. The X and Y coordinates, which had been stored in *prepare_data.py*, were used to assign the points to these blocks. Using *GroupShuffleSplit*, complete blocks were then assigned to either the training or test data. This ensured that points from the same spatial block could not appear in both datasets simultaneously.

The model was then trained on the spatially separated training data using the previously optimised hyperparameters before being evaluated on the corresponding test data. The same evaluation metrics as in the previous evaluation were stored in *spatial_results* and compared with the results of the random split.

As the spatial blocks contain varying numbers of samples and the results may depend on a single partition, the evaluation was repeated 30 times using different *random_state* values. For comparison, the random split was also repeated 30 times using the same optimised hyperparameters and different *random_state* values. The mean and standard deviation of the respective evaluation metrics were then calculated for both methods. This enabled a comparison of both the average model performance and its variation across different data splits.


### 3.6 Model interpretation

To interpret the final model, Permutation Feature Importance (PFI) was first calculated using the test dataset from the random split. For each feature, it was examined how much the model's performance changed when its values were randomly permuted. As with the hyperparameter optimisation, ROC-AUC was used as the evaluation metric. The permutation was repeated 30 times for each feature. Subsequently, the mean feature importance and its standard deviation were visualised.

Additionally, Partial Dependence Plots (PDPs) were generated for the four selected continuous features: *slope*, *spi*, *elevation* and *twi*. These show how the model’s prediction changes on average across different values of a feature while averaging over the remaining features. This makes it possible to examine not only the general importance of a feature, but also the direction and shape of its relationship with predicted landslide susceptibility.

### 3.7 Landslide susceptibility mapping

At the end of the workflow, a landslide susceptibility map was produced for the study area. For this purpose, the previously optimised model was retrained using all available point data, since no separate test data were required for the final spatial prediction.

The eleven raster files were then loaded and checked to determine which raster cells had valid values for all required features. Pixels for which at least one raster contained a 'NoData' value were excluded from the prediction. The spatial distribution of the remaining valid pixels was first visualised and checked.

The same feature transformations used during model training were then applied to all valid raster cells. Finally, the trained model was applied to determine the predicted landslide susceptibility for each raster cell. Due to the large number of raster cells, the prediction was carried out in several chunks. The calculated susceptibility values were then reassigned to their original spatial positions and saved as the GeoTIFF file *landslide_susceptibility.tif*.



