# Arabic-Dialect-Identification

Araibic Dialect Identification is a NLP Problem to classify different 18 arabic dialect, 'IQ', 'LY', 'QA', 'PL', 'SY', 'TN', 'JO', 'MA', 'SA', 'YE', 'DZ',
       'EG', 'LB', 'KW', 'OM', 'SD', 'AE' and 'BH'.
       
The dataset used is **QADI** dataset, that initially consists of 2 columns id and dialect.
       
Abstractly, the Project is divided into **4 stages**.

 # Stage 1 (Data Fetching)
 
 This stage mainly fetch the tweets from an API using the given ids.
 
 # Stage 2 (Data Pre-processing and Analysis)
  
 in this stage, a different data sceience techniques is used to make  **Analysis, Visualizion, Encoding, Feature extractinon** on the dataset.
 
 # Stage 3 (Model Training)
 
 **4 different models** are trained on the preprocessed dataset in **2 different approaches**
 
The **1st approach** is to train on the **Features Extracted** on pre-processing stage.

3 models are included in this approach: **Logisitic Regression** , **Naive Bayes** , **Deep Learning Model**

the **2nd approach** is to train on encoded text it self, in other words let the model extract the features his self.

Reccurrent Neural Network is the one used in this approach

**Accuracy** is measureb based on Categorical accuracy and F1-Score

in 1 approach the DL could achieve about 38% categorical accuracy, wheras the Naive Bayes and logisitic regressing could achieve about 37%

However 2nd approach performs better with accuracy about 50. (For more details check the training notebook)

# Stage 4

Deploying the model using Flask as a back end. HTML and CSS as front end.

![Alt text](dialect_identification_app.png "dialect identification  app")









 

 
  
 

 

