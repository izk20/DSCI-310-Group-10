# Investment Outcome Predictor

## DSCI 310 Group 10 Project

### Authors: 
1.Nikko Dumrique  
2.Mahdi Heydar  
3.Ahmed Rizk  
4.Harry Zhang  

### Summary:
  The KNN-Classification model was applied to 2018 Canadian census data to predict whether an individual made money on their investments (true class) or broke even or lost money (false class) using their family size and whether they are the major income earner in their family as features.
  
  All investments contain a risk, so the rationale for this analysis was to gain insight into whether the pressures of being the main income earner in a family and having a larger family size have influence on predicting someones investment outcomes. This could be used to further analyze the risks associated within the specific investments for further analysis.
  
  The KNN-model was tuned for the nearest neighbors hyperparameter. A value of 26 was used yielding approximately 57% accuracy. Therefore, the model did not perform much better compared to a dummy classifier. The KNN-classification model was not able to distinguish between individuals in the same family size group unlike the pattern found in the actual data.
  
  It is important to build other models such as a support vector machine model (SVM) or carry out feature engineering or add other features that may serve as better predictors to gain more solid results regarding the original research question of how family size and whether an individual is the major income earner in their family can be used ot predict investment outcomes.
  
### Dataset Description

  Canadian Income Survey (CIS) is a cross-sectional survey sponsored both by the Government of Canada and Statistics Canada. The purpose of this survey is to collect information from all citizens and households within Canada, however, around 2% of the residing on the reserve, aboriginal settlements or extremely remote areas with extremely small populations is not included in this survey. This survey collects the data from several different characteristics including labour market activity, school attendance, disability, support payments, child care expenses, inter-household transfers. This dataset also combines some information from the Labour Force Survey(LFS), such as the information about the education level ogeography information. This data set is available to all of the organizations, different levels of the government, and individuals. Different governments could use this dataset to make economic policies to all canadians.

### Usage

Docker info

### Dependencies

* Python 3.9.7 with packages:
  - pandas=1.3.4 
  - scikit-learn=0.24.2 
  - seaborn=0.11.2 
  - matplotlib=3.4.3 
