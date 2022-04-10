# Investment Outcome Predictor

## DSCI 310 Group 10 Project

### Authors: 
1.Nikko Dumrique  (nkoda)  
2.Mahdi Heydar (mahdiheydar)  
3.Ahmed Rizk  (izk20)  
4.Harry Zhang  (harrysyz99)  

### Summary:
  The KNN-Classification model was applied to 2017 Canadian census data to predict whether an individual made money on their investments (true class) or broke even or lost money (false class) using their family size, and whether they are the major income earner in their family as features.
  
  All investments contain a risk, so the rationale for this analysis was to gain insight into whether the pressures of being the main income earner in a family and having a larger family size have influence on predicting someones investment outcomes. This could be used to further analyze the risks associated within the specific investments for further analysis.
  
  The KNN-model was tuned for the nearest neighbors hyperparameter. A value of 26 was used yielding approximately 57% accuracy. Therefore, the model did not perform much better compared to a dummy classifier. The KNN-classification model was not able to distinguish between individuals in the same family size group unlike the pattern found in the actual data.
  
  It is important to build other models such as a support vector machine model (SVM), or carry out feature engineering or add other features that may serve as better predictors to gain more solid results. This will enhance the investigation of the original research question of how family size, and whether an individual is the major income earner in their family, can be used to predict investment outcomes.
  
### Dataset Description

  Canadian Income Survey (CIS) is a cross-sectional survey sponsored both by the Government of Canada and Statistics Canada. The purpose of this survey is to collect information from all citizens and households within Canada, however, around 2% of the residing on the reserve, aboriginal settlements or extremely remote areas with extremely small populations is not included in this survey. This survey collects the data from several different characteristics including labour market activity, school attendance, disability, support payments, child care expenses, inter-household transfers. This dataset also combines some information from the Labour Force Survey(LFS), such as the information about the education level and geography information. This data set is available to all of the organizations, different levels of the government, and individuals. Different governments could use this dataset to make economic policies for all Canadians.

### Usage

This project can be replicated using Docker.

Steps:

* Install Docker if you have not already done so. This is hardware specific and is specified here: https://docs.docker.com/get-docker/
* Clone this repository to your local machine:

    -**```git clone git@github.com:DSCI-310/DSCI-310-Group-10.git```**
* Using the command line/terminal, navigate to the repository directory
* Type the following:
* Navigate to the root directory of the project
    - **```docker run -d -p 8787:8787 -v /$(pwd):/home/rstudio -e PASSWORD=pass mahdiheydar/dsci-310-10-image:v3.4.0```**
* Once this is finished running, open a browser and type localhost:8787
* Use the following information to enter the container
       **Username: rstudio**
       **Password: pass**
* After this, rstudio will open. Open a terminal and type the following commands:
   - **```make clean```**
   - **```make all```**
* Navigate to the doc folder using the directory to find the Analysis_of_Investment_Outcomes.pdf and Analysis_of_Investment_Outcomes.html


### Dependencies

* Python 3.9.7 with packages:
  - numpy==1.22.3
  - pandas==1.4.1
  - jupyterlab==3.2.9
  - docopt==0.6.2
  - pytest==7.0.1
  - scikit-learn=0.24.2 
  - seaborn==0.11.2
  - matplotlib==3.5.1
  - group10pack==0.1.6

### License
This project is licensed under the MIT License and [Creative Commons Attribution-NonCommerical-NoDerivatives 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/)
<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:1" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />
