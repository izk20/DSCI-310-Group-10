# This driver script completes the graphical and textual analysis 
# for the Analysis of Investment Outcome Predictor Report.
# author: Nikko Dumrique, Mahdi Heydar, Ahmed Rizk, Harry Zhang
# date: 2022-04-01

#example usage:
# make all

all : doc/Analysis_of_Investment_Outcome_Predictor_report.md

!!!TODO
all: results/final_model.rds results/accuracy_vs_k.png results/predictor_distributions_across_class.png results/final_model_quality.rds doc/breast_cancer_predict_report.md

# download data
data/raw/<!!!>: src/<!!!>.py
	python src/<!!!>.py 

# pre-process data 
data/processed/<!!!> data/processed/<!!!> : src/<!!!>.py

# exploratory data analysis - Histograms
results/predictor_distributions_across_class.png: src/eda_wisc.r data/processed/training.feather
	Rscript src/eda_wisc.r --train=data/processed/training.feather --out_dir=results

# Hyperparameter tuning (here, find K for k-nn using 30 fold cv with Cohen's Kappa)
results/<!!!>.png results/<!!!>.png : src/<!!!>.py data/processed/
	src/<!!!>.py data/processed/

# Visualized results
results/<!!!>.png: src/<!!!>.py data/processed/
	src/<!!!>.py data/processed/
	src/<!!!>.py data/processed/
	src/<!!!>.py data/processed/

# render report
doc/Analysis_of_Investment_Outcome_report.md : doc/breast_cancer_predict_report.Rmd doc/breast_cancer_refs.bib
	Rscript -e "rmarkdown::render('doc/breast_cancer_predict_report.Rmd')"

clean: 
	rm -rf data
	rm -rf results
	rm -rf doc/Analysis_of_Investment_Outcome_report.md doc/Analysis_of_Investment_Outcome_report.html
			