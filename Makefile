
# This driver script completes the graphical and textual analysis 
# for the Analysis of Investment Outcome Predictor Report.
# author: Nikko Dumrique, Mahdi Heydar, Ahmed Rizk, Harry Zhang
# date: 2022-04-01

#example usage:
# make all


all : data/raw/raw_data.csv data/processed/preprocessing results/hyperparamter_opti results/simple_linear_regression

# all: results/final_model.rds results/accuracy_vs_k.png results/predictor_distributions_across_class.png results/final_model_quality.rds doc/breast_cancer_predict_report.md

# download data
data/raw/raw_data.csv : 
	Rscript src/download_dataset_script.r --url="https://onedrive.live.com/download?cid=3186CCDB0C6495E0&resid=3186CCDB0C6495E0%2157273&authkey=AK4_vAlM4AFx7_M" --out_dir="data/raw/raw_data.csv"
# pre-process data
data/processed/preprocessing: src/read_process_script.py
	Python src/read_process_script.py --read_path="data/raw/raw_data.csv" --processed_path="data/processed/" --train_path="data/processed/" --test_path="data/processed/"

# # exploratory data analysis - Histograms
# results/predictor_distributions_across_class.png : src/eda_wisc.r data/processed/training.feather
# 	Rscript src/eda_wisc.r --train=data/processed/training.feather --out_dir=results

 # Hyperparameter tuning (here, find K for k-nn using 30 fold cv with Cohen's Kappa)
results/hyperparamter_opti: src/hyperparameter_optimization_new.py
	Python src/hyperparameter_optimization_new.py --xtrainpath="data/processed/X_train.pkl" --ytrainpath="data/processed/Y_train.pkl" --variables="EFSIZE,EFMJIE" --out_dir="result/"

results/simple_linear_regression: src/single_linear_regression.py
	python src/single_linear_regression.py --xtrainpath="data/processed/X_train.pkl" --ytrainpath="data/processed/Y_train.pkl" --preprocessorpath="result/preprocessor" --bestalpha="result/best_alpha" --path="result/"


# # Visualized results
# results/<!!!>.png : src/<!!!>.py data/processed/
# 	src/<!!!>.py data/processed/
# 	src/<!!!>.py data/processed/
# 	src/<!!!>.py data/processed/

# # render report
# doc/Analysis_of_Investment_Outcome_report.md : doc/breast_cancer_predict_report.Rmd doc/breast_cancer_refs.bib
# 	Rscript -e "rmarkdown::render('doc/breast_cancer_predict_report.Rmd')"


clean: 
	rm -rf data/raw/*
	rm -rf data/processed/*
	rm -rf results
	rm -rf doc/Analysis_of_Investment_Outcome_report.md doc/Analysis_of_Investment_Outcome_report.html

