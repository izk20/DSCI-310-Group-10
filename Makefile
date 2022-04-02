
# This driver script completes the graphical and textual analysis 
# for the Analysis of Investment Outcome Predictor Report.
# author: Nikko Dumrique, Mahdi Heydar, Ahmed Rizk, Harry Zhang
# date: 2022-04-01

#example usage:
# make all


all : data/raw/raw_data.csv data/processed/preprocessing results/hyperparamter_opti results/simple_linear_regression result/eda_figures result/knn_classification result/final_plot1 result/final_plot2 result/final_plot3 result/final_plot4 

# all: results/final_model.rds results/accuracy_vs_k.png results/predictor_distributions_across_class.png results/final_model_quality.rds doc/breast_cancer_predict_report.md

# download data
data/raw/raw_data.csv : 
	Rscript src/download_dataset_script.r --url="https://onedrive.live.com/download?cid=3186CCDB0C6495E0&resid=3186CCDB0C6495E0%2157273&authkey=AK4_vAlM4AFx7_M" --out_dir="data/raw/raw_data.csv"
# pre-process data
data/processed/preprocessing: src/read_process_script.py
	python3 src/read_process_script.py --read_path="data/raw/raw_data.csv" --processed_path="data/processed/"

 # exploratory data analysis - Histograms
result/eda_figures: src/eda.py
	python3 src/eda.py --processed_data_path="data/processed/reduced_data.pkl" --output_file="result/"

 # Hyperparameter tuning (here, find K for k-nn using 30 fold cv with Cohen's Kappa)
results/hyperparamter_opti: src/hyperparameter_optimization_new.py
	python3 src/hyperparameter_optimization_new.py --xtrainpath="data/processed/X_train.pkl" --ytrainpath="data/processed/Y_train.pkl" --variables="EFSIZE,EFMJIE" --out_dir="result/"

# Single linear regression
results/simple_linear_regression: src/single_linear_regression.py
	python3 src/single_linear_regression.py --xtrainpath="data/processed/X_train.pkl" --ytrainpath="data/processed/Y_train.pkl" --preprocessorpath="result/preprocessor" --bestalpha="result/best_alpha" --path="result/"

# knn classification

result/knn_classification: src/knn_classification_script.py
	python3 src/knn_classification_script.py --processed="data/processed/processed.pkl" --out_dir="result/"

# final plot1
result/final_plot1: src/plot-stacked-chart.py
	python3 src/plot-stacked-chart.py --pipeline_path="data/processed/pipe_final" --test_2_p="data/processed/test_2" --X_test_2_p="data/processed/X_test_2" --plot_number="1" --output_file="result/"

# final plot2
result/final_plot2: src/plot-stacked-chart.py
	python3 src/plot-stacked-chart.py --pipeline_path="data/processed/pipe_final" --test_2_p="data/processed/test_2" --X_test_2_p="data/processed/X_test_2" --plot_number="2" --output_file="result/"

# final plot3
result/final_plot3: src/plot-stacked-chart.py
	python3 src/plot-stacked-chart.py --pipeline_path="data/processed/pipe_final" --test_2_p="data/processed/test_2" --X_test_2_p="data/processed/X_test_2" --plot_number="3" --output_file="result/"

# final plot4
result/final_plot4: src/plot-stacked-chart.py
	python3 src/plot-stacked-chart.py --pipeline_path="data/processed/pipe_final" --test_2_p="data/processed/test_2" --X_test_2_p="data/processed/X_test_2" --plot_number="4" --output_file="result/"

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
	rm -rf result/*
	rm -rf doc/Analysis_of_Investment_Outcome_report.md doc/Analysis_of_Investment_Outcome_report.html
