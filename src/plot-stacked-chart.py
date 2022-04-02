""" generates a normalized stacked bar chart showing proportions of 
    major earner/non major earner individuals that made money on investments 
    and proportion of individuals that lost money on investments among families 
    of different size

Usage: plot-stacked-chart.py --pipeline_path=<pipeline_path> --test_2_p=<test_2_p> --X_test_2_p=<X_test_2_p> --plot_number=<plot_number> --output_file=<output_file>

Options:
--pipeline_path=<pipeline_path>         Path to final pipeline object
--test_2_p=<test_2_p>                   Path to test_2 dataframe object
--X_test_2_p=<X_test_2_p>               Path to training set made from test_2 dataframe object
--plot_number=<plot_number>             Natural number as string in range ["1","4"] corresponding to which plot
--output_file=<output_file>             Path (including filename) of where to locally write the figures

"""
  
from docopt import docopt
import pandas as pd
import matplotlib.pyplot as plt
from analysis.inv_outcome_plot import inv_outcome_plot

opt = docopt(__doc__)

def main(pipeline_path, test_2_p, X_test_2_p, plot_number, output_file):
    pipe_final = pd.read_pickle(pipeline_path)
    test_2 = pd.read_pickle(test_2_p)
    X_test_2 = pd.read_pickle(X_test_2_p)
    
    predictions_df = pd.DataFrame(data=pipe_final.predict(X_test_2),
                             columns = ["Pred_EFINVA_Made_Money"],
                             index=test_2.index)

    true_and_pred = pd.concat([test_2, predictions_df], axis=1)
    true_and_pred.drop(columns=['EFINVA'])
    grouped_true_pred = true_and_pred.groupby(['EFSIZE','EFMJIE', 'EFINVA_Made_Money', 'Pred_EFINVA_Made_Money']).size().reset_index()
    grouped_true_pred = pd.DataFrame(grouped_true_pred)
    grouped_true_pred = grouped_true_pred.rename(columns={0:"counts"})
    

    grouped_df = grouped_true_pred

    
    title_1 = """
    The Relationship Between Family Size and 
    Investment Outcome Among Individuals
    Who are Major Earners In Their Family
    """
    ylabel_1 = """Proportion of Major Earners 
    with Investment Outcome"""

    title_2 = """
    The Relationship Between Family Size and 
    Investment Outcome Among Individuals
    Who are Non-major Earners In Their Family
    """
    ylabel_2 = """Proportion of Non-major Earners 
    with Investment Outcome"""

    title_3 ="""
    The KNN-Classification Results For Predicting
    Investment Income Outcome among Major Earners In Their Family
    """
    ylabel_3 = """Proportion of Major Earners 
    Predicted with Investment Outcome"""

    title_4 ="""
    The KNN-Classification Results For Predicting
    Investment Income Outcome among Non-major Earners In Their Family
    """
    ylabel_4 = """Proportion of Non-major Earners 
    Predicted with Investment Outcome"""
    
    if plot_number == "1":
        print("works")
        fig1 = inv_outcome_plot(grouped_df,
                                 'EFSIZE',
                                 'EFMJIE',
                                 'EFINVA_Made_Money',
                                 'counts',
                                 True,
                                 title_1,
                                 ylabel_1)
        fig1.tight_layout()
        fig1.savefig(output_file + "plot1.png")
    
    elif plot_number == "2":
        fig2 = inv_outcome_plot(grouped_df,
                                 'EFSIZE',
                                 'EFMJIE',
                                 'EFINVA_Made_Money',
                                 'counts',
                                 False,
                                 title_2,
                                 ylabel_2)
        fig2.tight_layout()
        fig2.savefig(output_file + "plot2.png")                            
    elif plot_number == "3":
        fig3 = inv_outcome_plot(grouped_df,
                                 'EFSIZE',
                                 'EFMJIE',
                                 'Pred_EFINVA_Made_Money',
                                 'counts',
                                 True,
                                 title_3,
                                 ylabel_3)
        fig3.tight_layout()
        fig3.savefig(output_file + "plot3.png")                         
    elif plot_number == "4":
        fig4 = inv_outcome_plot(grouped_df,
                                 'EFSIZE',
                                 'EFMJIE',
                                 'Pred_EFINVA_Made_Money',
                                 'counts',
                                 False,
                                 title_4,
                                 ylabel_4)
        fig4.tight_layout()
        fig4.savefig(output_file + "plot4.png")
        
    
    
if __name__ == "__main__":
    print("1")
    main(opt["--pipeline_path"], opt["--test_2_p"], opt["--X_test_2_p"], opt["--plot_number"], opt["--output_file"])

    
    
