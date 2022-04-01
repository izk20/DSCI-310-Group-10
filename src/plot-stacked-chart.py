""" generates a normalized stacked bar chart showing proportions of 
    major earner/non major earner individuals that made money on investments 
    and proportion of individuals that lost money on investments among families 
    of different size

Usage: plot-stacked-chart.py <grouped_df_path> <plot_number> <output_file>

Options:
<grouped_df_path> Path to Dataframe of features grouped by all columns
<plot_number>    Natural number in range [1,4] corresponding to which plot
<output_file>    Path (including filename) of where to locally write the figure
"""
  
from docopt import docopt
import pandas as pd
import matplotlib.pyplot as plt
from analysis import inv_outcome_plot

opt = docopt(__doc__)

def main(grouped_df_path, plot_number, output_file):
    grouped_df = pd.read_pickle(grouped_df_path)
    title_1 = """
    Figure 6:
    The Relationship Between Family Size and 
    Investment Outcome Among Individuals
    Who are Major Earners In Their Family
    """
    ylabel_1 = """Proportion of Major Earners 
    with Investment Outcome"""

    title_2 = """
    Figure 7:
    The Relationship Between Family Size and 
    Investment Outcome Among Individuals
    Who are Non-major Earners In Their Family
    """
    ylabel_2 = """Proportion of Non-major Earners 
    with Investment Outcome"""

    title_3 ="""
    Figure 8:
    The KNN-Classification Results For Predicting
    Investment Income Outcome among Major Earners In Their Family
    """
    ylabel_3 = """Proportion of Major Earners 
    Predicted with Investment Outcome"""

    title_4 ="""
    Figure 9:
    The KNN-Classification Results For Predicting
    Investment Income Outcome among Non-major Earners In Their Family
    """
    ylabel_4 = """Proportion of Non-major Earners 
    Predicted with Investment Outcome"""
    
    raise ImplementError("made it to one")

    if plot_number == 1:
        print("1")
        raise ImplementError("made it to one")
        fig1 = inv_outcome_plot.inv_outcome_plot(grouped_df,
                                 'EFSIZE',
                                 'EFMJIE',
                                 'EFINVA_Made_Money',
                                 'counts',
                                 True,
                                 title_1,
                                 ylabel_1)
        fig1.tight_layout()
        fig1.savefig(output_file + "plot1.png")
    elif plot_number == 2:
        fig2 = inv_outcome_plot.inv_outcome_plot(grouped_df,
                                 'EFSIZE',
                                 'EFMJIE',
                                 'EFINVA_Made_Money',
                                 'counts',
                                 False,
                                 title_2,
                                 ylabel_2)
        fig2.tight_layout()
        fig2.savefig(output_file + "plot2.png")                            
    elif plot_number == 3:
        fig3 = inv_outcome_plot.inv_outcome_plot(grouped_df,
                                 'EFSIZE',
                                 'EFMJIE',
                                 'Pred_EFINVA_Made_Money',
                                 'counts',
                                 True,
                                 title_3,
                                 ylabel_3)
        fig3.tight_layout()
        fig3.savefig(output_file + "plot3.png")                         
    elif plot_number == 4:
        fig4 = inv_outcome_plot.inv_outcome_plot(grouped_df,
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
        main(opt["<grouped_df_path>"], opt["<plot_number>"], opt["<output_file>"])