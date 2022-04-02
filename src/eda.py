""" generates and saves histogram plots of processed data, as well as correlation matrix showing
    correlations between features in processed data

Usage: eda.py --processed_data_path=<processed_data_path> --output_file=<output_file>

Options:
--processed_data_path=<processed_data_path>         Path to processed data
--output_file=<output_file>                         Path (including filename) of where to locally write the figures

"""

from docopt import docopt
import pandas as pd
import matplotlib.pyplot as plt
from analysis.format_histograms import format_histograms

opt = docopt(__doc__)

def main(processed_data_path, output_file):
    
    reduced_data = pd.read_pickle(processed_data_path)
    
    texts = { 'titles':['Number of economic family members', 
                    'After Tax Income',
                    'Highest level of education of person',
                    'EF Investment Income',
                    'Major Income earner in the economic Family',
                    'EF After-Tax Income',
                    'Major Source of income for the economic family',
                    ''],
            'xaxes':['Number of People',
                        'Dollars [CAD]',
                        'Highest level of education',
                        'Dollars [CAD]',
                        'Major income earner',
                        'Dollars [CAD]',
                        'Major source of income',
                        '']
         }
    
    
    histograms = format_histograms(reduced_data, texts)
    histograms.tightlayout()
    histograms.savefig(output_file + "/histograms.png")
    
    plt.figure(figsize=(10,10))
    correlations = reduced_data.iloc[:,[1,3,5]].corr()

    corr_mat = sns.heatmap(correlations, cmap=plt.cm.Blues, annot=True)
    corr_mat.savefig(output_file + "/corr_mat.png")
        
    
    
if __name__ == "__main__":
    main(opt["--processed_data_path"], opt["--output_file"])

    
    