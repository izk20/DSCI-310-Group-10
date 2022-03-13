from src.analysis.inv_outcome_plot import inv_outcome_plot
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@pytest.fixture
def toy_dataset():
    return pd.DataFrame({'size':[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
                    'bar_split_var':[1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,2],
                    'actual_val':[False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,
                                 False,True,False,True,False,True,False,True,False,True,False,True],
                    'pred_val': [False,False,False,True,False,True,False,True,False,True,True,True,False,True,False,False,
                                 False,True,True,True,False,False,False,True,True,True,False,True],
                     'counts': [320, 300, 180, 500, 488, 600, 700, 800, 240, 500, 400, 120, 400, 300, 
                               500, 500, 320, 980, 890, 750, 210, 540, 320, 450, 560, 760, 580, 650]})
@pytest.fixture
def toy_dataset1():
    return pd.DataFrame({'size':[1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7],
                    'bar_split_var':[1,1,2,2,1,1,2,2,1,1,2,2,1,1,2,3,1,1,2,2,1,1,2,2,1,1,2,2],
                    'actual_val':[False,True,False,True,False,True,False,True,False,True,False,True,False,True,False,True,
                                 False,True,False,True,False,True,False,True,False,True,False,True],
                    'pred_val': [False,False,False,True,False,True,False,True,False,True,True,True,False,True,False,False,
                                 False,True,True,True,False,False,False,True,True,True,False,True],
                     'counts': [320, 300, 180, 500, 488, 600, 700, 800, 240, 500, 400, 120, 400, 300, 
                               500, 500, 320, 980, 890, 750, 210, 540, 320, 450, 560, 760, 580, 650]})

def test_grouped_df(toy_dataset):
    ''' Checks if passed in data is a dataframe '''
    not_df = 1
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(not_df, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4], 
                         True, 
                         "", 
                         "")
    assert "grouped_df is not a DataFrame" in str(e_info.value)

def test_size_col_name(toy_dataset):
    ''' Checks if size_col name is inputted as a string '''
    non_string_name = 1
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         non_string_name, 
                         list_of_col_names[1], 
                         list_of_col_names[3], 
                         list_of_col_names[4], 
                         False, 
                         "", 
                         "")
    assert "size_col is not inputted as String" in str(e_info.value)

def test_bar_split_col_name(toy_dataset):
    ''' Checks if passed in bar_split_col name is inputted as a string '''
    non_string_name = 1
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         non_string_name, 
                         list_of_col_names[2], 
                         list_of_col_names[4], 
                         True, 
                         "", 
                         "")
    assert "bar_split_col is not inputted as String" in str(e_info.value)

def test_count_col_name(toy_dataset):
    ''' Checks if counts_col name is inputted as a string '''
    non_string_name = 1
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[3],
                         non_string_name, 
                         True, 
                         "", 
                         "")
    assert "counts_col is not inputted as String" in str(e_info.value)
    
def test_val_col_name(toy_dataset):
    ''' Checks if val_col name is inputted as a string '''
    non_string_name = 1
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         non_string_name, 
                         list_of_col_names[4], 
                         False, 
                         "", 
                         "")
    assert "val_col is not inputted as String" in str(e_info.value)
    
def test_maj_earner(toy_dataset):
    '''Checks if maj_earner is inputted as a boolean'''
    non_bool_val = "false"
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4], 
                         non_bool_val, 
                         "", 
                         "")
    assert "major_earner is not inputted as Boolean" in str(e_info.value)

def test_fig_title(toy_dataset):
    '''Checks if fig_title is inputted as a string '''
    non_string_val = 1
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4], 
                         True, 
                         non_string_val, 
                         "")
    assert "fig_title is not inputted as String" in str(e_info.value)

def test_fig_ylabel(toy_dataset):
    ''' Checks if fig_ylabel is inputted as a string '''
    non_string_val = 1
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4], 
                         True, 
                         "", 
                         non_string_val)
    assert "fig_ylabel is not inputted as String" in str(e_info.value)

def test_size_col_type(toy_dataset):
    ''' Checks if data in size_col are integer values '''
    toy_dataset['size'] = toy_dataset['size'].to_string()
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4],
                         True,
                         "", 
                         "")
    assert "size_col must be column of integers" in str(e_info.value)

def test_bar_split_col_type(toy_dataset):
    ''' Checks if data in bar_split_col are integer values '''
    toy_dataset['bar_split_var'] = toy_dataset['bar_split_var'].to_string()
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4],
                         True,
                         "", 
                         "")
    assert "bar_split_col must be a binary column of integers with only 2 distinct values" in str(e_info.value)

def test_bar_split_binary(toy_dataset1):
    ''' Checks if data in bar_split_col are binary - only 2 distinct integers '''
    list_of_col_names = list(toy_dataset1.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset1, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4],
                         True,
                         "", 
                         "")
    assert "bar_split_col must be a binary column of integers with only 2 distinct values" in str(e_info.value)

def test_val_col_type(toy_dataset):
    ''' Checks if data in val_col are boolean '''
    toy_dataset['actual_val'] = toy_dataset['actual_val'].to_string()
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4],
                         True,
                         "", 
                         "")
    assert "val_col must be column of boolean" in str(e_info.value)

def test_counts_col_type(toy_dataset):
    ''' Checks if data in counts_col are integer values '''
    toy_dataset['counts'] = toy_dataset['counts'].to_string()
    list_of_col_names = list(toy_dataset.columns)
    with pytest.raises(TypeError) as e_info:
        inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4],
                         True,
                         "", 
                         "")
    assert "counts_col must be column of integers" in str(e_info.value)

def test_fig_title_output(toy_dataset):
    ''' Checks if given plot title is applied correctly '''
    list_of_col_names = list(toy_dataset.columns)
    sb_chart = inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4],
                         True,
                         "sample_title", 
                         "sample_ylable")

    assert sb_chart.axes[0].get_title() == "sample_title"

def test_fig_ylabel_output(toy_dataset):
    ''' Checks if given ploy y-axis label is applied correctly'''
    list_of_col_names = list(toy_dataset.columns)
    sb_chart = inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4],
                         True,
                         "sample_title", 
                         "sample_ylable")

    assert sb_chart.axes[0].get_ylabel() == "sample_ylable"

def test_fig_xlabel_output(toy_dataset):
    ''' Checks if given ploy x-axis label is applied correctly '''
    list_of_col_names = list(toy_dataset.columns)
    sb_chart = inv_outcome_plot(toy_dataset, 
                         list_of_col_names[0], 
                         list_of_col_names[1], 
                         list_of_col_names[2], 
                         list_of_col_names[4],
                         True,
                         "sample_title", 
                         "sample_ylable")

    assert sb_chart.axes[0].get_xlabel() == "Family Size"