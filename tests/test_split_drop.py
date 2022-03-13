from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.analysis.split_drop import split_drop
import pytest

def test_split_drop_correct():
    '''checks if the function works correctly for valid inputs'''
    
    test_data = DataFrame({
       'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
        
    train_X, train_Y, test_X, test_Y = split_drop(test_data, 0.3, 123, "num")
        
    assert len(train_X) == 7
    assert len(train_Y) == 7
    assert len(test_X) == 3
    assert len(test_Y) == 3
    assert train_X.columns.values.tolist() == ['let']
    assert DataFrame(train_Y).columns.values.tolist() == ['num']
    assert test_X.columns.values.tolist() == ['let']
    assert DataFrame(test_Y).columns.values.tolist() == ['num']
        
        
def test_split_drop_data():
    '''checks if the error message is given if data is not a dataframe'''

    with pytest.raises(TypeError) as e_info:
        split_drop("wdcwdcdc", 0.3, 123, "num")
    assert "data should be a dataframe" in str(e_info.value)
    
    
def test_split_drop_rn():
    '''checks if the error message is given if rn is not an integer'''
    
    test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })

    with pytest.raises(TypeError) as e_info:   
        split_drop(test_data, 0.3, 0.5, "num")
    assert "random number should be an integer" in str(e_info.value)
    
    
    
    
def test_split_col_str():
    '''checks if the error message is given if column name is not a string (is unquoted)'''
    
    test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
    
    

    with pytest.raises(TypeError) as e_info:
        split_drop(test_data, 0.5, 66276, 217)
    assert "column name should be quoted (a string)" in str(e_info.value)
    
    
    
def test_split_col_testsize_type():
    '''checks if the error message is given if the test set size is not a float'''
    
    test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })

    with pytest.raises(TypeError) as e_info:
        split_drop(test_data, "zeropointthree", 76767, "num")
    assert "the size of the testing set should be a float" in str(e_info.value)
    
    
    
def test_split_col_testsize_value():
    '''checks if the error message is given if the size of the test set is not a proportion'''
    
    test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })


    with pytest.raises(ValueError) as e_info:
        split_drop(test_data, 79.0, 2873, "num")
    assert "the size of the testing set should be a proportion" in str(e_info.value)
    
    
    
def test_split_col_value():
    '''checks if the error message is given if the provided column is not in data'''
    
    test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
    
    

    with pytest.raises(ValueError) as e_info:
        split_drop(test_data, 0.3, 276726, "sum")
    assert "the specified column is not in the provided dataframe" in str(e_info.value)
    

    
def test_split_small_data():
    '''checks if the error message is given if the dataset is too small'''
    
    test_data = DataFrame({
        'num': [1],
        'let': ['a']
    })
    
    
    

    with pytest.raises(ValueError) as e_info:
        split_drop(test_data, 0.3, 76265, "num")
    assert "please use a dataset with at least 10 observations" in str(e_info.value)