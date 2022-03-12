from pandas import DataFrame
from sklearn.model_selection import train_test_split

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
        assert list(train_X.columns) == ['let']
        assert list(train_Y.columns) == ['num']
        assert list(test_X.columns) == ['let']
        assert list(test_Y.columns) == ['num']
        
        
def test_split_drop_data():
    
    '''checks if the error message is given if data is not a dataframe'''

        with pytest.raises(TypeError) as e_info:
            
        split_drop("wdcwdcdc", 0.3, 123, "num")
        
    assert "data should be data frame" in str(e_info.value)
    
    
def test_split_drop_rn():
    
    
     test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
    
    '''checks if the error message is given if rn is not an integer'''

        with pytest.raises(TypeError) as e_info:
            
        split_drop(test_data, 0.3, 0.5, "num")
        
    assert "random number should be an integer" in str(e_info.value)
    
    
def test_split_col_str():
    
    
     test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
    
    '''checks if the error message is given if column name is not a string (is unquoted)'''

        with pytest.raises(TypeError) as e_info:
            
        split_drop(test_data, 62352, 0.5, num)
        
    assert "column name should be quoted (a string)" in str(e_info.value)
    
    
    
def test_split_col_testsize_type():
    
    
     test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
    
    
    '''checks if the error message is given if the test set size is not a float'''

        with pytest.raises(TypeError) as e_info:
            
        split_drop(test_data, 62352, "zeropointfour", "num")
        
    assert "the size of the testing set should be a proportion" in str(e_info.value)
    
    
    
def test_split_col_testsize_value():
    
    
     test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
    
    '''checks if the error message is given if the size of the test set is not a proportion'''

        with pytest.raises(ValueError) as e_info:
            
        split_drop(test_data, 62352, 79, "num")
        
    assert "the size of the testing set should be a proportion" in str(e_info.value)
    
    
    
def test_split_col_value():
    
    
     test_data = DataFrame({
        'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
        'let': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    })
    
    '''checks if the error message is given if the provided column is not in data'''

        with pytest.raises(ValueError) as e_info:
            
        split_drop(test_data, 62352, 0.3, "sum")
        
    assert "the specified column is not in the provided dataframe" in str(e_info.value)
    

    
def test_split_small_data():
    
    
     test_data = DataFrame({
        'num': [1],
        'let': ['a']
    })
    
    
    '''checks if the error message is given if the dataset is too small'''

        with pytest.raises(ValueError) as e_info:
            
        split_drop(test_data, 62352, 0.3, "num")
        
    assert "please use a dataset with at least 10 observations" in str(e_info.value)