from src.analysis.format_histograms import format_histograms
from pandas import DataFrame
import pytest

@pytest.fixture
def toy_dataset():
    return DataFrame({'x1':[1,2,3,4,5,6],
                    'x2':[1,1,1,2,3,3],
                    'x3':[20,40,21,34,56,53]
                    })

def test_raise_errors(toy_dataset):
    '''checks if the ValueError is thrown for empty parameter'''
    with pytest.raises(ValueError) as e_info:
        histograms = format_histograms(toy_dataset)
    assert "Parameter 'texts' is empty" in str(e_info.value)

def test_format_title(toy_dataset):
    '''Checks if the passed in title formats correctly'''
    assert toy_dataset is not None
    texts = { 'titles':['title1', 'title2', 'title3']}
    histograms = format_histograms(toy_dataset, texts)
    assert histograms[0].get_title() == 'title1'
    assert histograms[1].get_title() == 'title2'
    assert histograms[2].get_title() == 'title3'

def test_format_xlabel(toy_dataset):
    '''Checks if the passed in xlabel formats correctly'''
    texts = { 'xaxes':['xaxes1', 'xaxes2', 'xaxes3']}
    assert toy_dataset is not None
    histograms = format_histograms(toy_dataset, texts)
    assert histograms[0].get_xlabel() == 'xaxes1'
    assert histograms[1].get_xlabel() == 'xaxes2'
    assert histograms[2].get_xlabel() == 'xaxes3'

def test_format_ylabel(toy_dataset):
    '''Checks if the passed in ylabel formats correctly'''
    texts = { 'yaxes':['yaxes1', 'yaxes2', 'yaxes3']}
    assert toy_dataset is not None
    histograms = format_histograms(toy_dataset, texts)
    assert histograms[0].get_ylabel() == 'yaxes1'
    assert histograms[1].get_ylabel() == 'yaxes2'
    assert histograms[2].get_ylabel() == 'yaxes3'

def test_format_ylabel_default(toy_dataset):
    '''checks if the ylabel will default given no formatted value'''
    texts = { 'titles':['title1', 'title2', 'title3']}
    histograms = format_histograms(toy_dataset, texts)
    assert histograms[0].get_ylabel() == 'Frequency'
    assert histograms[1].get_ylabel() == 'Frequency'
    assert histograms[2].get_ylabel() == 'Frequency'
