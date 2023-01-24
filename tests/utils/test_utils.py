import pytest
from src.utils.utils import normalize_numeric, normalize_dates


def test_normalize_numeric():
    data = {'value1': '100', 'value2': 200, 'value3': '300'}
    expected = {'value1': 0.0, 'value2': 0.5, 'value3': 1.0}
    assert normalize_numeric(data) == pytest.approx(expected)


def test_normalize_dates():
    data = {'date1': '2021-01-01', 'date2': '2021-06-15', 'date3': 'invalid-date'}
    expected = {'date1': 0.0, 'date2': 1.0, 'date3': 0.0}
    result = normalize_dates(data)
    assert result['date1'] == pytest.approx(expected['date1'])
    assert result['date2'] == pytest.approx(expected['date2'])
    assert result['date3'] == expected['date3']


def test_normalize_dates_empty():
    data = {}
    expected = {}
    assert normalize_dates(data) == expected
