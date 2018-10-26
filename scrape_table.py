import numpy as np
import pandas as pd
import requests

from contextlib import closing
from bs4 import BeautifulSoup

def is_good_response(response):

    """
    is_good_response()

    Returns True if the response seems to be HTML, False otherwise

    INPUTS:
        response - response to a GET request

    OUTPUTS:
        good_response_flag

    """

    content_type = response.headers.get('content-type')

    return (
        response.status_code == 200
        and content_type is not None
        and content_type.lower().find('text/html') > -1
    )

def get_html_content(url):

    """
    get_html_content()

    INPUTS:
        url - URL of the page from which to scrape content

    OUTPUTS:
        raw_html

    """

    with closing(requests.get(url, stream=True)) as response:

        if is_good_response(response):
            return response.content

    return None

def expected_table_years():
    """ calculates number of tables (1 per year) to expect and returns year-list """
    last_election_in = 2017
    current_year = pd.datetime.today().year
    return current_year - np.arange(current_year - last_election_in + 1)


if __name__ == "__main__":

    raw_html = get_html_content(
        "https://en.wikipedia.org/wiki/"
        "Opinion_polling_for_the_next_United_Kingdom_general_election"
    )

    processed_html = BeautifulSoup(raw_html, 'html.parser')

    all_tables = pd.read_html(processed_html.encode('utf8'), header=0)

    expected_tables = expected_table_years()

    data_tables = []

    for year, table in zip(expected_tables, all_tables[:len(expected_tables)]):

        null_check_cols = table.columns[4:].values

        dropnull_table = table.copy().dropna(how='all', subset=null_check_cols)

        dc_with_year = (
            dropnull_table
                .loc[:, 'Date(s)conducted']
                .map(lambda s: s + ' ' + str(year))
        )

        dropnull_table['Date(s)conducted'] = dc_with_year.values

        data_tables.append(dropnull_table.rename({'Date(s)conducted': 'date'}, axis=1))

    all_polls = pd.concat(data_tables)
