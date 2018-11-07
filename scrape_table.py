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

def unwrap_merged_cells(raw_row):

    """
    unwrap_merged_cells

    """

    cells = raw_row.find_all(["td", "th"])

    colspans = [int(cell.get("colspan", 1)) - 1 for cell in cells]

    unwrapped_row = np.hstack([
        np.append([cell.text.strip('\n')], [np.nan for x in range(span)])
        for cell, span in zip(cells, colspans)
    ])

    return unwrapped_row

def html_to_dataframe(raw_table):

    """
    html_to_dataframe

    INPUTS:
        raw_table - HTML table as parsed by BeautifulSoup

    OUTPUTS:
        dataframe

    WORK IN PROGRESS

    """

    rows = raw_table.find_all("tr")

    raw_rows = [unwrap_merged_cells(row) for row in rows]


def read_html(processed_html):

    """
    read_html

    An alternative to the pandas read_html method, which can't handle
    merged cells

    INPUTS:
        processed_html

    OUTPUTS:
        all_tables - list of DataFrames parsed out from the HTML

    """

    all_tables_raw = processed_html.find_all("table")

    all_tables = [html_to_dataframe(raw_table) for raw_table in all_tables_raw]

    return all_tables

def expected_table_years():
    """ calculates number of tables (1 per year) to expect and returns year-list """
    last_election_in = 2017
    current_year = pd.datetime.today().year
    return current_year - np.arange(current_year - last_election_in + 1)

def unify_tables(all_tables):

    """
    unify_tables

    This function takes the list of HTML tables that have been read into
    a pandas DataFrames and unites them, appending the appropriate year
    to the dates as it goes (each table corresponds to a different year's
    polling data)

    INPUTS:
        all_tables - list of pandas DataFrames read from HTML tables

    OUTPUTS:
        all_polls - unified dataset returned as a single pandas DataFrame

    """

    # no input needed to get expected table years, works it out from current date:
    expected_tables = expected_table_years()

    year_table_pairs = zip(expected_tables, all_tables[:len(expected_tables)])

    data_tables = []

    for year, table in year_table_pairs:

        # first drop all events that aren't actually polling data:
        null_check_cols = table.columns[4:].values
        dropnull_table = table.copy().dropna(how='all', subset=null_check_cols)

        # next append the year to the 'Date(s)conducted' column:
        dc_with_year = (
            dropnull_table
                .loc[:, 'Date(s)conducted']
                .map(lambda s: s + ' ' + str(year))
        )

        dropnull_table['Date(s)conducted'] = dc_with_year.values

        # append this modified table, renaming column to the simpler 'date':
        data_tables.append(dropnull_table.rename({'Date(s)conducted': 'date'}, axis=1))

    all_polls = pd.concat(data_tables)

    return all_polls

def parse_polling_dates(dataframe):

    """
    parse_polling_dates

    Alas, polls are conducted over several days and so the dates are
    formatted in an inconvenient manner. For now, we're going to parse
    these out into a single date (the last in the data-collection period)
    but in future it may be more appropriate to identify a start and end
    date instead

    INPUTS:
        dataframe - table of polling data as a DataFrame, must have a
                    'date' column

    OUTPUTS:
        modified_dataframe - input with 'date' column parsed into proper
                             datetime type (final date in data-collection
                             period is used)

    """

    dates_conducted = dataframe.loc[:, 'date']

    final_date = (
        dates_conducted
            .str.split("-")
            .map(lambda L: L[-1])
            .str.split("\u2013") # I know, I know - why do we need two types of hyphen!?
            .map(lambda L: L[-1])
    )

    as_dt = pd.to_datetime(final_date)

    modified_dataframe = dataframe.copy()

    modified_dataframe.loc[:, 'date'] = as_dt

    return modified_dataframe

def parse_polling_org(dataframe):

    """
    parse_polling_org

    Take a DataFrame with 'Polling organisation/client' column, and parse
    this out into two separate columns: 'organisation' and 'client'

    INPUTS:
        dataframe - table of polling data as a pandas DataFrame, must
                    contain a 'Polling organisation/client' column

    OUTPUTS:
        modified_dataframe - input with 'Polling organisation/client'
                             column removed and replaced with two new
                             columns, 'polling_org' and 'polling_client'

    """

    polling_org = dataframe.loc[:, "Polling organisation/client"]

    attempted_split = polling_org.str.split('/')

    organisation = attempted_split.map(lambda L: L[0])

    client = attempted_split.map(lambda L: L[1] if len(L) == 2 else np.nan)

    modified_dataframe = dataframe.drop(["Polling organisation/client"], axis=1)

    col_order = list(modified_dataframe.copy().columns.values)
    col_order.insert(1, 'polling_client')
    col_order.insert(1, 'polling_org')

    modified_dataframe.loc[:, 'polling_org'] = organisation
    modified_dataframe.loc[:, 'polling_client'] = client

    return modified_dataframe.loc[:, col_order]

def parse_minor_parties(dataframe):

    """
    parse_minor_parties

    Different polling organisations have different approaches to handling
    the vote shares of minor parties. This function is a best-effort
    at rationalising the format, filling missing values where needed

    INPUTS:
        dataframe - table of polling data as a pandas DataFrame

    OUTPUTS:
        modified_dataframe - input with imputed values entered for minor
                             party vote shares


    WORK IN PROGRESS

    """

    pass

def download_and_transform():

    """
    download_and_transform

    The top level procedure for scrape_table.py

    Downloads HTML tables from the Wikipedia article

        "Opinion polling for the next United Kingdom general election"

    ... and transforms these into a single pandas DataFrame containing
    all the whole-country polling data aggregated there

    OUTPUTS:
        all_polls - pandas DataFrame containing all the whole-country
                    polling data since the 2017 general election

    """

    raw_html = get_html_content(
        "https://en.wikipedia.org/wiki/"
        "Opinion_polling_for_the_next_United_Kingdom_general_election"
    )

    processed_html = BeautifulSoup(raw_html, 'html.parser')

    all_tables = pd.read_html(processed_html.encode('utf8'), header=0)

    all_polls = unify_tables(all_tables)
    all_polls = parse_polling_dates(all_polls) # modify date column in place
    all_polls = parse_polling_org(all_polls) # split polling organisation/client out

    return all_polls


if __name__ == "__main__":

    all_polls = download_and_transform()
