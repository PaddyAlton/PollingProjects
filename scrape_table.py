import numpy as np
import pandas as pd
import requests
import janitor

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
        raw_html - HTML obtained via a GET request from the url

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

    # clearly mark missing entries
    cleaned_row = [
        cell if cell not in ["", "N/A", "nan"]
        else np.nan for cell in unwrapped_row
    ]

    return cleaned_row

def html_to_dataframe(raw_table):

    """
    html_to_dataframe

    INPUTS:
        raw_table - HTML table as parsed by BeautifulSoup

    OUTPUTS:
        table - the same data represented as a pandas DataFrame

    """

    rows = raw_table.find_all("tr")

    raw_rows = [unwrap_merged_cells(row) for row in rows]

    column_headers = raw_rows[0]

    table = pd.DataFrame(raw_rows[1:], columns=column_headers).dropna(how="all")

    return table

def read_html(processed_html):

    """
    read_html

    An alternative to the pandas read_html method, which can't handle
    merged cells

    INPUTS:
        processed_html - HTML as-parsed by BeautifulSoup

    OUTPUTS:
        tables - list of DataFrames extracted from the HTML

    """

    all_tables_raw = processed_html.find_all("table")

    data_tables = [
        table for table in all_tables_raw if "wikitable" in table.get("class")
    ]

    tables = [html_to_dataframe(raw_table) for raw_table in data_tables]

    return tables

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
    year_table_pairs = zip(expected_table_years(), [all_tables[i] for i in [0, 2, 3]])

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
        data_tables.append(
            dropnull_table.rename_column("Date(s)conducted", "date")
        )

    # join tables together, clean column names --> lower-case, spaces-to-underscores
    all_polls = pd.concat(data_tables, sort=False).clean_names()

    return all_polls

def render_numeric(dataframe):

    """
    render_numeric

    The data is parsed as character strings, but of course polling data
    is numeric. Here we identify the numeric columns and convert types

    INPUTS:
        dataframe - pandas DataFrame

    OUTPUTS:
        modified_dataframe - copy, data-types coerced where appropriate

    """

    modified_dataframe = dataframe.copy()

    modified_dataframe.loc[:, "sample_size"] = (
        dataframe
            .sample_size
            .map(lambda s: int(s.replace(",", "")) if isinstance(s, str) else s)
    )

    for col in dataframe.columns[4:]:

        numeric_series = (
            dataframe
                .loc[:, col] # extract series, convert percentage to float
                .map(lambda s: s.replace("%", "") if isinstance(s, str) else "")
                .str.replace("Tie", "0")
                .str.replace("\[.\]", "")
                .map(lambda s: float(s) if s != "" else np.nan)
        )

        modified_dataframe.loc[:, col] = numeric_series

    return modified_dataframe

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

    polling_org = dataframe.loc[:, "polling_organisation_client"]

    attempted_split = polling_org.str.split('/')

    organisation = attempted_split.map(lambda L: L[0])

    client = attempted_split.map(lambda L: L[1] if len(L) == 2 else np.nan)

    modified_dataframe = dataframe.remove_columns(["polling_organisation_client"])

    col_order = list(modified_dataframe.copy().columns.values)
    col_order.insert(1, 'polling_client')
    col_order.insert(1, 'polling_org')

    modified_dataframe.loc[:, 'polling_org'] = organisation
    modified_dataframe.loc[:, 'polling_client'] = client

    return modified_dataframe.loc[:, col_order]

def disaggregate_nationalist_data(dataframe):

    """
    disaggregate_nationalist_data

    YouGov (and sometimes Ipsos MORI) group the SNP and Plaid Cymru
    together, which is not something all other pollsters do. This
    function imputes the disaggregated values for these parties using the
    average ratio of their vote shares

    """

    modified_dataframe = dataframe.copy()

    # check which rows have complete data
    is_complete = dataframe.isna().apply(np.any, axis=1).map(np.logical_not)

    nationalists_complete = dataframe.loc[is_complete, ["plaid", "snp"]]

    ## YouGov data

    # impute SNP & Plaid Cymru values in YouGov data (combined share stored under snp)
    is_yg = dataframe.polling_org == "YouGov"

    ratio = ( # get the average ratio between the two nationalist parties
        nationalists_complete
            .plaid.div(nationalists_complete.snp).mean()
    )

    modified_dataframe.loc[is_yg, "snp"] = (
        dataframe.loc[is_yg, "snp"].mul(1 - ratio).map(np.round)
    )

    modified_dataframe.loc[is_yg, "plaid"] = (
        dataframe.loc[is_yg, "snp"].mul(ratio).map(np.round)
    )

    ## Ipsos MORI data - changes methodology, sometimes similar to YouGov

    is_im = dataframe.polling_org == "Ipsos MORI"

    needs_disaggregation = is_im & dataframe.plaid.isna()

    modified_dataframe.loc[needs_disaggregation, "snp"] = (
        dataframe.loc[needs_disaggregation, "snp"].mul(1 - ratio).map(np.round)
    )

    modified_dataframe.loc[needs_disaggregation, "plaid"] = (
        dataframe.loc[needs_disaggregation, "snp"].mul(ratio).map(np.round)
    )

    return modified_dataframe

def impute_small_parties(dataframe):

    """
    impute_small_parties

    """

    modified_dataframe = dataframe.copy()

    # update which rows have complete data
    is_complete = (
        dataframe
            .iloc[:, 5:]
            .notna()
            .apply(np.all, axis=1)
    )

    complete_data = dataframe.loc[is_complete]

    # let's work on a common pattern
    no_snp = dataframe.snp.isna()
    no_plaid = dataframe.plaid.isna()
    no_green = dataframe.green.isna()

    no_small_parties = no_snp & no_plaid & no_green

    small_parties_total = (
        complete_data.snp
        + complete_data.plaid
        + complete_data.green
        + complete_data.others
    )

    for party in ["snp", "plaid", "green", "others"]:
        modified_dataframe.loc[no_small_parties, party] = np.round(
            complete_data[party].mean() / small_parties_total.mean()
        )

    return modified_dataframe

def backfill_polls(dataframe):

    """
    backfill_polls

    This method uses the standard fillna method "bfill", but then
    compares the original values to the imputed ones and sums these
    across all parties. This tells us the total imputed to each party and
    we remove this from the 'others' column

    """

    modified_dataframe = dataframe.copy()

    numeric_subset = dataframe.iloc[:, 5:] # numeric columns only

    numeric_subset_corr = numeric_subset.fillna(method="bfill")

    imputed_vals = numeric_subset_corr.sub(numeric_subset.fillna(0)).apply(np.sum,axis=1)

    modified_dataframe.iloc[:, 5:] = numeric_subset_corr
    modified_dataframe.loc[:, "others"] = dataframe.others.sub(imputed_vals)

    return modified_dataframe

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

    """

    modified_dataframe = disaggregate_nationalist_data(dataframe)
    modified_dataframe = impute_small_parties(modified_dataframe)
    modified_dataframe = backfill_polls(modified_dataframe)

    return modified_dataframe

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

    all_tables = read_html(processed_html)

    all_polls = (
        unify_tables(all_tables)
            .pipe(render_numeric)
            .pipe(parse_polling_dates)
            .pipe(parse_polling_org)
            .pipe(parse_minor_parties)
            .set_index("date")
    )

    return all_polls


if __name__ == "__main__":

    all_polls = download_and_transform()
