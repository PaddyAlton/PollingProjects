import numpy as np
import pandas as pd
import janitor

def get_html_tables(url):

    """
    get_html_tables

    Wrapper function around the pandas read_html function to set the
    correct keywords and apply the pyjanitor name-cleaning routine

    INPUTS:
        url - target URL from which to retrieve HTML tables (we only want
              those that have the wikitable class)

    OUTPUTS:
        all_tables - list of retrieved tables

    """

    kwargs = {
        "attrs": {"class": "wikitable"},
        "header": 0, # use row 0 as header
        "skiprows": [1], # skip row 1 (0-based, i.e. the second row)
        "na_values": ["–"],
    }

    wikitables = pd.read_html(url, **kwargs)

    all_tables = [t.clean_names(strip_underscores=True) for t in wikitables]

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
    table_years = expected_table_years()

    year_table_pairs = zip(table_years, [all_tables[i] for i in range(len(table_years))])

    data_tables = []

    for year, table in year_table_pairs:

        # first drop all events that aren't actually polling data:
        dropnull_table = table.copy().dropna(subset=["pollster_client_s"])

        # next append the year to the 'Date(s)conducted' column:
        dc_with_year = (
            dropnull_table.date_s_conducted.map(lambda s: f"{s} {year}")
        )

        dropnull_table.loc[:, "date"] = dc_with_year.values

        # append this modified table, renaming column to the simpler 'date':
        data_tables.append(dropnull_table)

    # join tables together, drop old date column name
    all_polls = pd.concat(data_tables, sort=False).drop(columns=["date_s_conducted"])

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

    column_names = modified_dataframe.columns

    # identify columns containing values formatted as percentages
    numeric_columns = [
        col for col in column_names if modified_dataframe[col].str.endswith("%").any()
    ]

    for col in numeric_columns:

        numeric_series = (
            dataframe
                .loc[:, col] # extract series, convert percentage to float
                .map(lambda s: s.replace("%", "") if isinstance(s, str) else "")
                .str.replace("Tie", "0")
                .str.replace("\[.\]", "")
                .str.replace("<1", "0") # <1% usually means they found 0 in-sample but are hedging
                .map(lambda s: float(s) if s != "" else np.nan)
        )

        modified_dataframe.loc[:, col] = numeric_series


    # convert non-missing sample-sizes to numeric type
    modified_dataframe.loc[:, "samplesize"] = modified_dataframe.samplesize.astype(float)

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

    polling_org = dataframe.loc[:, "pollster_client_s"]

    attempted_split = polling_org.str.split('/')

    organisation = attempted_split.map(lambda L: L[0])

    client = attempted_split.map(lambda L: L[1] if len(L) == 2 else np.nan)

    modified_dataframe = dataframe.remove_columns(["pollster_client_s"])

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

    nationalists_complete = dataframe.loc[is_complete, ["plaid_cymru", "snp"]]

    ## YouGov data

    # impute SNP & Plaid Cymru values in YouGov data (combined share stored under snp)
    is_yg = dataframe.polling_org == "YouGov"

    ratio = ( # get the average ratio between the two nationalist parties
        nationalists_complete
            .plaid_cymru.div(nationalists_complete.snp).mean()
    )

    modified_dataframe.loc[is_yg, "snp"] = (
        dataframe.loc[is_yg, "snp"].mul(1 - ratio).map(np.round)
    )

    modified_dataframe.loc[is_yg, "plaid_cymru"] = (
        dataframe.loc[is_yg, "snp"].mul(ratio).map(np.round)
    )

    ## Ipsos MORI data - changes methodology, sometimes similar to YouGov

    is_im = dataframe.polling_org == "Ipsos MORI"

    needs_disaggregation = is_im & dataframe.plaid_cymru.isna()

    modified_dataframe.loc[needs_disaggregation, "snp"] = (
        dataframe.loc[needs_disaggregation, "snp"].mul(1 - ratio).map(np.round)
    )

    modified_dataframe.loc[needs_disaggregation, "plaid_cymru"] = (
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
    no_plaid = dataframe.plaid_cymru.isna()
    no_green = dataframe.green.isna()

    no_small_parties = no_snp & no_plaid & no_green

    small_parties_total = (
        complete_data.snp
        + complete_data.plaid_cymru
        + complete_data.green
        + complete_data.other
    )

    for party in ["snp", "plaid_cymru", "green", "other"]:
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
    we remove this from the 'other' column

    """

    numeric_columns = [
        column for column in dataframe.columns
        if (dataframe[column].dtype == np.float64) and (column != "samplesize")
    ]

    numeric_subset = dataframe.loc[:, numeric_columns] # numeric columns only

    numeric_subset_corr = numeric_subset.fillna(method="bfill")

    imputed_total = numeric_subset_corr.sub(numeric_subset.fillna(0)).apply(np.sum,axis=1)

    modified_dataframe = dataframe.copy()

    modified_dataframe.loc[:, numeric_columns] = numeric_subset_corr

    modified_dataframe.loc[:, "other"] = dataframe.other.sub(imputed_total)

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

    target = "Opinion_polling_for_the_next_United_Kingdom_general_election"

    url = "https://en.wikipedia.org/wiki/" + target

    all_tables = get_html_tables(url)

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
