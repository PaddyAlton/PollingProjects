 from __future__ import unicode_literals

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

if __name__ == "__main__":

    raw_html = get_html_content(
        "https://en.wikipedia.org/wiki/"
        "Opinion_polling_for_the_next_United_Kingdom_general_election"
    )

    processed_html = BeautifulSoup(raw_html, 'html.parser')

    table = pd.read_html(processed_html.encode('utf8'))

    trs = processed_html.select('table')[0].tbody.find_all('tr')

    raw_data = pd.DataFrame([
        [x.contents[0] for x in tr.find_all('td')] for tr in trs
    ])
