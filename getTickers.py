import requests
from bs4 import BeautifulSoup

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers = []
    for row in table.find_all('tr')[1:]:
        columns = row.find_all('td')
        if len(columns) > 0:
            ticker = columns[1].text.strip()
            tickers.append(ticker)

    return tickers