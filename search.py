from bs4 import BeautifulSoup
import requests


def search(question:str) -> str:
    params = {
        "q": question,
        "hl": "en",  # language
        "gl": "us"   # country of the search, US -> USA
    }

    # https://docs.python-requests.org/en/master/user/quickstart/#custom-headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
    }

    html = requests.get("https://www.google.com/search", params=params, headers=headers, timeout=30)
    soup = BeautifulSoup(html.text, "lxml")

    try:
        result = soup.find("div", class_="kno-rdesc")
        result = result.select_one('span').text
        if result: return result
    except:
        pass

    # first fallback strategy
    try:
        result = soup.find("span", class_='hgKElc').text
        if result: return result
    except:
        pass

    # final fallback strategy
    try:
        result = soup.find_all("span", class_="hgKElc").text
        return result
    except:
        return None

answer = search("what is gluten")
print(answer)
