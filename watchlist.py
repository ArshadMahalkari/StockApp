import json
import os

WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if not os.path.exists(WATCHLIST_FILE):
        return []
    with open(WATCHLIST_FILE, "r") as f:
        return json.load(f)

def save_to_watchlist(ticker):
    ticker = ticker.upper()
    watchlist = load_watchlist()
    if ticker not in watchlist:
        watchlist.append(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist, f, indent=4)
        return True
    return False

def remove_from_watchlist(ticker):
    watchlist = load_watchlist()
    if ticker in watchlist:
        watchlist.remove(ticker)
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlist, f, indent=4)
        return True
    return False
