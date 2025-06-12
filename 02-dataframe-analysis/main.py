import pandas as pd
import requests
from io import StringIO
import re

# Question 1

url = f"https://stockanalysis.com/ipos/withdrawn/"
headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/58.0.3029.110 Safari/537.3'
    )
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    # Wrap HTML text in StringIO to avoid deprecation warning
    # "Passing literal html to 'read_html' is deprecated and will be removed in a future version. To read from a literal string, wrap it in a 'StringIO' object."
    html_io = StringIO(response.text)
    tables = pd.read_html(html_io)

    if not tables:
        raise ValueError(f"No tables found for year {year}.")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
except ValueError as ve:
    print(f"Data error: {ve}")
except Exception as ex:
    print(f"Unexpected error: {ex}")

df = tables[0]

def get_company_class(company_name):
    if not isinstance(company_name, str):
        # Handle non-string inputs, e.g., NaN, None, numbers
        return "Other"

    # Convert to lowercase and split into words for robust matching
    name_lower = company_name.lower()
    words = name_lower.split()
    words = re.findall(r'\b\w+\b', name_lower) # \b is word boundary, \w+ is one or more word characters
    # Define the rules and their order
    # (Pattern check function, Class Name)
    rules = [
        (lambda n, w: ("acquisition" in w and ("corp" in w or "corporation" in w)), "Acq.Corp"),
        (lambda n, w: ("inc" in w or "incorporated" in w), "Inc"),
        (lambda n, w: "group" in w, "Group"),
        (lambda n, w: ("ltd" in w or "limited" in w), "Limited"),
        (lambda n, w: "holdings" in w, "Holdings"),
    ]

    for check_func, class_name in rules:
        if check_func(name_lower, words):
            return class_name

    # If no rule matches
    return "Other"

def parse_price_range(price_range):
    if not isinstance(price_range, str):
        # Handle non-string inputs (e.g., NaN, None) by returning None
        return None

    price_range = price_range.strip() # Remove leading/trailing whitespace

    if price_range == '-':
        return None
    
    # Remove '$' sign and potentially extra spaces
    cleaned_price_range = price_range.replace('$', '').strip()

    if '-' in cleaned_price_range:
        # It's a range, e.g., '8.00-10.00'
        try:
            low_str, high_str = cleaned_price_range.split('-')
            low = float(low_str.strip())
            high = float(high_str.strip())
            return (low + high) / 2.0
        except ValueError:
            # Handle cases where conversion to float fails
            return None
    else:
        # It's a single price, e.g., '5.00'
        try:
            return float(cleaned_price_range)
        except ValueError:
            # Handle cases where conversion to float fails
            return None

def clean_and_convert_shares_offered(value):
    if pd.isna(value) or value is None:
        return pd.NA # Explicitly return pandas' nullable NA for missing values

    s_value = str(value).strip() # Convert to string and remove leading/trailing whitespace

    if not s_value: # Handle empty strings
        return pd.NA

    # Remove commas (e.g., "1,000,000" -> "1000000")
    s_value = s_value.replace(',', '')

    try:
        # Attempt to convert to float. Using float to handle potential decimals,
        # but if you expect only integers, you can cast to int later if desired.
        return float(s_value)
    except ValueError:
        # If conversion fails (e.g., for "N/A", "Unknown", etc.)
        return pd.NA


df['Company Class'] = df['Company Name'].apply(get_company_class)
df['Avg. price'] = df['Price Range'].apply(parse_price_range)
df['Shares Offered'] = df['Shares Offered'].apply(clean_and_convert_shares_offered)
df['Withdrawn Value'] = df['Shares Offered'] * df['Avg. price']
total_withdrawn_by_class = df.groupby('Company Class')['Withdrawn Value'].sum()
print(total_withdrawn_by_class/1000000)