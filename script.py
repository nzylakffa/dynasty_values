from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
import streamlit as st

# Set up Chrome options for headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run Chrome in headless mode
chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration

# Use webdriver_manager to automatically download the correct version of ChromeDriver
driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
driver.get("https://keeptradecut.com/dynasty-rankings")

# Wait for a few seconds to let JavaScript execute and populate the rankings
time.sleep(5)

# Get the page source after JavaScript has executed
page_source = driver.page_source

# Close the Selenium WebDriver
driver.quit()

# Parse the HTML content of the page
soup = BeautifulSoup(page_source, 'html.parser')

# Find the div with the specified class and id
rankings_div = soup.find('div', {'class': 'rankings-page-rankings', 'id': 'rankings-page-rankings'})

# Find the table within the div
table = rankings_div.find('table')

# Use pandas to read the HTML table into a DataFrame
df = pd.read_html(str(table))[0]

# Display the DataFrame
st.dataframe(df)
