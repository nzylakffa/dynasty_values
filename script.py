# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
import beautifulsoup4 as bs4
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
import time
import streamlit as st
st.write("Hi")

st.sidebar.markdown("# Read This!")
st.sidebar.markdown("### You will see errors until you fill out the 4 items on this page and then select your team and a trade partner on the Trade Calculator tab!")
st.sidebar.markdown("1) Click Fullscreen at the bottom for a better user experience")
st.sidebar.markdown("2) Input Sleeper Username")
st.sidebar.markdown("3) Input Season")
st.sidebar.markdown("(This is the season you're looking at. Use 2023 for last season and 2024 once we draft our teams for 2024.)")
st.sidebar.markdown("4) Select the league you want to use")
st.sidebar.markdown("(This is a dropdown of all the league's you're in! If you don't know which is which then just pick one and check out the trade calculator tab to see which team of yours that is.)")
st.sidebar.markdown("5) Input your league's scoring format")
st.sidebar.markdown("6) Turn on the toggle if it's a dynasty league")
st.sidebar.markdown("7) Go to the Trade Calculator tab and select your display name along with your trade partner's display name")
st.sidebar.markdown("8) You'll need to wait a few seconds for the tool to pull all the information")

# # Set up Chrome options for headless mode
# chrome_options = Options()
# chrome_options.add_argument("--headless")  # Run Chrome in headless mode
# chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration

# # Use webdriver_manager to automatically download the correct version of ChromeDriver
# driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
# driver.get("https://keeptradecut.com/dynasty-rankings")

# # Wait for a few seconds to let JavaScript execute and populate the rankings
# time.sleep(5)

# # Get the page source after JavaScript has executed
# page_source = driver.page_source

# # Close the Selenium WebDriver
# driver.quit()

# # Parse the HTML content of the page
# soup = BeautifulSoup(page_source, 'html.parser')

# # Find the div with the specified class and id
# rankings_div = soup.find('div', {'class': 'rankings-page-rankings', 'id': 'rankings-page-rankings'})

# # Find the table within the div
# table = rankings_div.find('table')

# # Use pandas to read the HTML table into a DataFrame
# df = pd.read_html(str(table))[0]

# # Display the DataFrame
# st.dataframe(df)
