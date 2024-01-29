import requests
from bs4 import BeautifulSoup
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable import Table, ColumnDefinition
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
import matplotlib
import streamlit as st

# Add custom CSS to hide the GitHub icon
hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

###################
##### Sidebar #####
###################
st.sidebar.image('ffa_red.png', use_column_width=True)
st.sidebar.markdown("<h4 style='text-align: center;'>Click Fullscreen at the bottom for the best user experience</h4>", unsafe_allow_html=True)
st.sidebar.markdown("<h1 style='text-align: center;'>What is this page?</h1>", unsafe_allow_html=True)
st.sidebar.markdown("When you open this page, my script will scrape industry data about players ranking in dynasty leagues.")
st.sidebar.markdown("I've chosen to use [KeepTradeCut](https://keeptradecut.com/dynasty-rankings) as a resource for this. You can access their website through the hyperlink.")
st.sidebar.markdown("KeepTradeCut aggregates peoples opinions on players in real time, ensuring you always have a good sense of how the market as a whole is feeling about each player.")
st.sidebar.markdown("My script will also pull my own rankings on the backend, compare those to the realtime numbers from KeepTradeCut, and display the best and worst values...according to my rankings!")
st.sidebar.markdown("This doesn't mean you need to go out and trade for all the best values and sell the worst values. But it does mean that I value them a lot more or less than the overall market. So these are players you should heavily consider using when you think about trading!")
st.sidebar.markdown("Also note that these are POSITIONAL RANKINGS. This means you can use this table as a guide no matter if you're in a SuperFlex league, a TE Premium league, or a regular 1 QB league.")

tab_best, tab_worst = st.tabs(["Best Values", "Worst Values"])

def scrape_rankings(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    player_containers = soup.find_all('div', class_='single-ranking-wrapper')

    player_data = []

    for container in player_containers:
        rank_element = container.find('div', class_='rank-number')
        rank = rank_element.text.strip() if rank_element else ''
        name_element = container.find('div', class_='player-name')
        name = name_element.find('a').text.strip() if name_element and name_element.find('a') else ''
        rookie = 'Rookie' if name_element and name_element.find('span', class_='rookie-badge') else ''
        team_element = container.find('span', class_='player-team')
        team = team_element.text.strip() if team_element else ''

        player_info = {
            'Industry Rank': rank,
            'Name': name,
            'Rookie': rookie,
            'Team': team
        }
        player_data.append(player_info)

    return pd.DataFrame(player_data)

# Dictionary to store DataFrames for each position
rankings_dfs = {}
positions = ['qb', 'rb', 'wr', 'te']

# Scrape each page and store the results in a DataFrame
rankings_dfs = {position: scrape_rankings(f'https://keeptradecut.com/dynasty-rankings/{position}-rankings') for position in ['qb', 'rb', 'wr', 'te']}

# ... (other parts of the code remain unchanged)

# Load the CSV data from the GitHub URL into a DataFrame
github_df = pd.read_csv('https://raw.githubusercontent.com/nzylakffa/sleepercalc/main/All%20Dynasty%20Rankings.csv')

# Drop the specific columns named 'TEP', 'SF TEP', and 'SF'
columns_to_drop = ['TEP', 'SF TEP', 'SF']
github_df.drop(columns=columns_to_drop, errors='ignore', inplace=True)  # errors='ignore' will ignore any columns not found

# Create a new 'Rank' column based on sorting by '1 QB' within each position
github_df['Rank'] = github_df.groupby('Position')['1 QB'].rank(method='first', ascending=False)

# Function to apply fuzzy matching and find the best match for a given name
def fuzzy_merge(df1, df2, key1, key2, threshold=90, limit=1):
    """
    df1 is the left table to merge
    df2 is the right table to merge
    key1 is the key column of the left table
    key2 is the key column of the right table
    threshold is how close the matches should be to return a match, based on Levenshtein distance
    limit is the number of matches to return, we'll keep it at 1 to return the best match
    """
    s = df2[key2].tolist()

    m = df1[key1].apply(lambda x: process.extract(x, s, limit=limit))
    df1['matches'] = m
    
    m2 = df1['matches'].apply(lambda x: x[0][0] if x[0][1] >= threshold else None)
    df1['matched_name'] = m2
    
    return df1

# Perform the fuzzy matching and merge for each position DataFrame
final_rankings = pd.DataFrame()

for position in ['qb', 'rb', 'wr', 'te']:
    # Apply fuzzy matching
    rankings_dfs[position] = fuzzy_merge(rankings_dfs[position], github_df, 'Name', 'Player', threshold=80)
    
    # Perform the merge using the matched name
    merged_df = pd.merge(rankings_dfs[position], github_df, left_on='matched_name', right_on='Player', how='left')
    
    # Drop the columns we don't need after the merge
    merged_df.drop(columns=['matches', 'matched_name', 'Player'], inplace=True)
    
    # Append to the final DataFrame
    final_rankings = pd.concat([final_rankings, merged_df], ignore_index=True)

# Sort the final DataFrame by '1 QB' from highest to lowest
final_rankings.sort_values(by='1 QB', ascending=False, inplace=True)

# Reset index in the final DataFrame
final_rankings.reset_index(drop=True, inplace=True)

# Drop the specific columns named 'TEP', 'SF TEP', and 'SF'
columns_to_drop = ['Team_x']
final_rankings.drop(columns=columns_to_drop, errors='ignore', inplace=True)  # errors='ignore' will ignore any columns not found

# Rename:
final_rankings = final_rankings.rename(columns={'Team_y': 'Team',
                                                'Rank': 'FFA Rank',
                                                'Name': 'Player Name',
                                                '1 QB': 'Value'})

# Drop rows where 'FFA Rank' is NA
final_rankings.dropna(subset=['FFA Rank'], inplace=True)

# Convert 'FFA Rank' from float to int (after dropping NA to avoid type errors)
final_rankings['FFA Rank'] = final_rankings['FFA Rank'].astype(int)

# Keep these columns
final_rankings = final_rankings[['Value', 'Industry Rank', 'FFA Rank', 'Player Name', 'Rookie', 'Position', 'Team']]

# Sort by 1 QB
final_rankings.sort_values(by='Value', ascending=False, inplace=True)
sorted_final_rankings = final_rankings

#######################
##### Best Values #####
#######################

with tab_best:
    

    # Rename
    final_rankings = sorted_final_rankings

    # Ensure both 'Industry Rank' and 'FFA Rank' are integers
    final_rankings['Industry Rank'] = pd.to_numeric(final_rankings['Industry Rank'], errors='coerce')
    final_rankings['FFA Rank'] = pd.to_numeric(final_rankings['FFA Rank'], errors='coerce')

    # Drop rows where either 'Industry Rank' or 'FFA Rank' is NaN after conversion
    final_rankings.dropna(subset=['Industry Rank', 'FFA Rank'], inplace=True)

    # Now, let's calculate the rank difference
    final_rankings['Rank Difference'] = final_rankings['Industry Rank'] - final_rankings['FFA Rank']

    # Rename
    final_rankings = final_rankings.rename(columns={'Player Name': 'Player',
                                                    'Rank Difference': 'Diff',
                                                    'Position': 'Pos'})

    # Reorder
    final_rankings = final_rankings[['Value', 'Industry Rank', 'FFA Rank', 'Diff', 'Player', 'Rookie', 'Pos', 'Team']]

    # Save the top 50 & top 150
    final_rankings_t50 = final_rankings.head(50)
    final_rankings_t150 = final_rankings.head(150)

    # Create Chunks
    chunk_1 = final_rankings[0:30]
    chunk_2 = final_rankings[30:60]
    chunk_3 = final_rankings[60:90]
    chunk_4 = final_rankings[90:120]
    chunk_5 = final_rankings[120:150]

    chunk_1_top_rb_values = chunk_1[chunk_1['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_2_top_rb_values = chunk_2[chunk_2['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_3_top_rb_values = chunk_3[chunk_3['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_4_top_rb_values = chunk_4[chunk_4['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_5_top_rb_values = chunk_5[chunk_5['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_1_top_wr_values = chunk_1[chunk_1['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_2_top_wr_values = chunk_2[chunk_2['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_3_top_wr_values = chunk_3[chunk_3['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_4_top_wr_values = chunk_4[chunk_4['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_5_top_wr_values = chunk_5[chunk_5['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=False).head(3)
    chunk_1_top_te_values = chunk_1[chunk_1['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_2_top_te_values = chunk_2[chunk_2['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_3_top_te_values = chunk_3[chunk_3['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_4_top_te_values = chunk_4[chunk_4['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_5_top_te_values = chunk_5[chunk_5['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_1_top_qb_values = chunk_1[chunk_1['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_2_top_qb_values = chunk_2[chunk_2['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_3_top_qb_values = chunk_3[chunk_3['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_4_top_qb_values = chunk_4[chunk_4['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=False).head(1)
    chunk_5_top_qb_values = chunk_5[chunk_5['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=False).head(1)

    # Put all dfs into a list
    dfs = [chunk_1_top_rb_values, chunk_2_top_rb_values, chunk_3_top_rb_values, chunk_4_top_rb_values, chunk_5_top_rb_values,
           chunk_1_top_wr_values, chunk_2_top_wr_values, chunk_3_top_wr_values, chunk_4_top_wr_values, chunk_5_top_wr_values,
           chunk_1_top_te_values, chunk_2_top_te_values, chunk_3_top_te_values, chunk_4_top_te_values, chunk_5_top_te_values,
           chunk_1_top_qb_values, chunk_2_top_qb_values, chunk_3_top_qb_values, chunk_4_top_qb_values, chunk_5_top_qb_values]

    top_values = pd.concat(dfs).reset_index(drop=True)

    # Drop if Diff is less than 2
    top_values = top_values[top_values['Diff'] > 1]

    # Sort by value
    table_df = top_values.sort_values(by = 'Value', ascending=False).reset_index(drop=True)

    # Drop the value column
    table_df = table_df.drop('Value', axis=1)

    # Convert 'Rookie' column to integers: 1 for 'Rookie' (True), 0 for '' (False)
    table_df['Rookie'] = table_df['Rookie'].apply(lambda x: 1 if x == 'Rookie' else 0)

    # Define the mapping from position to number
    pos_to_number = {'RB': 0, 'WR': 1, 'QB': 2, 'TE': 3}

    # Apply the mapping to the 'Pos' column
    table_df['Pos'] = table_df['Pos'].apply(lambda x: pos_to_number[x])

    rookie_cmap = LinearSegmentedColormap.from_list(
        name='rookie_cmap', 
        colors=['white', 'gold'],
        N=2  # We only need two colors
    )

    pos_cmap = LinearSegmentedColormap.from_list(
        name='pos_cmap', 
        colors=['#A9DFBF', '#A9CCE3', '#D7BDE2', '#FAD7A0'],
        N=4  # We only need two colors
    )

    def format_rookie(value):
        return "Rookie" if value == 1 else ""

    # Define the inverse mapping from number to position for formatting
    number_to_pos = {0: 'RB', 1: 'WR', 2: 'QB', 3: 'TE'}

    def format_pos(value):
        # Use the get method to return the position string; if value is not found, return an empty string
        return number_to_pos.get(value, "")

    # Define a normalization function for rank values
    def normalize_rank(value, cutoff=50, max_value=100):
        # Normalize the value based on the cutoff to a range of [0, 1]
        # Values above the cutoff will be closer to 1, and below the cutoff closer to 0
        return min(max((value - 1) / (cutoff - 1), 0), 1)

    def cmap_map(function, cmap):
        """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
        This routine will break any discontinuous points in a colormap.
        """
        cdict = cmap._segmentdata
        step_dict = {}
        # Firt get the list of points where the segments start or end
        for key in ('red', 'green', 'blue'):
            step_dict[key] = list(map(lambda x: x[0], cdict[key]))
        step_list = sum(step_dict.values(), [])
        step_list = np.array(list(set(step_list)))
        # Then compute the LUT, and apply the function to the LUT
        reduced_cmap = lambda step : np.array(cmap(step)[0:3])
        old_LUT = np.array(list(map(reduced_cmap, step_list)))
        new_LUT = np.array(list(map(function, old_LUT)))
        # Now try to make a minimal segment definition of the new LUT
        cdict = {}
        for i, key in enumerate(['red','green','blue']):
            this_cdict = {}
            for j, step in enumerate(step_list):
                if step in step_dict[key]:
                    this_cdict[step] = new_LUT[j, i]
                elif new_LUT[j,i] != old_LUT[j, i]:
                    this_cdict[step] = new_LUT[j, i]
            colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
            colorvector.sort()
            cdict[key] = colorvector

        return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

    # Create a custom colormap for ranks
    rank_cmap = LinearSegmentedColormap.from_list(
        name="rank_cmap", 
        colors=["#f2fbd2", "#93d3ab"],  # Use a gradient from light to more intense
        N=256
    )

    # Create a custom colormap for Diff
    diff_cmap = LinearSegmentedColormap.from_list(
        name="rank_cmap", 
        colors=["#f0fced", "#5ede37"],  # Use a gradient from light to more intense
        N=256
    )

    def rank_formatter(value):
        # Assuming the ranks are integers, we don't format them as percentages
        # Instead, we can return them as they are, or apply any other formatting you need
        return str(value)

    # Define a function to create a normed colormap for ranks
    def normed_rank_cmap(series, cmap, cutoff):
        max_value = series.max()
        normed_data = series.apply(lambda x: normalize_rank(x, cutoff, max_value))
        return normed_cmap(normed_data, cmap=cmap)

    # Apply the custom colormap functions to the rank columns
    norm_industry = normed_rank_cmap(table_df["Industry Rank"], rank_cmap, 50)
    norm_ffa = normed_rank_cmap(table_df["FFA Rank"], rank_cmap, 50)
    norm_diff = normed_rank_cmap(table_df["Diff"], diff_cmap, 50)

    # Now, use this function to create lighter colormaps
    light_reds = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.Reds)
    light_blues = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.Blues)
    light_rdylgn = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.Reds)

    # Then use these lightened colormaps with your normed_cmap functions
    norm_ffa = normed_cmap(table_df["FFA Rank"], cmap=light_reds, num_stds=3)
    norm_industry = normed_cmap(table_df["Industry Rank"], cmap=light_blues, num_stds=3)
    norm_diff = normed_cmap(table_df["Diff"], cmap=diff_cmap, num_stds=2)


    # Define column definitions for creating the table with specific properties
    # circle_pad = 0.1  # Adjust padding to make circles smaller
    text_font_size = 10  # Adjust font size for the content

    col_defs = [
            ColumnDefinition(
            name="Value",
            textprops={"ha": "center"},
            width=0.9,
        ),
        ColumnDefinition(
            name="Rookie",
            formatter=format_rookie,  # Use the custom formatter
            cmap=rookie_cmap,
            width=0.9,
            textprops={"ha": "center", "va": "center"},  # Center both horizontally and vertically
        ),
        ColumnDefinition(
            name="Pos",
            formatter=format_pos,  # Use the custom formatter
            cmap=pos_cmap,
            textprops={"ha": "center"},
            width=0.7,
        ),
            ColumnDefinition(
            name="Player",
            textprops={"ha": "center"},
            width=2.2,
        ),
        ColumnDefinition(
            name="Team",
            textprops={"ha": "center"},
            width=0.7,
        ),
        ColumnDefinition(
            name="Diff",
            formatter=lambda x: str(int(x)),  # Convert to integer and then to string
            textprops={"ha": "center"},
            cmap=norm_diff,
            width=0.6,
            border='right',  # This adds a border to the right side of the "Diff" column
        ),
        ColumnDefinition(
            name="FFA Rank",
            formatter=lambda x: str(int(x)),  # Convert to integer and then to string
            cmap=norm_ffa,
            width=1,
            textprops={"ha": "center"},
        ),
        ColumnDefinition(
            name="Industry Rank",
            formatter=lambda x: str(int(x)),  # Convert to integer and then to string
            cmap=norm_industry,
            width=1,
            textprops={"ha": "center"},
        ),
    ]

    # Create the figure and plot the table
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.axis('off')  # Hide the axes

    # Instantiate the Table object with the dataframe and column definitions
    table = Table(
        table_df,
        index_col = 'Industry Rank',
        column_definitions=col_defs,
        ax=ax,
        textprops={"fontsize": 10},
        row_divider_kw={"linewidth": 0.5, "linestyle": (0, (1, 5))},
        column_border_kw={"linewidth": 0.5, "linestyle": "-"},  # Set a thinner border here

    )

    # Display the table
    plt.title('FFA vs Industry Best Values', fontsize=16, fontweight='bold')
    st.pyplot(fig)


########################
##### Worst Values #####
########################
with tab_worst:
    
    # Rename
    final_rankings = sorted_final_rankings

    # Ensure both 'Industry Rank' and 'FFA Rank' are integers
    final_rankings['Industry Rank'] = pd.to_numeric(final_rankings['Industry Rank'], errors='coerce')
    final_rankings['FFA Rank'] = pd.to_numeric(final_rankings['FFA Rank'], errors='coerce')

    # Drop rows where either 'Industry Rank' or 'FFA Rank' is NaN after conversion
    final_rankings.dropna(subset=['Industry Rank', 'FFA Rank'], inplace=True)

    # Now, let's calculate the rank difference
    final_rankings['Rank Difference'] = final_rankings['Industry Rank'] - final_rankings['FFA Rank']

    # Rename
    final_rankings = final_rankings.rename(columns={'Player Name': 'Player',
                                                    'Rank Difference': 'Diff',
                                                    'Position': 'Pos'})

    # Reorder
    final_rankings = final_rankings[['Value', 'Industry Rank', 'FFA Rank', 'Diff', 'Player', 'Rookie', 'Pos', 'Team']]

    # Save the top 50 & top 150
    final_rankings_t50 = final_rankings.head(50)
    final_rankings_t150 = final_rankings.head(150)

    # Create Chunks
    chunk_1 = final_rankings[0:30]
    chunk_2 = final_rankings[30:60]
    chunk_3 = final_rankings[60:90]
    chunk_4 = final_rankings[90:120]
    chunk_5 = final_rankings[120:150]

    chunk_1_top_rb_values = chunk_1[chunk_1['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_2_top_rb_values = chunk_2[chunk_2['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_3_top_rb_values = chunk_3[chunk_3['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_4_top_rb_values = chunk_4[chunk_4['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_5_top_rb_values = chunk_5[chunk_5['Pos'] == 'RB'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_1_top_wr_values = chunk_1[chunk_1['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_2_top_wr_values = chunk_2[chunk_2['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_3_top_wr_values = chunk_3[chunk_3['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_4_top_wr_values = chunk_4[chunk_4['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_5_top_wr_values = chunk_5[chunk_5['Pos'] == 'WR'].sort_values(by = 'Diff', ascending=True).head(3)
    chunk_1_top_te_values = chunk_1[chunk_1['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_2_top_te_values = chunk_2[chunk_2['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_3_top_te_values = chunk_3[chunk_3['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_4_top_te_values = chunk_4[chunk_4['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_5_top_te_values = chunk_5[chunk_5['Pos'] == 'TE'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_1_top_qb_values = chunk_1[chunk_1['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_2_top_qb_values = chunk_2[chunk_2['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_3_top_qb_values = chunk_3[chunk_3['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_4_top_qb_values = chunk_4[chunk_4['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=True).head(1)
    chunk_5_top_qb_values = chunk_5[chunk_5['Pos'] == 'QB'].sort_values(by = 'Diff', ascending=True).head(1)

    # Put all dfs into a list
    dfs = [chunk_1_top_rb_values, chunk_2_top_rb_values, chunk_3_top_rb_values, chunk_4_top_rb_values, chunk_5_top_rb_values,
           chunk_1_top_wr_values, chunk_2_top_wr_values, chunk_3_top_wr_values, chunk_4_top_wr_values, chunk_5_top_wr_values,
           chunk_1_top_te_values, chunk_2_top_te_values, chunk_3_top_te_values, chunk_4_top_te_values, chunk_5_top_te_values,
           chunk_1_top_qb_values, chunk_2_top_qb_values, chunk_3_top_qb_values, chunk_4_top_qb_values, chunk_5_top_qb_values]

    top_values = pd.concat(dfs).reset_index(drop=True)

    # Drop if Diff is less than 2
    top_values = top_values[top_values['Diff'] < -1]

    # Sort by value
    table_df = top_values.sort_values(by = 'Value', ascending=False).reset_index(drop=True)

    # Drop the value column
    table_df = table_df.drop('Value', axis=1)

    # Convert 'Rookie' column to integers: 1 for 'Rookie' (True), 0 for '' (False)
    table_df['Rookie'] = table_df['Rookie'].apply(lambda x: 1 if x == 'Rookie' else 0)

    # Define the mapping from position to number
    pos_to_number = {'RB': 0, 'WR': 1, 'QB': 2, 'TE': 3}

    # Apply the mapping to the 'Pos' column
    table_df['Pos'] = table_df['Pos'].apply(lambda x: pos_to_number[x])

    rookie_cmap = LinearSegmentedColormap.from_list(
        name='rookie_cmap', 
        colors=['white', 'gold'],
        N=2  # We only need two colors
    )

    pos_cmap = LinearSegmentedColormap.from_list(
        name='pos_cmap', 
        colors=['#A9DFBF', '#A9CCE3', '#D7BDE2', '#FAD7A0'],
        N=4  # We only need two colors
    )

    def format_rookie(value):
        return "Rookie" if value == 1 else ""

    # Define the inverse mapping from number to position for formatting
    number_to_pos = {0: 'RB', 1: 'WR', 2: 'QB', 3: 'TE'}

    def format_pos(value):
        # Use the get method to return the position string; if value is not found, return an empty string
        return number_to_pos.get(value, "")

    # Define a normalization function for rank values
    def normalize_rank(value, cutoff=50, max_value=100):
        # Normalize the value based on the cutoff to a range of [0, 1]
        # Values above the cutoff will be closer to 1, and below the cutoff closer to 0
        return min(max((value - 1) / (cutoff - 1), 0), 1)

    def cmap_map(function, cmap):
        """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
        This routine will break any discontinuous points in a colormap.
        """
        cdict = cmap._segmentdata
        step_dict = {}
        # Firt get the list of points where the segments start or end
        for key in ('red', 'green', 'blue'):
            step_dict[key] = list(map(lambda x: x[0], cdict[key]))
        step_list = sum(step_dict.values(), [])
        step_list = np.array(list(set(step_list)))
        # Then compute the LUT, and apply the function to the LUT
        reduced_cmap = lambda step : np.array(cmap(step)[0:3])
        old_LUT = np.array(list(map(reduced_cmap, step_list)))
        new_LUT = np.array(list(map(function, old_LUT)))
        # Now try to make a minimal segment definition of the new LUT
        cdict = {}
        for i, key in enumerate(['red','green','blue']):
            this_cdict = {}
            for j, step in enumerate(step_list):
                if step in step_dict[key]:
                    this_cdict[step] = new_LUT[j, i]
                elif new_LUT[j,i] != old_LUT[j, i]:
                    this_cdict[step] = new_LUT[j, i]
            colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
            colorvector.sort()
            cdict[key] = colorvector

        return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

    # Create a custom colormap for ranks
    rank_cmap = LinearSegmentedColormap.from_list(
        name="rank_cmap", 
        colors=["#f2fbd2", "#93d3ab"],  # Use a gradient from light to more intense
        N=256
    )

    # Create a custom colormap for Diff
    diff_cmap = LinearSegmentedColormap.from_list(
        name="rank_cmap", 
        colors=["#ff1e1e", "#ffeaea"],  # Use a gradient from light to more intense
        N=256
    )

    def rank_formatter(value):
        # Assuming the ranks are integers, we don't format them as percentages
        # Instead, we can return them as they are, or apply any other formatting you need
        return str(value)

    # Define a function to create a normed colormap for ranks
    def normed_rank_cmap(series, cmap, cutoff):
        max_value = series.max()
        normed_data = series.apply(lambda x: normalize_rank(x, cutoff, max_value))
        return normed_cmap(normed_data, cmap=cmap)

    # Apply the custom colormap functions to the rank columns
    norm_industry = normed_rank_cmap(table_df["Industry Rank"], rank_cmap, 50)
    norm_ffa = normed_rank_cmap(table_df["FFA Rank"], rank_cmap, 50)
    norm_diff = normed_rank_cmap(table_df["Diff"], diff_cmap, 50)

    # Now, use this function to create lighter colormaps
    light_reds = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.Reds)
    light_blues = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.Blues)
    light_rdylgn = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.Reds)

    # Then use these lightened colormaps with your normed_cmap functions
    norm_ffa = normed_cmap(table_df["FFA Rank"], cmap=light_reds, num_stds=3)
    norm_industry = normed_cmap(table_df["Industry Rank"], cmap=light_blues, num_stds=3)
    norm_diff = normed_cmap(table_df["Diff"], cmap=diff_cmap, num_stds=2)


    # Define column definitions for creating the table with specific properties
    # circle_pad = 0.1  # Adjust padding to make circles smaller
    text_font_size = 10  # Adjust font size for the content

    col_defs = [
            ColumnDefinition(
            name="Value",
            textprops={"ha": "center"},
            width=0.9,
        ),
        ColumnDefinition(
            name="Rookie",
            formatter=format_rookie,  # Use the custom formatter
            cmap=rookie_cmap,
            width=0.9,
            textprops={"ha": "center", "va": "center"},  # Center both horizontally and vertically
        ),
        ColumnDefinition(
            name="Pos",
            formatter=format_pos,  # Use the custom formatter
            cmap=pos_cmap,
            textprops={"ha": "center"},
            width=0.7,
        ),
            ColumnDefinition(
            name="Player",
            textprops={"ha": "center"},
            width=2.2,
        ),
        ColumnDefinition(
            name="Team",
            textprops={"ha": "center"},
            width=0.7,
        ),
        ColumnDefinition(
            name="Diff",
            formatter=lambda x: str(int(x)),  # Convert to integer and then to string
            textprops={"ha": "center"},
            cmap=norm_diff,
            width=0.6,
            border='right',  # This adds a border to the right side of the "Diff" column
        ),
        ColumnDefinition(
            name="FFA Rank",
            formatter=lambda x: str(int(x)),  # Convert to integer and then to string
            cmap=norm_ffa,
            width=1,
            textprops={"ha": "center"},
        ),
        ColumnDefinition(
            name="Industry Rank",
            formatter=lambda x: str(int(x)),  # Convert to integer and then to string
            cmap=norm_industry,
            width=1,
            textprops={"ha": "center"},
        ),
    ]

    # Create the figure and plot the table
    fig, ax = plt.subplots(figsize=(10, 15))
    ax.axis('off')  # Hide the axes

    # Instantiate the Table object with the dataframe and column definitions
    table = Table(
        table_df,
        index_col = 'Industry Rank',
        column_definitions=col_defs,
        ax=ax,
        textprops={"fontsize": 10},
        row_divider_kw={"linewidth": 0.5, "linestyle": (0, (1, 5))},
        column_border_kw={"linewidth": 0.5, "linestyle": "-"},  # Set a thinner border here

    )

    # Display the table
    plt.title('FFA vs Industry Worst Values', fontsize=16, fontweight='bold')
    st.pyplot(fig)
