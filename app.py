#Initialization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import linregress
import streamlit as st

games = pd.read_csv('games.csv')


#Pre-processing Step
games.columns = games.columns.str.lower()

games = games[games['name'].notna()]
games = games[~games['name'].str.contains('.hack//')]

games_year_clean = games.dropna(subset=['year_of_release'])

games_critic_score_clean = games.dropna(subset=['critic_score'])

games_user_score_clean = games.dropna(subset=['user_score'])
games_user_score_clean = games.drop(games[games['user_score'] == 'tbd'].index)

games_rating_clean = games.dropna(subset=['rating'])


#Calculation of Total Global Sales for Each Game
total_games_sales = games.groupby('name')[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum().reset_index()

total_games_sales['ww_total'] = total_games_sales[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)

st.write(total_games_sales)


#Exploration of CumReleases
# Create a histogram of the number of games released per year
plt.figure(figsize=(16,6))
plt.hist(games_year_clean['year_of_release'], bins=range(int(games_year_clean['year_of_release'].min()), int(games_year_clean['year_of_release'].max()+1)), align='left')
plt.xlabel('Year of Release')
plt.ylabel('Number of Games')
plt.title('Number of Games Released per Year')
plt.xticks(rotation=45)

# Add data labels
for i, v in zip(games_year_clean['year_of_release'].value_counts().sort_index().index, games_year_clean['year_of_release'].value_counts().sort_index().values):
    plt.text(i, v, str(v), ha='center')

plt.show()


#Exploration of Sales by Patform
total_platform_sales = games.groupby('platform')[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum().reset_index()

total_platform_sales['ww_total'] = total_platform_sales[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)

st.write(total_platform_sales)

# Sort the data from highest to lowest
sorted_data = sorted(zip(total_platform_sales['platform'], total_platform_sales['ww_total']), key=lambda x: x[1], reverse=True)

# Create a horizontal bar chart
fig, ax = plt.subplots()
ax.barh([x[0] for x in sorted_data], [x[1] for x in sorted_data])

# Add labels to each bar
for i, v in enumerate(sorted_data):
    ax.text(v[1], i - 0.15, str(v[1]), color='white', fontweight='bold')

# Set the title and labels
ax.set_xlabel('Sales (millions)')
ax.set_ylabel('Platforms')
ax.set_title('Total Sales by Platform (2017)')
ax.set_yticks([x[0] for x in sorted_data])
ax.set_yticklabels([x[0] for x in sorted_data])

# Adjust the size of the figure
fig.set_size_inches(11, 8)

# Display the plot
plt.show()



# Find platforms that used to be popular but now have zero sales
popular_platforms = games[(games['na_sales'] > 0) | (games['eu_sales'] > 0) | (games['jp_sales'] > 0) | (games['other_sales'] > 0)]['platform'].unique()
zero_sales_platforms = games[(games['na_sales'] == 0) & (games['eu_sales'] == 0) & (games['jp_sales'] == 0) & (games['other_sales'] == 0)]['platform'].unique()
former_popular_platforms = [platform for platform in popular_platforms if platform in zero_sales_platforms]

# Calculate the total sales for each platform
games['total_sales'] = games[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)



# fill NaN values with 0 (or any other value that makes sense for your data)
games['year_of_release'].fillna(0, inplace=True)
games['total_sales'].fillna(0, inplace=True)

# convert 'year_of_release' and 'total_sales' to int
games['year_of_release'] = games['year_of_release'].astype(int)
games['total_sales'] = games['total_sales'].astype(int)

games = games[games['year_of_release']!= 0]

# group by 'year_of_release' and 'platform', and sum 'total_sales'
grouped_df = games.groupby(['year_of_release', 'platform'])['total_sales'].sum().reset_index()

# plot 'total_sales' by 'year_of_release' with 'platform' as separate series
for platform in games['platform'].unique():
    platform_df = grouped_df[grouped_df['platform'] == platform]
    plt.plot(platform_df['year_of_release'], platform_df['total_sales'], label=platform)

plt.xlabel('Year of Release')
plt.ylabel('Total Sales')
plt.title('Total Sales by Year of Release and Platform')
plt.show()



games_slice = games[games['year_of_release'] >= 1995]

games_slice.groupby('platform')['total_sales'].sum().nlargest(10)

from scipy.stats import linregress

# Assuming 'games' is your original dataframe
games_by_year = games_slice.groupby(['platform', 'year_of_release'])['total_sales'].sum().unstack().reset_index()
games_by_year.set_index('platform', inplace=True)

games_by_year['slope'] = games_by_year.apply(lambda x: linregress(range(len(x)), x)[0] if len(x) >= 2 else np.nan, axis=1)

st.write(games_by_year)



#Statistical
# Fill NaN values in the 'total_sales' column with 0
games['total_sales'].fillna(0, inplace=True)

# Convert the 'total_sales' column to integer
games['total_sales'] = games['total_sales'].astype(int)

# Group the data by platform
grouped_data = games.groupby('platform')

# Create a list of arrays, one for each platform
data_to_plot = [group['total_sales'] for _, group in grouped_data]

# Create a list of labels, one for each platform
labels = [platform for platform, _ in grouped_data]

# Create a box plot of the total sales by platform
plt.figure(figsize=(12,6))
plt.boxplot(data_to_plot, labels=labels)
plt.xlabel('')
plt.ylabel('Global Sales (millions)')
plt.title('Global Sales of Video Games by Platform')
plt.show()


# Group the data by platform and calculate the total sales for each region
platform_sales = games.groupby('platform')[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum().reset_index()

# Convert the sales columns to integer
platform_sales[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']] = platform_sales[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].astype(int)

# Calculate the total sales for each platform
platform_sales['total_sales'] = platform_sales[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)

# Create a stacked bar chart of the sales for each platform broken down by region
plt.figure(figsize=(15,6))
plt.bar(platform_sales['platform'], platform_sales['na_sales'], label='North America', color='red')
plt.bar(platform_sales['platform'], platform_sales['eu_sales'], label='Europe', color='green')
plt.bar(platform_sales['platform'], platform_sales['jp_sales'], label='Japan', color='blue')
plt.bar(platform_sales['platform'], platform_sales['other_sales'], label='Other', color='orange')
plt.xlabel('Platform')
plt.ylabel('Sales (millions)')
plt.title('Sales of Video Games by Platform and Region')
plt.legend()
plt.show()


x360_games = games[games['platform'] == 'X360']

# Display the resulting dataframe
st.write(x360_games)

# Create a single dataframe with both critic scores and user scores
x360_games = x360_games.replace('tbd', np.nan).dropna()
scores = x360_games[['critic_score', 'user_score', 'total_sales']].copy()

# Remove NaN and Inf values from the scores dataframe
scores = scores.replace([np.inf, -np.inf], np.nan).dropna()

# Divide user_score by e+10 and multiply by 10
scores['user_score'] = scores['user_score'].astype(float)
scores['user_score'] = scores['user_score'] *10

plt.figure(figsize=(12, 6))

# Create a scatter plot of both critic scores and user scores vs. sales
sns.scatterplot(x='critic_score', y='total_sales', data=scores, color='#406d9c', label='Critic Reviews', alpha=0.5)
sns.scatterplot(x='user_score', y='total_sales', data=scores, color='#cf850c', label='User Reviews', alpha=0.5)

# Add a legend
plt.legend()

# Add axis labels and a title
plt.xlabel('Score')
plt.ylabel('Sales (millions)')
plt.title('X360 Sales vs. Reviews')

# Set the x-axis limits to the minimum and maximum values in the data
plt.xlim(np.nanmin([scores['critic_score'].min(), scores['user_score'].min()]),
         np.nanmax([scores['critic_score'].max(), scores['user_score'].max()]))

# Calculate the slope and r-squared for critic scores
critic_regression = linregress(scores['critic_score'], scores['total_sales'])
critic_slope = critic_regression.slope
critic_r_squared = critic_regression.rvalue ** 2

# Calculate the slope and r-squared for user scores
user_regression = linregress(scores['user_score'], scores['total_sales'])
user_slope = user_regression.slope
user_r_squared = user_regression.rvalue ** 2

# Add slope lines and r-squared values
plt.plot(scores['critic_score'], critic_slope * scores['critic_score'] + critic_regression.intercept, color='#406d9c', linestyle='--', label=f'Critic Reviews Slope: {critic_slope:.2f}, R²: {critic_r_squared:.2f}')
plt.plot(scores['user_score'], user_slope * scores['user_score'] + user_regression.intercept, color='#cf850c', linestyle='--', label=f'User Reviews Slope: {user_slope:.2f}, R²: {user_r_squared:.2f}')

# Show the plot
plt.show()


st.write('The r-squared value for User Reviews to Sales is : ', round(user_r_squared,2))
st.write('The r-squared value for Critic Reviews to Sales is : ', round(critic_r_squared,2))



# Select the columns of interest
cols = ['na_sales', 'eu_sales', 'jp_sales', 'other_sales', 'total_sales', 'user_score', 'critic_score']

# Create a new DataFrame with the selected columns
df = games[cols].copy()

# Drop rows with NaN values
df = df.dropna()

# Replace 'tbd' with 0
df = df.replace('tbd', 0)

# Convert the columns to float
df = df.astype(float)

# Create a correlation matrix
corr_matrix = df.corr()

# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()



# Group the data by game name and platform, and sum the total sales
grouped_df = games.groupby(['name', 'platform'])['total_sales'].sum().reset_index()

# Pivot the data to create separate columns for each platform
pivot_df = grouped_df.pivot(index='name', columns='platform', values='total_sales')

# Display the pivot table
st.write(pivot_df)



genre_counts = games['genre'].value_counts()

# Create a bar chart
plt.figure(figsize=(12, 6))
genre_counts.plot(kind='bar')
plt.title('Distribution of Games by Genre (1980 – 2016)')
plt.xlabel('Genre')
plt.ylabel('Number of Games')
plt.xticks(rotation=0)

# Add data labels to the bars
for i, v in enumerate(genre_counts):
    plt.text(i, v + 0.5, str(v), ha='center')

plt.show()



# Group by genre and calculate sum of total sales
genre_sales = games.groupby('genre')['total_sales'].sum().reset_index()

# Sort by total sales in descending order
genre_sales = genre_sales.sort_values('total_sales', ascending=False)

st.write(genre_sales)



# Pivot the data to get total sales by genre and year
genre_year_sales = games.pivot_table(index=['genre', 'year_of_release'], values='total_sales', aggfunc='sum').reset_index()

# Sort by total sales in descending order
genre_year_sales = genre_year_sales.sort_values('total_sales', ascending=False)

# Get the top 10 most profitable genres
top_genres = genre_year_sales.groupby('genre')['total_sales'].sum().reset_index().sort_values('total_sales', ascending=False).head(10)

# Create a figure with a specified width
plt.figure(figsize=(12, 6))  # adjust the width and height as needed

# Plot the top 10 genres over time
sns.lineplot(x='year_of_release', y='total_sales', hue='genre', data=genre_year_sales[genre_year_sales['genre'].isin(top_genres['genre'])])

plt.title('Top 10 Most Profitable Genres Over Time')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.legend(title='Genre')

plt.show()




# assume 'games' is a Pandas DataFrame with columns 'na_sales', 'eu_sales', 'jp_sales', 'other_sales'

# create separate DataFrames for each region
north_america_games = games[games['na_sales'] > 0].copy()
europe_games = games[games['eu_sales'] > 0].copy()
jp_games = games[games['jp_sales'] > 0].copy()
other_games = games[games['other_sales'] > 0].copy()

# drop rows with NaN, 'tbd', or 0 sales in each region
north_america_games = north_america_games.replace('tbd', np.nan).dropna()
europe_games = europe_games.replace('tbd', np.nan).dropna()
jp_games = jp_games.replace('tbd', np.nan).dropna()
other_games = other_games.replace('tbd', np.nan).dropna()


# assume north_america_games, europe_games, jp_games, other_games are DataFrames

regions = {
    'north_america': {'df': north_america_games, 'sales_col': 'na_sales'},
    'europe': {'df': europe_games, 'sales_col': 'eu_sales'},
    'jp': {'df': jp_games, 'sales_col': 'jp_sales'},
    'other': {'df': other_games, 'sales_col': 'other_sales'}
}


# create a Streamlit dashboard
st.title("Games Sales Dashboard")

# add a region selector
region_selector = st.selectbox("Select a region:", list(regions.keys()), key='region_selector')

# get the selected region's DataFrame and sales column
region_df = regions[region_selector]['df']
sales_col = regions[region_selector]['sales_col']

# add a sales column selector
sales_cols = ['na_sales', 'eu_sales', 'jp_sales', 'other_sales']
sales_col_selector = st.selectbox("Select a sales column:", sales_cols)

# create a function to calculate top platforms by sales
def top_platforms_by_sales(df, sales_col):
    return df.groupby('platform')[sales_col].sum().sort_values(ascending=False).head(5)

# display the top platforms by sales
st.write(f"Top 5 platforms by games sales in {region_selector}:")
st.dataframe(top_platforms_by_sales(region_df, sales_col))

# create a function to calculate top platforms by number of games
def top_platforms_by_games(df):
    return df['platform'].value_counts().head(5)

# display the top platforms by number of games
st.write(f"Top 5 platforms by number of games in {region_selector}:")
st.dataframe(top_platforms_by_games(region_df))

# create a function to calculate top genres by region
def top_genres_by_region(df):
    return df['genre'].value_counts().head(5)

# display the top genres by region
st.write(f"Top 5 genres in {region_selector}:")
st.dataframe(top_genres_by_region(region_df))

# Create a function to analyze ESRB ratings and sales
def analyze_esrb_ratings(df, sales_col):
    # Calculate cumulative sales for each rating
    esrb_ratings = df.groupby('rating')[sales_col].sum().reset_index()
        
    # Create a bar chart to visualize the results
    fig, ax = plt.subplots()
    ax.bar(esrb_ratings['rating'], esrb_ratings[sales_col])
    ax.set_xlabel('ESRB Rating')
    ax.set_ylabel('Cumulative Sales')
    ax.set_title('ESRB Ratings vs. Cumulative Sales')
        
    return fig

# Add a new section to the dashboard
st.write(f"ESRB Ratings vs. Cumulative Sales in {region_selector}:")
esrb_fig = analyze_esrb_ratings(region_df, sales_col)
st.pyplot(esrb_fig)




st.title("Hypothesis Testing Examples")

# 1. Hypothesis Test: Average user ratings for Xbox One and PC platforms
st.header("Hypothesis Test 1: Xbox One vs PC")
st.write("**Null Hypothesis:** Average user ratings for Xbox One and PC platforms are the same.")
st.write("**Alternative Hypothesis:** Average user ratings for Xbox One and PC platforms are different.")

xbox_ratings = games_user_score_clean[games_user_score_clean["platform"] == "XOne"]["user_score"]
pc_ratings = games_user_score_clean[games_user_score_clean["platform"] == "PC"]["user_score"]

t_statistic, p_value = stats.ttest_ind(xbox_ratings, pc_ratings)
st.write(f"**T-statistic:** {t_statistic:.2f}")
st.write(f"**P-value:** {p_value:.3f}")

if p_value < 0.05:
    st.write("**Conclusion:** Reject the null hypothesis. There is statistically significant evidence to suggest that average user ratings for Xbox One and PC platforms are different.")
else:
    st.write("**Conclusion:** Fail to reject the null hypothesis. There is not enough evidence to suggest that average user ratings for Xbox One and PC platforms are different.")

# 2. Hypothesis Test: Average user ratings for Action and Sports genres
st.header("Hypothesis Test 2: Action vs Sports")
st.write("**Null Hypothesis:** Average user ratings for Action and Sports genres are the same.")
st.write("**Alternative Hypothesis:** Average user ratings for Action and Sports genres are different.")

action_ratings = games_user_score_clean[games_user_score_clean["genre"] == "Action"]["user_score"]
sports_ratings = games_user_score_clean[games_user_score_clean["genre"] == "Sports"]["user_score"]

t_statistic, p_value = stats.ttest_ind(action_ratings, sports_ratings)
st.write(f"**T-statistic:** {t_statistic:.2f}")
st.write(f"**P-value:** {p_value:.3f}")

if p_value < 0.05:
    st.write("**Conclusion:** Reject the null hypothesis. There is statistically significant evidence to suggest that average user ratings for Action and Sports genres are different.")
else:
    st.write("**Conclusion:** Fail to reject the null hypothesis. There is not enough evidence to suggest that average user ratings for Action and Sports genres are different.")
