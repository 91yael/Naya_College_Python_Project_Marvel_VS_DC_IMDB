import pandas as pd
import os 
import re
import numpy as np
import seaborn as sns

from kaggle.api.kaggle_api_extended import KaggleApi
from matplotlib import pyplot as plt


# Get user input for download path
download_path = input("\033[1m Enter the path where you want to download the dataset: \033[0m")  # Example: c:\Users\97254

# Add '\.kaggle' to the download path
download_path = os.path.join(download_path, r'.kaggle')

### Get data set ###

api = KaggleApi()
api.authenticate()  

dataset_slug = 'hetulmehta/marvel-vs-dc-imdb-dataset'


api.dataset_download_files(dataset_slug, path=download_path, unzip=True)

files = os.listdir(download_path)
csv_file_path = os.path.join(download_path, files[1])
print(csv_file_path)
df = pd.read_csv(csv_file_path)
print("\033[1mCSV file loaded successfully!\033[0m")

# Another option:

# csv_file_path = r"C:\Users\97254\OneDrive\Desktop\Naya College\Python\Naya_College_Python_Project_Marvel_VS_DC_IMDB\Marvel_DC_imdb.csv"
# df = pd.read_csv(csv_file_path)

df.head((1))

#Get info
df.info()

# Delete columns
df.drop(columns=["Metascore", "USA_Gross"], inplace=True)
df.columns

# Rename columns
df.rename(columns={df.columns[0]: 'Id'}, inplace=True)
df.columns = df.columns.str.replace(r'RunTime', 'Run_Time_Minutes')
df.columns = df.columns.str.replace(r' ', '')
df.columns

# Convert all column data types to string
df = df.astype(str)
print(df.dtypes)

# Remove not a digit character
def remove_not_digit(val):
  new_val = re.sub(r'\D', '', str(val))
  return new_val


# Remove non-digit characters from specified columns
columns_to_fix = ['Run_Time_Minutes', 'Votes', 'IMDB_Score', 'Rating','Year']
for col in columns_to_fix:
    df[col] = df[col].apply(remove_not_digit)

# Remove not a digit character and fix the values in the year column
def format_year(Year):
  if len(Year)>4:
    start_year=Year[:4]
    end_year=Year[4:]
    return start_year+"-"+end_year
  else:
    return Year

df['Year'].apply(format_year)

# Index columns
index_columns = [col for col in columns_to_fix if col != 'Year']

#Fill empty values
df[index_columns] = df[index_columns].replace('nan', np.nan)
print(df[index_columns].isnull().sum())

#Get summary statistics
print(df[index_columns].describe())

index_columns.append('Category')

df[index_columns] = df[index_columns].replace('', np.nan)
df[index_columns] = df[index_columns].dropna(axis=0)
df = df.dropna(subset=index_columns, how='all')
df[index_columns]

selected_columns = df[index_columns]

# Group the dataframe by 'Category'
grouped_df = selected_columns.groupby('Category')

# Describe for each group
df_describe = grouped_df.describe().T
print(df_describe)


### Visualization ###

# Count Plot for Number of Movies in Each Category
plt.figure(figsize=(8, 6))
sns.countplot(x='Category', data=df[index_columns])
plt.title('Number of Movies in Each Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.grid(axis='y')  
plt.show()

# Distribution of rating by Ctegory
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Rating', data=df[index_columns])
plt.title('Distribution of Rating by Category')
plt.xlabel('Category')
plt.ylabel('Rating')
plt.grid(True)  

# Top 10 movies by IMDB_Score
max_imdb_per_movie = df.groupby('Movie')['IMDB_Score'].max().reset_index()
top_movies = max_imdb_per_movie.sort_values('IMDB_Score', ascending=False)
top_movies = top_movies.head(10)
plt.figure(figsize=(15,3))
sns.barplot(x='IMDB_Score', y='Movie', data=top_movies, palette='viridis', orient='h', hue='Movie', legend=False)
plt.title('Best Movies according to Votes given by Users',weight='bold')
plt.xlabel('Weighted Average Score',weight='bold')
plt.ylabel('Movie Title',weight='bold')
plt.show()

# Save DataFrame to CSV
csv_file_path = os.path.join(download_path, 'processed_data.csv')
df.to_csv(csv_file_path, index=False)  
print(f"CSV file saved successfully at: {csv_file_path}")

# Save DataFrame to JSON
json_file_path = os.path.join(download_path, 'processed_data.json')
df.to_json(json_file_path, orient='records') 
print(f"JSON file saved successfully at: {json_file_path}")
