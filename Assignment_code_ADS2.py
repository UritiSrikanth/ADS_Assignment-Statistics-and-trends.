# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 23:25:16 2023

@author: 91905
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    df = pd.read_csv(
        'D:\\Datasets\\Dataset\\P_Data_Extract_From_World_Development_Indicators\\Metadata.csv')
    return df
    """
    Load data from a CSV file into a pandas DataFrame.
    Returns:
        df (pandas.DataFrame): DataFrame containing the loaded data.
    """


# Load the data into a DataFrame
df = load_data()

# Create a DataFrame with years as columns
df_years = df.set_index(
    ['Country Name', 'Country Code', 'Series Name', 'Series Code']).T

# Create a DataFrame with countries as columns
df_countries = df.set_index(['Series Name', 'Series Code']).T

df_years = df_years.iloc[4:]

df_countries = df_countries.iloc[:-1]

# Subset the data for the selected countries and series
countries = ['Canada', 'United States']
series = [
    'Fuel exports (% of merchandise exports)',
    'Agricultural raw materials exports (% of merchandise exports)',
    'Arms exports (SIPRI trend indicator values)',
    'Labor force with intermediate education, female (% of female working-age population with intermediate education)',
    'Labor force with intermediate education, male (% of male working-age population with intermediate education)']
ticklabels = [
    'Fuel exports',
    'Agricultural raw materials exports',
    'Arms exports',
    'Labor force with intermediate education, female',
    'Labor force with intermediate education, male']

df_selected = df.loc[(df['Country Name'].isin(countries))
                     & (df['Series Name'].isin(series))]

# Rename the columns
df.columns = [
    "Country Name",
    "Country Code",
    "Series Name",
    "Series Code",
    "2012 [2K12]",
    "2013 [2K13]",
    "2014 [2K14]",
    "2015 [2K15]",
    "2016 [2K16]",
    "2017 [2K17]",
    "2018 [2K18]",
    "2019 [2K19]",
    "2020 [2K20]",
    "2021 [2K21]"]

# Transpose_data


def add_transpose(df):
    """
    Transpose the input DataFrame and set the index to "Country Name".

    Args:
        df (pandas.DataFrame): Input DataFrame to be transposed.

    Returns:
        df_T (pandas.DataFrame): Transposed DataFrame with "Country Name" as the index.
    """
    df = df.set_index("Country Name")
    df_T = df.T
    df_T.columns.name = "Country Name"
    return df_T


# Call the add_transpose function to transpose the DataFrame
df_T = add_transpose(df)
print(df_T)

# describe_data


def describe_data(df):
    """
    Compute descriptive statistics for the input DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame to be described.

    Returns:
        stats (pandas.DataFrame): DataFrame containing the descriptive statistics.

    """
    return df.describe()


# Call the describe_data function to display the descriptive statistics
# Describe the data
    print(df_years.describe())
    print(df.describe())
    
# Replace all instances of ".." with NaN
df.replace("..", np.nan, inplace=True)

# Transpose the data to have the years as columns and the countries as rows
df_T = df.set_index('Country Name').T.drop(
    ['Country Code', 'Series Name', 'Series Code'])

# Convert the values to numeric data type
df_T = df_T.apply(pd.to_numeric)

# Convert the values to numeric data type
df_years = df_years.apply(pd.to_numeric, errors='coerce')

# Transpose the data for better visualization
df_years_transposed = df_years.T

# Calculate the statistical properties for each indicator
for indicator in df_years.columns:
    print(f"Indicator: {indicator}")
    print(f"Mean: {df_years[indicator].mean()}")
    print(f"Median: {df_years[indicator].median()}")
    print(f"Minimum: {df_years[indicator].min()}")
    print(f"Maximum: {df_years[indicator].max()}")
    print(f"Standard Deviation: {df_years[indicator].std()}")
    print()

# Calculate the correlation matrix for each country
corr_matrices = {}
for country in countries:
    corr_matrix = df_years[country].corr()
    corr_matrices[country] = corr_matrix
ticklabels = [
    'Fuel exports',
    'Agricultural raw materials exports',
    'Arms exports',
    'Labor force with intermediate education, female',
    'Labor force with intermediate education, male']

# Define shortened tick labels for x and y axes
# If the label is longer than 20 characters, truncate it and add "..." at the end
xticklabels = [s[:20] + '...' if len(s) > 20 else s for s in ticklabels]
yticklabels = [s[:20] + '...' if len(s) > 20 else s for s in ticklabels]

# Plot the correlation heatmap for Canada
# Set the size of the figure to be 8 inches by 8 inches
fig, ax = plt.subplots(figsize=(8, 8))
heatmap = ax.pcolor(corr_matrices['Canada'], cmap='YlGnBu')

# Add values to the heatmap
# Loop through each cell in the heatmap and add the corresponding value
for i in range(len(corr_matrices['Canada'].index)):
    for j in range(len(corr_matrices['Canada'].columns)):
        text = ax.text(j + 0.5,
                       i + 0.5,
                       round(corr_matrices['Canada'].iloc[i,
                                                          j],
                             2),
                       ha="center",
                       va="center",
                       color="black",
                       fontsize=8)

ax.set_xticks(np.arange(corr_matrices['Canada'].shape[0]) + 0.5, minor=False)
ax.set_yticks(np.arange(corr_matrices['Canada'].shape[1]) + 0.5, minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(corr_matrices['Canada'].index, fontsize=8, rotation=90)
ax.set_yticklabels(corr_matrices['Canada'].columns, fontsize=8)
ax.set_xticklabels([s[:20] + '...' if len(s) >
                   20 else s for s in xticklabels], fontsize=8, rotation=90)
ax.set_yticklabels(
    [s[:20] + '...' if len(s) > 20 else s for s in yticklabels], fontsize=8)
ax.set_title('Correlation Heatmap - Canada', fontsize=10)

plt.colorbar(heatmap)
plt.show()

# Plot the correlation heatmap for the United States
# Set the size of the figure to be 8 inches by 8 inches
fig, ax = plt.subplots(figsize=(8, 8))
heatmap = ax.pcolor(corr_matrices['United States'], cmap='YlGnBu')

# Add values to the heatmap
# Loop through each cell in the heatmap and add the corresponding value
for i in range(len(corr_matrices['United States'].index)):
    for j in range(len(corr_matrices['United States'].columns)):
        text = ax.text(j + 0.5,
                       i + 0.5,
                       round(corr_matrices['United States'].iloc[i,
                                                                 j],
                             2),
                       ha="center",
                       va="center",
                       color="black",
                       fontsize=8)

ax.set_xticks(
    np.arange(
        corr_matrices['United States'].shape[0]) +
    0.5,
    minor=False)
ax.set_yticks(
    np.arange(
        corr_matrices['United States'].shape[1]) +
    0.5,
    minor=False)
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(
    corr_matrices['United States'].index,
    fontsize=8,
    rotation=90)
ax.set_yticklabels(corr_matrices['United States'].columns, fontsize=8)
ax.set_title('Correlation Heatmap - United States', fontsize=10)
ax.set_xticklabels([s[:20] + '...' if len(s) >
                   20 else s for s in xticklabels], fontsize=8, rotation=90)
ax.set_yticklabels(
    [s[:20] + '...' if len(s) > 20 else s for s in yticklabels], fontsize=8)
plt.colorbar(heatmap)
plt.show()

# Load the data
df = pd.read_csv(
    'D:\\Datasets\\Dataset\\P_Data_Extract_From_World_Development_Indicators\\Metadata.csv')

# Filter the DataFrame to only include the two series of interest
df_filtered = df[df["Series Name"].isin(["Labor force with intermediate education, female (% of female working-age population with intermediate education)",
                                        "Labor force with intermediate education, male (% of male working-age population with intermediate education)"])]

# Pivot the DataFrame to have the series names as columns and the country
# names as rows
df_pivoted = pd.pivot_table(
    df_filtered,
    index="Country Name",
    columns="Series Name",
    values="2019 [YR2019]")

# Create the plot for female education
# Select the data for female labor force with intermediate education, drop
# the missing values
female_data = df_pivoted["Labor force with intermediate education, female (% of female working-age population with intermediate education)"].dropna(
)

# Plot the data with purple color, solid line style and 2 points line width
plt.plot(
    female_data.values,
    label="Female",
    color="purple",
    linestyle="solid",
    linewidth=2)

# Set the plot title, x-label and y-label
plt.title("Labor Force with Intermediate Education - Female")
plt.xlabel("Country")
plt.ylabel("% of Female Working-Age Population with Intermediate Education")

# Set the x-ticks labels to the country names and rotate them 90 degrees
plt.xticks(np.arange(len(female_data.index)), female_data.index, rotation=90)

# Display the legend and grid
plt.legend()
plt.grid(True, alpha=0.5, color='lightgray')
plt.show()

# Create the plot for male education
# Select the data for male labor force with intermediate education, drop
# the missing values
male_data = df_pivoted["Labor force with intermediate education, male (% of male working-age population with intermediate education)"].dropna(
)

# Plot the data with green color, solid line style and 2 points line width
plt.plot(
    male_data.values,
    label="Male",
    color="green",
    linestyle="solid",
    linewidth=2)

# Set the plot title, x-label and y-label
plt.title("Labor Force with Intermediate Education - Male")
plt.xlabel("Country")
plt.ylabel("% of Male Working-Age Population with Intermediate Education")

# Set the x-ticks labels to the country names and rotate them 90 degrees
plt.xticks(np.arange(len(male_data.index)), male_data.index, rotation=90)

# Display the legend and grid
plt.legend()
plt.grid(True, alpha=0.5, color='lightgray')
plt.show()

# Filter the DataFrame to only include the two series of interest
df_filtered = df[df["Series Name"].isin(
    ["Fuel exports (% of merchandise exports)", "Agricultural raw materials exports (% of merchandise exports)"])]

# Pivot the DataFrame to have the series names as columns and the country
# names as rows
df_pivoted = df_filtered.pivot(
    index="Country Name",
    columns="Series Name",
    values='2019 [YR2019]')

# Convert the values to numeric data type
df_pivoted["Fuel exports (% of merchandise exports)"] = pd.to_numeric(
    df_pivoted["Fuel exports (% of merchandise exports)"])
df_pivoted["Agricultural raw materials exports (% of merchandise exports)"] = pd.to_numeric(
    df_pivoted["Agricultural raw materials exports (% of merchandise exports)"])

# Set up figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# First bar chart
ax1.bar(
    df_pivoted.index,  # x-axis values
    df_pivoted["Fuel exports (% of merchandise exports)"]  # y-axis values
)
ax1.set_title("Fuel exports (% of merchandise exports)")  # set title
ax1.set_xlabel("Country Name")  # set x-axis label
ax1.set_ylabel("% of Fuel exports")  # set y-axis label
ax1.set_xticks(range(len(df_pivoted.index)))  # set x-axis tick positions
# set x-axis tick labels with 90 degree rotation
ax1.set_xticklabels(df_pivoted.index, rotation=90)
ax1.grid(axis='y', alpha=0.4)  # set grid lines on y-axis
ax1.grid(axis='x', alpha=0.4)  # set grid lines on x-axis

# Second bar chart
ax2.bar(
    df_pivoted.index,  # x-axis values
    # y-axis values
    df_pivoted["Agricultural raw materials exports (% of merchandise exports)"]
)
# set title
ax2.set_title("Agricultural raw materials exports (% of merchandise exports)")
ax2.set_xlabel("Country Name")  # set x-axis label
ax2.set_ylabel("% of Raw materials exports")  # set y-axis label
ax2.set_xticks(range(len(df_pivoted.index)))  # set x-axis tick positions
# set x-axis tick labels with 90 degree rotation
ax2.set_xticklabels(df_pivoted.index, rotation=90)
ax2.grid(axis='y', alpha=0.4)  # set grid lines on y-axis
ax2.grid(axis='x', alpha=0.4)  # set grid lines on x-axis

# Display the figure
plt.show()
