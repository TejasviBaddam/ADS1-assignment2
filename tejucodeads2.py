
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def read_data(filename):
    """
    Read data from a CSV file and perform necessary preprocessing.

    Parameters:
    - filename (str): The path to the CSV file.

    Returns:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - df_countries (pd.DataFrame): DataFrame with countries as columns.
    """

    # Read data from the CSV file, skipping the first 4 rows.
    df = pd.read_csv(filename, skiprows=4)

    # Drop unnecessary columns.
    cols_to_drop = ['Country Code', 'Indicator Code', 'Unnamed: 66']
    df = df.drop(cols_to_drop, axis=1)

    # Rename remaining columns.
    df = df.rename(columns={'Country Name': 'Country'})

    # Melt the dataframe to convert years to a single column.
    df = df.melt(id_vars=['Country', 'Indicator Name'],
                 var_name='Year', value_name='Value')

    # Convert year column to integer and value column to float.
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Separate dataframes with years and countries as columns.
    df_years = df.pivot_table(
        index=['Country', 'Indicator Name'], columns='Year', values='Value')
    df_countries = df.pivot_table(
        index=['Year', 'Indicator Name'], columns='Country', values='Value')

    # Clean the data by removing columns with all NaN values.
    df_years = df_years.dropna(how='all', axis=1)
    df_countries = df_countries.dropna(how='all', axis=1)

    return df_years, df_countries


def calculate_summary_stats(df_years, countries, indicators):
    """
    Calculate summary statistics for specified countries and indicators.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - countries (list): List of countries to calculate statistics for.
    - indicators (list): List of indicators to calculate statistics for.

    Returns:
    - summary_stats (dict): Dictionary containing calculated summary statistics.
    """

    # Create a dictionary to store the summary statistics.
    summary_stats = {}

    # Calculate summary statistics for each indicator and country.
    for indicator in indicators:
        for country in countries:
            # Summary statistics for individual countries.
            stats = df_years.loc[(country, indicator)].describe()
            summary_stats[f'{country} - {indicator}'] = stats

        # Summary statistics for the world.
        stats = df_years.loc[('World', indicator)].describe()
        summary_stats[f'World - {indicator}'] = stats

    return summary_stats


def print_summary_stats(summary_stats):
    """
    Print the calculated summary statistics.

    Parameters:
    - summary_stats (dict): Dictionary containing calculated summary statistics.
    """

    # Print the summary statistics.
    for key, value in summary_stats.items():
        print(key)
        print(value)
        print()


def create_scatter_plots(df_years, indicators, countries, figsize=(10, 8)):
    for country in countries:
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                plt.figure(figsize=figsize)  # Set the figure size here
                x = df_years.loc[(country, indicators[i])]
                y = df_years.loc[(country, indicators[j])]
                plt.scatter(x, y, alpha=0.7, edgecolors='w', linewidth=0.5, s=80, cmap='winter')  # Adjust point style and transparency
                plt.xlabel(indicators[i], fontsize=12)
                plt.ylabel(indicators[j], fontsize=12)
                plt.title(f"{country} - Scatter Plot: {indicators[i]} vs {indicators[j]}", fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.show()



def subset_data(df_years, countries, indicators, start_year, end_year):
    """
    Subsets the data to include only the selected countries, indicators, and specified year range.
    Returns the subsetted data as a new DataFrame.
    """
    # Create a boolean mask for the specified year range
    mask_year = (df_years.columns.get_level_values('Year').astype(int) >= start_year) & (df_years.columns.get_level_values('Year').astype(int) <= end_year)

    # Apply masks to subset the data
    df = df_years.loc[(countries, indicators), mask_year].transpose()

    return df


def calculate_correlations(df):
    """
    Calculates the correlations between the indicators in the input DataFrame.
    Returns the correlation matrix as a new DataFrame.
    """
    corr = df.corr()
    return corr


def visualize_correlations(corr):
    """
    Plots the correlation matrix as a heatmap using Seaborn.
    """
    sns.heatmap(corr, cmap='winter', annot=True, square=True)
    plt.title('Correlation Matrix of Indicators')
    plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_line_Electric_power_Production(df_years, selected_years):
    """
    Plot a line chart for electricity production from coal sources (% of total) for selected years.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - selected_years (list): List of years to be plotted.

    Returns:
    - None
    """

    country_list = ['United States', 'India', 'China', 'Canada']
    indicator = 'Electricity production from coal sources (% of total)'

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Set the color palette
    colors = sns.color_palette("husl", n_colors=len(country_list))

    # Set the line styles
    line_styles = ['-', '--', '-.', ':']
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, country in enumerate(country_list):
        df_subset = df_years.loc[(country, indicator), selected_years]
        plt.plot(df_subset.index, df_subset.values, label=country, linestyle=line_styles[i % len(line_styles)], color=colors[i])
        
        # Print results for the selected years
        print(f"\nResults for {country} - {indicator} for the selected years:")
        print(df_subset)

    plt.xlabel('Year', fontsize=12)
    plt.ylabel(indicator, fontsize=12)
    plt.title('Electricity production from coal sources (% of total)', fontsize=12)

    # Move the legend to the right side
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()


def plot_line_Population_total(df_years, selected_years):
    """
    Plot a line chart for renewable electricity output (% of total electricity output) for selected years.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.
    - selected_years (list): List of years to be plotted.

    Returns:
    - None
    """

    country_list = ['United States', 'India', 'China', 'Canada']
    indicator = 'Renewable electricity output (% of total electricity output)'

    # Set Seaborn style
    sns.set(style="whitegrid")

    # Set the color palette
    colors = sns.color_palette("Set1", n_colors=len(country_list))

    # Set the line styles
    line_styles = ['-', '--', '-.', ':']

    # Set the figure size
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, country in enumerate(country_list):
        df_subset = df_years.loc[(country, indicator), selected_years]
        ax.plot(df_subset.index, df_subset.values, label=country, linestyle=line_styles[i % len(line_styles)], color=colors[i])
        
        # Print results for the selected years
        print(f"\nResults for {country} - {indicator} for the selected years:")
        print(df_subset)

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel(indicator, fontsize=12)
    ax.set_title('Renewable electricity output (% of total electricity output)', fontsize=12)

    # Move the legend to the right side
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()


def plot_group(df_years):
    """
    Plot a grouped bar chart for urban population (% of total population) from 1995 to 2000.

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.

    Returns:
    - None
    """

    country_list = ['United States', 'India', 'China', 'Canada']
    urban_population_indicator = 'Urban population (% of total population)'
    years = [1995, 1996, 1997, 1998, 1999, 2000]
    x = np.arange(len(country_list))
    total_width = 0.8  # Total width for each group of bars
    bar_width = total_width / len(years)  # Width for each individual bar

    # Set Seaborn style
    sns.set(style="whitegrid", palette="pastel")

    fig, ax = plt.subplots()

    # Set the color palette
    colors = sns.color_palette("viridis", n_colors=len(years))

    for i, year in enumerate(years):

        urban_population_values = []
        for country in country_list:
            value = df_years.loc[(country, urban_population_indicator), year]
            urban_population_values.append(value)
            print(f"{country} - {urban_population_indicator} ({year}): {value}")

        # Adjust the x positions to create grouped bars
        x_positions = x + (i - len(years) / 2 + 0.5) * bar_width

        rects2 = ax.bar(x_positions, urban_population_values,
                        bar_width, label=str(year)+" ", color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title('Urban population (% of total population) from 1995 to 2000')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)

    # Add grid lines
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Move the legend to the right side
    ax.legend(loc='center left', bbox_to_anchor=(0.5, 0.8))

    # Set background color
    ax.set_facecolor('#f9f9f9')

    fig.tight_layout()
    plt.show()

def plot_Population_total(df_years):
    """
    Plot a bar chart for urban population growth (annual %).

    Parameters:
    - df_years (pd.DataFrame): DataFrame with years as columns.

    Returns:
    - None
    """

    country_list = ['United States', 'India', 'China', 'Canada']
    Population_total_indicator = 'Urban population growth (annual %)'

    years = [1995, 1996, 1997, 1998, 1999, 2000]
    x = np.arange(len(country_list))
    width = 0.35

    fig, ax = plt.subplots()
    for i, year in enumerate(years):
        Population_total_values = []
        for country in country_list:
            value = df_years.loc[(country, Population_total_indicator), year]
            Population_total_values.append(value)

            # Print the data used for each bar
            print(f"{country} - {Population_total_indicator} ({year}): {value}")

        rects1 = ax.bar(x - width/2 + i*width/len(years), Population_total_values,
                        width/len(years), label=str(year)+" ")

    ax.set_xlabel('Country')
    ax.set_ylabel('Value')
    ax.set_title('Urban population growth (annual %)')
    ax.set_xticks(x)
    ax.set_xticklabels(country_list)
    ax.legend()

    fig.tight_layout()
    plt.show()



def explore_indicators(df_years, countries, indicators):
    """
    Explore statistical properties of indicators for individual countries and cross-compare.

    Parameters:
    - df_years (pandas.DataFrame): DataFrame containing the yearly data for indicators.
    - countries (list of str): List of countries for analysis.
    - indicators (list of str): List of indicators to explore.
    """
    # Create a dictionary to store summary statistics
    summary_stats = {}

    # Explore indicators for individual countries
    for country in countries:
        for indicator in indicators:
            # Get data for the specific country and indicator
            data = df_years.loc[(country, indicator), :]

            # Calculate summary statistics using .describe() and two additional statistical methods
            stats = {
                'describe': data.describe(),
                'median': data.median(),
                'std_dev': data.std(),
            }

            # Store the statistics in the dictionary
            summary_stats[f'{country} - {indicator}'] = stats

    # Explore indicators for aggregated regions or categories
    for indicator in indicators:
        # Get data for the world (you can modify this for other regions/categories)
        data_world = df_years.loc[('World', indicator), :]

        # Calculate summary statistics using .describe() and two additional statistical methods
        stats_world = {
            'describe': data_world.describe(),
            'median': data_world.median(),
            'std_dev': data_world.std(),
        }

        # Store the statistics in the dictionary
        summary_stats[f'World - {indicator}'] = stats_world

    # Print the summary statistics
    for key, stats in summary_stats.items():
        print(f"Summary Statistics for {key}:")
        print(stats['describe'])
        print(f"Median: {stats['median']}")
        print(f"Standard Deviation: {stats['std_dev']}")
        print("\n" + "=" * 50 + "\n")

def explore_correlations(df_years, countries, indicators, start_year, end_year):
    """
    Explore and understand correlations between indicators within countries and across time.

    Parameters:
    - df_years (pandas.DataFrame): DataFrame containing the yearly data for indicators.
    - countries (list of str): List of countries for analysis.
    - indicators (list of str): List of indicators to explore.
    - start_year (int): Start year for the analysis.
    - end_year (int): End year for the analysis.
    """
    # Subset the data for the specified year range
    df_filtered = subset_data(df_years, countries, indicators, start_year, end_year)

    # Calculate correlations
    corr_matrix = calculate_correlations(df_filtered)

    # Visualize correlations as a heatmap
    visualize_correlations(corr_matrix)

def main():
    # Read data from the specified CSV file
    df_years, df_countries = read_data(r"C:\Users\ACER\Downloads\wbdata (3).csv")

    # Task 1: Calculate and print summary statistics
    indicators_task1 = ['Electric power consumption (kWh per capita)', 'Electricity production from hydroelectric sources (% of total)']
    countries_task1 = ['United States', 'China', 'India', 'Canada']
    start_year_task1 = 1990
    end_year_task1 = 2000

    # Calculate summary statistics for Task 1 and print the results
    summary_stats_task1 = calculate_summary_stats(df_years, countries_task1, indicators_task1)
    print_summary_stats(summary_stats_task1)

    # Task 2: Explore and visualize correlations
    indicators_task2 = ['Forest area (% of land area)', 'Agricultural land (% of land area)']
    countries_task2 = ['United States', 'China', 'India', 'Canada']
    start_year_task2 = 1990
    end_year_task2 = 2000

    # Explore and visualize correlations for Task 2
    explore_correlations(df_years, countries_task1, indicators_task1, start_year_task1, end_year_task1)
    explore_correlations(df_years, countries_task2, indicators_task2, start_year_task2, end_year_task2)

    # Task 3: Explore indicators with six plots
    # explore_indicators(df_years, countries_task1, indicators_task1)
    explore_indicators(df_years, countries_task2, indicators_task2)
    
    # Specify the selected years for plotting line charts
    selected_years = [1994, 1997, 1999, 2001, 2003, 2006]

    # Plot electricity production from coal sources for selected years
    plot_line_Electric_power_Production(df_years, selected_years)
    
    # Plot urban population growth for selected years
    plot_line_Population_total(df_years, selected_years)
   
    # Plot grouped bar chart for urban population (% of total population) from 1995 to 2000
    plot_group(df_years)
    
    # Plot bar chart for urban population growth (annual %)
    plot_Population_total(df_years)
    
    # Task 4: Create scatter plots
    indicators_scatter = ['Electricity production from renewable sources, excluding hydroelectric (% of total)', 'Electricity production from renewable sources, excluding hydroelectric (kWh)']
    countries_scatter = ['China', 'India']
    
    # Create scatter plots for Task 4
    create_scatter_plots(df_years, indicators_scatter, countries_scatter, figsize=(12, 10))


if __name__ == '__main__':
    main()
