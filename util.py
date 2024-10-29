from nba_api.stats.endpoints import commonplayerinfo as cm
from nba_api.stats.endpoints import playercareerstats as cr
from nba_api.stats.static import players
from nba_api.stats.endpoints import playerdashboardbyyearoveryear as dash
from nba_api.stats.endpoints import playergamelogs as logs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import networkx as nx
from sklearn.decomposition import PCA


# Helper function to get the id of an athlete
def get_player_id(full_name):
    id = players.find_players_by_full_name(full_name)[0].get("id")
    return id


# Helper function to get the yearly dashboard of an athlete
def get_dashboard(id, yearly=True, seasons=None):
    if yearly:
        # I get most of my advanced statistics here (percentages)
        py_dash = dash.PlayerDashboardByYearOverYear(player_id=id, season_type_playoffs="Regular Season",
                                                     measure_type_detailed="Advanced")
        headers = py_dash.get_dict()["resultSets"][1]["headers"]
        adv = pd.DataFrame(py_dash.get_dict()["resultSets"][1]["rowSet"][::-1], columns=headers)

        # Most of the counts come from here
        py_basic_dash = dash.PlayerDashboardByYearOverYear(player_id=id, season_type_playoffs="Regular Season",
                                                           measure_type_detailed="Base", per_mode_detailed="PerGame")
        headers = py_basic_dash.get_dict()["resultSets"][1]["headers"]
        basic = pd.DataFrame(py_basic_dash.get_dict()["resultSets"][1]["rowSet"][::-1], columns=headers)

        common_col = adv.columns.intersection(basic.columns).delete(1)  # Deletes "GROUP_VALUE" so I can merge on that
        full = adv.merge(basic.drop(common_col, axis=1), how="left", on="GROUP_VALUE")

    # A more in depth game-by-game analysis
    else:
        adv = pd.DataFrame()
        basic = pd.DataFrame()
        full = pd.DataFrame()
        for season in seasons:
            py_logs = logs.PlayerGameLogs(player_id_nullable=id, season_nullable=season,
                                          measure_type_player_game_logs_nullable="Advanced")
            headers = py_logs.get_dict()["resultSets"][0]["headers"]
            adv_logs = pd.DataFrame(py_logs.get_dict()["resultSets"][0]["rowSet"][::-1], columns=headers)
            adv = pd.concat([adv, adv_logs]).reset_index(drop=True)

            py_basic_dash = logs.PlayerGameLogs(player_id_nullable=id, season_nullable=season,
                                                measure_type_player_game_logs_nullable="Base")
            headers = py_basic_dash.get_dict()["resultSets"][0]["headers"]
            basic_logs = pd.DataFrame(py_basic_dash.get_dict()["resultSets"][0]["rowSet"][::-1], columns=headers)
            basic = pd.concat([basic, basic_logs]).reset_index(drop=True)

            common_col = adv.columns.intersection(basic.columns).delete(7)  # Deletes "GAME_ID" so I can merge on that
            full_logs = adv.merge(basic.drop(common_col, axis=1), how="left", on="GAME_ID")
            full = pd.concat([full, full_logs]).reset_index(drop=True)

        full = full.sort_values(by="GAME_DATE")

    basic = basic.drop_duplicates().reset_index(drop=True)
    adv = adv.drop_duplicates().reset_index(drop=True)
    full = full.drop_duplicates().reset_index(drop=True)

    return [basic, adv, full]


def pct_change(data, col):
    return data[col].diff() / data[col].abs().shift()


def plot_metrics(name, metrics, data, rows=2, cols=3, yoy=False, yearly=True):
    """
    Plot multiple figures with 6 subplots each for the given metrics.

    :param name: Name of athlete
    :param metrics: List of metrics to plot
    :param data: DataFrame containing the data
    :param rows: Number of rows in each figure (default 2)
    :param cols: Number of columns in each figure (default 3)
    :param yoy: If True, plot year-over-year changes instead of raw values
    :param yearly: If True, plot yearly values instead of per-game values
    """
    num_plots = rows * cols
    num_figures = (len(metrics) + num_plots - 1) // num_plots
    common = "GROUP_VALUE" if yearly else "GAME_ID"  # y-value

    for fig_num in range(num_figures):
        fig, axes = plt.subplots(rows, cols, figsize=(25, 16))
        title_suffix = "Year-over-Year Changes" if yoy else "Statistics Over Time"
        fig.suptitle(f'{name} {title_suffix} (Figure {fig_num + 1})', fontsize=16)

        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i in range(num_plots):
            metric_index = fig_num * num_plots + i
            if metric_index < len(metrics):
                metric = metrics[metric_index]
                ax = axes[i]

                if yoy:
                    yoy_data = pct_change(data, metric)
                    valid_data = data[~np.isnan(yoy_data)]
                    valid_yoy = yoy_data[~np.isnan(yoy_data)]
                    sns.barplot(x=valid_data[common], y=valid_yoy, ax=ax)
                    ax.axhline(y=0, color='r', linestyle='--')
                    ax.set_title(f'{metric} Year-over-Year Change')
                    ax.set_ylabel('Percent Change')
                else:
                    sns.lineplot(data=data, x=common, y=metric, ax=ax)
                    ax.scatter(data[common], data[metric], color='red', s=30)
                    ax.set_title(f'{metric} Over Time')
                    ax.set_ylabel(metric)

                    if not yearly:
                        sns.scatterplot(data=data, x=common, y=metric, hue='WL', ax=ax,
                                        palette={'W': 'green', 'L': 'red'})
                    else:
                        sns.scatterplot(data=data, x=common, y=metric, color="red", ax=ax)

                    ax.set_title(f'{metric} Over Time')
                    ax.set_ylabel(metric)

                if not yearly:
                    ax.set_xlabel('Season')
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

                    # Add vertical lines and labels for new seasons
                    season_starts = data.groupby('SEASON_YEAR').first()
                    for season, start in season_starts.iterrows():
                        ax.axvline(x=start[common], color='green', linestyle='--', alpha=0.7)
                        ax.text(start[common], ax.get_ylim()[1], f' {season}',
                                rotation=90, va='top', ha='right', fontsize=8, color='green')

                    ax.set_xticks([])
                    if not yoy:
                        ax.get_legend().remove()
                else:
                    ax.set_xlabel('Year')
                    ax.tick_params(axis='x', rotation=45)

        if not yearly and not yoy:
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Adjust to prevent title overlap
        plt.show()


def perform_correlation_analysis(df, categories, correlation_threshold=0.5, figsize=(20, 16)):
    """
    Perform correlation analysis including heatmap, hierarchical clustering, and network graph.

    :param df: pandas DataFrame containing the data
    :param categories: list of column names to include in the analysis
    :param correlation_threshold: threshold for including edges in the network graph (default 0.5)
    :param figsize: tuple specifying the figure size (default (20, 16))
    """
    correlation_matrix = df[categories].corr()

    # Correlation heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                linewidths=0.5, mask=np.triu(np.ones_like(correlation_matrix, dtype=bool)))
    plt.title('Correlation Heatmap with Lower Triangle', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Hierarchical Clustering
    plt.figure(figsize=(figsize[0], figsize[1] // 2))
    corr_linkage = hierarchy.ward(squareform(1 - np.abs(correlation_matrix)))
    dendro = hierarchy.dendrogram(corr_linkage, labels=categories, leaf_rotation=90)
    plt.title('Hierarchical Clustering of Metrics', fontsize=16)
    plt.tight_layout()
    plt.show()

    # Correlation network graph
    G = nx.Graph()
    # Add nodes
    for i, metric in enumerate(categories):
        G.add_node(i, name=metric)

    # Add edges (only for correlations above threshold)
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
                G.add_edge(i, j, weight=abs(correlation_matrix.iloc[i, j]))

    # Set up colors for positive and negative correlations
    pos_edges = [(i, j) for (i, j, d) in G.edges(data=True) if correlation_matrix.iloc[i, j] > 0]
    neg_edges = [(i, j) for (i, j, d) in G.edges(data=True) if correlation_matrix.iloc[i, j] < 0]

    # Create layout
    pos = nx.spring_layout(G, k=0.8, iterations=50)

    # Plot
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)
    nx.draw_networkx_edges(G, pos, edgelist=pos_edges, edge_color='blue', width=2)
    nx.draw_networkx_edges(G, pos, edgelist=neg_edges, edge_color='red', width=2)

    # Add labels
    labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    plt.title('Correlation Network Graph', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return correlation_matrix


def analyze_significant_changes(df, categories, year_column='GROUP_VALUE', threshold=0.3, top_n=6):
    """
    Analyze and visualize significant year-over-year changes for given categories.

    :param df: pandas DataFrame containing the data
    :param categories: list of column names to analyze
    :param year_column: name of the column containing year information (default 'GROUP_VALUE')
    :param threshold: threshold for significant change (default 0.3)
    :param top_n: number of top variable metrics to visualize (default 6)
    :return: tuple containing significant changes dict and change matrix DataFrame
    """

    # Calculate year-over-year changes
    yoy_changes = pct_change(df, categories).dropna()
    yoy_changes[year_column] = df[year_column][1:]  # Add year column to yoy_changes

    def get_significant_change_years(metric):
        significant_changes = yoy_changes[abs(yoy_changes[metric]) > threshold]
        return list(significant_changes[year_column])

    # Analyze significant changes for each metric
    significant_changes = {}
    for metric in categories:
        years = get_significant_change_years(metric)
        if years:
            significant_changes[metric] = years

    # Sort metrics by number of significant change years
    sorted_metrics = sorted(significant_changes.items(), key=lambda x: len(x[1]), reverse=True)

    # Print results
    print(f"Years with Significant Changes (>{threshold} change) for Each Metric:")
    for metric, years in sorted_metrics:
        print(f"{metric}: {years} (Total: {len(years)})")

    # Visualize the changes over time for top N metrics with most significant changes
    top_metrics = [metric for metric, _ in sorted_metrics[:top_n]]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Year-over-Year Changes for Top {top_n} Variable Metrics', fontsize=16)

    for i, metric in enumerate(top_metrics):
        ax = axes[i // 3, i % 3]
        sns.lineplot(data=df, x=year_column, y=metric, ax=ax)
        ax.set_title(f'{metric} Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel(metric)

        # Highlight years with significant changes
        for year in significant_changes[metric]:
            ax.axvline(x=year, color='r', linestyle='--', alpha=0.5)

        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust to prevent title overlap
    plt.show()

    # Create a matrix of significant changes
    change_matrix = pd.DataFrame(index=df[year_column][1:], columns=categories)
    for metric in categories:
        # Only include changes above the threshold, otherwise set to NaN
        change_matrix[metric] = np.where(yoy_changes[metric].abs() > threshold,
                                         yoy_changes[metric],
                                         np.nan)

    print(f"\nSummary of Significant Changes (>{threshold * 100}% change) for Each Metric:")
    for metric in categories:
        significant_years = change_matrix[metric].dropna()
        if not significant_years.empty:
            print(f"{metric}:")
            for year, change in significant_years.items():
                print(f"  {year}: {change:.2%}")
            print(f"  Total significant changes: {len(significant_years)}\n")

    return significant_changes, change_matrix
