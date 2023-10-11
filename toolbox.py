import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats as st
from matplotlib import pyplot as plt
from matplotlib.cbook import boxplot_stats
import sklearn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from math import ceil
from skimage import io
import textwrap
import ssl
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


def normalize_string(string_to_normalize: str, withspace=False):
    """
    Retourne une chaine de caractères en minuscule, sans ponctuation

    Positional arguments : 
    -------------------------------------
    string_to_normalize: str: chaine de caractères à modifier

    Optional arguments : 
    -------------------------------------
    withspace : bool : True remplace ponctuation par un espace
    """

    normalized_string = string_to_normalize.lower()
    if withspace:
        normalized_string = re.sub(r'[^\w]', ' ', normalized_string)
    else:
        normalized_string = re.sub(r'[^\w]', '', normalized_string)

    return normalized_string


def get_closest_in_(values: [float], value: float):
    """
    Retourne la valeur la plus proche de la valeur renseignée parmis les valeurs d'une liste

    Positional arguments : 
    -------------------------------------
    values: list of floats: Liste des valeurs parmi lesquelles on recherche la plus proche
    value : float : valeur comparée à chaque élément de la liste
    """
    closest_in_values = min(values, key=lambda x: abs(x-value))

    return closest_in_values


def get_columns_contains(dataset: pd.DataFrame, regex_str: str, ignore=None):
    """
    Retourne la liste des noms des colonnes d'un dataframe contenant la chaîne de caractères renseignée

    Positional arguments : 
    -------------------------------------
    dataset:  pd.DataFrame: le dataframe contenant les colonnes que l'on souhaite extraire
    regex_str : str : chaîne de caractères recherchée

    Optional arguments : 
    -------------------------------------
    ignore : str or list of strings : nom(s) de la/des colonne(s) à ignorer (car on ne les veut pas dans la liste)
    """

    mask = dataset.columns.str.contains(regex_str, regex=True)
    if ignore:
        columns_list = [
            col for col in dataset.columns[mask].values if col not in ignore]
    else:
        columns_list = dataset.columns[mask].values

    return columns_list


def missing_values_by_column(dataset: pd.DataFrame):
    """
    Retourne un dataframe avec le nombre et le pourcentage de valeurs manquantes par colonnes

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les colonnes dont on veut connaitre le pourcentage de vide
    """

    missing_values_series = dataset.isnull().sum()
    missing_values_df = missing_values_series.to_frame(
        name='Number of Missing Values')
    missing_values_df = missing_values_df.reset_index().rename(
        columns={'index': 'VARIABLES'})

    missing_values_df['Missing Values (%)'] = round(
        missing_values_df['Number of Missing Values'] / (dataset.shape[0]) * 100, 2)

    missing_values_df = missing_values_df.sort_values(
        'Number of Missing Values')

    return missing_values_df


def clean_duplicates(dataset: pd.DataFrame, key_column: str, date_column: str):
    """
    Retourne un dataframe sans doublons (garde l'individu avec le moins de valeurs manquantes et le plus récent parmis les doublons)

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe dont on souhaite retirer les doublons
    key_column : str : nom de la colonne contenant la clé d'unicité (doublons = individus avec la même valeur dans cette colonne)
    date_column : str : nom de la colonne contenant les dates utilisées pour garder l'individu le plus récent parmis les doublons
    """

    duplicates_all = dataset.loc[dataset.duplicated(subset=key_column, keep=False)].copy()
    print('Il y a', 
          duplicates_all.shape[0] - len(duplicates_all[key_column].unique()), 
          'doublon(s)')

    if duplicates_all.empty:
        return dataset

    subset = dataset.copy()
    duplicates_all['MissingValues'] = duplicates_all.isnull().sum(axis=1)
    duplicates_all = duplicates_all.sort_values(
        ['MissingValues', date_column], ascending=[True, False])

    duplicates_to_drop = duplicates_all.loc[duplicates_all.duplicated(
        subset=key_column)]
    subset.drop(index=duplicates_to_drop.index.values, inplace=True)

    print(dataset.shape[0] - subset.shape[0], 'ligne(s) supprimée(s)')
    print('Il reste', subset.loc[subset.duplicated(
        subset=key_column)].shape[0], 'doublon(s)')

    return subset


def build_mask_outlier(dataset: pd.DataFrame, column: str, limits: dict):
    """
    Retourne une condition pour sélectionner les individus d'un jeu de données ayant une valeur aberrante dans la variable choisie 
    au regard des limites renseignées

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données à filtrer
    column : str : nom de la colonne dans laquelle on recherche des valeurs aberrantes
    limits: dict : dictionnaire contenant les bornes en dehors desquelles la valeur est considérée comme aberrante (ex: {'min':0.0, 'max': 3_765})
    """
    mask_max = dataset[column] > limits['max']
    mask_min = dataset[column] < limits['min']
    mask_outlier = (mask_max) | (mask_min)

    if np.isnan(limits['max']):
        mask_outlier = mask_min
    elif np.isnan(limits['min']):
        mask_outlier = mask_max

    return mask_outlier


def replace_outliers_by_mean(dataset: pd.DataFrame, columns_to_check: dict, category: str):
    """
    Retourne un dataframe dans lequel les valeurs aberrantes, dans les colonnes choisies, 
    ont été imputées avec la moyenne de la catégorie renseignée
    Les individus, avec une valeur aberrante, qui n'ont pas de catégorie renseignée sont supprimés.

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données à nettoyer

    columns_to_check : dict : dictionnaire contenant les noms des colonnes dans 
    lesquelles on recherche des valeurs aberrantes. Pour chaque colonne sont aussi renseignées 
    les bornes en dehors desquelles la valeur est considérée comme aberrante
    (ex : {'energy_100g': {'min':0.0, 'max': 3_765}, 'saturated-fat_100g' : {'min': 0.0, 'max': 100.0}})

    category: str : nom de la colonne contenant les catégories selon lesquelles calculer la moyenne
    """
    mask_outliers_all = np.logical_or.reduce(
        [build_mask_outlier(dataset, c, l) for c, l in columns_to_check.items()])
    mask_missing_category = dataset[category].isnull()

    outliers_nb = dataset.loc[mask_outliers_all].shape[0]
    subset = dataset.loc[~((mask_missing_category) &
                           (mask_outliers_all))].copy()

    print('{:_}'.format(outliers_nb), 'valeur(s) aberrante(s)')
    print('{:_}'.format(dataset.shape[0] -
          subset.shape[0]), 'ligne(s) supprimée(s)')

    if (outliers_nb == 0) or (dataset.shape[0] - subset.shape[0] == outliers_nb):
        return subset

    for column, limits in columns_to_check.items():
        mask_outlier = build_mask_outlier(subset, column, limits)
        mean_df = subset.loc[~(mask_outlier)].groupby(
            category)[[column]].mean()

        if subset.loc[mask_outlier].shape[0] > 0:
            subset.loc[mask_outlier, column] = subset.loc[mask_outlier].apply(
                lambda row: mean_df.loc[row[category]][column], axis=1)

    return subset


def replace_outliers_by_limits(dataset: pd.DataFrame, columns_to_check: dict):
    """
    Retourne un dataframe dans lequel les valeurs aberrantes, dans les colonnes choisies, 
    ont été remplacées par la limite la plus proche

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données à nettoyer

    columns_to_check : dict : dictionnaire contenant les noms des colonnes dans lesquelles 
    on recherche des valeurs aberrantes. Pour chaque colonne sont aussi renseignées les 
    bornes en dehors desquelles la valeur est considérée comme aberrante
    (ex : {'energy_100g': {'min':0.0, 'max': 3_765}, 'saturated-fat_100g' : {'min': 0.0, 'max': 100.0}})
    """
    mask_outliers_all = np.logical_or.reduce(
        [build_mask_outlier(dataset, c, l) for c, l in columns_to_check.items()])
    outliers_nb = dataset.loc[mask_outliers_all].shape[0]

    print('{:_}'.format(outliers_nb), 'valeur(s) aberrante(s)')

    if (outliers_nb == 0):
        return dataset

    subset = dataset.copy()
    for column, limits in columns_to_check.items():
        mask_outlier = build_mask_outlier(subset, column, limits)

        if subset.loc[mask_outlier].shape[0] > 0:
            subset.loc[mask_outlier, column] = subset.loc[mask_outlier].apply(
                lambda row: get_closest_in_(limits.values, row[column]), axis=1)

    return subset


def drop_uncorrect_nutriscore(dataset: pd.DataFrame, limits_grade: dict, grade_column: str, score_column: str):
    """
    Retourne un dataframe sans les lignes pour lesquelles le nutriscore et le nutrigrade ne correspondent pas

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données à nettoyer

    limits_grade : dict : dictionnaire contenant les nutrigrades avec les nutriscores 
    minimum et maximum possibles associés
    (ex : {'a': {'min':-15, 'max': -1}, 'b': {'min': 0, 'max': 2}})

    grade_column : str : nom de la colonne contenant les nutrigrades
    score_column : str : nom de la colonne contenant les nutriscores
    """
    subset = dataset.copy()

    for grade in subset[grade_column].dropna().unique():
        mask_n = subset[grade_column] == grade
        mask_max = subset[score_column] > limits_grade[grade]['max']
        mask_min = subset[score_column] < limits_grade[grade]['min']

        mask_outlier = (mask_n) & ((mask_max) | (mask_min))
        print(grade, ':',
              subset.loc[mask_outlier].shape[0], 'valeur(s) aberrante(s)')

        subset = subset.loc[~mask_outlier].copy()

    print(dataset.shape[0] - subset.shape[0], 'valeur(s) supprimée(s)')

    return subset


def fill_missing_values_with_mean_by_(dataset: pd.DataFrame, columns_to_fill: [str], category: str):
    """
    Retourne un dataframe dans lequel les valeurs manquantes, dans les colonnes choisies, 
    ont été imputées avec la moyenne de la catégorie renseignée
    Les individus, avec une valeur manquante, qui n'ont pas de catégorie renseignée sont supprimés.

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données à nettoyer
    columns_to_fill : list of strings : liste contenant les noms des colonnes dans lesquelles 
    on recherche des valeurs manquantes. 
    category: str : nom de la colonne contenant les catégories selon lesquelles calculer la moyenne
    """

    mask_missing_category = dataset[category].isnull()
    mask_missing_value = np.logical_or.reduce(
        [dataset[c].isnull() for c in columns_to_fill])

    mean_df = dataset.groupby(category)[columns_to_fill].mean()
    subset = dataset.loc[~((mask_missing_category) &
                           (mask_missing_value))].copy()

    print('{:_}'.format(dataset.shape[0] - dataset.dropna(
        subset=columns_to_fill).shape[0]), 'valeur(s) manquante(s)')
    print('{:_}'.format(dataset.shape[0] -
          subset.shape[0]), 'lignes(s) supprimée(s)')

    for column in columns_to_fill:
        mask_column = subset[column].isnull()
        if subset.loc[mask_column].shape[0] > 0:
            subset.loc[mask_column, column] = subset.loc[mask_column].apply(
                lambda row: mean_df.loc[row[category]][column], axis=1)

    print('Il reste', '{:_}'.format(
        subset.shape[0] - subset.dropna(subset=columns_to_fill).shape[0]), 'valeur(s) manquante(s)')

    return subset


def filter_outlier(dataset: pd.DataFrame, x_column: str):
    """
    Retourne un dataframe sans outliers et la liste des outliers

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données dont on souhaite extraire les outliers
    x_column : str : nom de la colonne contenant possiblement des outliers
    """

    outliers = [y for stat in boxplot_stats(
        dataset[x_column]) for y in stat['fliers']]
    mask_outliers = dataset[x_column].isin(outliers)
    subset = dataset.loc[~mask_outliers]

    return subset, outliers


def plot_distribution_with_hue(dataset: pd.DataFrame, x_column: str, hue_column: str, figsize: tuple, titles: dict, palette: str):
    """
    Affiche la distribution d'une variable sous forme d'histogramme coloré 
    en fonction d'une autre variable, avec les effectifs par classe

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données à afficher
    x_column : str : nom de la colonne contenant la variable dont on souhaite afficher la distribution
    hue_column : str : nom de la colonne contenant la variable selon laquelle colorer l'histogramme
    figsize: tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    titles: dict : dictionnaire contenant les différents titres à 
    afficher sur le graphique {'chart_title': '...' , 'y_title': '...', 'x_title': '...'}
    palette: str : nom de la palette seaborn utilisée pour colorer l'histogramme (ex: 'husl', 'Set2', 'Pastel2' ...)
    """

    sns.set_theme(style='whitegrid', palette=palette)
    plt.figure(figsize=figsize)
    plt.rcParams['axes.labelpad'] = '20'
    plt.title(titles['chart_title'], fontname='Corbel', fontsize=20, pad=40)
    plt.ylabel(titles['y_title'], fontsize=18, fontname='Corbel')
    plt.xlabel(titles['x_title'], fontsize=18, fontname='Corbel')

    ax = sns.histplot(data=dataset, x=x_column, hue=hue_column, shrink=.7)
    ax.margins(y=0.1)
    plt.grid(False, axis='x')
    plt.xticks(rotation=45, ha='right')

    for p in ax.patches:
        _x = p.get_x() + p.get_width() / 2
        _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
        value = p.get_height()
        if value != 0:
            ax.text(_x, _y, value, ha="center")

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()


def plot_dist_comparison(data_before: pd.DataFrame, data_after: pd.DataFrame, 
                         dimensions: [str], title: str, figsize: tuple, without_outliers=False):
    """
    Affiche la distribution de chaque variable choisie avant et après modification

    Positional arguments : 
    -------------------------------------
    data_before : pd.DataFrame : dataframe contenant les données avant modification
    data_after : pd.DataFrame : dataframe contenant les données après modification
    dimensions : list of strings : liste des variables dont on souhaite afficher la distribution
    title : str : titre principal (suptitle)
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    without_outliers : bool : True affiche la distribution sans outliers 
    mais indique le nombre d'outliers non pris en compte, 
    False affiche la distribution avec les outliers
    """

    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    data = {0: {'dataset': data_before, 'title': 'avant'},
            1: {'dataset': data_after, 'title': 'après'}}

    with plt.style.context('seaborn-white'):
        plt.rcParams['axes.labelpad'] = '40'
        sns.set_theme(style='whitegrid', palette='Set2')

        fig, axes = plt.subplots(
            len(dimensions), 2, figsize=figsize, sharey=False)
        fig.tight_layout()
        suptitle_text = 'Distribution ' + title
        fig.suptitle(suptitle_text, fontname='Arial Rounded MT Bold',
                     fontsize=60, color=rgb_text)
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=0.93, wspace=0.2, hspace=0.5)

        (l, c) = (0, 0)

    for dimension in dimensions:
        for c in range(0, 2):
            dataset = data[c]['dataset']
            if without_outliers:
                subset_without_outliers, outliers = filter_outlier(
                    dataset.loc[~dataset[dimension].isnull()], dimension)
                sns.histplot(data=subset_without_outliers,
                             x=dimension, ax=axes[l, c])
                axes[l, c].text(1, 1, '\n outliers: {:_} \n'.format(len(outliers)), 
                                transform=axes[l, c].transAxes, fontsize=50,
                                verticalalignment='top', horizontalalignment='right',
                                bbox={'facecolor': sns.color_palette(
                                    "husl", 8)[0], 'alpha': 0.3, 'pad': 0, 'boxstyle': 'round'},
                                style='italic', fontname='Open Sans')
            else:
                sns.histplot(data=dataset, x=dimension, ax=axes[l, c])

            axes[l, c].set_title("Distribution \"{}\" {}".format(
                dimension, data[c]['title']), fontname='Corbel', color=rgb_text, fontsize=45, pad=50)
            axes[l, c].set_xlabel(dimension, fontsize=40,
                                  fontname='Corbel', color=rgb_text)
            axes[l, c].set_ylabel(
                'Nombre de produits', fontsize=40, fontname='Corbel', color=rgb_text)

            axes[l, c].tick_params(
                axis='both', which='major', labelsize=40, labelcolor=rgb_text)
            axes[l, c].xaxis.offsetText.set_fontsize(40)

        (c, l) = (0, l+1)

    plt.show()


def display_confusion_matrix(classifier, x_test: np.array, y_test: np.array, class_names: [str], x_label: str, y_label: str):
    """
    Affiche la matrice de confusion normalisée

    Positional arguments : 
    -------------------------------------
    classifier : estimator instance : modèle à tester
    x_test : np.array : données d'entrée (testing set)
    y_test : np.array : données de sortie (testing set)
    class_names : list of strings : liste des classes dans lesquelles on cherche à ranger les variables
    x_label : str : titre de l'axe des abscisses
    y_label : str : titre de l'axe des ordonnées
    """

    sns.set_theme(style="white")
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        x_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.YlGnBu,
        normalize="true",
    )
    disp.ax_.set_title("Matrice de confusion (normalisée)",
                       fontsize=20, pad=15, fontname='Corbel')
    disp.ax_.set_ylabel(y_label, fontsize=16, fontname='Corbel')
    disp.ax_.set_xlabel(x_label, fontsize=16, fontname='Corbel')
    disp.ax_.tick_params(axis='both', which='major', labelsize=14)

    disp.figure_.set_figheight(6)
    disp.figure_.set_figwidth(10)

    plt.show()


def plot_bar_to_compare_frequency(dataset: pd.DataFrame, title: str, x_label: str, hue: str):
    """
    Affiche un diagramme à barre de la fréquence de chaque variable

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données à afficher
    title : str : titre du graphique
    x_label : str : titre de l'axe des abscisses
    hue : str : nom de la colonne selon laquelle colorer le graphique
    """

    sns.set_theme(style='white')
    plt.figure(figsize=(10, 6))

    colors = sns.color_palette("husl", 8)[0:3]

    ax = sns.barplot(x=x_label, y='fréquence', hue=hue,
                     data=dataset, palette=colors)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', size=14)

    plt.title(title, size=20, fontname='Corbel', pad=40)
    ax.set(yticklabels=[])
    sns.despine(left=True)
    plt.ylabel('Fréquence(%)', fontsize=18, fontname='Corbel')
    plt.xlabel(x_label, fontsize=18, fontname='Corbel')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

    plt.show()


def plot_bar_to_compare_frequency_(dataset1: pd.DataFrame, dataset2: pd.DataFrame, 
                                   column_to_compare: str, categ1: str, categ2: str, title: str, 
                                   x_label: str, hue: str, colors=sns.color_palette("husl", 8)[0:3]):
    """
    Affiche un diagramme à barre de la fréquence de chaque variable

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : dataframe contenant les données à afficher
    title : str : titre du graphique
    x_label : str : titre de l'axe des abscisses
    hue : str : nom de la colonne selon laquelle colorer le graphique
    """

    sns.set_theme(style='white')
    plt.figure(figsize=(10, 6))

    comparison_df = pd.DataFrame(dataset1).merge(
        pd.DataFrame(dataset2), right_index=True, left_index=True)
    comparison_df = comparison_df.sort_index().reset_index()
    comparison_df.rename(columns={column_to_compare + '_x': categ1,
                         column_to_compare + '_y': categ2, 'index': x_label}, inplace=True)
    comparison_df = pd.melt(comparison_df, id_vars=x_label,
                            var_name=hue, value_name="fréquence")
    comparison_df['fréquence'] = comparison_df['fréquence'] * 100

    ax = sns.barplot(x=x_label, y='fréquence', hue=hue,
                     data=comparison_df, palette=colors)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', size=14)

    plt.title(title, size=20, fontname='Corbel', pad=40)
    ax.set(yticklabels=[])
    sns.despine(left=True)
    plt.ylabel('Fréquence(%)', fontsize=18, fontname='Corbel')
    plt.xlabel(x_label, fontsize=18, fontname='Corbel')
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

    plt.show()


def plot_empirical_distribution(column_to_plot: pd.Series, color: tuple, titles: dict, figsize: tuple, vertical=True):
    """
    Affiche un histogramme de la distribution empirique de la variable choisie

    Positional arguments : 
    -------------------------------------
    column_to_plot : np.array : valeurs observées
    color : tuple : couleur des barres de l'histogramme
    titles : dict : titres du graphique et des axes -
    ex: {'chart_title': 'blabla', 'y_title': 'blabla', 'x_title': 'blabla'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    vertical : bool : True pour afficher l'histogramme à la verticale, False à l'horizontale
    """

    plt.figure(figsize=figsize)
    rgb_text = sns.color_palette('Greys', 15)[12]

    with sns.axes_style('white'):
        if vertical:
            ax = sns.histplot(column_to_plot, stat="percent", discrete=True,
                              shrink=.9, edgecolor=color, linewidth=3, alpha=0.4, color=color)
            ax.set(yticklabels=[])
            sns.despine(left=True)
        else:
            ax = sns.histplot(y=column_to_plot, stat="percent", discrete=True,
                              shrink=.6, edgecolor=color, linewidth=3, alpha=0.4, color=color)
            ax.set(xticklabels=[])
            sns.despine(bottom=True)

    for container in ax.containers:
        ax.bar_label(container, size=18, fmt='%.1f%%',
                     fontname='Open Sans', padding=5)

    plt.title(titles['chart_title'], size=24,
              fontname='Corbel', pad=40, color=rgb_text)
    plt.ylabel(titles['y_title'], fontsize=20,
               fontname='Corbel', color=rgb_text)
    ax.set_xlabel(titles['x_title'], rotation=0, labelpad=20,
                  fontsize=20, fontname='Corbel', color=rgb_text)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.tight_layout()
    plt.show()


def build_frequency_df_with_thresh(dataset: pd.DataFrame, column_to_count: str, thresh: float, other_label: str):
    """
    Retourne un dataframe avec la fréquence empirique de chaque modalité et 
    regroupe les modalités peu représentées (i.e. fréquence < limite choisie)

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant la variable à étudier
    column_to_count : str : nom de la colonne contenant la variable à étudier
    thresh : float : fréquence limite en dessous de laquelle les modalités 
    peu représentées sont regroupées en une seule classe
    other_label : str : nom de la nouvelle modalité (qui regroupe les modalités peu représentées)
    """
    frequency_df = dataset[[column_to_count]].copy()
    effectifs = dataset[column_to_count].value_counts(normalize=True).to_dict()
    frequency_df['frequency'] = frequency_df.apply(
        lambda row: effectifs[row[column_to_count]], axis=1)

    other = other_label.format(str(thresh * 100))

    frequency_df[column_to_count] = frequency_df.apply(lambda row: other if (
        row['frequency'] < thresh) else row[column_to_count], axis=1)
    frequency_df = frequency_df.sort_values('frequency', ascending=False)

    return frequency_df


def count_countries(dataset: pd.DataFrame, column_country: str, thresh: float, countries_dict=None):
    """
    Retourne un dataframe dans lequel chaque individu correspond à un pays avec la fréquence empirique associée

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant la variable "pays" à nettoyer et étudier
    column_country : str : nom de la colonne contenant les valeurs "pays" (qui peuvent être une liste de pays)
    thresh : float : fréquence limite en dessous de laquelle les pays peu représentés sont regroupées en une seule classe

    Optional arguments : 
    -------------------------------------
    countries_dict : dict : dictionnaire des pays à corriger - 
    ex : countries_dict = {'espagne': 'spain', 'spanien': 'spain'}
    si renseigné nettoie variable 'countries' : supprime tags et met tous les noms de pays dans une même langue
    """
    subset = dataset[[column_country]].copy()
    subset['count sep'] = subset[column_country].str.count(',')

    countries_df = subset[column_country].str.split(
        ',', n=int(subset['count sep'].max()), expand=True)
    all_countries = []

    for column in countries_df.columns:
        countries = countries_df[column].dropna().values

        all_countries.extend(countries)

    count_df = pd.DataFrame(data=all_countries, columns=['countries'])

    if countries_dict:
        count_df['countries'] = count_df['countries'].str.replace(
            r'^[a-z]{2}:', "", regex=True)
        count_df['countries'] = count_df.apply(lambda row: countries_dict[row['countries']]
                                               if row['countries'] in countries_dict else row['countries'], axis=1)

    count_df = build_frequency_df_with_thresh(
        count_df, 'countries', thresh, 'other\n(< {} %)')

    return count_df


def plot_boxplot_by_dimension(dataset: pd.DataFrame, dimensions: [str], column_nb: int, 
                              title: str, figsize: tuple, y_column=None, sharey=False):
    """
    Affiche des boxplots (un graphique différent par variable étudiée) 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les variables à afficher
    dimensions : list of strings : listes des variables à afficher 
    column_nb : int : nombre de graphique par ligne
    title : str : titre principal (suptitle)
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    y_column : str : nom de la colonne contenant les catégories 
    (pour agréger les données par catégorie et afficher plusieurs boxplots par graphique)
    sharey : bool : True pour que les graphiques partagent tous le même axe des ordonnées
    """

    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    if y_column:
        dico_param = {'wspace': 0.1, 'hspace': 0.5, 'top': 0.91}
    else:
        dico_param = {'wspace': 0.2, 'hspace': 1.8, 'top': 0.85}

    with plt.style.context('seaborn-white'):
        plt.rcParams['axes.labelpad'] = '40'
        sns.set_theme(style='whitegrid', palette='deep')

        fig, axes = plt.subplots(
            ceil(len(dimensions)/column_nb), column_nb, figsize=figsize, sharey=sharey)
        fig.tight_layout()
        suptitle_text = 'Boxplot ' + title
        fig.suptitle(suptitle_text, fontname='Arial Rounded MT Bold',
                     fontsize=60, color=rgb_text)
        plt.subplots_adjust(left=None, bottom=None, right=None,
                            top=dico_param['top'], wspace=dico_param['wspace'], hspace=dico_param['hspace'])

        (l, c) = (0, 0)

        for dimension in dimensions:
            sns.boxplot(data=dataset, x=dimension, y=y_column, ax=axes[l, c],
                        orient='h',
                        showfliers=False,
                        medianprops={"color": "coral", 'linewidth': 4.0},
                        showmeans=True,
                        meanprops={'marker': 'o', 'markeredgecolor': 'black',
                        'markerfacecolor': 'coral', 'markersize': 20},
                        boxprops={'edgecolor': 'black', 'linewidth': 4.0},
                        capprops={'color': 'black', 'linewidth': 4.0},
                        whiskerprops={'color': 'black', 'linewidth': 4.0})

            axes[l, c].set_title(dimension, fontname='Corbel',
                                 color=rgb_text, fontsize=45, pad=50)
            axes[l, c].set_xlabel(
                None, fontsize=40, fontname='Corbel', color=rgb_text)
            axes[l, c].set_ylabel(y_column, fontsize=40,
                                  fontname='Corbel', color=rgb_text)

            axes[l, c].tick_params(
                axis='both', which='major', labelsize=40, labelcolor=rgb_text)
            axes[l, c].xaxis.offsetText.set_fontsize(40)

            (c, l) = (0, l+1) if c == column_nb-1 else (c+1, l)

    plt.show()


def build_xi_table(contingence_table: pd.DataFrame):
    """
    Retourne un dataframe (contenant la contribution à la non-indépendance
    pour chaque case du tableau de contingence) et la statistique du chi-2 associée

    Positional arguments : 
    -------------------------------------
    contingence_table : pd.DataFrame : tableau de contingence
    """

    distribution_marginale_x = contingence_table.loc[:, ['Total']]
    distribution_marginale_y = contingence_table.loc[['Total'], :]

    independance_table = distribution_marginale_x.dot(
        distribution_marginale_y) / contingence_table['Total'].loc['Total']

    xi_ij = (contingence_table-independance_table)**2 / independance_table
    xi_n = xi_ij.sum().sum()
    xi_table = xi_ij/xi_n

    return xi_table, xi_n


def plot_heatmap(data: pd.DataFrame, vmax: float, titles: dict, figsize: tuple, fmt: str, 
                 annotation=True, vmin=0.0, palette="rocket_r", square=False):
    """
    Affiche une heatmap 

    Positional arguments : 
    -------------------------------------
    data : pd.DataFrame : jeu de données contenant les valeurs pour colorer la heatmap
    vmax : float : valeur maximale de l'échelle des couleurs
    titles : dict : titres du graphique et des axes - 
    ex: {'chart_title': 'blabla', 'y_title': 'blabla', 'x_title': 'blabla'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    fmt : str : format annotations 

    Optional arguments : 
    -------------------------------------
    annotation : bool or pd.DataFrame : valeurs à afficher dans les cases de la heatmap - True : utilise data 
    vmin : float : valeur minimale de l'échelle des couleurs
    palette : str : couleurs de la heatmap
    square : bool : affiche les cases de la heatmap en carré
    """

    plt.figure(figsize=figsize)

    with sns.axes_style('white'):
        ax = sns.heatmap(data, annot=annotation, vmin=vmin, vmax=vmax, cmap=sns.color_palette(palette, as_cmap=True),
                         annot_kws={"fontsize": 16, 'fontname': 'Open Sans'}, 
                         linewidth=1, linecolor='w', fmt=fmt, square=square)

    if fmt == 'd':
        for t in ax.texts:
            t.set_text('{:_}'.format(int(t.get_text())))
    plt.title(titles['chart_title'], size=28, fontname='Corbel', pad=40)
    plt.xlabel(titles['x_title'], fontname='Corbel', fontsize=24, labelpad=20)
    ax.xaxis.set_label_position('top')
    plt.ylabel(titles['y_title'], fontname='Corbel', fontsize=24, labelpad=20)
    plt.tick_params(axis='both', which='major', labelsize=14,
                    labeltop=True,  labelbottom=False)

    plt.show()


def plot_hist_and_donut(dataset: pd.DataFrame, numeric_var: str, categ_var: str, 
                        palette: [str], text_color: str, titles: dict, figsize: tuple):
    """
    Affiche un histogramme de la distribution empirique de la variable 
    quantitative coloré en fonction de la variable qualitative 
    et un donut de la répartition de la variable qualitative

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    numeric_var : str : nom de la colonne contenant les valeurs de la variable quantitative
    categ_var : str : nom de la colonne contenant les valeurs de la variable qualitative

    palette : list of strings : couleurs des modalités prises par la variable qualitative
    text_color : str : couleur du texte
    titles : dict : titres du graphique et des axes - 
    ex: {'suptitle': 'blabla', 'hist_title': 'blabla', 'donut_title': 'blabla'}
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    """
    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(titles['suptitle'], fontname='Corbel', fontsize=30)
        plt.rcParams.update({'axes.labelcolor': text_color, 
                             'axes.titlecolor': text_color, 
                             'legend.labelcolor': text_color,
                             'axes.titlesize': 16, 'axes.labelpad': 10})

    sns.histplot(data=dataset, x=numeric_var,
                 hue=categ_var, palette=palette, ax=axes[0])
    axes[0].set_title(titles['hist_title'],
                      fontname='Corbel', fontsize=20, pad=20)
    axes[0].set_xticks([-15, 0, 3, 11, 19, 40])
    axes[0].tick_params(axis='both', which='major',
                        labelsize=14, labelcolor=text_color)

    pie_series = dataset[categ_var].value_counts(sort=False, normalize=True)
    patches, texts, autotexts = axes[1].pie(pie_series, labels=pie_series.index, autopct='%.0f%%', colors=palette,
                                            explode=(0.05, 0.05, 0.05, 0.05, 0.05), startangle=90, pctdistance=0.85,
                                            textprops={
                                                'fontsize': 20, 'color': text_color, 'fontname': 'Open Sans'},
                                            wedgeprops={'alpha': 0.6})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(23)

    axes[1].set_title(titles['donut_title'],
                      fontname='Corbel', fontsize=20, pad=20)
    axes[1].axis('equal')
    axes[1].add_artist(plt.Circle((0, 0), 0.7, fc='white'))

    plt.tight_layout()
    plt.show()


def extract_country_data(dataset: pd.DataFrame, column_country: str, list_countries: [list], column_to_keep: str):
    """
    Retourne un dataframe dans lequel chaque individu correspond à un pays 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant la variable "pays" à nettoyer et étudier
    column_country : str : nom de la colonne contenant les valeurs "pays" (qui peuvent être une liste de pays)
    list_countries : list of lists : liste des pays à conserver
    column_to_keep : str or list of strings : nom variable(s) à reporter dans ce nouveau dataframe
    """
    mask_no_missing_country = ~dataset[column_country].isnull()
    countries_df = pd.DataFrame()

    for country in list_countries:
        mask_country = dataset[column_country].str.contains(
            '|'.join(country), case=False, regex=True)
        country_df = pd.DataFrame(
            dataset.loc[(mask_country) & (mask_no_missing_country)][column_to_keep])
        country_df['country'] = country[0]
        country_df['country_label'] = country[0] + \
            '\n(n={:_})'.format(country_df.shape[0])
        countries_df = pd.concat([countries_df, country_df], ignore_index=True)

    return countries_df


def plot_boxplot(dataset: pd.DataFrame, numeric_var: str, title: str, figsize: tuple, categ_var=None, palette='Set2'):
    """
    Affiche un graphique avec un ou plusieurs boxplot

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    numeric_var : str : nom de la colonne contenant les valeurs dont on veut étudier la distribution
    titles : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    categ_var : str : nom de la colonne contenant les catégories 
    (si on souhaite regrouper les variables numériques par catégorie)
    palette : str or list of strings : nom de la palette seaborn utilisée ou liste de couleurs personnalisées
    """
    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=figsize)
    plt.rcParams['axes.labelpad'] = '30'

    ax = sns.boxplot(data=dataset, x=numeric_var, y=categ_var,
                     orient='h', palette=palette, saturation=0.95,
                     showfliers=False,
                     medianprops={"color": "#c2ecff", 'linewidth': 3.0},
                     showmeans=True,
                     meanprops={'marker': 'o', 'markeredgecolor': 'black',
                                'markerfacecolor': '#c2ecff', 'markersize': 10},
                     boxprops={'edgecolor': 'black', 'linewidth': 1.5},
                     capprops={'color': 'black', 'linewidth': 1.5},
                     whiskerprops={'color': 'black', 'linewidth': 1.5})

    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .6))

    plt.title(title, fontname='Corbel', color=rgb_text, fontsize=26, pad=20)
    plt.xlabel(numeric_var, fontsize=20, fontname='Corbel', color=rgb_text)
    plt.ylabel(categ_var, fontsize=20, fontname='Corbel', color=rgb_text)
    ax.tick_params(axis='both', which='major',
                   labelsize=16, labelcolor=rgb_text)

    plt.show()


def plot_violinplot(dataset: pd.DataFrame, numeric_var: str, title: str, 
                    figsize: tuple, categ_var=None, palette='Set2'):
    """
    Affiche un graphique avec un ou plusieurs boxplot

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à afficher
    numeric_var : str : nom de la colonne contenant les valeurs dont on veut étudier la distribution
    titles : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    categ_var : str : nom de la colonne contenant les catégories 
    (si on souhaite regrouper les variables numériques par catégorie)
    palette : str or list of strings : nom de la palette seaborn utilisée ou liste de couleurs personnalisées
    """
    color_list_text = sns.color_palette('Greys', 15)
    rgb_text = color_list_text[12]

    sns.set_theme(style='whitegrid')
    plt.figure(figsize=figsize)

    ax = sns.violinplot(data=dataset, y=numeric_var, x=categ_var,
                        orient='v', linewidth=2.5, width=1, palette=palette)

    plt.title(title, fontname='Corbel', color=rgb_text, fontsize=26, pad=20)
    plt.xlabel(categ_var, fontsize=20, fontname='Corbel', color=rgb_text)
    plt.ylabel(numeric_var, fontsize=20, fontname='Corbel', color=rgb_text)
    ax.tick_params(axis='both', which='major',
                   labelsize=16, labelcolor=rgb_text)

    plt.show()


def get_anova_stats(dataset: pd.DataFrame, numeric_var: str, categ_var: str):
    """
    Retourne des statistiques utiles pour faire un test statistique ANOVA 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à tester
    numeric_var : str : nom de la colonne contenant les valeurs observées de la variable quantitative 
    categ_var : str : nom de la colonne contenant les modalités observées de la variables qualitative
    """
    subset = dataset[[categ_var, numeric_var]].copy().rename(
        columns={numeric_var: 'y', categ_var: 'category'})
    y_mean = subset['y'].mean()

    SCE = 0

    for category in subset['category'].unique():

        mask_c = subset['category'] == category
        subset.loc[mask_c, 'y_estimated'] = subset.loc[mask_c, 'y'].mean()

        SCE += subset.loc[mask_c].shape[0] * \
            (subset.loc[mask_c, 'y'].mean() - y_mean)**2

    subset['y-y_mean'] = subset.apply(lambda row: (row['y']-y_mean)**2, axis=1)
    subset['y-y_estimated'] = subset.apply(
        lambda row: (row['y']-row['y_estimated'])**2, axis=1)

    SCT = subset['y-y_mean'].sum()
    SCR = subset['y-y_estimated'].sum()

    k = len(subset['category'].unique())
    N = subset.shape[0]

    eta_squared = SCE / SCT
    omega_squared = (SCE - (k-1 * SCR/(N-k))) / (SCT + SCR/(N-k))

    F = SCE/(k-1) * (N-k)/SCR
    p_value = st.f.sf(F, k-1, N-k)

    return [numeric_var, SCT, SCE, SCR, F, p_value, eta_squared, omega_squared]


def build_anova_stats_df(dataset: pd.DataFrame, numeric_vars: [str], categ_var: str):
    """
    Retourne un dataframe contenant des statistiques utiles pour faire un ou plusieurs test(s) statistique(s) ANOVA 

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données contenant les valeurs à tester
    numeric_var : list of strings : liste des variables quantitatives 
    categ_var : str : nom de la colonne contenant les modalités observées de la variables qualitative
    """
    stats = [get_anova_stats(dataset, n_v, categ_var) for n_v in numeric_vars]
    anova_stats_df = pd.DataFrame(stats, columns=[
                                  'numeric_var', 'SCT', 'SCE', 'SCR', 'F', 'p_value', 'eta_squared', 'omega_squared'])

    return anova_stats_df


def plot_heatmap_correlation_matrix(correlation_matrix: pd.DataFrame, title: str, figsize: tuple, palette: str):
    """
    Affiche la matrice de corrélation sous forme de heatmap

    Positional arguments : 
    -------------------------------------
    correlation_matrix : pd.DataFrame : matrice de corrélation
    titles : str : titres du graphique
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    palette : str : palette seaborn utilisée
    """

    sns.set_theme(style='white')
    plt.figure(figsize=figsize)

    mask_upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    ax = sns.heatmap(correlation_matrix, annot=True, mask=mask_upper_triangle, vmin=-1, vmax=1, center=0,
                     cmap=sns.color_palette(palette, as_cmap=True),
                     annot_kws={"fontsize": 16, 'fontname': 'Open Sans'},
                     cbar_kws={"shrink": .5},
                     linewidth=1.5, linecolor='w', fmt='.2f', square=True)

    plt.title(title, size=20, fontname='Corbel', pad=20)

    plt.show()


def plot_screeplot(pca: sklearn.decomposition.PCA, n_components: int, figsize: tuple, titles: dict, color_bar: str):
    """
    Affiche l'éboulis des valeurs propres avec la courbe de la somme cumulée des inertie

    Positional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : modèle d'analyse en composantes principales (déjà entrainé)
    n_components : int : nombre d'axes d'inertie
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    titles : dict : titres du graphique et des axes - 
    ex: {'chart_title': 'blabla', 'y_title': 'blabla', 'x_title': 'blabla'}
    color_bar : str : couleur utilisée pour le diagramme à bar
    """
    scree = (pca.explained_variance_ratio_ * 100).round(2)
    scree_cum = scree.cumsum().round(2)
    x_list = range(1, n_components + 1)

    rgb_text = sns.color_palette('Greys', 15)[12]

    with plt.style.context('seaborn-white'):
        sns.set_theme(style='whitegrid')
        plt.rcParams.update(
            {'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.titlesize': 18})
        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(x_list, scree, color=color_bar)
        ax.plot(x_list, scree_cum, color='coral', marker='o', markerfacecolor='white',
                markeredgecolor='coral', markersize=18, markeredgewidth=2)
        ax.text(6, 89, "variance cumulée", fontsize=20,
                color='coral', fontname='Corbel')

    plt.title(titles['chart_title'], fontname='Corbel',
              fontsize=23, pad=20, color=rgb_text)
    plt.ylabel(titles['y_label'], color=rgb_text, fontsize=18)
    plt.xlabel(titles['x_label'], color=rgb_text, fontsize=18)
    plt.grid(False, axis='x')

    plt.show()


def adjust_text(texts: list):
    """
    Retourne la liste des annotations d'un graphique avec des 
    coordonnées ajustées pour éviter que les annotations ne se superposent

    Positional arguments : 
    -------------------------------------
    texts : list of matplotlib Text objects : liste des annotations à ajuster
    """
    for index, text in enumerate(texts):
        for text_next in texts[index + 1:]:
            x_text = text.get_position()[0]
            y_text = text.get_position()[1]

            x_text_next = text_next.get_position()[0]
            y_text_next = text_next.get_position()[1]

            if abs(x_text - x_text_next) < 0.12 and abs(y_text-y_text_next) < 0.12:
                text.set_position((x_text - 0.20, y_text - 0.20))
    return texts


def plot_correlation_circle(pca: sklearn.decomposition.PCA, x: int, y: int, figsize: tuple, features: [str], arrow_color: str):
    """
    Affiche le cercle des corrélations

    Positional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : modèle d'analyse en composantes principales (déjà entrainé)
    x : int : index de l'axe d'inertie affiché en abscisse 
    y : int : index de l'axe d'inertie affiché en ordonnée
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)
    features : list of strings : liste des variables initiales à projeter sur le cercle des corrélations
    arrow_color : str : couleur utilisée pour les flèches
    """
    with plt.style.context('seaborn-white'):
        plt.rcParams['axes.labelpad'] = '20'
        sns.set_theme(style='ticks', palette='colorblind')
        fig, ax = plt.subplots(figsize=figsize)
        texts = []

        # pour chaque variable on trace une flèche
        for i in range(0, pca.components_.shape[1]):
            ax.arrow(0, 0,  # abscisse, ordonnée origine de la flèche
                     # abscisse, ordonnée du bout de la flèche (coef correlation avec Fx, Fy)
                     pca.components_[x, i], pca.components_[y, i],
                     head_width=0.07,
                     head_length=0.07,
                     width=0.01,
                     linewidth=.5,
                     color=arrow_color)

            texts.append(plt.text(pca.components_[x, i] + 0.02,
                         pca.components_[y, i] + 0.05,
                         features[i], fontname='Corbel', fontsize=14, color='black'))  
            # ajoute une étiquette avec le nom de la variable

        texts = adjust_text(texts)

        # affichage des axes du graph
        plt.plot([-1, 1], [0, 0], color='#D3D3D3', ls='--')
        plt.plot([0, 0], [-1, 1], color='#D3D3D3', ls='--')

        # nom des axes, avec le pourcentage d'inertie expliqué par l'axe
        plt.xlabel('F{} ({}%)'.format(
            x+1, round(100*pca.explained_variance_ratio_[x], 1)), fontsize=16, fontname='Corbel')
        plt.ylabel('F{} ({}%)'.format(
            y+1, round(100*pca.explained_variance_ratio_[y], 1)), fontsize=16, fontname='Corbel')

        plt.title("Cercle des corrélations (F{} et F{})".format(
            x+1, y+1), fontsize=20, fontname='Corbel')

        # Trace le cercle
        # renvoie une liste de 100 angles entre 0° et 360° régulièrement espacés
        an = np.linspace(0, 2 * np.pi, 100)
        # relie les points du cercle (abscisse = cos(angle), ordonnée=sin(angle))
        plt.plot(np.cos(an), np.sin(an))
        plt.axis('equal')
        sns.despine()

        plt.show()


def display_pca_scatterplot(pca: sklearn.decomposition.PCA, x: int, y: int, 
                            X_proj: np.array, nb_components: int, figsize: tuple,
                            with_hue=False, hue_data=None, hue_palette=None):
    """
    Affiche le nuage des individus projeté dans un plan factoriel et retourne le dataframe associé

    Positional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : modèle d'analyse en composantes principales (déjà entrainé)
    x : int : index de l'axe d'inertie affiché en abscisse 
    y : int : index de l'axe d'inertie affiché en ordonnée
    X_proj : np.array : tableau des individus projetés sur les axes d'inertie 
    nb_components : int : nombres de composantes de l'ACP
    figsize : tuple : taille de la zone d'affichage du graphique (largeur, hauteur)

    Optional arguments : 
    -------------------------------------
    with_hue : bool : True pour colorer le nuage des individus selon les modalités prises par une variable qualitative
    hue_data : np.array : valeurs prises par la variable qualitative
    hue_palette : list of strings or string : palette seaborn utilisée pour colorer données ou liste de couleurs personnalisées
    """

    plt.rcParams['axes.labelpad'] = '20'
    sns.set_theme(style='whitegrid', palette='bright')
    plt.figure(figsize=figsize)

    X_proj_df = pd.DataFrame(data=X_proj, columns=[
                             'F' + str(s) for s in range(1, nb_components + 1)])

    if with_hue:
        X_proj_df[hue_data.name] = hue_data
        X_proj_df = X_proj_df.sort_values(hue_data.name)
        sns.scatterplot(data=X_proj_df, x=X_proj_df.columns[x], y=X_proj_df.columns[y],
                        s=50, hue=hue_data.name, palette=hue_palette, alpha=0.6)
    else:
        sns.scatterplot(
            data=X_proj_df, x=X_proj_df.columns[x], y=X_proj_df.columns[y], s=50)

    x_lim = ceil(X_proj_df['F' + str(x+1)].abs().max())
    y_lim = ceil(X_proj_df['F' + str(y+1)].abs().max())
    plt.plot([-x_lim, x_lim], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-y_lim, y_lim], color='grey', ls='--')
    plt.xlim(-x_lim, x_lim)
    plt.ylim(-y_lim, y_lim)

    plt.title('Projection du nuage des individus (F{} et F{})'.format(
        x+1, y+1), fontsize=20, fontname='Corbel', pad=10)
    plt.xlabel('F{} ({}%)'.format(
        x+1, round(100*pca.explained_variance_ratio_[x], 1)), fontsize=16, fontname='Corbel')
    plt.ylabel('F{} ({}%)'.format(
        y+1, round(100*pca.explained_variance_ratio_[y], 1)), fontsize=16, fontname='Corbel')
    sns.despine(left=True, right=True)

    plt.show()

    return X_proj_df


def filter_words(words: [str]):
    """
    Retourne une liste de mots sans les stopwords i.e. les mots qui peuvent être ignorés sans altérer le sens d'une phrase

    Positional arguments : 
    -------------------------------------
    words : list of strings : liste des mots à filtrer
    """
    stopwords_english = set(stopwords.words('english'))
    stopwords_french = set(stopwords.words('french'))
    wordsFiltered = []
    for w in words:
        if (w not in stopwords_english) and (w not in stopwords_french):
            wordsFiltered.append(w)

    return wordsFiltered


def build_vocab(sentences: [str]):
    """
    Retourne une liste de mots en minuscules, sans ponctuation, sans stopwords à partir d'une liste de phrases

    Positional arguments : 
    -------------------------------------
    sentences : list of strings : liste de phrases à transformer en une liste de vocabulaire
    """
    vocab = []
    for sentence in sentences:
        words = word_tokenize(normalize_string(sentence, True))
        vocab.extend(w for w in filter_words(words) if w not in vocab)

    return vocab


def build_bag_of_words_df(sentences_vocab: [str], sentences_to_check: pd.Series, index_series: pd.Series):
    """
    Retourne un tableau représentant les textes selon l'approche bags of words

    Positional arguments : 
    -------------------------------------
    sentences_vocab : list of strings : liste des phrases à mettre en colonne 
    sentences_to_check : pd.Series : phrases à mettre en ligne 
    index_series : pd.Series : clé d'unicité (à mettre en index)
    """
    vocab = build_vocab(sentences_vocab)
    bag_vectors = []

    for sentence in sentences_to_check:
        words = word_tokenize(normalize_string(sentence, True))
        bag_vector = np.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] = 1
        bag_vectors.append(bag_vector)

    bag_of_words_df = pd.DataFrame(
        data=bag_vectors, columns=vocab, index=index_series)

    return bag_of_words_df


def display_better_products(dataset: pd.DataFrame, nutrigrade_colors: dict, 
                            categ_col: str, nutri_col: str, name_col: str, image_col: str):
    """
    Affiche les produits avec une meilleure valeur nutritionnelle

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données 
    nutrigrade_colors : dict : couleurs associée à chaque nutrigrade - 
    ex : {'A':'#00BF00', 'B':'#73FF00', 'C':'#FFCC00', 'D':'#FF6600','E':'#FF0000'}
    categ_col : str : nom de la colonne contenant les familles d'aliments
    nutri_col : str : nom de la colonne contenant les nutrigrades
    name_col : str : nom de la colonne contenant les noms des produits
    image_col : str : nom de la colonne contenant les images des produits
    """
    dataset.reset_index(inplace=True)

    fig, axes = plt.subplots(dataset.shape[0], 3, figsize=(17, 17))
    fig.suptitle('Meilleures Alternatives ({})'.format(
        dataset[categ_col].unique()[0]), fontname='Corbel', fontsize=30)
    sns.set_theme(style='white')
    plt.subplots_adjust(left=None, bottom=None, right=None,
                        top=0.91, wspace=0.1, hspace=0.5)

    for i, product in dataset.iterrows():
        axes[i, 0].set_facecolor(sns.desaturate(
            nutrigrade_colors[product[nutri_col]], 0.85))
        axes[i, 0].set_title(
            'NutriScore', fontname='Corbel', fontsize=20, pad=10)
        axes[i, 0].text(0.5, 0.45, product[nutri_col], fontsize=150, fontname='Corbel', fontweight='bold',
                        color='white', horizontalalignment='center', verticalalignment='center')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        product_name = textwrap.fill(product[name_col], 14)
        axes[i, 1].text(0.5, 0.5, product_name, fontsize=30, fontname='Corbel',
                        horizontalalignment='center', verticalalignment='center')
        axes[i, 1].set_title(
            'Nom du produit', fontname='Corbel', fontsize=20, pad=10)
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        url = product[image_col]
        if isinstance(url, str):
            ssl._create_default_https_context = ssl._create_unverified_context
            img = io.imread(url)
            axes[i, 2].imshow(img)
        else:
            axes[i, 2].text(0.5, 0.5, 'No Image', fontsize=30, fontname='Corbel',
                            horizontalalignment='center', verticalalignment='center')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        axes[i, 2].set_title('Image', fontname='Corbel', fontsize=20, pad=10)

    plt.show()


def application_nutriscore(dataset: pd.DataFrame, better_products_n: int, 
                           nutrigrade_colors: dict, key_col: str, categ_col: str,
                           nutri_col: str, additives_col: str, name_col: str, image_col: str):
    """
    Idée d'application au service de la santé publique

    Positional arguments : 
    -------------------------------------
    dataset : pd.DataFrame : jeu de données 
    better_products_n : int : nombre de produits avec une meilleure valeur nutritionnelle que l'on souhaite afficher
    nutrigrade_colors : dict : couleurs associée à chaque nutrigrade - 
    ex : {'A':'#00BF00', 'B':'#73FF00', 'C':'#FFCC00', 'D':'#FF6600','E':'#FF0000'}
    key_col : str : nom de la colonne contenant les codes barres 
    categ_col : str : nom de la colonne contenant les familles d'aliments
    nutri_col : str : nom de la colonne contenant les nutrigrades
    additives_col : str : nom de la colonne contenant le nombre d'additifs
    name_col : str : nom de la colonne contenant les noms des produits
    image_col : str : nom de la colonne contenant les images des produits
    """
    try:
        product_code = float(input("Veuillez entrer le code-barres: "))
    except ValueError:
        print("Le code-barres doit être un nombre")
        return

    mask_code = (dataset[key_col] == product_code)
    product_info = dataset.loc[mask_code].copy()

    if product_info.shape[0] == 0:
        print('Le produit demandé n\'existe pas dans la base de données \U0001F615')
    else:
        print('Ce code barre correspond à :',
              product_info[name_col].values[0], '- NutriScore', product_info[nutri_col].values[0])

        mask_nutrigrade = (dataset[nutri_col] <
                           product_info[nutri_col].values[0])
        mask_category = (dataset[categ_col].isin(product_info[categ_col]))

        better_products = dataset.loc[mask_category & mask_nutrigrade]
        if better_products.shape[0] == 0:
            print(
                'Pas de produits avec une meilleure valeur nutritionnelle dans la base de donnée \U0001F622')
        else:
            print('Recherche de meilleures alternatives \U0001F9D0 \n')
            bag_of_words = build_bag_of_words_df(
                [product_info[name_col].values[0]], better_products[name_col], better_products[key_col])
            bag_of_words['common_words_n'] = bag_of_words.sum(axis=1)
            better_products = better_products.merge(bag_of_words, on=key_col)
            better_products = better_products.sort_values(
                ['common_words_n', nutri_col, additives_col], ascending=[False, True, True])

            if better_products.shape[0] > better_products_n:
                display_better_products(
                    better_products.iloc[:better_products_n, :], nutrigrade_colors, categ_col, nutri_col, name_col, image_col)
            else:
                display_better_products(
                    better_products, nutrigrade_colors, categ_col, nutri_col, name_col, image_col)