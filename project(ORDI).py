import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# A. Загрузка и первичный обзор
plt.rcParams['figure.figsize'] = (10, 6)
sns.set(style="whitegrid")

df = pd.read_csv('google_books_dataset.csv')

print(df.head())  # вывести первые/последние строки;
print(df.tail())
print(df.info())  # посмотреть структуру данных;
print(df.dtypes)

# проверить типы данных и при необходимости привести их к корректным типам.
num_cols = ['average_rating', 'ratings_count', 'text_reviews_count', 'num_pages']

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

if 'publication_date' in df.columns:
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')

# B. Обработка данных
# Пропущенные значения
print("Пропуски по столбцам:\n", df.isnull().sum())

for col in ['average_rating', 'ratings_count', 'text_reviews_count', 'num_pages']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# категориальные — модой
for col in ['publisher', 'language', 'authors']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

# Дубликаты
dups = df[df.duplicated()]
print("Кол-во дубликатов:", len(dups))

df = df.drop_duplicates()

# Выбросы(пример: ratings_count)
if 'ratings_count' in df.columns:
    z = np.abs(stats.zscore(df['ratings_count'].dropna()))
    # оставляем только записи с |z| < 3
    df_no_outliers = df.loc[df['ratings_count'].dropna().index[z < 3]]

# Статистики данных
print(df.describe(include='all'))  # описательная статистика

# уникальные значения и частоты для издателей
if 'publisher' in df.columns:
    print("Уникальных издателей:", df['publisher'].nunique())
    print(df['publisher'].value_counts().head(10))

# корреляции между числовыми признаками
corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

# C. Визуализация данных(5 графиков)

# 1. Гистограмма распределения рейтингов

if 'average_rating' in df.columns:
    sns.histplot(df['average_rating'], bins=20, kde=True)
    plt.title('Распределение среднего рейтинга книг')
    plt.xlabel('Средний рейтинг')
    plt.ylabel('Количество книг')
    plt.show()

# 2. Линейный график: средний рейтинг по годам

# if 'publication_date' in df.columns and 'average_rating' in df.columns:
#     df_year = df.copy()
#     df_year['year'] = df_year['publication_date'].dt.year
#     year_rating = df_year.groupby('year')['average_rating'].mean().dropna()
#
#     plt.plot(year_rating.index, year_rating.values, marker='o')
#     plt.title('Средний рейтинг книг по годам')
#     plt.xlabel('Год')
#     plt.ylabel('Средний рейтинг')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# Линейный график ломается из-за дат с типом object или много данных NaN.

# 3. Столбчатая диаграмма: топ‑10 издателей по количеству книг

if 'publisher' in df.columns:
    top_publishers = df['publisher'].value_counts().head(10)

    top_publishers.plot(kind='bar')
    plt.title('Топ-10 издателей по количеству книг')
    plt.xlabel('Издатель')
    plt.ylabel('Количество книг')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4. boxplot(число страниц по языку)

if 'num_pages' in df.columns and 'language' in df.columns:
    sns.boxplot(x='language', y='num_pages', data=df[df['language'].isin(df['language'].value_counts().head(5).index)])
    plt.title('Распределение числа страниц по языкам (топ‑5)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 5. heatmap корреляций
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица числовых признаков')
plt.show()

# D. Группировки и агрегации

# группировка по языку
if {'language', 'average_rating', 'ratings_count'}.issubset(df.columns):
    grp_lang = df.groupby('language').agg(
        avg_rating=('average_rating', 'mean'),
        median_rating=('average_rating', 'median'),
        books_count=('title', 'count'),
        total_ratings=('ratings_count', 'sum')
    ).sort_values('books_count', ascending=False)
    print(grp_lang.head(10))

# сводная таблица: средний рейтинг по языку и году
if 'publication_date' in df.columns and 'average_rating' in df.columns and 'language' in df.columns:
    df['year'] = df['publication_date'].dt.year
    pivot = pd.pivot_table(
        df,
        values='average_rating',
        index='language',
        columns='year',
        aggfunc='mean'
    )
    print(pivot)
