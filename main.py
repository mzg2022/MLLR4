import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

matplotlib.use('Agg')
os.makedirs('output_figures', exist_ok=True)

# Импорт данных
client_transactions = pd.read_csv('customer_segmentation_project.csv', encoding='ISO-8859-1')
print("Размер таблицы:", client_transactions.shape)
print("\nСтруктура и типы данных:")
client_transactions.info()
client_transactions.head(5)

print("Период транзакций:", client_transactions['InvoiceDate'].min(), "до", client_transactions['InvoiceDate'].max())
print("Уникальных клиентов:", client_transactions['CustomerID'].nunique())
print("Уникальных товаров:", client_transactions['StockCode'].nunique())
print("Уникальных стран:", client_transactions['Country'].nunique())
print("\nТоп-5 стран по активности:")
print(client_transactions['Country'].value_counts().head())
print("\nДетали заказа (InvoiceNo=536365):")
print(client_transactions[client_transactions['InvoiceNo'] == '536365'].shape[0], "позиций")
client_transactions[client_transactions['InvoiceNo'] == '536365'].head()

# Обработка пропусков
print(client_transactions.isna().sum())
print("Удаление неполных записей...")
initial_count = client_transactions.shape[0]
client_transactions = client_transactions.dropna(subset=['CustomerID', 'Description'])
print(f"Удалено строк: {initial_count - client_transactions.shape[0]}")
print("Поиск дубликатов...")
initial_count = client_transactions.shape[0]
client_transactions = client_transactions.drop_duplicates()
print(f"Удалено дубликатов: {initial_count - client_transactions.shape[0]}")
print("Размер после очистки:", client_transactions.shape)

# Анализ числовых полей
print(client_transactions[['Quantity','UnitPrice']].describe(percentiles=[0.01,0.05,0.95,0.99]))
print("Возвраты (Quantity < 0):", (client_transactions['Quantity'] < 0).sum())
print("Бесплатные товары (UnitPrice = 0):", (client_transactions['UnitPrice'] == 0).sum())
print("Топ-5 по количеству:")
print(client_transactions.nlargest(5, 'Quantity')[['InvoiceNo','StockCode','Description','Quantity','UnitPrice']])

# Фильтрация данных
initial_count = client_transactions.shape[0]
client_transactions = client_transactions[(client_transactions['Quantity'] != 0) & (client_transactions['UnitPrice'] != 0)]
print(f"Удалено невалидных строк: {initial_count - client_transactions.shape[0]}")
print("Остаток возвратов:", (client_transactions['Quantity'] < 0).sum())

# Расчёт финансовых показателей
client_transactions['LineTotal'] = client_transactions['UnitPrice'] * client_transactions['Quantity']
print("Средний чек:", round(client_transactions['LineTotal'].mean(), 2))
print("Общий объём продаж:", round(client_transactions['LineTotal'].sum(), 2))

# Формирование RFM-матрицы
client_transactions['InvoiceDate'] = pd.to_datetime(client_transactions['InvoiceDate'])
cutoff_date = client_transactions['InvoiceDate'].max() + pd.Timedelta(days=1)
valid_orders = client_transactions[client_transactions['InvoiceNo'].str.startswith('C') == False]

customer_rfm = valid_orders.groupby('CustomerID').agg(
    DaysSinceLast=('InvoiceDate', lambda d: (cutoff_date - d.max()).days),
    OrderCount=('InvoiceNo', 'nunique')
).reset_index()

customer_spending = client_transactions.groupby('CustomerID').agg(TotalSpent=('LineTotal','sum')).reset_index()
customer_rfm = pd.merge(customer_rfm, customer_spending, on='CustomerID', how='left').fillna({'DaysSinceLast': np.nan, 'OrderCount': 0, 'TotalSpent': 0})
print("RFM-датасет:")
customer_rfm.head(5)
print("Всего клиентов в RFM:", customer_rfm.shape[0])
print("Клиенты с отрицательными тратами:", (customer_rfm['TotalSpent'] < 0).any())

# Визуализация распределений
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
sns.histplot(customer_rfm['DaysSinceLast'], bins=30)
plt.title("Распределение Recency")
plt.subplot(1,3,2)
sns.histplot(customer_rfm['OrderCount'], bins=30)
plt.title("Распределение Frequency")
plt.subplot(1,3,3)
sns.histplot(customer_rfm['TotalSpent'], bins=30)
plt.title("Распределение Monetary")
plt.savefig('output_figures/rfm_histograms.png', bbox_inches='tight')
plt.close()

print(customer_rfm[['DaysSinceLast','OrderCount','TotalSpent']].describe(percentiles=[0.95,0.99]))
filtered_rfm = customer_rfm[(customer_rfm['OrderCount'] <= 30) & (customer_rfm['TotalSpent'] <= 20000)].copy()
print("Удалено клиентов-аномалий:", customer_rfm.shape[0] - filtered_rfm.shape[0])
print("Остаток клиентов:", filtered_rfm.shape[0])
print(filtered_rfm[['DaysSinceLast','OrderCount','TotalSpent']].describe())

# Нормализация данных
standardizer = StandardScaler()
scaled_features = standardizer.fit_transform(filtered_rfm[['DaysSinceLast','OrderCount','TotalSpent']])

# Анализ корреляций
rfm_corr = filtered_rfm[['DaysSinceLast','OrderCount','TotalSpent']].corr()
sns.heatmap(rfm_corr, annot=True, cmap="coolwarm")
plt.title("Корреляции RFM-метрик")
plt.savefig('output_figures/rfm_correlations.png', bbox_inches='tight')
plt.close()

# Определение оптимального числа кластеров
cluster_range = range(2, 11)
wcss_values = []
silhouette_scores = []
for n in cluster_range:
    cluster_model = KMeans(n_clusters=n, init='k-means++', random_state=42)
    cluster_model.fit(scaled_features)
    wcss_values.append(cluster_model.inertia_)
    cluster_labels = cluster_model.labels_
    silhouette_scores.append(silhouette_score(scaled_features, cluster_labels))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(list(cluster_range), wcss_values, marker='o')
plt.title("Метод локтя")
plt.xlabel("Число кластеров")
plt.ylabel("WCSS")
plt.subplot(1,2,2)
plt.plot(list(cluster_range), silhouette_scores, marker='o', color='green')
plt.title("Оценка силуэта")
plt.xlabel("Число кластеров")
plt.ylabel("Средний индекс силуэта")
plt.savefig('output_figures/cluster_metrics.png', bbox_inches='tight')
plt.close()

optimal_k = cluster_range[silhouette_scores.index(max(silhouette_scores))]
print("Оптимальное число кластеров:", optimal_k, "с индексом силуэта", round(max(silhouette_scores), 3))