# %%
import pandas as pd

# The previous cell failed to download the file, so this cell will also fail.
# Once the file permissions are updated and the download is successful, this cell can be uncommented and run.
df_downloaded_2 = pd.read_csv('/opt/airflow/data/clean_data2.csv')

# %%
df = df_downloaded_2.head(100000)

# %%
df.head()

# %%
df.columns

# %%
df['star'].value_counts(dropna=False)

# %%
drop_df = df.drop(columns=['Unnamed: 0', 'ticket_id', 'photo', 'photo_after', 'address', 'subdistrict', 'province', 'timestamp', 'star', 'last_activity', 'latitude', 'longitude',
                 'type 2', 'type 3', 'organization_1', 'organization_2', 'organization_3', 'comment',
                           'district'])

# %%
drop_df.columns
drop_df.head()

# %%
for col in drop_df.columns:
  print(col, drop_df[drop_df[col].isna()].shape)

# %%
for col in drop_df.columns:
  print(col, drop_df[col].value_counts())

# %% [markdown]
# # **Clustering**

# %%
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
# ======== 1) เตรียม column ที่ใช้ =========

# Categorical ที่ควร one-hot
categorical_cols = ['state', 'type 1']

# Numerical ที่ต้อง scale (รวม coords)
numerical_cols = ['count_reopen']

# ======== 2) ตัวแปลงข้อมูล =========

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
])

# ======== 3) Fit + Transform =========

# แปลง data (จะได้เป็น array พร้อมใช้กับ K-Means)
X = preprocessor.fit_transform(drop_df)

# X คือ data matrix แบบ numerical ล้วน
# สามารถใช้ X ไป KMeans.fit(X) ได้เลย


# %%
# ======== Get feature names =========
ohe = preprocessor.named_transformers_['cat']
ohe_cols = ohe.get_feature_names_out(categorical_cols)

all_cols = numerical_cols + list(ohe_cols)

# ======== Convert to DataFrame ========
X_df = pd.DataFrame(X, columns=all_cols)

# %%
print(X_df.columns.shape)

# %%
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, make_scorer
import numpy as np

# custom scorer สำหรับ clustering
def silhouette_scorer(estimator, X):
    labels = estimator.fit_predict(X)
    # silhouette_score ต้องมี cluster มากกว่า 1
    if len(np.unique(labels)) > 1:
        return silhouette_score(X, labels)
    else:
        return -1  # กรณี cluster = 1

scorer = make_scorer(silhouette_scorer, greater_is_better=True)

# KMeans
kmeans = KMeans(random_state=42)

# Grid search parameter
param_grid = {
    'n_clusters': [5],
    'init': ['k-means++', 'random'],
    'n_init': [10]
}

grid = GridSearchCV(kmeans, param_grid, scoring=scorer, cv=3, n_jobs=-1)
grid.fit(X_df)  # ใช้ dataframe ทั้งหมดที่เตรียมแล้ว

print("Best params:", grid.best_params_)
print("Best silhouette score:", grid.best_score_)


# %%
drop_df['cluster'] = grid.best_estimator_.predict(X_df)

# %%
drop_df['district'] = df_downloaded_2['district']
drop_df['latitude'] = df_downloaded_2['latitude']
drop_df['longitude'] = df_downloaded_2['longitude']
drop_df['star'] = df_downloaded_2['star']

# %%

# %%
cluster_df = drop_df

# %%
# กำหนด columns
num_cols = ['count_reopen']
cat_cols = ['state', 'type 1', 'district']

# loop แต่ละ cluster
for cluster_id in sorted(cluster_df['cluster'].unique()):
    print(f"\n===== Cluster {cluster_id} =====")

    # 1️⃣ Numerical features (mean)
    num_means = cluster_df.loc[cluster_df['cluster']==cluster_id, num_cols].mean()
    print("Numerical features (mean):")
    print(num_means)

    # 2️⃣ Categorical features (% ของ cluster)
    print("\nCategorical features (% of cluster):")
    for col in cat_cols:
        counts = cluster_df.loc[cluster_df['cluster']==cluster_id, col].value_counts(normalize=True) * 100
        print(f"\n{col}:")
        print(counts.sort_values(ascending=False).head(5))


# %%
cluster_df.to_csv('/opt/airflow/data/clusterd_df.csv', encoding='utf-8')

