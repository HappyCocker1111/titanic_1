このノートブックは、タイタニックの生存予測を行う機械学習モデルを構築、訓練、評価するためのものです。以下は、ノートブックの内容の要約です：

# ライブラリのインポート：

必要なPythonライブラリ（例：Pandas、NumPy、LightGBMなど）をインポートしています。
データの読み込みと前処理：

# タイタニックの訓練データとテストデータを読み込んでいます。
データの欠損値を処理し、カテゴリカルデータを数値データに変換しています。

特徴量エンジニアリング：データから新しい特徴量を作成したり、既存の特徴量を変換したりすることで、モデルの性能を向上させるプロセスです。

## 欠損値の補完：

Age、Embarked、Fareなどの欠損値を補完しています。
例えば、Ageは平均値や中央値、あるいはグループごとの中央値などで補完されることがあります。
```
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
```
## カテゴリカルデータの変換：

Sex、Embarkedなどのカテゴリカルデータを数値データに変換しています。
通常、これにはOne-Hotエンコーディングやラベルエンコーディングが使われます。

## 新しい特徴量の作成：

既存の特徴量から新しい特徴量を作成することがあります。
例えば、家族の人数を表すFamilySize、タイトルを抽出したTitle、一人で乗船したかを示すIsAloneなど。
```
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

train_df['IsAlone'] = 1
train_df['IsAlone'].loc[train_df['FamilySize'] > 1] = 0
test_df['IsAlone'] = 1
test_df['IsAlone'].loc[test_df['FamilySize'] > 1] = 0

train_df['Title'] = train_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
```

## スケーリング：

数値データを標準化や正規化することで、モデルの訓練を効率化します。
例えば、StandardScalerを使って標準化します。

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_df[['Age', 'Fare']] = scaler.fit_transform(train_df[['Age', 'Fare']])
test_df[['Age', 'Fare']] = scaler.transform(test_df[['Age', 'Fare']])
```

## ドロップする特徴量：

モデルに不要な特徴量や情報漏洩につながる特徴量をドロップします。
例えば、PassengerId、Ticket、Cabinなど。
```
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
train_df = train_df.drop(drop_columns, axis=1)
test_df = test_df.drop(drop_columns, axis=1)
```

これらの特徴エンジニアリング手法により、モデルがより良い予測を行えるようにデータが準備されています。具体的なコードやプロセスは、実際のデータに応じて適宜調整されることが一般的です。

# 新しい特徴量を作成し、既存の特徴量を変換・選択しています。
モデルの訓練：このノートブックでは、LightGBMという勾配ブースティングフレームワークを使ってモデルを訓練しています。LightGBMは、効率的で高速な訓練を可能にするため、多くの機械学習タスクで人気があります。

## 1. ライブラリのインポート
まず、必要なライブラリをインポートします。
```
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, log_loss
```

##2. データの準備
訓練データとテストデータを準備します。特徴量とターゲット変数を分けます。
```
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
test_X = test_df
```

##3. データの分割
訓練データを訓練セットと検証セットに分割します。これにより、モデルの性能を評価することができます。
```
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
```

##4. モデルの訓練
LightGBMのデータセット形式に変換します。
```
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
```

モデルのパラメータを設定します。
```
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
```

モデルを訓練します。
```
model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], num_boost_round=1000, early_stopping_rounds=50)
```


##5. モデルの評価
訓練が完了したら、検証データを使ってモデルの性能を評価します。
```
y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
logloss = log_loss(y_valid, y_pred)
accuracy = accuracy_score(y_valid, (y_pred > 0.5).astype(int))

print(f'Validation Log Loss: {logloss}')
print(f'Validation Accuracy: {accuracy}')
```

##6. 予測の実行
テストデータに対して予測を行い、提出用のファイルを作成します。
```
test_pred = model.predict(test_X, num_iteration=model.best_iteration)
submission = pd.DataFrame({'PassengerId': test_df.index, 'Survived': (test_pred > 0.5).astype(int)})
submission.to_csv('submission.csv', index=False)
```

##7. 交差検証
KFoldを使って交差検証を行い、モデルの性能を安定化させます。
```
kf = KFold(n_splits=5)
models = []

for train_index, valid_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], num_boost_round=1000, early_stopping_rounds=50)
    models.append(model)
```

複数のモデルを使用して予測を行い、平均化します。
```
preds = []
for model in models:
    pred = model.predict(test_X)
    preds.append(pred)

preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis=0)
```

このように、ノートブックではLightGBMを使ってデータの前処理からモデルの訓練、評価、予測まで一連の流れを実施しています。これにより、タイタニックの生存予測を行う高性能なモデルを構築しています。


# LightGBMを使用してモデルを訓練しています。
モデルのパフォーマンスを評価するために交差検証を行っています。
予測：交差検証（Cross-Validation）は、データを複数の分割に分けて訓練と評価を行い、モデルの性能をより信頼性高く評価する方法です。最も一般的な方法はK分割交差検証（K-Fold Cross-Validation）です。K分割交差検証では、データセットをK個の「フォールド」に分割し、各フォールドを1つずつ検証セットとして使用し、残りのフォールドを訓練セットとして使用します。これをK回繰り返して、各回の評価結果を平均して最終的なモデルの性能を評価します。

訓練済みモデルを使用してテストデータの予測を行っています。
複数のモデルの予測結果を平均化して最終的な予測を行っています

##K分割交差検証の具体的な手順
以下は、具体的にLightGBMを使用してK分割交差検証を行う手順です。

##1. ライブラリのインポート
```
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, log_loss
import numpy as np
```

##2. データの準備
特徴量とターゲット変数を準備します。
```
X = train_df.drop('Survived', axis=1)
y = train_df['Survived']
test_X = test_df

```

##3. パラメータの設定
LightGBMのパラメータを設定します。
```
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
```

##4. KFoldの設定
KFoldを使用してデータをK個に分割します。
```
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

##5. モデルの訓練と評価
各フォールドごとにモデルを訓練し、検証します。
```
models = []
log_losses = []
accuracies = []

for train_index, valid_index in kf.split(X):
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
    
    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], num_boost_round=1000, early_stopping_rounds=50)
    models.append(model)
    
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    logloss = log_loss(y_valid, y_pred)
    accuracy = accuracy_score(y_valid, (y_pred > 0.5).astype(int))
    
    log_losses.append(logloss)
    accuracies.append(accuracy)

# 平均のログロスと精度を計算
average_logloss = np.mean(log_losses)
average_accuracy = np.mean(accuracies)

print(f'Average Log Loss: {average_logloss}')
print(f'Average Accuracy: {average_accuracy}')
```

##6. テストデータの予測
全てのモデルを使用してテストデータの予測を行い、その結果を平均化します。
```
preds = []
for model in models:
    pred = model.predict(test_X, num_iteration=model.best_iteration)
    preds.append(pred)

preds_array = np.array(preds)
preds_mean = np.mean(preds_array, axis=0)

submission = pd.DataFrame({'PassengerId': test_df.index, 'Survived': (preds_mean > 0.5).astype(int)})
submission.to_csv('submission.csv', index=False)
```

##まとめ
この方法では、K分割交差検証を使用して複数のモデルを訓練し、それぞれのモデルの性能を評価することで、データ全体に対するモデルの一般化性能を高めることができます。さらに、最終的な予測は複数のモデルの予測を平均化することで安定化されます。これにより、過剰適合（オーバーフィッティング）を防ぎ、より信頼性の高い予測が可能となります。





