# Data

本專案的資料**未包含在此 repository 中**，請依下方說明自行下載後放在此資料夾：

```
data/
├── train.csv
└── test.csv
```

## 資料來源

資料源自 **UCI Human Activity Recognition Using Smartphones Dataset**：

- UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

30 位受試者將 Samsung Galaxy S II 配戴於腰部，執行六種日常活動（WALKING、WALKING_UPSTAIRS、WALKING_DOWNSTAIRS、SITTING、STANDING、LAYING），由手機內建的 accelerometer 與 gyroscope 以 50Hz 取樣，並以 2.56 秒（128 個讀數）、50% 重疊的滑動視窗切段，每個視窗萃取出 561 個時間域與頻率域特徵。

### Citation

> Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).
> A Public Domain Dataset for Human Activity Recognition Using Smartphones.
> *21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN 2013)*, Bruges, Belgium.

## 本專案使用的資料格式

本專案使用的是課程 Kaggle InClass 競賽所提供的版本。該版本沿用 UCI 原始的 train/test 切分，但做了以下處理：

| | UCI 原始資料 | 本專案（課程競賽版本） |
|---|---|---|
| 筆數 | train 7,352 / test 2,947 | 相同 |
| 特徵 | 561 個 | 相同，欄位名稱取自 `features.txt`；42 個重複名稱（共 126 欄，皆為 `bandsEnergy`）加上 `...{欄位位置}` 後綴 |
| 受試者編號 `subject` | 有 | **已移除** |
| 識別欄位 | 無 | 新增 `id` 欄位 |
| test 標籤 | 有 | **隱藏**（需預測後提交至 Kaggle） |

因此 CSV 欄位結構為：

- `train.csv`：`id`, 561 個特徵, `Activity`
- `test.csv`：`id`, 561 個特徵

## 從 UCI 原始資料重建

課程競賽為邀請制，無法公開取得。若要重現本專案，可從 UCI 下載原始資料後轉換成相同格式：

```python
import pandas as pd

root = "UCI HAR Dataset"
features = pd.read_csv(f"{root}/features.txt", sep=r"\s+", header=None)[1]

# features.txt 中有重複名稱；與競賽版本一致，重複者一律加上 "...{CSV 欄位位置}"
# （位置從 1 起算，第 1 欄為 id，故特徵 i 位於第 i + 2 欄）
dup = features.duplicated(keep=False)
features = [f"{name}...{i + 2}" if d else name
            for i, (name, d) in enumerate(zip(features, dup))]

labels = pd.read_csv(f"{root}/activity_labels.txt", sep=r"\s+", header=None, index_col=0)[1]

def load(split):
    X = pd.read_csv(f"{root}/{split}/X_{split}.txt", sep=r"\s+", header=None)
    X.columns = features
    y = pd.read_csv(f"{root}/{split}/y_{split}.txt", header=None)[0].map(labels)
    return X, y

X_train, y_train = load("train")
X_test, y_test = load("test")

train = X_train.assign(Activity=y_train)
train.insert(0, "id", range(len(train)))
test = X_test.copy()
test.insert(0, "id", range(len(train), len(train) + len(test)))

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
```

> 注意：`id` 的編號方式與課程競賽不同，因此重建後的 `submission.csv` 無法直接與 [`../submissions/submission.csv`](../submissions/submission.csv) 逐筆比對；但因為 test 標籤在 UCI 原始資料中是公開的，可以直接用 `y_test` 計算 accuracy。
