---
title: Human Activity Recognition with Smartphone Sensor Features
---

# Human Activity Recognition with Smartphone Sensor Features

## 1. Introduction

本專案為多變量分析期末資料專案，目標是利用智慧型手機感測器訊號所萃取出的特徵，預測受試者當下所執行的人類活動類別。資料來源為 Human Activity Recognition with Smartphones dataset，原始資料由受試者將智慧型手機配戴於腰部，並透過手機內建的 accelerometer 與 gyroscope 記錄日常活動中的運動訊號。

本研究的任務是建立一個 supervised classification model，將測試資料中的每一筆觀測值分類為以下六種活動之一：

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

由於資料包含大量由時間域與頻率域訊號萃取出的 sensor features，本專案除了建立分類模型外，也透過 exploratory data analysis 與 PCA 觀察高維特徵資料的結構，並比較不同線性分類模型與降維模型在 cross-validation 上的表現。最終主要比較的模型包括 LDA、Linear SVM、Linear SVM with PCA，以及 regularized logistic regression。

---

## 2. Related Work

本次研究主要參考以下方法：

### 2.1 Linear Discriminant Analysis

Linear Discriminant Analysis，簡稱 LDA，是多變量分析中常用的線性分類方法。LDA 假設不同類別具有相同 covariance matrix，並建立 linear discriminant score 進行分類。由於本專案為六類活動分類問題，LDA 可作為一個具備統計解釋性的 baseline model。

### 2.2 Linear Support Vector Machine

Support Vector Machine，簡稱 SVM，是一種以最大化 margin 為核心概念的分類方法。本研究改用 Linear SVM 作為主要比較模型，而非 RBF SVM。Linear SVM 使用線性決策邊界，但不需要像 LDA 一樣假設各類別服從 multivariate normal distribution 或具有相同 covariance matrix。因此，Linear SVM 可以視為一個介於 LDA 與更複雜非線性 SVM 之間的穩定分類模型。

本研究使用 Linear SVM 的原因是：先前 RBF SVM 雖然在 cross-validation 上表現較高，但在競賽測試資料上的表現未必優於 LDA。因此，本研究改用較簡單且較穩定的 Linear SVM，以降低非線性模型對訓練資料分布過度貼合的風險。

### 2.3 Principal Component Analysis

Principal Component Analysis，簡稱 PCA，是一種常用的降維與視覺化方法。本研究使用 PCA 作為 EDA 工具，觀察 561 個 sensor features 是否存在低維結構，並透過 scree plot 與 score plot 分析不同活動類別在主成分空間中的分布情形。

此外，本研究也將 PCA 放入 Linear SVM pipeline 中，測試降維後的主成分特徵是否能維持分類能力。不過 PCA 是 unsupervised method，只保留最大變異方向，不一定保留最有分類能力的方向，因此 PCA-based model 被視為補充比較模型，而非主要模型。

### 2.4 Logistic Regression

Logistic Regression 是常見的線性分類模型。本研究額外比較 regularized logistic regression，作為 LDA 與 Linear SVM 之外的線性分類基準。Regularization 可限制模型係數大小或進行稀疏化，降低 overfitting 風險。

---

## 3. Material

### 3.1 Dataset Description

本研究使用 Kaggle 提供的訓練資料與測試資料：

| Dataset | Shape |
|---|---:|
| train.csv | 7352 × 563 |
| test.csv | 2947 × 562 |

其中 `train.csv` 包含：

- `id`
- 561 個 sensor-derived features
- 目標變數 `Activity`

`test.csv` 包含：

- `id`
- 561 個 sensor-derived features

測試資料中的 `Activity` 標籤為隱藏，需要由模型預測後產生 submission file。

### 3.2 Target Variable

目標變數為 `Activity`，共有六個類別：

| Activity | Count | Proportion |
|---|---:|---:|
| LAYING | 1407 | 0.1914 |
| STANDING | 1374 | 0.1869 |
| SITTING | 1286 | 0.1749 |
| WALKING | 1226 | 0.1668 |
| WALKING_UPSTAIRS | 1073 | 0.1459 |
| WALKING_DOWNSTAIRS | 986 | 0.1341 |

整體來看，六個活動類別的分布沒有嚴重不平衡，因此本研究主要使用 accuracy 作為模型評估指標。

### 3.3 Data Quality Check

資料檢查結果如下：

| Check Item | Result |
|---|---:|
| Missing values in train | 0 |
| Missing values in test | 0 |
| Duplicated rows in train | 0 |
| Duplicated rows in test | 0 |

因此，本資料集不需要進行 missing value imputation 或 duplicated data removal。

### 3.4 Feature Preprocessing

本研究將 `id` 排除於模型訓練之外，僅作為 submission file 中對應測試樣本的識別欄位。模型輸入只使用 561 個 sensor-derived features。

對於 LDA、SVM、Logistic Regression 等對尺度敏感的模型，本研究使用 `StandardScaler` 將所有特徵標準化，使每個特徵具有平均數 0 與標準差 1。

### 3.5 PCA for EDA

PCA 僅作為 EDA 與視覺化工具，不直接作為主要模型的輸入。

PCA 結果如下：

| Item | Result |
|---|---:|
| Original feature dimension | 561 |
| Number of PCs for 90% explained variance | 63 |
| Cumulative explained variance | 0.9005 |
| PC1 explained variance ratio | 0.5078 |
| PC2 explained variance ratio | 0.0658 |

PCA scree plot 顯示第一主成分解釋了約 50.78% 的總變異，代表 sensor features 中存在明顯的主要變異方向。前 63 個 principal components 可解釋約 90% 的總變異。

PCA score plot 顯示，PC1 大致可以分離靜態活動與動態活動。靜態活動如 LAYING、SITTING、STANDING 主要位於 PC1 的一側，而動態活動如 WALKING、WALKING_UPSTAIRS、WALKING_DOWNSTAIRS 主要位於另一側。不過，SITTING 與 STANDING 仍有明顯重疊，代表僅使用前兩個 principal components 無法完全區分所有活動類別。


---

## 4. Experimental

### 4.1 Experimental Setup

本研究使用 5-fold Stratified Cross-Validation 評估模型表現。Stratified K-fold 可以確保每個 fold 中六個活動類別的比例大致相同。

評估指標包括：

- Cross-validation accuracy
- Out-of-fold accuracy
- Macro F1-score
- Confusion matrix
- Classification report

最後使用完整 training data 重新訓練模型，並對 test data 預測 `Activity`，輸出格式為：

| id | Activity |
|---|---|
| test id | predicted activity |

### 4.2 Models

本研究比較以下模型：

| Model | Description |
|---|---|
| Model 1 | LDA baseline |
| Model 2 | Linear SVM with all sensor features |
| Model 3 | Linear SVM with 90% PCA |
| Model 4 | L1-Regularized Logistic Regression |

---

### 4.3 Model 1: LDA Baseline

LDA 使用標準化後的全部 sensor features 作為輸入。模型設定為：

```python
LinearDiscriminantAnalysis(
    solver="lsqr",
    shrinkage=0.003
)
```

Cross-validation 結果如下：

| Fold | Accuracy |
|---|---:|
| Fold 1 | 0.9776 |
| Fold 2 | 0.9837 |
| Fold 3 | 0.9728 |
| Fold 4 | 0.9816 |
| Fold 5 | 0.9782 |

| Metric | Value |
|---|---:|
| Mean CV accuracy | 0.9788 |
| Std CV accuracy | 0.0037 |
| OOF accuracy | 0.9788 |
| Final training accuracy | 0.9850 |

LDA 的分類表現已經相當好，尤其對於 LAYING、WALKING、WALKING_DOWNSTAIRS、WALKING_UPSTAIRS 等類別有接近完美的分類效果。然而，confusion matrix 顯示 SITTING 與 STANDING 之間仍存在較明顯的混淆。

---

### 4.4 Model 2: Linear SVM with All Sensor Features

Linear SVM 使用標準化後的全部 sensor features 作為輸入。模型設定為：

```python
SVC(
    kernel="linear",
    C=3,
    cache_size=1000
)
```

Cross-validation 結果如下：

| Fold | Accuracy |
|---|---:|
| Fold 1 | 0.9789 |
| Fold 2 | 0.9884 |
| Fold 3 | 0.9823 |
| Fold 4 | 0.9871 |
| Fold 5 | 0.9830 |

| Metric | Value |
|---|---:|
| Mean CV accuracy | 0.9840 |
| Std CV accuracy | 0.0034 |
| OOF accuracy | 0.9839 |
| OOF Macro F1 | 0.9850 |
| Final training accuracy | 0.9985 |

Linear SVM 在 cross-validation 中取得比 LDA 更高的 accuracy，顯示最大化 margin 的線性分類器能有效利用高維 sensor features。相較於 LDA，Linear SVM 不需要常態分布與共同 covariance matrix 的假設，因此在保留線性模型穩定性的同時，也具有較強的分類彈性。

從 classification report 來看，LAYING、WALKING、WALKING_DOWNSTAIRS、WALKING_UPSTAIRS 皆有接近完美的表現。主要錯誤仍集中在 SITTING 與 STANDING，這兩類活動皆屬於靜態姿勢，手機感測器訊號差異相對較小。

Linear SVM 的 final training accuracy 為 0.9985，高於 cross-validation accuracy，因此仍需要注意模型是否對訓練資料產生一定程度的 overfitting。不過，相較於 RBF SVM，Linear SVM 的決策邊界較簡單，通常具有較穩定的泛化能力。

Linear SVM 對測試資料的預測分布如下：

| Predicted Activity | Count |
|---|---:|
| STANDING | 569 |
| LAYING | 539 |
| WALKING | 517 |
| WALKING_UPSTAIRS | 469 |
| SITTING | 451 |
| WALKING_DOWNSTAIRS | 402 |

---

### 4.5 Model 3: Linear SVM with 90% PCA

此模型將 PCA 放入 pipeline 中，以避免 cross-validation 中產生 data leakage。PCA 保留 90% explained variance，接著使用 Linear SVM 進行分類。模型設定為：

```python
Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.90, svd_solver="full")),
    ("svm", SVC(kernel="linear", C=3, cache_size=1000))
])
```

PCA 降維結果如下：

| Item | Value |
|---|---:|
| PCA components | 63 |
| Cumulative explained variance | 0.9005 |

Cross-validation 結果如下：

| Fold | Accuracy |
|---|---:|
| Fold 1 | 0.9504 |
| Fold 2 | 0.9667 |
| Fold 3 | 0.9592 |
| Fold 4 | 0.9592 |
| Fold 5 | 0.9524 |

| Metric | Value |
|---|---:|
| Mean CV accuracy | 0.9576 |
| Std CV accuracy | 0.0058 |
| OOF accuracy | 0.9576 |
| OOF Macro F1 | 0.9596 |
| Final training accuracy | 0.9699 |

Linear SVM with 90% PCA 的表現明顯低於使用全部 sensor features 的 Linear SVM。這表示雖然 PCA 可以保留 90% 的總變異，但被丟棄的低變異方向仍可能包含重要的分類資訊，尤其是區分 SITTING 與 STANDING 這類相似靜態活動所需的細節特徵。

因此，本研究認為 PCA 適合作為 EDA 與資料結構視覺化工具，但不適合作為最終分類模型的主要前處理步驟。

Linear SVM with 90% PCA 對測試資料的預測分布如下：

| Predicted Activity | Count |
|---|---:|
| STANDING | 534 |
| LAYING | 531 |
| WALKING | 520 |
| SITTING | 495 |
| WALKING_UPSTAIRS | 452 |
| WALKING_DOWNSTAIRS | 415 |

---

### 4.6 Model 4: L1-Regularized Logistic Regression

Logistic Regression 使用標準化後的全部 sensor features 作為輸入，並加入 L1 regularization 以控制模型複雜度並產生稀疏係數。

Cross-validation 結果如下：

| Fold | Accuracy |
|---|---:|
| Fold 1 | 0.9755 |
| Fold 2 | 0.9878 |
| Fold 3 | 0.9721 |
| Fold 4 | 0.9776 |
| Fold 5 | 0.9782 |

| Metric | Value |
|---|---:|
| Mean CV accuracy | 0.9782 |
| Std CV accuracy | 0.0052 |
| OOF accuracy | 0.9782 |
| OOF Macro F1 | 0.9793 |
| Final training accuracy | 0.9880 |

Logistic Regression 的表現與 LDA 接近，顯示此資料具有強烈的線性可分結構。雖然其 cross-validation accuracy 略低於 Linear SVM，但其模型形式簡單且具有良好的可解釋性，適合作為線性模型比較。

Logistic Regression 對測試資料的預測分布如下：

| Predicted Activity | Count |
|---|---:|
| STANDING | 567 |
| LAYING | 537 |
| WALKING | 534 |
| WALKING_UPSTAIRS | 470 |
| SITTING | 452 |
| WALKING_DOWNSTAIRS | 387 |

---

### 4.7 Overall Model Comparison

| Model | Mean CV Accuracy | Std | OOF Accuracy | OOF Macro F1 | Training Accuracy |
|---|---:|---:|---:|---:|---:|
| LDA | 0.9788 | 0.0037 | 0.9788 | 0.9788 | 0.9850 |
| Linear SVM, all features | 0.9840 | 0.0034 | 0.9839 | 0.9850 | 0.9985 |
| Linear SVM, 90% PCA | 0.9576 | 0.0058 | 0.9576 | 0.9596 | 0.9699 |
| L1 Logistic Regression | 0.9782 | 0.0052 | 0.9782 | 0.9793 | 0.9880 |

從 cross-validation 結果來看，Linear SVM with all sensor features 取得最高的 mean CV accuracy。LDA 與 L1-regularized logistic regression 表現接近，顯示此資料具有強烈的線性可分結構。Linear SVM with 90% PCA 則明顯較差，代表 PCA 降維雖然能保留大部分變異，但可能移除了部分關鍵分類資訊。

### 4.8 Final Prediction and Submission

最終使用 LDA 作為 final submission model。雖然 Linear SVM with all sensor features 在 cross-validation 中取得最高的 mean CV accuracy，但實際提交至 Kaggle 後，LDA 的 leaderboard 表現較佳。因此，本研究選擇 LDA 作為最終預測模型，以取得較好的 test set 泛化能力。

---

## 5. Discussion

### 5.1 Main Findings

研究結果顯示，smartphone sensor-derived features 對人類活動分類具有高度辨識能力。LDA baseline 的 mean CV accuracy 已達 0.9788，表示六種活動在原始 sensor feature space 中已具有明顯的線性可分結構。

在模型比較中，Linear SVM with all sensor features 取得最高的 cross-validation 表現，mean CV accuracy 為 0.9840，略高於 LDA 與 L1 Logistic Regression。這表示在不使用非線性 kernel 的情況下，最大化 margin 的線性分類器已能有效區分不同活動類別。因此，本資料集不一定需要複雜的非線性模型，也能達到良好的分類效果。

相較之下，Linear SVM with 90% PCA 的 mean CV accuracy 下降至 0.9576，明顯低於使用完整特徵的模型。這表示 PCA 雖然保留了大部分總變異，但主要變異方向不一定等同於最有分類能力的方向。部分對分類重要的低變異資訊，尤其是區分 SITTING 與 STANDING 這類相似靜態活動的資訊，可能在降維過程中被削弱。


### 5.2 Difficulty

本作業主要困難包括：

1. 從 PCA score plot 與 confusion matrix 可觀察到，SITTING 與 STANDING 是最容易混淆的兩類。原因可能是兩者皆屬於低動態活動，手機感測器訊號差異較小。

2. 90% PCA 版本的 Linear SVM accuracy 明顯下降，代表 PCA 雖然保留主要變異方向，但不一定保留最有分類能力的方向。尤其對於 SITTING 與 STANDING 這種細微差異，低變異方向可能仍包含重要分類資訊。

3. LDA、Linear SVM 與 Logistic Regression 都是線性分類模型，但假設與目標函數不同。LDA 基於 covariance structure 與 discriminant score，Linear SVM 最大化 margin，Logistic Regression 則直接建模 posterior probability。三者表現相近，說明此資料具有明顯線性結構。

### 5.3 Future Work

未來可從以下方向改進：

1. 若能取得 subject information 或資料收集順序資訊，可使用 group-based validation，以更接近真實測試情 境。

2. 進行 feature group analysis。資料集中的特徵來自不同感測器訊號與不同訊號處理方式，例如 accelerometer、gyroscope、time-domain features 與 frequency-domain features。未來可以分別使用不同 feature groups 訓練模型，以分析哪些類型的 sensor information 對活動分類最有幫助。


---

## 6. Conclusion

本研究以 smartphone sensor-derived features 建立人類活動分類模型，並完成資料檢查、PCA 視覺化、模型訓練、cross-validation 評估與最終 submission prediction。EDA 結果顯示，資料具有明顯的低維結構，PCA score plot 可大致區分靜態活動與動態活動。

模型實驗顯示，LDA、Linear SVM 與 L1 Logistic Regression 皆能取得良好的分類表現，代表此資料集具有明顯的線性分類結構。雖然 Linear SVM 在 cross-validation 中取得最高 accuracy，但最終模型選擇仍需考慮實際 submission 表現與泛化能力。因此，本研究最終採用 LDA 作為 final submission model。

整體而言，smartphone sensor features 對六類活動具有高度辨識能力。不過，SITTING 與 STANDING 仍是最主要的混淆來源，可能原因是兩者皆屬於靜態姿勢，sensor pattern 較為相似。未來若能進一步分析錯誤樣本、使用更接近真實測試情境的 validation strategy，或結合多個線性模型進行 ensemble，可能有助於提升模型穩定性與分類表現。
