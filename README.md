# judge-num
手書き数字の判定

## ロジスティック回帰（digits）
sklearnのデータセット(digits)とロジスティック回帰を用い、手書き数字の判定を行う

**digits**  
mnist を加工して作成した、 縦8ピクセル・横8ピクセル、1797枚の小さなデータセット。0~15の16段階で作られている。一番明るい部分（文字）が15, 一番暗い部分が0。

**ロジスティック回帰**  

### 流れ
1. パッケージのインポート
2. 画像データ読み込み、加工
3. 教師データから学習
4. 画像データの判別

---

1. パッケージのインポート  
  - scikit-learn(sklearn) : 機械学習のモジュール
  - os : ディレクトリの操作を簡単にするモジュール
  - numpy : 配列計算の効率化モジュール
  - PIL : 画像データ処理のモジュール
  - matplotlib : 画像を表示するためのモジュール

2. 画像データ読み込み、加工  
  元画像
  ![](/handwrite_numbers/5_1.png)
  ↓  
  教師データにあわせるため、8×8にリサイズ & 白黒反転  
  ↓  
  加工後
  ![](/edited_numbers/edited_5_1.png)

3. 教師データから学習  
  ロジスティック回帰  

  教師データ  
  ![](/sklearn_numbers/sklearn_5.png)
  ```
  教師データのスコア： 0.9988864142538976
  テストデータのスコア： 0.9443826473859844
  [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
  ```

4. 画像データの判別  
  ```
  判定結果
  観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
  予測： [1 1 1 8 1 1 4 8 4 4 1 1 1 1 1 1 1 1 6 1 6 7 1 1 6 1 1 4 4 1]
  正答率： 0.16666666666666666
  ```

## 正答率の向上を試みる
### 画像データの変更
画像データを[handwrite_numbers](/handwrite_numbers)から、[handwrite2_numbers](/handwrite2_numbers)へと変更した。  
handwrite_numbers

![handwrite_numbers](https://user-images.githubusercontent.com/20394831/61602921-83458300-ac76-11e9-9859-948e079050b3.png)

文字を大きくした。
handwrite2_numbers

![handwrite2_numbers](https://user-images.githubusercontent.com/20394831/61602919-7e80cf00-ac76-11e9-88e6-63a82ad401d1.png)

すると、
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [0 0 0 6 1 1 2 2 2 3 6 9 6 4 7 5 5 1 5 4 4 7 1 7 9 5 5 5 9 9]
正答率： 0.5333333333333333
```
正答率が16%から53%へと大きく向上した。  
文字の大きさを大きくしたことにより、教師データに近づいたと考える。

文字を太くした。
handwrite3_numbers

![handwrite3_numbers](https://user-images.githubusercontent.com/20394831/61602853-35c91600-ac76-11e9-8978-bd3e3ee45381.png)


予測された値に変化はあったが、正答率に変化はなかった。
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [0 0 0 6 1 1 2 8 2 9 9 1 6 6 4 5 1 5 4 4 6 7 7 7 3 1 4 1 9 9]
正答率： 0.5333333333333333
```

sklearn_numbers

![sklearn_numbers](https://user-images.githubusercontent.com/20394831/61603014-e20afc80-ac76-11e9-8b94-7656f0ddaa58.png)


## ロジスティック回帰(MNIST)
**MNIST**  
MNIST(エムニスト / Mixed National Institute of Standards and Technology database)とは、手書き数字画像60,000枚と、テスト画像10,000枚を集めた画像データセット。
縦28ピクセル・横28ピクセルで構成され、0~255の256段階で作られている。
一番明るい部分(文字)が255、一番暗い部分が0

mnist_numbers

![mnist_numbers](https://user-images.githubusercontent.com/20394831/61603045-06ff6f80-ac77-11e9-9088-8f0e34bebb87.png)

handwrite_numbersを、mnistに合うように加工した  
edited_mnist_numbers

![edited_mnist_numbers](https://user-images.githubusercontent.com/20394831/61603098-3f9f4900-ac77-11e9-80f2-3647c4746b93.png)


### logy.pyとの差異
- 8ピクセルから28ピクセルへ
- 16段階から256段階(255で割ることにより、全ての値を0~1の間に収める[規格化])

### 結果
#### train_size = 500,test_size = 100
```
教師データのスコア： 1.0
テストデータのスコア： 0.84
```
##### handwrite_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [0 7 5 5 5 5 8 2 2 7 5 5 4 4 4 4 5 5 6 6 6 7 7 7 5 5 8 9 9 9]
正答率： 0.6
```
##### handwrite2_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [3 3 3 8 1 8 7 0 8 3 9 4 4 2 1 4 1 5 5 4 3 2 7 7 5 9 3 1 4 4]
正答率： 0.2
```
##### handwrite3_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [2 0 2 8 1 8 2 2 7 3 3 8 2 4 8 3 5 4 4 4 6 2 4 2 3 8 5 1 3 3]
正答率： 0.3333333333333333
```
#### train_size = 50000 test_size = 10000
```
教師データのスコア： 0.93044
テストデータのスコア： 0.9158
```
##### handwrite_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [0 7 5 5 5 5 2 2 2 5 1 5 5 5 4 5 5 5 6 6 6 7 7 7 5 5 5 9 9 7]
正答率： 0.5333333333333333
```
##### handwrite2_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [3 3 3 5 1 5 3 3 3 3 6 1 6 2 3 5 3 5 3 3 3 3 3 3 3 3 5 1 3 3]
正答率： 0.13333333333333333
```
##### handwrite3_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [3 0 3 5 1 5 2 2 3 3 3 8 2 2 4 3 5 3 3 3 6 7 3 3 3 8 3 3 3 3]
正答率： 0.36666666666666664
```

### 考察
学習のデータ数を増やすことで、モデルの精度を高めることができる。  
学習したデータと手書き数字の特徴が似ているほど正答率が高くなる。  

## サポートベクターマシン(digits)
**サポートベクターマシン**

### 変更点
logi.py -> svm.py

logi.py
```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver='liblinear', multi_class='auto')
logreg_model = logreg.fit(X_train, y_train)
```
handwrite2_numbers利用
```
教師データのスコア： 0.9988864142538976
テストデータのスコア： 0.9443826473859844
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [0 0 0 6 1 1 2 2 2 3 6 9 6 4 7 5 5 1 5 4 4 7 1 7 9 5 5 5 9 9]
正答率： 0.5333333333333333
```
svm.py
```python
from sklearn import svm

clf = svm.SVC(gamma=0.001)
svm_model = clf.fit(X_train, y_train)
```
handwrite2_numbers
```
教師データのスコア： 0.9988864142538976
テストデータのスコア： 0.9899888765294772
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [1 1 0 1 1 1 2 2 2 3 9 9 4 4 7 5 5 4 5 4 5 7 7 7 8 5 3 1 9 7]
正答率： 0.5666666666666667
```
テストデータのスコアが94%から98%へと、正答率が53%から56%へと向上(handwrite2_numbersの場合)

### 結果
#### test_size = 0.5
```
教師データのスコア： 0.9988864142538976
テストデータのスコア： 0.9899888765294772
```
##### handwrite_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [4 4 4 1 1 1 4 4 4 9 4 1 4 4 4 4 4 4 4 4 4 4 4 4 4 4 1 4 4 1]
正答率： 0.2
```
##### handwrite2_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [1 1 0 1 1 1 2 2 2 3 9 9 4 4 7 5 5 4 5 4 5 7 7 7 8 5 3 1 9 7]
正答率： 0.5666666666666667
```
##### handwrite3_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [4 4 0 1 1 1 2 2 2 9 9 4 4 4 4 5 4 5 4 4 4 7 7 7 8 4 8 4 9 9]
正答率： 0.6333333333333333
```

## サポートベクターマシン(MNIST)
### 結果
#### train_size = 500,test_size = 100
```
学習時間： 0.31701250020000005
教師データのスコア： 0.782
テストデータのスコア： 0.81
```
##### handwrite_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [7 7 7 1 1 1 1 7 1 1 1 1 1 1 1 1 9 9 1 6 5 1 1 1 1 1 1 1 1 1]
正答率： 0.13333333333333333
```
##### handwrite2_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [1 1 1 1 1 1 1 5 1 1 9 5 1 1 1 1 1 9 9 1 1 1 1 7 1 9 1 1 1 9]
正答率： 0.16666666666666666
```
##### handwrite3_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [0 0 3 1 1 1 2 2 7 3 3 3 2 6 6 3 3 9 3 9 6 7 7 2 3 3 3 1 7 9]
正答率： 0.4666666666666667
```
#### train_size = 5000,test_size = 1000
```
学習時間： 13.5781102763
教師データのスコア： 0.91
テストデータのスコア： 0.893
```
##### handwrite_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [7 7 7 5 1 5 7 2 2 5 5 5 5 4 4 5 5 5 6 6 6 7 7 7 5 5 1 1 1 7]
正答率： 0.4666666666666667
```
##### handwrite2_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [5 5 5 1 1 1 7 5 7 5 4 5 6 6 1 5 5 5 5 5 5 7 7 7 5 5 5 1 4 4]
正答率： 0.3
```
##### handwrite3_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [3 0 3 5 1 1 2 2 7 3 3 8 2 6 6 3 5 4 3 5 6 2 5 3 5 5 3 1 7 7]
正答率： 0.3
```
#### train_size = 50000, test_size = 10000
```
学習時間： 535.932242031
教師データのスコア： 0.93824
テストデータのスコア： 0.9342
```
##### handwrite_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [7 7 7 5 5 5 2 2 2 5 5 5 5 4 4 6 5 5 6 6 6 7 7 7 5 5 8 9 9 9]
正答率： 0.5666666666666667
```
##### handwrite2_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [5 5 5 5 1 5 7 5 7 3 4 5 6 2 2 5 5 5 3 5 5 2 7 7 5 5 5 1 4 4]
正答率： 0.23333333333333334
```
##### handwrite3_numbers
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [2 0 3 5 1 5 2 2 2 3 3 3 2 2 2 3 5 5 3 2 6 2 5 3 3 5 8 7 7 7]
正答率： 0.4
```

### 考察
学習データを50000にした際、ロジスティック回帰よりも学習に時間がかかった（よって500, 5000で学習をした）。アルゴリズムによって学習にかかる時間が異なることがわかる。

## 学習時間の比較
### digits
#### train_size = 1000, test_size = 700
- ロジスティック回帰 : 約100ms
- SVM : 約60ms

### MNIST
#### train_size = 1000, test_size = 700
- ロジスティック回帰 : 約280ms
- SVM : 約990ms

#### train_size = 500, test_size = 100
- ロジスティック回帰 : 約120ms
- SVM : 約310ms

#### train_size = 5000, test_size = 1000
- ロジスティック回帰 : 約3400ms
- SVM : 約13000ms

#### train_size = 50000, test_size = 10000
- ロジスティック回帰 : 約7200ms
- SVM : 約535000ms

### 考察
SVMと比べ、ロジスティック回帰のほうがかかる時間が少ない。  
しかし、SVMの方が少ないデータで高い精度のモデルをつくることができる。

## 参考記事
[【機械学習初心者向け】ロジスティック回帰で手書き文字認識【機械学習の実装】 | Aidemy Blog](https://blog.aidemy.net/entry/2017/07/11/214635)  
[scikit-learnのSVMでMNISTの手書き数字データを分類](https://note.nkmk.me/python-scikit-learn-svm-mnist/)
