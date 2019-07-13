# judge-num
sklearnのデータセットとロジスティック回帰を用い、手書き数字の判定を行う

## 流れ
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
画像データを[handwrite_numbers](/handwrite_numbers)から、[handwrite2_numbers](/handwrite2_numbers)へと変更した。  
handwrite_numbers
![handwrite_numbers](https://user-images.githubusercontent.com/20394831/61168931-9f6b6500-a590-11e9-98c2-5d480997476e.png)

handwrite2_numbers
![handwrite2_numbers](https://user-images.githubusercontent.com/20394831/61168932-a1cdbf00-a590-11e9-9431-1f62ba2ef6da.png)

すると、
```
判定結果
観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
予測： [0 0 0 6 1 1 2 2 2 3 6 9 6 4 7 5 5 1 5 4 4 7 1 7 9 5 5 5 9 9]
正答率： 0.5333333333333333
```
正答率が16%から53%へと大きく向上した。  
文字の大きさを大きくしたことにより、教師データに近づいたと考える。

## 参考記事
[【機械学習初心者向け】ロジスティック回帰で手書き文字認識【機械学習の実装】 | Aidemy Blog](https://blog.aidemy.net/entry/2017/07/11/214635)
