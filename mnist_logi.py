### パッケージ（os, numpy, PIL, matplotlib）のインポート
# os: ディレクトリの操作を簡単にするモジュール
import os
# numpy(np): 配列計算の効率化モジュール
import numpy as np
# pillow(PIL): 画像データ処理のモジュール
from PIL import Image
# matplotlib(plt): 画像を表示するためのモジュール
import matplotlib.pyplot as plt

# sklearnのインポート
import sklearn
# 手書き文字のデータ(MNIST)
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
# ロジスティック回帰のモジュール(LogisticRegression)
from sklearn.linear_model import LogisticRegression
# 教師データの分割をするモジュール(train_test_split)
from sklearn.model_selection import train_test_split

# timeit: 処理時間を測定するためのモジュール
import timeit

### 画像データ読み込み、加工

# 画像の入っているフォルダを指定し、中身のファイル名を取得
# filenames = sorted(os.listdir('handwrite_numbers'))
filenames = sorted(os.listdir('handwrite_numbers'))
# filenames = sorted(os.listdir('handwrite3_numbers'))
# print(filenames)
# ['eight1.png', 'eight2.png', 'eight3.png', 'five1.png', 'five2.png', 'five3.png', 'four1.png', 'four2.png', 'four3.png', 'nine1.png', 'nine2.png', 'nine3.png', 'one1.png', 'one2.png', 'one3.png', 'seven1.png', 'seven2.png', 'seven3.png', 'six1.png', 'six2.png', 'six3.png', 'three1.png', 'three2.png', 'three3.png', 'two1.png', 'two2.png', 'two3.png', 'zero1.png', 'zero2.png', 'zero3.png']

# フォルダ内の全画像をデータ化
# np.empty() 要素の値が不定の状態のままで，指定した大きさの配列を生成する関数
img_test = np.empty((0, 784))
for filename in filenames:
    # 画像ファイルを取得、グレースケール（モノクロ）にしてサイズ変更
    # img = Image.open('handwrite_numbers/' + filename).convert('L')

    img = Image.open('handwrite_numbers/' + filename).convert('L')
    # 画像の表示
    # img.show()
    resize_img = img.resize((784, 784))
    # リサイズされていることを確認
    # resize_img.show()

    # サイズを更に縮めて配列を作り、MNISTと同じ型（28 × 28）にする
    img_data256 = np.array([])
    # 784画素の1画素ずつ明るさをプロット
    for y in range(28):
        for x in range(28):
            # 1画素に縮小される範囲の明るさの二乗平均をとり、白黒反転(sklearnのdigitsに合わせるため)
            # crop()は、画像の一部の領域を切り抜くメソッド。切り出す領域を引数(left, upper, right, lower)（要は左上と右下の座標）で指定する。
            crop = np.asarray(resize_img.crop((x * 28, y * 28, x * 28 + 28, y * 28 + 28)))
            bright = 255 - crop.mean()**2 / 255
            img_data256 = np.append(img_data256, bright)

    img_data = img_data256 / 255

    # 加工した画像データをmnist_edited_imagesに出力する
    # cmap='gray'でグレースケールで表示
    # plt.imshow(img_data.reshape(28, 28), cmap='gray')
    # plt.savefig("/Users/ryuto/works/judge-num/mnist_edited3_numbers/edited_" + filename.replace(".png", "") + ".png")

    img_test = np.r_[img_test, img_data.reshape(1, -1)]

X = mnist.data / 255
y = mnist.target

#
# # mnistのデータセットを画像として保存する
# # リスト（配列）などのイテラブルオブジェクトの要素とインデックス（カウンタ）を同時に取得したい場合は、enumerate()関数を使う。
# for i, x in enumerate(X[:10]):
#     plt.imshow(x.reshape(28, 28), cmap='gray')
#     plt.savefig("/Users/ryuto/works/judge-num/mnist_numbers/mnist_" + str(i) + ".png")

train_size = 50000
test_size = 10000
# # 教師データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=0)
# # ロジスティック回帰のモデルを作る。教師データを使って学習
# # FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
# #   FutureWarning)
# # solver='liblinear'を追記することで、上記のWarningが出ないようになった。
#
# # FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
# #   "this warning.", FutureWarning)
# # 同様に、multi_class='auto'を追記することで、上記のWarningが出ないようになった。
# # 参考記事： https://machinelearningmastery.com/how-to-fix-futurewarning-messages-in-scikit-learn/
logreg = LogisticRegression(solver='liblinear', multi_class='auto')

loop_count = 10
print("学習時間：", timeit.timeit(lambda: logreg.fit(X_train, y_train), number=loop_count) / loop_count)
print("教師データのスコア：", logreg.score(X_train, y_train))
print("テストデータのスコア：", logreg.score(X_test, y_test))

# 画像データの正解を配列にしておく
X_true = []
for filename in filenames:
    X_true = X_true + [int(filename[:1])]

X_true = np.array(X_true)

pred_logreg = logreg.predict(img_test)

# 結果の出力
print("判定結果")
print("観測：", X_true)
print("予測：", pred_logreg.astype(np.uint8))
print("正答率：", logreg.score(img_test, X_true))

# uint8 8ビットの符号なし整数型 8ビットの値の範囲は0～255、16ビットでは0～65,535、32ビットでは0～4,294,967,295となる

### train_size = 500,test_size = 100
# 学習時間： 0.1249850742
# 教師データのスコア： 1.0
# テストデータのスコア： 0.84
## handwrite_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [0 7 5 5 5 5 8 2 2 7 5 5 4 4 4 4 5 5 6 6 6 7 7 7 5 5 8 9 9 9]
# 正答率： 0.6
#
## handwrite2_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [3 3 3 8 1 8 7 0 8 3 9 4 4 2 1 4 1 5 5 4 3 2 7 7 5 9 3 1 4 4]
# 正答率： 0.2
#
## handwrite3_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [2 0 2 8 1 8 2 2 7 3 3 8 2 4 8 3 5 4 4 4 6 2 4 2 3 8 5 1 3 3]
# 正答率： 0.3333333333333333

### train_size = 5000, test_size = 1000
# 学習時間： 3.4751767758
# 教師データのスコア： 0.9622
# テストデータのスコア： 0.891
## handwrite_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [0 9 0 5 5 5 2 2 2 5 5 5 4 4 5 6 5 6 6 5 6 7 7 7 5 5 1 1 9 9]
# 正答率： 0.5
## handwrite2_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [7 7 7 5 1 5 3 3 3 3 9 3 7 2 3 3 5 5 3 3 3 7 7 7 3 3 7 7 7 7]
# 正答率： 0.26666666666666666
## handwrite3_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [3 0 3 5 1 5 2 2 7 3 3 8 2 2 5 3 9 3 3 3 6 7 3 7 3 5 5 7 7 3]
# 正答率： 0.3

### train_size = 50000 test_size = 10000
# 学習時間： 72.44821099129999
# 教師データのスコア： 0.93044
# テストデータのスコア： 0.9158
## handwrite_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [0 7 5 5 5 5 2 2 2 5 1 5 5 5 4 5 5 5 6 6 6 7 7 7 5 5 5 9 9 7]
# 正答率： 0.5333333333333333
#
## handwrite2_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [3 3 3 5 1 5 3 3 3 3 6 1 6 2 3 5 3 5 3 3 3 3 3 3 3 3 5 1 3 3]
# 正答率： 0.13333333333333333
#
## handwrite3_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [3 0 3 5 1 5 2 2 3 3 3 8 2 2 4 3 5 3 3 3 6 7 3 3 3 8 3 3 3 3]
# 正答率： 0.36666666666666664

### train_size = 1000, test_size = 700
# 学習時間： 0.2877770871
# 教師データのスコア： 1.0
# テストデータのスコア： 0.8614285714285714
## handwrite_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [0 7 0 5 5 5 2 2 2 5 6 5 6 4 1 6 9 6 6 5 6 7 7 7 5 5 8 9 9 3]
# 正答率： 0.4666666666666667
## handwrite2_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [3 3 3 6 1 6 7 6 3 3 6 5 6 2 2 5 3 7 3 3 3 3 6 7 3 5 9 1 3 3]
# 正答率： 0.13333333333333333
## handwrite3_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [3 0 3 6 1 8 2 2 3 3 3 9 2 2 4 5 5 5 3 2 6 4 3 4 3 5 5 4 7 3]
# 正答率： 0.36666666666666664
