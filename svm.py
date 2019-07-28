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
# 手書き文字のデータ(load_digits)
from sklearn.datasets import load_digits
# SVMのモジュール(svm)
from sklearn import svm
# 教師データの分割をするモジュール(train_test_split)
# 参考文献では、`from sklearn.cross_validation import train_test_split`だったが、現在は以下が正しい
from sklearn.model_selection import train_test_split

# timeit: 処理時間を測定するためのモジュール
import timeit

### 画像データ読み込み、加工

# 画像の入っているフォルダを指定し、中身のファイル名を取得
# filenames = sorted(os.listdir('handwrite_numbers'))
# filenames = sorted(os.listdir('handwrite2_numbers'))
filenames = sorted(os.listdir('handwrite3_numbers'))
# print(filenames)
# ['eight1.png', 'eight2.png', 'eight3.png', 'five1.png', 'five2.png', 'five3.png', 'four1.png', 'four2.png', 'four3.png', 'nine1.png', 'nine2.png', 'nine3.png', 'one1.png', 'one2.png', 'one3.png', 'seven1.png', 'seven2.png', 'seven3.png', 'six1.png', 'six2.png', 'six3.png', 'three1.png', 'three2.png', 'three3.png', 'two1.png', 'two2.png', 'two3.png', 'zero1.png', 'zero2.png', 'zero3.png']

# フォルダ内の全画像をデータ化
# np.empty() 要素の値が不定の状態のままで，指定した大きさの配列を生成する関数
img_test = np.empty((0, 64))
for filename in filenames:
    # 画像ファイルを取得、グレースケール（モノクロ）にしてサイズ変更
    # img = Image.open('handwrite_numbers/' + filename).convert('L')

    img = Image.open('handwrite3_numbers/' + filename).convert('L')
    # 画像の表示
    # img.show()
    resize_img = img.resize((64, 64))
    # リサイズされていることを確認
    # resize_img.show()

    # サイズを更に縮めて配列を作り、sklearnのdigitsと同じ型（8 × 8）にする
    # 通常のpng画像の明度は0〜255なので、サンプルデータに合わせて0〜15に変換
    img_data256 = np.array([])
    # 64画素の1画素ずつ明るさをプロット
    for y in range(8):
        for x in range(8):
            # 1画素に縮小される範囲の明るさの二乗平均をとり、白黒反転(sklearnのdigitsに合わせるため)
            # crop()は、画像の一部の領域を切り抜くメソッド。切り出す領域を引数(left, upper, right, lower)（要は左上と右下の座標）で指定する。
            crop = np.asarray(resize_img.crop((x * 8, y * 8, x * 8 + 8, y * 8 + 8)))
            bright = 255 - crop.mean()**2 / 255
            img_data256 = np.append(img_data256, bright)

    # 画像データ内の最小値が0, 最大値が16になるようになるように計算(sklearnのdigitsに合わせるため)
    min_bright = img_data256.min()
    max_bright = img_data256.max()
    img_data16 = (img_data256 - min_bright) / (max_bright - min_bright) * 16

    # 加工した画像データをedited_imagesに出力する
    # reshape(8, 8)で8 × 8にする
    # cmap='gray'でグレースケールで表示
    # plt.imshow(img_data16.astype(np.uint8).reshape(8, 8), cmap='gray')
    # plt.savefig("/home/ryuto/judge-num/edited2_numbers/edited_" + filename.replace(".png", "") + ".png")

    # 加工した画像データの配列をまとめる
    # np._r 配列同士の結合
    # astype() データ型の変換（キャスト）
    # np.uint8 符号なし8ビット整数型（整数に変換）
    # reshape(1, -1) -1を使用すると、元の要素数に合わせて自動で適切な値が設定される
    img_test = np.r_[img_test, img_data16.astype(np.uint8).reshape(1, -1)]

# img_testを調べる。
# print("img_test.shape", img_test.shape)
# > img_test.shape (30, 64)


# sklearn のデータセットから取得、目的変数Xと説明変数yに分ける
digits = load_digits()
X = digits.data
y = digits.target

# X, yを調べる
# print("X:", X.shape)
# print("y:", y.shape)
# > X: (1797, 64)
# > y: (1797,)

# sklearnのデータセットを画像として保存する
# リスト（配列）などのイテラブルオブジェクトの要素とインデックス（カウンタ）を同時に取得したい場合は、enumerate()関数を使う。
# for i, x in enumerate(X[:10]):
#     plt.imshow(x.reshape(8, 8), cmap='gray')
#     plt.savefig("/home/ryuto/judge-num/sklearn_numbers/sklearn_" + str(i) + ".png")

# 教師データとテストデータに分ける
train_size = 1000
test_size = 700
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, test_size=test_size, random_state=0)

# サポートベクターマシンのモデルを作る
clf = svm.SVC(gamma=0.001)
loop_count = 10
print("学習時間：", timeit.timeit(lambda: clf.fit(X_train, y_train), number=loop_count) / loop_count)
print("教師データのスコア：", clf.score(X_train, y_train))
print("テストデータのスコア：", clf.score(X_test, y_test))

# 画像データの正解を配列にしておく
X_true = []
for filename in filenames:
    X_true = X_true + [int(filename[:1])]

X_true = np.array(X_true)

pred_svm = clf.predict(img_test)

# 結果の出力
print("判定結果")
print("観測：", X_true)
print("予測：", pred_svm)
print("正答率：", clf.score(img_test, X_true))

### test_size = 0.5
# 教師データのスコア： 0.9988864142538976
# テストデータのスコア： 0.9899888765294772
## handwrite_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [4 4 4 1 1 1 4 4 4 9 4 1 4 4 4 4 4 4 4 4 4 4 4 4 4 4 1 4 4 1]
# 正答率： 0.2
#
## handwrite2_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [1 1 0 1 1 1 2 2 2 3 9 9 4 4 7 5 5 4 5 4 5 7 7 7 8 5 3 1 9 7]
# 正答率： 0.5666666666666667
#
## handwrite3_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [4 4 0 1 1 1 2 2 2 9 9 4 4 4 4 5 4 5 4 4 4 7 7 7 8 4 8 4 9 9]
# 正答率： 0.6333333333333333

### train_size = 1000, test_size = 700
# 学習時間： 0.06385449170000002
# 教師データのスコア： 0.998
# テストデータのスコア： 0.9928571428571429
## handwrite_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [4 4 4 1 1 1 4 4 4 9 1 1 4 4 4 4 4 4 4 4 4 4 4 1 4 4 1 1 4 1]
# 正答率： 0.2
## handwrite2_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [1 1 0 1 1 1 2 2 2 3 9 9 4 4 7 5 9 4 5 4 5 7 7 7 8 9 9 8 9 7]
# 正答率： 0.5333333333333333
## handwrite3_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [4 4 0 1 1 1 2 2 2 9 9 4 4 4 4 5 4 5 4 4 4 7 7 7 8 1 8 9 9 9]
# 正答率： 0.6666666666666666
