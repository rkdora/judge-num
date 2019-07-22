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
# SVMのモジュール(svm)
from sklearn import svm
# 教師データの分割をするモジュール(train_test_split)
from sklearn.model_selection import train_test_split

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

    img = Image.open('handwrite3_numbers/' + filename).convert('L')
    # 画像の表示
    # img.show()
    resize_img = img.resize((784, 784))
    # リサイズされていることを確認
    # resize_img.show()

    # サイズを更に縮めて配列を作り、sklearnのdigitsと同じ型（8 × 8）にする
    # 通常のpng画像の明度は0〜255なので、サンプルデータに合わせて0〜15に変換
    img_data256 = np.array([])
    # 64画素の1画素ずつ明るさをプロット
    for y in range(28):
        for x in range(28):
            # 1画素に縮小される範囲の明るさの二乗平均をとり、白黒反転(sklearnのdigitsに合わせるため)
            # crop()は、画像の一部の領域を切り抜くメソッド。切り出す領域を引数(left, upper, right, lower)（要は左上と右下の座標）で指定する。
            crop = np.asarray(resize_img.crop((x * 28, y * 28, x * 28 + 28, y * 28 + 28)))
            bright = 255 - crop.mean()**2 / 255
            img_data256 = np.append(img_data256, bright)

    img_data = img_data256 / 255
    # 画像データ内の最小値が0, 最大値が16になるようになるように計算(sklearnのdigitsに合わせるため)
    # min_bright = img_data256.min()
    # max_bright = img_data256.max()
    # img_data16 = (img_data256 - min_bright) / (max_bright - min_bright) * 16

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

train_size = 5000
test_size = 1000
# # 教師データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=train_size, random_state=0)
# サポートベクターマシンのモデルを作る
clf = svm.SVC(gamma=0.001)
svm_model = clf.fit(X_train, y_train)


print("教師データのスコア：", svm_model.score(X_train, y_train))
print("テストデータのスコア：", svm_model.score(X_test, y_test))

# 画像データの正解を配列にしておく
X_true = []
for filename in filenames:
    X_true = X_true + [int(filename[:1])]

X_true = np.array(X_true)

pred_svm = svm_model.predict(img_test)

# 結果の出力
print("判定結果")
print("観測：", X_true)
print("予測：", pred_svm.astype(np.uint8))
print("正答率：", svm_model.score(img_test, X_true))

# uint8 8ビットの符号なし整数型 8ビットの値の範囲は0～255、16ビットでは0～65,535、32ビットでは0～4,294,967,295となる

### train_size = 500,test_size = 100
# 教師データのスコア： 0.782
# テストデータのスコア： 0.81
## handwrite_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [7 7 7 1 1 1 1 7 1 1 1 1 1 1 1 1 9 9 1 6 5 1 1 1 1 1 1 1 1 1]
# 正答率： 0.13333333333333333
## handwrite2_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [1 1 1 1 1 1 1 5 1 1 9 5 1 1 1 1 1 9 9 1 1 1 1 7 1 9 1 1 1 9]
# 正答率： 0.16666666666666666
## handwrite3_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [0 0 3 1 1 1 2 2 7 3 3 3 2 6 6 3 3 9 3 9 6 7 7 2 3 3 3 1 7 9]
# 正答率： 0.4666666666666667

### train_size = 5000,test_size = 1000
# 教師データのスコア： 0.91
# テストデータのスコア： 0.893
## handwrite_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [7 7 7 5 1 5 7 2 2 5 5 5 5 4 4 5 5 5 6 6 6 7 7 7 5 5 1 1 1 7]
# 正答率： 0.4666666666666667
## handwrite2_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [5 5 5 1 1 1 7 5 7 5 4 5 6 6 1 5 5 5 5 5 5 7 7 7 5 5 5 1 4 4]
# 正答率： 0.3
## handwrite3_numbers
# 判定結果
# 観測： [0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6 7 7 7 8 8 8 9 9 9]
# 予測： [3 0 3 5 1 1 2 2 7 3 3 8 2 6 6 3 5 4 3 5 6 2 5 3 5 5 3 1 7 7]
# 正答率： 0.3
