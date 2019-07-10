### パッケージ（os, numpy, PIL）のインポート
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
# ロジスティック回帰のモジュール(LogisticRegression)
from sklearn.linear_model import LogisticRegression
# 教師データの分割をするモジュール(train_test_split)
# 参考文献では、`from sklearn.cross_validation import train_test_split`だったが、現在は以下が正しい
from sklearn.model_selection import train_test_split

### 画像データ読み込み、加工

# 画像の入っているフォルダを指定し、中身のファイル名を取得
filenames = sorted(os.listdir('handwrite_numbers'))
# print(filenames)
# ['eight1.png', 'eight2.png', 'eight3.png', 'five1.png', 'five2.png', 'five3.png', 'four1.png', 'four2.png', 'four3.png', 'nine1.png', 'nine2.png', 'nine3.png', 'one1.png', 'one2.png', 'one3.png', 'seven1.png', 'seven2.png', 'seven3.png', 'six1.png', 'six2.png', 'six3.png', 'three1.png', 'three2.png', 'three3.png', 'two1.png', 'two2.png', 'two3.png', 'zero1.png', 'zero2.png', 'zero3.png']

# フォルダ内の全画像をデータ化
# np.empty() 要素の値が不定の状態のままで，指定した大きさの配列を生成する関数
img_test = np.empty((0, 64))
for filename in filenames:
    # 画像ファイルを取得、グレースケール（モノクロ）にしてサイズ変更
    img = Image.open('handwrite_numbers/' + filename).convert('L')
    # 画像の表示
    # img.show()
    resize_img = img.resize((64, 64))
    # リサイズされていることを確認
    # resize_img.show()

    # サイズを更に縮めて配列を作り、sklearnのdigitsと同じ型（8 × 8）にする
    # 256は0から255までの合計256種類という意味
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
    # plt.savefig("/home/ryuto/judge-num/edited_numbers/edited_" + filename.replace(".png", "") + ".png")

    # 加工した画像データの配列をまとめる
    # np._r 配列同士の結合
    # astype() データ型の変換（キャスト）
    # np.uint8 符号なし8ビット整数型（整数に変換）
    # reshape(1, -1) -1を使用すると、元の要素数に合わせて自動で適切な値が設定される
    img_test = np.r_[img_test, img_data16.astype(np.uint8).reshape(1, -1)]

# sklearn のデータセットから取得、目的変数xと説明変数yに分ける
digits = load_digits()
x = digits.data
y = digits.target
# 教師データとテストデータに分ける
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)
# ロジスティック回帰のモデルを作る。教師データを使って学習
logreg = LogisticRegression()
logreg_model = logreg.fit(x_train, y_train)

print("教師データのスコア：", logreg_model.score(x_train, y_train))
print("テストデータのスコア：", logreg_model.score(x_test, y_test))

# 画像データの正解を配列にしておく
x_true = []
for filename in filenames:
    x_true = x_true + [int(filename[:1])]

x_true = np.array(x_true)
print(x_true)
