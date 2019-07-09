# os, numpy, PILのインポート
# os: ディレクトリの操作を簡単にするモジュール
import os
# numpy(np): 配列計算の効率化モジュール
import numpy as np
# pillow(PIL): 画像データ処理のモジュール
from PIL import Image

# sklearnのインポート
import sklearn
# 手書き文字のデータ(load_digits)
from sklearn.datasets import load_digits
# ロジスティック回帰のモジュール(LogisticRegression)
from sklearn.linear_model import LogisticRegression
# 教師データの分割をするモジュール(train_test_split)
# 参考文献では、`from sklearn.cross_validation import train_test_split`だったが、現在は以下が正しい
from sklearn.model_selection import train_test_split

# 画像の入っているフォルダを指定し、中身のファイル名を取得
filenames = sorted(os.listdir('handwrite_numbers'))
# print(filenames)
# ['eight1.png', 'eight2.png', 'eight3.png', 'five1.png', 'five2.png', 'five3.png', 'four1.png', 'four2.png', 'four3.png', 'nine1.png', 'nine2.png', 'nine3.png', 'one1.png', 'one2.png', 'one3.png', 'seven1.png', 'seven2.png', 'seven3.png', 'six1.png', 'six2.png', 'six3.png', 'three1.png', 'three2.png', 'three3.png', 'two1.png', 'two2.png', 'two3.png', 'zero1.png', 'zero2.png', 'zero3.png']
