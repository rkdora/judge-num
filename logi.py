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
