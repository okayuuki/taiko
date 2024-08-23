# Pythonのインストール
#pyenv install 3.6.0

# バージョンの切り替え
pyenv local 3.6.0

# 仮想環境の作成
python -m venv FSA-Netenv

#venvで作成した仮想環境を使用するためには「Activate」、中断するときは「Deactivate」する必要があります

# 仮想環境のアクティベート
#pyenv activate FSA-Netenv

#仮想環境に入った状態でpip installすればＯＫ

#SSL証明書の検証に失敗する問題を回避するために、--trusted-hostオプションを使用してパッケージをインストールする手順を以下に示します。

#pip install keras-applications==1.0.4
#pip install keras==2.1.0
#pip install tensorflow==1.10.0
#pip install tensorflow-gpu==1.10.0
#pip install cudnn==7.1.3# ダウンロードで
#pip install numpy==1.15.2

# requirements.txtのパッケージインストール
#pip install -r C:\Users\chach\Desktop\FSA-Net\requirements.txt

#１：pythonのインストール、pyenvで
#２：バージョン指定、pyenvで
#３：仮想環境作成
#４：有効化
#５；必要ライブラリを入れて
#６：プログラムを実行

#注意：gitbashでは実行できないものもある？