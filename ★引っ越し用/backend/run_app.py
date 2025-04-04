# 実行コマンド：python run_app.py

import subprocess
import sys
import os

if __name__ == "__main__":

    if getattr(sys, 'frozen', False):
        # EXEで実行されているとき（PyInstaller）
        base_path = sys._MEIPASS
    else:
        # 通常のPythonスクリプト実行
        base_path = os.path.dirname(__file__)

    # 今実行しているスクリプトがあるディレクトリの絶対パス/src/backend/app.pyのを取得
    script_path = os.path.join(base_path, 'src', 'backend', 'app.py')

    # app.py の ディレクトリパス（src/backend/ の絶対パス） を取得
    app_dir = os.path.dirname(script_path)

    # Python から「ターミナルで streamlit run app.py と打つ」のと同じことをやってる
    # cwd=app_dir によって、実行時のカレントディレクトリを app.py の場所に強制
    subprocess.run(['streamlit', 'run', script_path], cwd=app_dir)