import os
import ast
import chardet

class FunctionCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []
        self.functions = []

    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append((node.func.id, node.lineno))
        self.generic_visit(node)

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

def parse_python_file(file_path):
    # まずエンコーディングを検出する
    encoding = detect_encoding(file_path)
    print(f"Detected encoding: {encoding}")

    # 複数のエンコーディングを試す
    encodings_to_try = [encoding, 'utf-8', 'cp932', 'shift_jis', 'iso-8859-1']

    for enc in encodings_to_try:
        try:
            with open(file_path, "r", encoding=enc) as file:
                decoded_data = file.read()
                tree = ast.parse(decoded_data, filename=file_path)
                break  # 成功したらループを抜ける
        except (UnicodeDecodeError, SyntaxError) as e:
            print(f"Failed to decode {file_path} with {enc}: {e}")
            continue
    else:
        # 全てのリトライが失敗した場合
        print(f"Skipping {file_path} due to encoding issues.")
        return [], []  # 空のリストを返す

    visitor = FunctionCallVisitor()
    visitor.visit(tree)
    return visitor.functions, visitor.calls

def create_dot_file(file_path, functions, calls):
    with open(file_path, "w") as dot_file:
        dot_file.write("digraph G {\n")
        
        # 関数ノードの定義
        for func in functions:
            dot_file.write(f'    {func} [label="{func}"];\n')

        # 関係の定義
        for caller, callee in calls:
            dot_file.write(f'    {caller} -> {callee};\n')

        dot_file.write("}\n")

def main():
    # 現在のディレクトリを取得
    current_directory = os.getcwd()
    functions = []
    calls = []

    # ディレクトリ内のPythonファイルをすべて解析
    for root, _, files in os.walk(current_directory):
        for file in files:
            if file.endswith(".py") and file != "generate_dot.py":
                file_path = os.path.join(root, file)
                file_functions, file_calls = parse_python_file(file_path)
                functions.extend(file_functions)
                calls.extend(file_calls)

    # .dotファイルの作成
    create_dot_file("function_relations.dot", functions, calls)
    print("Graphviz .dot file created as 'function_relations.dot'")

if __name__ == "__main__":
    main()
