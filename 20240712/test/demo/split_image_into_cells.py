import cv2
import numpy as np

def draw_grid(image, line_spacing=10, line_color=(0, 255, 0)):
    """
    画像に指定したピクセル間隔でグリッド線を描画する関数。
    
    :param image: 入力画像
    :param line_spacing: グリッド線の間隔（ピクセル単位）
    :param line_color: グリッド線の色 (BGR形式)
    :return: グリッド線が描画された画像
    """
    img_h, img_w, _ = image.shape
    
    # 縦線を描画
    for x in range(0, img_w, line_spacing):
        cv2.line(image, (x, 0), (x, img_h), line_color, 1)
    
    # 横線を描画
    for y in range(0, img_h, line_spacing):
        cv2.line(image, (0, y), (img_w, y), line_color, 1)
    
    return image

def main(image_path, output_path, line_spacing=10):
    # 画像を読み込む
    image = cv2.imread(image_path)
    
    # グリッド線を描画
    image_with_grid = draw_grid(image, line_spacing)
    
    # 結果を保存
    cv2.imwrite(output_path, image_with_grid)
    print(f"Processed image with grid saved at {output_path}")

if __name__ == '__main__':
    image_path = 'test2_workman_zoom.png'  # Specify your input image path
    output_path = 'result_split_into_cells.png'  # Specify your output image path
    
    line_spacing = 10  # グリッド線の間隔を設定（ピクセル単位）

    main(image_path, output_path, line_spacing)
