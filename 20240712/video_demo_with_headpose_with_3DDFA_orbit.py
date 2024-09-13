#ライブラリのimport
import math
from collections import defaultdict

#! mmpose
import logging
from argparse import ArgumentParser
import cv2
import pickle
import numpy as np
from mmengine.logging import print_log
from mmpose.apis import inference_topdown, init_model
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from tqdm import tqdm

#! 3DDFA
import mobilenet_v1
import torch
import torchvision.transforms as transforms
from utils.ddfa import ToTensorGjz, NormalizeGjz #追加
import scipy.io as sio
from utils.inference import (get_suffix,
    parse_roi_box_from_landmark,
    crop_img,
    predict_68pts
)
from utils.cv_plot import plot_kpt, plot_pose_box, plot_pose_arrow

#! PRNet、頭部方向実行モジュール
from utils.estimate_pose import parse_pose

STD_SIZE = 120

overlap_info = defaultdict(list)

# 2つのフレームを横に並べて表示する関数
def display_side_by_side(frame1, frame2):
    """
    2つのフレームを横に並べて表示する関数。

    Parameters:
    - frame1: 左側に表示するフレーム
    - frame2: 右側に表示するフレーム
    """
    # 2つのフレームの高さが同じか確認し、異なる場合はサイズを合わせる
    if frame1.shape[0] != frame2.shape[0]:
        height = min(frame1.shape[0], frame2.shape[0])  # 小さい方の高さに合わせる
        frame1 = cv2.resize(frame1, (frame1.shape[1], height))
        frame2 = cv2.resize(frame2, (frame2.shape[1], height))

    # 2つのフレームを横に連結
    combined_frame = cv2.hconcat([frame1, frame2])

    # 横に並べたフレームを表示
    return combined_frame

def draw_orbit(frame, pose_history, start_point_history, frame_w, frame_h):
    """
    過去の軌道から現在までの軌道を描画し、点を線でつなぐ関数。

    Parameters:
    - frame: 現在のフレーム
    - pose_history: 過去の顔向き推定結果のリスト
    - start_point_history: 過去の各フレームの鼻の位置のリスト
    - frame_w: フレームの横幅
    - frame_h: フレームの縦幅
    """

    # 白い背景を作成 (フルHDであれば 255 * 3 で白)
    white_background = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 255

    # 前回の終点を覚えておくための変数
    previous_end_point = None

    for past_pose, past_start_point in zip(pose_history, start_point_history):
        yaw_rad, pitch_rad, roll_rad = past_pose

        # 顔向きベクトルに基づく終点の計算
        x = np.sin(yaw_rad)
        z = np.cos(pitch_rad) * np.cos(yaw_rad)
        y = np.sin(pitch_rad)

        direction_vector = np.array([x, y, z])
        direction_vector /= np.linalg.norm(direction_vector)

        # 現在の終点
        past_end_point = (int(past_start_point[0] - direction_vector[0] * frame_w),
                          int(past_start_point[1] + direction_vector[1] * frame_h))

        # 前回の終点と現在の終点を線で結ぶ
        if previous_end_point is not None:
            cv2.line(white_background, previous_end_point, past_end_point, (0, 0, 255), 2)
            cv2.circle(white_background, past_end_point, 4, (0, 0, 255), -1)

        # 現在の終点を次のフレームの開始点として保存
        previous_end_point = past_end_point

    return white_background


def highlight_direction_vector(frame, pose, start_point, split_h, split_w, frame_count):
    frame_h, frame_w = frame.shape[:2]
    step_h = frame_h // split_h
    step_w = frame_w // split_w

    yaw_rad = pose[0]
    pitch_rad = pose[1]
    roll_rad = pose[2]

    x = np.sin(yaw_rad)
    z = np.cos(pitch_rad) * np.cos(yaw_rad)
    y = np.sin(pitch_rad)

    direction_vector = np.array([x, y, z])
    direction_vector /= np.linalg.norm(direction_vector)

    end_point = (int(start_point[0] - direction_vector[0] * frame_w),
                 int(start_point[1] + direction_vector[1] * frame_h))

    start_point_2d = (int(start_point[0]), int(start_point[1]))
    end_point_2d = (int(end_point[0]), int(end_point[1]))

    #顔向き推定
    cv2.line(frame, start_point_2d, end_point_2d, (0, 255, 0), 6)

    #顔向き終点だけ描画
    #cv2.circle(frame, end_point_2d, 5, (0, 255, 0), -1)

    # 方向ベクトルの終点に最も近い領域をハイライト
    highlighted_frame = frame.copy()
    overlay = frame.copy()
    alpha = 0.4  # 透明度

    closest_region = None
    min_distance = float('inf')

    for i in range(split_h):
        for j in range(split_w):
            region_number = i * split_w + j + 1  # 領域に番号を付ける
            top_left = (j * step_w, i * step_h)
            bottom_right = ((j + 1) * step_w, (i + 1) * step_h)
            rect_top_left = (top_left[0], top_left[1])
            rect_bottom_right = (bottom_right[0], bottom_right[1])
            rect_center = ((rect_top_left[0] + rect_bottom_right[0]) // 2, (rect_top_left[1] + rect_bottom_right[1]) // 2)

            # 終点がフレーム外にある場合、最も近い領域を計算
            distance = np.linalg.norm(np.array(rect_center) - np.array(end_point_2d))
            if distance < min_distance:
                min_distance = distance
                closest_region = (top_left, bottom_right, region_number)

            # 領域の番号を描画
            cv2.putText(highlighted_frame, str(region_number), (top_left[0] + 20, top_left[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            # 領域の外枠に線を描画
            cv2.rectangle(highlighted_frame, top_left, bottom_right, (0, 0, 255), 2)

    # 最も近い領域をハイライト
    if closest_region:
        top_left, bottom_right, region_number = closest_region
        cv2.rectangle(overlay, top_left, bottom_right, (0, 255, 255), -1)
        overlap_info[f'Region {region_number}'].append(frame_count)  # 重なった領域の番号とフレーム数を記録

    # 半透明の矩形をフレームに重ねる
    cv2.addWeighted(overlay, alpha, highlighted_frame, 1 - alpha, 0, highlighted_frame)
    
    return highlighted_frame, yaw_rad, pitch_rad, roll_rad

def get_direction_vector(yaw, pitch, roll):
    # オイラー角（Yaw, Pitch, Roll）を3D方向ベクトルに変換
    # ラジアン単位で角度を想定
    
    # Yaw, Pitch, Rollに基づく回転行列を生成
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    
    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(pitch), -np.sin(pitch)],
                        [0, np.sin(pitch), np.cos(pitch)]])
    
    R_roll = np.array([[np.cos(roll), 0, np.sin(roll)],
                       [0, 1, 0],
                       [-np.sin(roll), 0, np.cos(roll)]])
    
    # 初期ベクトル。顔の前方向をZ軸（0, 0, 1）に沿って正規化ベクトルとして使用。
    initial_vector = np.array([0, 0, 1])
    
    # 3つの回転行列を適用して、向きベクトルを取得
    direction_vector = R_yaw @ R_pitch @ R_roll @ initial_vector
    
    return direction_vector

def highlight_attention_region_on_split_image(img, yaw):
    img_h, img_w = img.shape[:2]

    # 画像を3つの領域に分割
    left_region = img[:, :int(img_w / 3)]
    middle_region = img[:, int(img_w / 3):int(2 * img_w / 3)]
    right_region = img[:, int(img_w / 3 * 2):]

    # 頭部方向に応じて対応する領域をハイライト
    if yaw > 0.5: #約28.6479度
        # 左を向いているときは左の領域をハイライト
        overlay = left_region.copy()
        cv2.rectangle(overlay, (0, 0), (left_region.shape[1], left_region.shape[0]), (255, 255, 0), -1)
        alpha = 0.3  # 透明度
        cv2.addWeighted(overlay, alpha, left_region, 1 - alpha, 0, left_region)
    elif yaw < -0.5:
        # 右を向いているときは右の領域をハイライト
        overlay = right_region.copy()
        cv2.rectangle(overlay, (0, 0), (right_region.shape[1], right_region.shape[0]), (255, 255, 0), -1)
        alpha = 0.3  # 透明度
        cv2.addWeighted(overlay, alpha, right_region, 1 - alpha, 0, right_region)
    else:
        # 真ん中を向いているときは真ん中の領域をハイライト
        overlay = middle_region.copy()
        cv2.rectangle(overlay, (0, 0), (middle_region.shape[1], middle_region.shape[0]), (255, 255, 0), -1)
        alpha = 0.3  # 透明度
        cv2.addWeighted(overlay, alpha, middle_region, 1 - alpha, 0, middle_region)

    # 3つの領域を再結合して1つの画像に戻す
    highlighted_img = np.hstack((left_region, middle_region, right_region))

    return highlighted_img

def highlight_attention_region_on_split_image9(img, yaw, pitch):
    img_h, img_w = img.shape[:2]

    # 画像を9つの領域に分割（縦横3分割）
    left = int(img_w / 3)
    right = int(2 * img_w / 3)
    top = int(img_h / 3)
    bottom = int(2 * img_h / 3)

    # 9つの領域の定義 (左上、中上、右上, 左中、真ん中、右中, 左下、中央下、右下)
    regions = {
        "top_left": img[0:top, 0:left],
        "top_middle": img[0:top, left:right],
        "top_right": img[0:top, right:],
        "middle_left": img[top:bottom, 0:left],
        "middle_middle": img[top:bottom, left:right],
        "middle_right": img[top:bottom, right:],
        "bottom_left": img[bottom:, 0:left],
        "bottom_middle": img[bottom:, left:right],
        "bottom_right": img[bottom:, right:]
    }

    # ハイライトする透明度
    alpha = 0.3

    # 頭部方向に応じて領域をハイライト
    if yaw > 0.5:  # 左側に向いている場合
        if pitch > 0.5:  # 上を向いている場合
            overlay = regions["top_left"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["top_left"].shape[1], regions["top_left"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["top_left"], 1 - alpha, 0, regions["top_left"])
        elif pitch < -0.5:  # 下を向いている場合
            overlay = regions["bottom_left"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["bottom_left"].shape[1], regions["bottom_left"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["bottom_left"], 1 - alpha, 0, regions["bottom_left"])
        else:  # 正面を向いている場合
            overlay = regions["middle_left"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["middle_left"].shape[1], regions["middle_left"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["middle_left"], 1 - alpha, 0, regions["middle_left"])
    elif yaw < -0.5:  # 右側に向いている場合
        if pitch > 0.5:  # 上を向いている場合
            overlay = regions["top_right"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["top_right"].shape[1], regions["top_right"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["top_right"], 1 - alpha, 0, regions["top_right"])
        elif pitch < -0.5:  # 下を向いている場合
            overlay = regions["bottom_right"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["bottom_right"].shape[1], regions["bottom_right"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["bottom_right"], 1 - alpha, 0, regions["bottom_right"])
        else:  # 正面を向いている場合
            overlay = regions["middle_right"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["middle_right"].shape[1], regions["middle_right"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["middle_right"], 1 - alpha, 0, regions["middle_right"])
    else:  # 真ん中を向いている場合
        if pitch > 0.5:  # 上を向いている場合
            overlay = regions["top_middle"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["top_middle"].shape[1], regions["top_middle"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["top_middle"], 1 - alpha, 0, regions["top_middle"])
        elif pitch < -0.5:  # 下を向いている場合
            overlay = regions["bottom_middle"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["bottom_middle"].shape[1], regions["bottom_middle"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["bottom_middle"], 1 - alpha, 0, regions["bottom_middle"])
        else:  # 正面を向いている場合
            overlay = regions["middle_middle"].copy()
            cv2.rectangle(overlay, (0, 0), (regions["middle_middle"].shape[1], regions["middle_middle"].shape[0]), (255, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, regions["middle_middle"], 1 - alpha, 0, regions["middle_middle"])

    # 分割された領域を元に戻して再結合
    top = np.hstack((regions["top_left"], regions["top_middle"], regions["top_right"]))
    middle = np.hstack((regions["middle_left"], regions["middle_middle"], regions["middle_right"]))
    bottom = np.hstack((regions["bottom_left"], regions["bottom_middle"], regions["bottom_right"]))

    highlighted_img = np.vstack((top, middle, bottom))

    return highlighted_img

def parse_args():
    #! ArgumentParserオブジェクトを作成し、コマンドライン引数を解析する準備をする
    parser = ArgumentParser()

    #! 必須の位置引数を定義（位置引数は順番に指定する必要がある）
    parser.add_argument('video', help='Video file')# ビデオファイルのパス
    parser.add_argument('config', help='Config file')# 設定ファイルのパス
    parser.add_argument('checkpoint', help='Checkpoint file')# チェックポイントファイルのパス

    #! オプション引数を定義（指定しなかった場合にはデフォルト値が使われる）
    parser.add_argument('--out-file', default=None, help='Path to output file')

    #! 使用するデバイスを指定するオプション（デフォルトは 'cuda:0' つまり最初のGPU）
    #! 「--device cpu」と明示的に与えればCPUも利用できる
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    
    #! キーポイントのインデックスを表示するかどうかを指定するオプション
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',# 引数が指定された場合、Trueに設定
        default=False,# デフォルトではFalse
        help='Whether to show the index of keypoints')
    
    #! スケルトンのスタイルを選択するオプション（mmposeかopenposeを選べる）
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    
    #! キーポイントのしきい値を指定するオプション（しきい値を下回るキーポイントは表示しない）
    parser.add_argument(
        '--kpt-thr',
        type=float,# 浮動小数点数型
        default=0.3,# デフォルト値は 0.3
        help='Visualizing keypoint thresholds')
    
    #! 可視化のためのキーポイントの円の半径を指定するオプション
    parser.add_argument(
        '--radius',
        type=int,# 整数型
        default=3,# デフォルトは 3 ピクセル
        help='Keypoint radius for visualization')
    
    #! スケルトンリンクの太さを指定するオプション
    parser.add_argument(
        '--thickness',
        type=int,# 整数型
        default=2,# デフォルトは 1 ピクセル
        help='Link thickness for visualization')
    
    #! バウンディングボックスの透明度を指定するオプション
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    
    #! 処理結果を表示するかどうかを指定するオプション
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    
    #! コマンドライン引数を解析し、結果をargsオブジェクトに格納
    args = parser.parse_args()

    return args

def main():

    #! parse_args関数は、argparseモジュールを利用して、コマンドライン引数を解析する
    args = parse_args()

    #! 推論で利用するモデルの定義
    #! init_model 関数は、与えられた構成ファイル、チェックポイントファイル、デバイスに基づいて、モデルを初期化します。
    # args.config：コマンドライン引数から取得した config の値です。これは構成ファイルのパスで、モデルのアーキテクチャや設定が記述されています
    # args.checkpoint：コマンドライン引数から取得した checkpoint の値です。これは学習済みのモデルの重みが保存されているファイルのパスです
    # device=args.device：コマンドライン引数から取得した device の値です。通常は 'cuda:0'（GPUを使用する場合）や 'cpu'（CPUを使用する場合）などのデバイス名が指定されます。
    model = init_model(
        args.config,# モデルの構成ファイル（configファイル）のパスを引数として渡します
        args.checkpoint,# モデルの重み（checkpointファイル）のパスを引数として渡します
        device=args.device)# 使用するデバイス（例: 'cuda:0' または 'cpu'）を引数として渡します

    #! モデルの可視化設定を調整 
    model.cfg.visualizer.radius = args.radius# キーポイントを描画する際の円の半径を設定
    model.cfg.visualizer.alpha = args.alpha# バウンディングボックスの透明度を設定
    model.cfg.visualizer.line_width = args.thickness # スケルトンのリンクの太さを設定

    #! 事前学習済みの3D顔姿勢推定モデル（Mobilenet V1 ベースのモデル）をロード
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    #! モデルのアーキテクチャ設定 (arch = 'mobilenet_1')
    arch = 'mobilenet_1'
    #! 三角メッシュの読み込み
    tri = sio.loadmat('visualize/tri.mat')['tri']
    #! 画像をテンソルに変換し、さらに平均値 127.5 と標準偏差 128 を用いて正規化する処理を定義
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    #! 学習済みモデルの重みを読み込み (torch.load)
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    #! モデルの構築
    model_3DFFA = getattr(mobilenet_v1, arch)(
        num_classes=62
    )  # 62 = 12(pose) + 40(shape) +10(expression)
    #!モデルの重みを更新
    model_dict = model_3DFFA.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model_3DFFA.load_state_dict(model_dict)
    model_3DFFA.eval()

    #! 可視化のためのオブジェクトを構築
    visualizer = VISUALIZERS.build(model.cfg.visualizer)# モデルの可視化設定に基づいて、ビジュアライザーオブジェクトを作成

    #! データセットのメタ情報とスケルトンスタイルを可視化オブジェクトに設定
    visualizer.set_dataset_meta(
        model.dataset_meta, skeleton_style=args.skeleton_style)# スケルトンのスタイル（例: mmposeやopenpose）を設定

    #! ビデオファイルを開いてキャプチャオブジェクトを作成
    cap = cv2.VideoCapture(args.video)
    # 出力ファイルが指定されている場合、ビデオライターを初期化
    # if args.out_file is not None:
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     out = cv2.VideoWriter(args.out_file, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
    #                           (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #                            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    if args.out_file is not None:
        # 横幅を2倍にする
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.out_file, fourcc, fps, (frame_width, frame_height))


    keypoints_results = []

    #! キーポイント保存用ファイル名定義
    pickle_file = 'keypoints_results.pkl'

    #! フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_count = 0

    #vc = cv2.VideoCapture(args.video)
    #success, frame = vc.read()

    # 顔向き推定データを保存するリスト
    pose_history = []
    start_point_history = []

    #! tqdmを使って進捗を表示
    for _ in tqdm(range(total_frames), desc="Processing frames", unit="frame"):
        ret, frame = cap.read()
        #ret, _ = cap.read()
        if not ret:
            break

        frame_count += 1

        #! mmpose
        img = frame[:, :, ::-1]  # BGR to RGB
        batch_results = inference_topdown(model, img)
        #! resultsが全身のキーポイントデータ
        results = merge_data_samples(batch_results)

        if results.pred_instances.keypoints is not None:
            keypoints = results.pred_instances.keypoints.squeeze()
            face_keypoints = keypoints[23:91, :2].T

            pts_res = []
            Ps = []
            poses = []
            pts68 = []

            roi_box = parse_roi_box_from_landmark(face_keypoints)
            cropimg = crop_img(frame, roi_box)
            cropimg = cv2.resize(cropimg, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)

            input = transform(cropimg).unsqueeze(0)

            with torch.no_grad():
                param = model_3DFFA(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            vertex_0, vertex = predict_68pts(param, roi_box)
            # pts68_ori = vertex_0
            pts68 = vertex
            # face_keypoints[:] = pts68[:2]

            pts_res.append(pts68)

            P, pose = parse_pose(param)
            Ps.append(P)
            poses.append(pose)
            frame2 = plot_kpt(frame, pts68.T)
            frame2 = plot_pose_box(frame, Ps, pts_res)

            #注視領域分割
            start_point = (pts_res[0][0][30], pts_res[0][1][30]) #鼻頭
            frame, yaw_rad, pitch_rad, roll_rad = highlight_direction_vector(frame, pose, start_point, 3, 3, frame_count)                        

            # 実験
            # 顔向きデータを保存
            if pose is not None:
                pose_history.append(pose)
                start_point = (pts_res[0][0][30], pts_res[0][1][30])  # 鼻頭
                start_point_history.append(start_point)
            #print(len(pose_history))

        #! mmposeの結果を描画
        visualizer.add_datasample(
            'result',
            img,
            data_sample=results,
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=args.kpt_thr,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show)

        vis_frame = visualizer.get_image()
        vis_frame = vis_frame[:, :, ::-1]  # RGB to BGR for OpenCV

        frame_h, frame_w = vis_frame.shape[:2]

        print(len(frame),len(vis_frame))

        # 軌道を描画
        #start_point = (pts_res[0][0][30], pts_res[0][1][30])  # 鼻頭
        frame = draw_orbit(frame, pose_history, start_point_history, frame_w, frame_h)

        # ここでフレームをマージする
        combined_frame = cv2.addWeighted(frame, 0, vis_frame, 1, 0)
        combined_frame, yaw_rad, pitch_rad, roll_rad = highlight_direction_vector(combined_frame, pose, start_point, 3, 3, frame_count)                        
        cv2.putText(combined_frame, f'Yaw: {yaw_rad * (180 / math.pi):.2f} ', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, f'Pitch: {pitch_rad * (180 / math.pi):.2f} ', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, f'Roll: {roll_rad * (180 / math.pi):.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # 2つのフレームを横に並べる
        combined_frame2 = display_side_by_side(combined_frame, frame)

        if args.out_file is not None:
            out.write(combined_frame2)  # 書き込みを1回に統合

        if args.show:
            cv2.imshow('Frame', combined_frame2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

    #ビデオファイルの書き込みを終了
    #out.release()
    if args.out_file is not None:
        out.release()
    if args.show:
        cv2.destroyAllWindows()

    with open(pickle_file, 'wb') as f:
        pickle.dump(keypoints_results, f)

if __name__ == '__main__':
    main()

