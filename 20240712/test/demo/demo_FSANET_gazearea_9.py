# The demo credit belongs to Yi-Ting Chen

import os
import cv2
import sys
sys.path.append('..')
import numpy as np
from math import cos, sin
# from moviepy.editor import *
from lib.FSANET_model import *
# from moviepy.editor import *
from keras import backend as K
from keras.layers import Average

def highlight_attention_region_on_split_image(img, yaw, pitch):
    img_h, img_w = img.shape[:2]

    # 画像を9つの領域に分割（縦3分割、横3分割）
    regions = []
    for i in range(3):
        for j in range(3):
            region = img[int(i * img_h / 3):int((i + 1) * img_h / 3), int(j * img_w / 3):int((j + 1) * img_w / 3)]
            regions.append(region)

    # 頭部方向に応じて対応する領域をハイライト
    if pitch > 15:  # 上を向いている場合
        if yaw > 15:
            region_to_highlight = regions[2]  # 右上
        elif yaw < -15:
            region_to_highlight = regions[0]  # 左上
        else:
            region_to_highlight = regions[1]  # 上中央
    elif pitch < -15:  # 下を向いている場合
        if yaw > 15:
            region_to_highlight = regions[8]  # 右下
        elif yaw < -15:
            region_to_highlight = regions[6]  # 左下
        else:
            region_to_highlight = regions[7]  # 下中央
    else:  # 水平方向
        if yaw > 15:
            region_to_highlight = regions[5]  # 右中央
        elif yaw < -15:
            region_to_highlight = regions[3]  # 左中央
        else:
            region_to_highlight = regions[4]  # 中央

    # ハイライトの描画
    overlay = region_to_highlight.copy()
    cv2.rectangle(overlay, (0, 0), (region_to_highlight.shape[1], region_to_highlight.shape[0]), (0, 255, 255), -1)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, region_to_highlight, 1 - alpha, 0, region_to_highlight)

    # 9つの領域を再結合して1つの画像に戻す
    top_row = np.hstack((regions[0], regions[1], regions[2]))
    middle_row = np.hstack((regions[3], regions[4], regions[5]))
    bottom_row = np.hstack((regions[6], regions[7], regions[8]))
    highlighted_img = np.vstack((top_row, middle_row, bottom_row))

    return highlighted_img


def color_attention_region(img, yaw, tdx=None, tdy=None):
    img_h, img_w = img.shape[:2]
    
    # 頭部方向に応じて注視領域を色分け
    if yaw < -15:
        # 左を向いているときは青
        color = (255, 0, 0)
    elif yaw > 15:
        # 右を向いているときは赤
        color = (0, 0, 255)
    else:
        # 真ん中を向いているときは緑
        color = (0, 255, 0)

    overlay = img.copy()
    output = img.copy()

    cv2.rectangle(overlay, (0, 0), (img_w, img_h), color, -1)
    
    # 透明度を設定
    alpha = 0.3  # 透明度
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    
    return output

def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 80):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img
    
def draw_results(detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot):
    
    if len(detected) > 0:
        for i, (x, y, w, h) in enumerate(detected):
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)
            
            faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)        
            
            face = np.expand_dims(faces[i, :, :, :], axis=0)
            p_result = model.predict(face)
            
            face = face.squeeze()

            # 注視領域をハイライト
            # 元画像全体を9分割してハイライト
            # 関数呼び出し時に yaw と pitch を渡す
            highlighted_img = highlight_attention_region_on_split_image(input_img, p_result[0][0], p_result[0][1])
            img = draw_axis(highlighted_img[yw1:yw2 + 1, xw1:xw2 + 1, :], p_result[0][0], p_result[0][1], p_result[0][2])

            input_img[yw1:yw2 + 1, xw1:xw2 + 1, :] = img
            
    cv2.imshow("result", input_img)
    
    return input_img

import time

def main():
    try:
        os.mkdir('./img')
    except OSError:
        pass
    
    K.set_learning_phase(0) # make sure its testing mode
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    
    # load model and weights
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    img_idx = 0
    detected = '' #make this not local variable
    time_detection = 0
    time_network = 0
    time_plot = 0
    skip_frame = 5 # every 5 frame do 1 detection and network forward propagation
    ad = 0.6

    #Parameters
    num_capsule = 3
    dim_capsule = 16
    routings = 2
    stage_num = [3,3,3]
    lambda_d = 1
    num_classes = 3
    image_size = 64
    num_primcaps = 7*3
    m_dim = 5
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    num_primcaps = 8*8*3
    S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    
    print('Loading models ...')

    weight_file1 = '../pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
    model1.load_weights(weight_file1)
    print('Finished loading model 1.')
    
    weight_file2 = '../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
    model2.load_weights(weight_file2)
    print('Finished loading model 2.')

    weight_file3 = '../pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
    model3.load_weights(weight_file3)
    print('Finished loading model 3.')

    inputs = Input(shape=(64,64,3))
    x1 = model1(inputs) #1x1
    x2 = model2(inputs) #var
    x3 = model3(inputs) #w/o
    avg_model = Average()([x1,x2,x3])
    model = Model(inputs=inputs, outputs=avg_model)
    
    # capture video
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)
    
    # Define the codec and create VideoWriter object
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
    #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1024, 768))

    # 動画ファイル保存用の設定
    fps = int(cap.get(cv2.CAP_PROP_FPS))                    # カメラのFPSを取得
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))              # カメラの横幅を取得
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))             # カメラの縦幅を取得
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')        # 動画保存時のfourcc設定（mp4用）
    out = cv2.VideoWriter('video.mp4', fourcc, fps, (w, h))  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

    print('Start detecting pose ...')
    detected_pre = []

    record_duration = 30  # 録画時間を15秒に設定
    end_time = time.time() + record_duration  # 録画終了時間を設定

    while time.time() < end_time:
        # get video frame
        ret, input_img = cap.read()

        if ret:
            img_idx += 1
            img_h, img_w, _ = np.shape(input_img)

            if img_idx == 1 or img_idx % skip_frame == 0:
                time_detection = 0
                time_network = 0
                time_plot = 0

                # detect faces using LBP detector
                gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
                detected = face_cascade.detectMultiScale(gray_img, 1.1)

                if len(detected_pre) > 0 and len(detected) == 0:
                    detected = detected_pre

                faces = np.empty((len(detected), img_size, img_size, 3))

                input_img = draw_results(detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot)
                cv2.imwrite('img/' + str(img_idx) + '.png', input_img)
            else:
                input_img = draw_results(detected, input_img, faces, ad, img_size, img_w, img_h, model, time_detection, time_network, time_plot)

            # Write the frame to the output video file
            out.write(input_img)

            if len(detected) > len(detected_pre) or img_idx % (skip_frame * 3) == 0:
                detected_pre = detected

            key = cv2.waitKey(1)

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
        
if __name__ == '__main__':
    main()
