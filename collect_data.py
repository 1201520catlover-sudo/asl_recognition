import cv2
import mediapipe as mp
import math
import csv

# --- 初始化 MediaPipe 手部偵測模組 ---
mp_hands = mp.solutions.hands
# static_image_mode=False 代表追蹤動態視訊，min_detection_confidence 為偵測信心門檻
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def get_angle(v1, v2):
    """計算兩組二維向量之間的夾角 (0~180度)"""
    try:
        # 使用內積公式計算夾角：cos(theta) = (v1·v2) / (|v1|*|v2|)
        angle = math.degrees(math.acos((v1[0]*v2[0] + v1[1]*v2[1]) / 
            (((v1[0]**2 + v1[1]**2)**0.5) * ((v2[0]**2 + v2[1]**2)**0.5))))
    except: 
        angle = 180  # 若計算出錯則回傳 180 度 (代表彎曲)
    return angle

def get_hand_angles(landmarks):
    """提取手部 21 個節點座標，並計算五隻手指的關鍵夾角"""
    points = [(i.x, i.y) for i in landmarks.landmark]
    angle_list = []
    # 定義每隻手指由哪三個節點組成向量 (根部, 中間, 尖端)
    # indices 代表：大拇指、食指、中指、無名指、小指
    indices = [[0,2,4], [0,6,8], [0,10,12], [0,14,16], [0,18,20]]
    for idx in indices:
        # 建立兩個向量：根部到中間 (v1)，中間到尖端 (v2)
        v1 = (points[idx[0]][0]-points[idx[1]][0], points[idx[0]][1]-points[idx[1]][1])
        v2 = (points[idx[1]][0]-points[idx[2]][0], points[idx[1]][1]-points[idx[2]][1])
        angle_list.append(round(get_angle(v1, v2), 2))
    return angle_list

cap = cv2.VideoCapture(0)
print("【數據蒐集模式】比好手勢後，按下該字母鍵紀錄。按 空白鍵 退出。")

# 使用 'a' 模式追加寫入 CSV，避免覆蓋舊資料
with open('hand_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1) # 鏡像翻轉，讓操作更直覺
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        current_angles = []
        if results.multi_hand_landmarks:
            # 取得當前手部的五個角度
            current_angles = get_hand_angles(results.multi_hand_landmarks[0])
            # 在畫面左上角即時顯示角度數據
            cv2.putText(frame, str(current_angles), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Data Collector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '): # 按下空白鍵退出程式
            break
        elif key != 255: # 當按下任何字母鍵時 (key 不等於預設值 255)
            char = chr(key)
            if current_angles:
                # 將字母標籤與五個角度寫入 CSV
                writer.writerow([char] + current_angles)
                print(f"成功存入字母 {char} 的特徵數據: {current_angles}")

cap.release()
cv2.destroyAllWindows()