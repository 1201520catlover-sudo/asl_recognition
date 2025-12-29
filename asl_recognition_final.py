import cv2
import mediapipe as mp
import math
import csv
import numpy as np

# --- 1. 載入個人化手語資料庫 ---
hand_samples = []
try:
    with open('hand_data.csv', mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[0]
            angles = [float(x) for x in row[1:]]
            hand_samples.append({'label': label, 'angles': angles})
    print(f"已讀取 {len(hand_samples)} 組字母特徵基準。")
except FileNotFoundError:
    print("錯誤：找不到 hand_data.csv，請先執行採樣程式。")
    exit()

# --- 2. 初始化偵測與繪圖工具 ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

def get_angle(v1, v2):
    try:
        angle = math.degrees(math.acos((v1[0]*v2[0] + v1[1]*v2[1]) / 
            (((v1[0]**2 + v1[1]**2)**0.5) * ((v2[0]**2 + v2[1]**2)**0.5))))
    except: angle = 180
    return angle

def get_current_angles(landmarks):
    """即時計算畫面上手部的五指角度向量"""
    pts = [(i.x, i.y) for i in landmarks.landmark]
    indices = [[0,2,4], [0,6,8], [0,10,12], [0,14,16], [0,18,20]]
    angles = []
    for idx in indices:
        v1 = (pts[idx[0]][0]-pts[idx[1]][0], pts[idx[0]][1]-pts[idx[1]][1])
        v2 = (pts[idx[1]][0]-pts[idx[2]][0], pts[idx[1]][1]-pts[idx[2]][1])
        angles.append(get_angle(v1, v2))
    return angles

def find_closest_letter(current_angles, pts):
    """核心算法：比對當前特徵與資料庫的相似度"""
    best_match = "?"
    min_dist = float('inf')
    
    # 對資料庫進行遍歷，尋找「歐幾里得距離」最短的樣本
    for sample in hand_samples:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_angles, sample['angles'])))
        if dist < min_dist:
            min_dist = dist
            best_match = sample['label']
    
    # --- 空間方向校正機制 (針對 P 與 K) ---
    index_finger_tip_y = pts[8][1] # 食指尖 y 座標
    wrist_y = pts[0][1]            # 手腕 y 座標
    
    # 如果最接近 K 但手指朝下，則根據 ASL 定義修正為 P
    if best_match == "K" and index_finger_tip_y > wrist_y:
        return "P"

    # 若最小距離大於門檻值 (60)，代表手勢不夠標準或不在資料庫內
    return best_match if min_dist < 60 else "Scanning..."

# --- 4. 打字機系統變數 ---
typed_text = ""         # 螢幕上顯示的總文字
current_letter = ""     # 目前正鎖定的字母
confirm_counter = 0     # 穩定度計數器
CONFIRM_THRESHOLD = 30  # 設定需穩定偵測 30 幀才正式輸入一個字

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    detected_now = ""
    
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        # 繪製手部節點連線圖
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        
        c_angles = get_current_angles(lm)
        pts = [(i.x, i.y) for i in lm.landmark]
        
        # 進行字母特徵匹配
        detected_now = find_closest_letter(c_angles, pts)
        
        # 在手部上方顯示即時辨識結果
        cv2.putText(frame, f"Match: {detected_now}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # --- 穩定打字輸入邏輯 ---
    if detected_now != "" and detected_now != "Scanning...":
        # 如果偵測到的字母與當前鎖定的一致
        if detected_now == current_letter:
            confirm_counter += 1
        else:
            # 若手勢改變，重新計時
            current_letter = detected_now
            confirm_counter = 0
            
        # 當穩定度達標 (例如持續比了 1 秒)
        if confirm_counter == CONFIRM_THRESHOLD:
            typed_text += current_letter
            confirm_counter = 0 
    else:
        confirm_counter = 0

    # --- UI 顯示介面 ---
    # 繪製底部的黑色文字背景區
    cv2.rectangle(frame, (0, h-100), (w, h), (0, 0, 0), -1)
    # 顯示目前已輸入的所有文字
    cv2.putText(frame, f"TEXT: {typed_text}", (20, h-40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.imshow('ASL Typing System', frame)
    
    # --- 鍵盤控制邏輯 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break     # 退出程式
    elif key == ord('r'):         # 重置 (清空文字)
        typed_text = ""
    elif key == ord(' '):         # 新增空格
        typed_text += " "
    elif key == ord('.'):         # 新增句點
        typed_text += "."
    elif key == ord(','):         # 新增逗點
        typed_text += ","
    elif key == ord('b'):         # Backspace (刪除最後一個字)
        typed_text = typed_text[:-1]

cap.release()
cv2.destroyAllWindows()