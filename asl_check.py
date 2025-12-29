import cv2
import mediapipe as mp
import math
import csv
import datetime

# --- 1. 讀取 CSV 數據 ---
hand_samples = []
try:
    with open('hand_data.csv', mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[0]
            angles = [float(x) for x in row[1:]]
            hand_samples.append({'label': label, 'angles': angles})
    print(f"成功載入 {len(hand_samples)} 筆數據！")
except FileNotFoundError:
    print("找不到 hand_data.csv")
    exit()

# --- 2. 初始化統計字典 ---
# 格式: {'a': {'total': 0, 'correct': 0}, 'b': ...}
stats = {chr(i): {'total': 0, 'correct': 0} for i in range(ord('a'), ord('z') + 1)}
check_msg = ""
msg_timer = 0

# --- 3. 初始化 MediaPipe ---
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
    pts = [(i.x, i.y) for i in landmarks.landmark]
    indices = [[0,2,4], [0,6,8], [0,10,12], [0,14,16], [0,18,20]]
    angles = []
    for idx in indices:
        v1 = (pts[idx[0]][0]-pts[idx[1]][0], pts[idx[0]][1]-pts[idx[1]][1])
        v2 = (pts[idx[1]][0]-pts[idx[2]][0], pts[idx[1]][1]-pts[idx[2]][1])
        angles.append(get_angle(v1, v2))
    return angles

def find_closest_letter(current_angles, pts):
    best_match = "?"
    min_dist = float('inf')
    for sample in hand_samples:
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_angles, sample['angles'])))
        if dist < min_dist:
            min_dist = dist
            best_match = sample['label']
    
    # K/P 修正
    index_finger_tip_y = pts[8][1]
    wrist_y = pts[0][1]
    if best_match == "K" and index_finger_tip_y > wrist_y:
        best_match = "P"
    
    return best_match if min_dist < 60 else "Scanning..."

# --- 4. 主程式迴圈 ---
cap = cv2.VideoCapture(0)

print("【測試模式開啟】")
print("比出手勢後按下對應字母鍵，程式將自動統計正確率。按 'q' 結束並輸出報告。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    detected_now = ""
    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
        c_angles = get_current_angles(lm)
        pts = [(i.x, i.y) for i in lm.landmark]
        detected_now = find_closest_letter(c_angles, pts)
        
        cv2.putText(frame, f"Match: {detected_now}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # --- 顯示 Check 訊息 ---
    if msg_timer > 0:
        color = (0, 255, 0) if "Correct" in check_msg else (0, 0, 255)
        cv2.putText(frame, check_msg, (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        msg_timer -= 1

    cv2.imshow('ASL Accuracy Tester', frame)
    
    # --- 監聽按鍵 ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '): 
        break
    elif ord('a') <= key <= ord('z'):
        user_ans = chr(key)
        if detected_now != "" and detected_now != "Scanning...":
            stats[user_ans]['total'] += 1
            if detected_now.lower() == user_ans:
                stats[user_ans]['correct'] += 1
                check_msg = f"Check: {user_ans} -> Correct!"
            else:
                check_msg = f"Check: {user_ans} -> Wrong! (Got {detected_now})"
            msg_timer = 20 # 訊息顯示 20 幀
            print(f"測試字母 {user_ans}: 目前正確次數 {stats[user_ans]['correct']}/{stats[user_ans]['total']}")

cap.release()
cv2.destroyAllWindows()

# --- 5. 輸出統計結果至 result.txt ---
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
with open('result.txt', mode='w', encoding='utf-8') as f:
    f.write(f"=== 手語辨識正確率測試報告 ({now}) ===\n\n")
    f.write(f"{'字母':<5} | {'測試次數':<8} | {'正確次數':<8} | {'正確率':<8}\n")
    f.write("-" * 45 + "\n")
    
    total_all = 0
    correct_all = 0
    
    for char in sorted(stats.keys()):
        s = stats[char]
        if s['total'] > 0:
            acc = (s['correct'] / s['total']) * 100
            f.write(f"{char.upper():<7} | {s['total']:<12} | {s['correct']:<12} | {acc:.2f}%\n")
            total_all += s['total']
            correct_all += s['correct']
    
    f.write("-" * 45 + "\n")
    if total_all > 0:
        total_acc = (correct_all / total_all) * 100
        f.write(f"總計    | {total_all:<12} | {correct_all:<12} | {total_acc:.2f}%\n")
    else:
        f.write("未進行任何測試。\n")

print("\n測試結束！結果已儲存至 result.txt。")