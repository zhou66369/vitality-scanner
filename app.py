import av
import cv2
import numpy as np
import mediapipe as mp
import time
import math
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image, ImageDraw, ImageFont

# --- 页面基础配置 ---
st.set_page_config(page_title="馀芮园 AI 元气检测", page_icon="🌿")

# 隐藏 Streamlit 默认菜单
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# --- 字体加载 (适配 Linux 服务器) ---
# 请技术人员务必在服务器根目录放入中文字体文件并重命名为 'font.ttf'
FONT_PATH = "font.ttf" 

# --- 核心参数 ---
SCAN_DURATION = 10
EYE_SENSITIVITY = 6.5

class YuRuiYuanProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.state = "IDLE" 
        self.start_time = 0
        self.buffer = {'focus': [], 'plump': [], 'conf': [], 'stress': [], 'joy': [], 'char': []}
        self.result_card = None
        self.scan_completed = False

    def put_text_cn(self, img, text, pos, color=(255, 255, 255), size=20, align="left"):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype(FONT_PATH, size)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        
        draw_x, draw_y = pos
        if align == "center":
            draw_x = pos[0] - w // 2
        elif align == "right":
            draw_x = pos[0] - w

        # 黑色描边
        draw.text((draw_x+1, draw_y+1), text, font=font, fill=(0,0,0))
        draw.text((draw_x, draw_y), text, font=font, fill=color)
        
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def calculate_metrics(self, landmarks, w, h, image_hsv):
        metrics = {}
        # 1. 专注力
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        def get_ear(indices):
            coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
            v = np.linalg.norm(coords[1] - coords[5]) + np.linalg.norm(coords[2] - coords[4])
            hor = np.linalg.norm(coords[0] - coords[3])
            return v / (2.0 * hor)
        avg_ear = (get_ear(LEFT_EYE) + get_ear(RIGHT_EYE)) / 2.0
        metrics['focus'] = np.clip((avg_ear - 0.18) * 100 * EYE_SENSITIVITY, 30, 98)

        # 2. 充盈度
        mean_s = np.mean(image_hsv[:, :, 1])
        if 40 < mean_s < 90: raw_plump = 85 + (mean_s - 40)/2
        else: raw_plump = 60
        metrics['plump'] = np.clip(raw_plump, 35, 95)

        # 3. 自信力
        nose_y = landmarks[1].y
        brow_y = landmarks[10].y
        chin_y = landmarks[152].y
        ratio = (nose_y - brow_y) / (chin_y - brow_y)
        if ratio < 0.48: pose_score = 95
        elif ratio > 0.55: pose_score = 45
        else: pose_score = 70
        metrics['conf'] = pose_score

        # 4. 抗压力
        metrics['stress'] = np.clip(metrics['focus'] * 0.6 + metrics['conf'] * 0.4, 40, 90)

        # 5. 愉悦值
        mouth_l = landmarks[61].y
        mouth_r = landmarks[291].y
        mouth_c = landmarks[0].y 
        smile_ratio = (mouth_l + mouth_r) / 2 - mouth_c
        metrics['joy'] = 90 if smile_ratio < 0 else 65

        # 6. 魅力值
        nose_x = landmarks[1].x
        metrics['char'] = 92 if 0.4 < nose_x < 0.6 else 60
        return metrics

    def draw_radar_chart(self, image, center, size, data, color=(0, 255, 255), label_scale=1.0):
        labels_cn = ["专注力", "充盈度", "自信力", "抗压力", "愉悦值", "魅力值"]
        labels_en = ["Focus", "Plumpness", "Confidence", "Resilience", "Joy", "Charisma"]
        keys = ['focus', 'plump', 'conf', 'stress', 'joy', 'char']
        values = [data[k] for k in keys]
        
        num_vars = len(labels_cn)
        angle_step = 2 * math.pi / num_vars
        
        # 绘制网格
        for r_step in [0.3, 0.6, 1.0]:
            pts = []
            for i in range(num_vars):
                angle = i * angle_step - math.pi / 2
                r = size * r_step
                x = int(center[0] + r * math.cos(angle))
                y = int(center[1] + r * math.sin(angle))
                pts.append((x, y))
            pts = np.array(pts, np.int32)
            cv2.polylines(image, [pts], True, (100, 100, 100), 1, cv2.LINE_AA)

        # 绘制数据填充
        data_pts = []
        for i in range(num_vars):
            angle = i * angle_step - math.pi / 2
            val_r = (values[i] / 100.0) * size
            x = int(center[0] + val_r * math.cos(angle))
            y = int(center[1] + val_r * math.sin(angle))
            data_pts.append((x, y))
        
        data_pts = np.array(data_pts, np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [data_pts], (0, 150, 0))
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv2.polylines(image, [data_pts], True, color, 2, cv2.LINE_AA)

        # 绘制标签 (双语)
        for i in range(num_vars):
            angle = i * angle_step - math.pi / 2
            label_r = size * 1.35
            lx = int(center[0] + label_r * math.cos(angle))
            ly = int(center[1] + label_r * math.sin(angle))
            
            image = self.put_text_cn(image, labels_cn[i], (lx, ly-18), (255,255,255), int(18*label_scale), align="center")
            image = self.put_text_cn(image, labels_en[i], (lx, ly+4), (180,180,180), int(14*label_scale), align="center")
            image = self.put_text_cn(image, str(int(values[i])), (lx, ly+22), color, int(16*label_scale), align="center")
        return image

    def generate_result_card(self, score, metrics):
        # 黑色背景竖图 (1280x720)
        card = np.zeros((1280, 720, 3), dtype=np.uint8)
        
        # 1. 品牌区
        card = self.put_text_cn(card, "馀芮园 AI 生物元气光谱分析", (360, 100), (255, 255, 255), 32, align="center")
        card = self.put_text_cn(card, "YuRuiYuan AI Bio-Vitality Spectral Analysis", (360, 150), (150, 150, 150), 18, align="center")
        
        # 2. 分数区
        card = self.put_text_cn(card, str(int(score)), (360, 280), (0, 255, 0), 100, align="center")
        card = self.put_text_cn(card, "VITALITY INDEX / 元气指数", (360, 360), (255, 255, 255), 24, align="center")
        
        # 评级
        if score > 80: level_cn, level_en, col = "元气充盈", "EXCELLENT", (0, 255, 0)
        elif score > 60: level_cn, level_en, col = "状态平稳", "NORMAL", (0, 255, 255)
        else: level_cn, level_en, col = "能量透支", "CRITICAL", (0, 0, 255)
        
        # 状态条
        card = self.put_text_cn(card, f"{level_cn} | {level_en}", (360, 430), col, 30, align="center")

        # 3. 雷达图 (V3.4 优化版坐标)
        card = self.draw_radar_chart(card, (360, 760), 175, metrics, color=col, label_scale=1.1)
        
        # 4. 底部隐私声明
        card = self.put_text_cn(card, "隐私保护声明 / PRIVACY NOTICE", (360, 1050), (180, 180, 180), 16, align="center")
        card = self.put_text_cn(card, "本检测数据仅在本地运行，系统不保存任何个人信息。", (360, 1080), (120, 120, 120), 14, align="center")
        card = self.put_text_cn(card, "请自行截图保存结果", (360, 1130), (255, 255, 0), 24, align="center")
        
        # 5. 二维码占位
        cv2.rectangle(card, (600, 1160), (700, 1260), (255, 255, 255), 1)
        card = self.put_text_cn(card, "扫码内测", (650, 1230), (255, 255, 255), 12, align="center")
        return card

    # === WebRTC 每一帧的回调处理 ===
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        
        # 状态机逻辑
        if self.state == "IDLE":
            image = self.put_text_cn(image, "点击下方按钮开始", (w//2, h//2), (0, 255, 0), 40, align="center")
            
        elif self.state == "SCANNING":
            if self.start_time == 0: self.start_time = time.time()
            elapsed = time.time() - self.start_time
            remaining = int(SCAN_DURATION - elapsed)
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            results = self.face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # 画网格
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image, landmark_list=results.multi_face_landmarks[0],
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0))
                
                curr_metrics = self.calculate_metrics(landmarks, w, h, image_hsv)
                for k in curr_metrics: self.buffer[k].append(curr_metrics[k])
                image = self.draw_radar_chart(image, (w//2, h-250), 100, curr_metrics)

            image = self.put_text_cn(image, str(remaining), (w//2, 150), (0,255,0), 80, align="center")

            if remaining <= 0:
                self.state = "COMPLETED"
                # 计算结果
                final_m = {}
                for k in self.buffer:
                    if self.buffer[k]:
                        vals = sorted(self.buffer[k])
                        mid = vals[len(vals)//4 : -len(vals)//4] if len(vals)>4 else vals
                        final_m[k] = sum(mid)/len(mid)
                    else: final_m[k] = 50
                
                raw = (final_m['focus']*0.25 + final_m['plump']*0.2 + final_m['conf']*0.2 + 
                       final_m['stress']*0.1 + final_m['joy']*0.1 + final_m['char']*0.15)
                if raw > 90: raw = 90 + (raw-90)*0.5
                if raw < 35: raw = 35
                
                # 生成卡片并存入 session_state
                self.result_card = self.generate_result_card(int(raw), final_m)
                st.session_state["final_card"] = self.result_card
                self.scan_completed = True

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# --- Streamlit 前端 ---
def main():
    if "processor" not in st.session_state:
        st.session_state["processor"] = YuRuiYuanProcessor()

    st.title("YuRuiYuan AI Bio-Scan")
    st.caption("请授权摄像头权限 · 建议使用微信或 Safari 打开")

    ctx = webrtc_streamer(
        key="vitality-scanner",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": {"facingMode": "user"}, "audio": False},
        video_processor_factory=lambda: st.session_state["processor"],
        async_processing=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 开始检测 / START"):
            if st.session_state["processor"]:
                st.session_state["processor"].state = "SCANNING"
                st.session_state["processor"].start_time = time.time()
                st.session_state["processor"].buffer = {k:[] for k in ['focus', 'plump', 'conf', 'stress', 'joy', 'char']}
    
    with col2:
        if st.button("🔄 重置 / RESET"):
            if st.session_state["processor"]:
                st.session_state["processor"].state = "IDLE"
                st.session_state["processor"].scan_completed = False

    if "final_card" in st.session_state and st.session_state["processor"].scan_completed:
        st.success("检测完成！请长按下方图片保存")
        card_rgb = cv2.cvtColor(st.session_state["final_card"], cv2.COLOR_BGR2RGB)
        st.image(card_rgb, caption="YuRuiYuan AI-LABS Report", use_column_width=True)

if __name__ == "__main__":
    main()