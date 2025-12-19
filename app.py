import av
import cv2
import numpy as np
import mediapipe as mp
import time
import math
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import Image, ImageDraw, ImageFont

# --- 页面基础配置 ---
st.set_page_config(page_title="馀芮园 AI 元气检测", page_icon="🌿")
hide_st_style = """<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)

FONT_PATH = "font.ttf" 
SCAN_DURATION = 10
EYE_SENSITIVITY = 6.5

class YuRuiYuanProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.state = "IDLE" 
        self.start_time = 0
        self.buffer = {'focus': [], 'plump': [], 'conf': [], 'stress': [], 'joy': [], 'char': []}
        self.result_card = None
        self.scan_completed = False

    def put_text_cn(self, img, text, pos, color=(255, 255, 255), size=20, align="left"):
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try: font = ImageFont.truetype(FONT_PATH, size)
        except: font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw_x, draw_y = pos
        if align == "center": draw_x = pos[0] - w // 2
        elif align == "right": draw_x = pos[0] - w
        draw.text((draw_x+1, draw_y+1), text, font=font, fill=(0,0,0))
        draw.text((draw_x, draw_y), text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def calculate_metrics(self, landmarks, w, h, image_hsv):
        metrics = {}
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        def get_ear(indices):
            coords = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
            v = np.linalg.norm(coords[1] - coords[5]) + np.linalg.norm(coords[2] - coords[4])
            hor = np.linalg.norm(coords[0] - coords[3])
            return v / (2.0 * hor)
        avg_ear = (get_ear(LEFT_EYE) + get_ear(RIGHT_EYE)) / 2.0
        metrics['focus'] = np.clip((avg_ear - 0.18) * 100 * EYE_SENSITIVITY, 30, 98)
        mean_s = np.mean(image_hsv[:, :, 1])
        metrics['plump'] = np.clip(85 + (mean_s - 40)/2 if 40 < mean_s < 90 else 60, 35, 95)
        nose_y, brow_y, chin_y = landmarks[1].y, landmarks[10].y, landmarks[152].y
        ratio = (nose_y - brow_y) / (chin_y - brow_y)
        metrics['conf'] = 95 if ratio < 0.48 else (45 if ratio > 0.55 else 70)
        metrics['stress'] = np.clip(metrics['focus'] * 0.6 + metrics['conf'] * 0.4, 40, 90)
        metrics['joy'] = 90 if ((landmarks[61].y + landmarks[291].y) / 2 - landmarks[0].y) < 0 else 65
        metrics['char'] = 92 if 0.4 < landmarks[1].x < 0.6 else 60
        return metrics

    def draw_radar_chart(self, image, center, size, data, color=(0, 255, 255)):
        keys = ['focus', 'plump', 'conf', 'stress', 'joy', 'char']
        values = [data[k] for k in keys]
        num_vars = 6
        angle_step = 2 * math.pi / num_vars
        for r_step in [0.3, 0.6, 1.0]:
            pts = []
            for i in range(num_vars):
                angle = i * angle_step - math.pi / 2
                pts.append((int(center[0] + size * r_step * math.cos(angle)), int(center[1] + size * r_step * math.sin(angle))))
            cv2.polylines(image, [np.array(pts, np.int32)], True, (100, 100, 100), 1, cv2.LINE_AA)
        data_pts = []
        for i in range(num_vars):
            angle = i * angle_step - math.pi / 2
            data_pts.append((int(center[0] + (values[i] / 100.0) * size * math.cos(angle)), int(center[1] + (values[i] / 100.0) * size * math.sin(angle))))
        data_pts = np.array(data_pts, np.int32)
        overlay = image.copy()
        cv2.fillPoly(overlay, [data_pts], (0, 150, 0))
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        cv2.polylines(image, [data_pts], True, color, 2, cv2.LINE_AA)
        labels = ["专注力", "充盈度", "自信力", "抗压力", "愉悦值", "魅力值"]
        for i in range(num_vars):
            angle = i * angle_step - math.pi / 2
            lx, ly = int(center[0] + size * 1.35 * math.cos(angle)), int(center[1] + size * 1.35 * math.sin(angle))
            image = self.put_text_cn(image, labels[i], (lx, ly), (255,255,255), 18, align="center")
            image = self.put_text_cn(image, str(int(values[i])), (lx, ly+20), color, 16, align="center")
        return image

    def generate_result_card(self, score, metrics):
        card = np.zeros((1280, 720, 3), dtype=np.uint8)
        card = self.put_text_cn(card, "馀芮园 AI 生物元气检测", (360, 100), (255, 255, 255), 32, align="center")
        col = (0, 255, 0) if score > 80 else ((0, 255, 255) if score > 60 else (0, 0, 255))
        card = self.put_text_cn(card, str(int(score)), (360, 280), (0, 255, 0), 100, align="center")
        card = self.put_text_cn(card, "元气指数", (360, 360), (255, 255, 255), 24, align="center")
        card = self.draw_radar_chart(card, (360, 700), 180, metrics, color=col)
        card = self.put_text_cn(card, "扫码关注", (360, 1100), (180, 180, 180), 20, align="center")
        return card

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)
        h, w, _ = image.shape
        if self.state == "IDLE":
            image = self.put_text_cn(image, "点击下方按钮开始", (w//2, h//2), (0, 255, 0), 40, align="center")
        elif self.state == "SCANNING":
            if self.start_time == 0: self.start_time = time.time()
            remaining = int(SCAN_DURATION - (time.time() - self.start_time))
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                mp.solutions.drawing_utils.draw_landmarks(image, results.multi_face_landmarks[0], mp.solutions.face_mesh.FACEMESH_TESSELATION, None, mp.solutions.drawing_styles.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=0))
                curr = self.calculate_metrics(landmarks, w, h, cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                for k in curr: self.buffer[k].append(curr[k])
                image = self.draw_radar_chart(image, (w//2, h-250), 100, curr)
            image = self.put_text_cn(image, str(remaining), (w//2, 150), (0,255,0), 80, align="center")
            if remaining <= 0:
                self.state = "COMPLETED"
                final_m = {k: (sum(v)/len(v) if v else 50) for k,v in self.buffer.items()}
                raw = (final_m['focus']*0.25 + final_m['plump']*0.2 + final_m['conf']*0.2 + final_m['stress']*0.1 + final_m['joy']*0.1 + final_m['char']*0.15)
                self.result_card = self.generate_result_card(int(90 + (raw-90)*0.5 if raw>90 else (raw if raw>35 else 35)), final_m)
                self.scan_completed = True
        return av.VideoFrame.from_ndarray(image, format="bgr24")

def main():
    st.title("YuRuiYuan AI Bio-Scan")
    st.caption("中国网络优化版 · 请使用 Firefox 或开启 VPN")
    
    # === 关键修改：混合了全球通用和中国友好的连接服务器 ===
    rtc_config = RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}, 
            {"urls": ["stun:global.stun.twilio.com:3478"]},
            {"urls": ["stun:stun.miwifi.com"]}, 
            {"urls": ["stun:stun.qq.com"]}, 
        ]
    })

    ctx = webrtc_streamer(
        key="vitality-scanner", mode=WebRtcMode.SENDRECV, rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=YuRuiYuanProcessor, async_processing=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 开始检测 / START"):
            if ctx.video_processor:
                ctx.video_processor.state = "SCANNING"
                ctx.video_processor.start_time = time.time()
                ctx.video_processor.buffer = {k:[] for k in ['focus', 'plump', 'conf', 'stress', 'joy', 'char']}
    with col2:
        if st.button("🔄 重置 / RESET"):
            if ctx.video_processor:
                ctx.video_processor.state = "IDLE"
                ctx.video_processor.scan_completed = False
                
    if ctx.video_processor and ctx.video_processor.scan_completed and ctx.video_processor.result_card is not None:
        st.success("检测完成！")
        st.image(cv2.cvtColor(ctx.video_processor.result_card, cv2.COLOR_BGR2RGB), use_column_width=True)

if __name__ == "__main__":
    main()
