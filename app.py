import tkinter as tk
import cv2
from PIL import Image, ImageTk
import requests
from datetime import datetime
import mediapipe as mp
import math

# ------------------- CONFIG -------------------
WEATHER_API_KEY = "5e554023cc149f2ad2fb5518b6be04e2"
NEWS_API_KEY = "c323a0a45eff4786bc8b31e519aa478c"
UPDATE_INTERVAL = 60000
REMINDERS = ["Meeting at 3 PM", "Call Mom", "Buy groceries"]

# Map number of fingers to cities
finger_city_map = {1: "Delhi", 2: "Mumbai", 3: "London", 4: "New York", 5: "Tokyo"}

# ---------------- Weather & News ----------------
def get_weather(location):
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url).json()
        if response.get("cod") != 200:
            return f"Weather data not available for {location}"
        temp = response["main"]["temp"]
        desc = response["weather"][0]["description"].title()
        city_name = response["name"]
        country_code = response["sys"]["country"]
        return f"Weather in {city_name}, {country_code}: {temp}°C, {desc}"
    except Exception as e:
        print("Weather error:", e)
        return "Weather data not available."

def get_news(location):
    try:
        url = f"https://newsapi.org/v2/top-headlines?q={location}&apiKey={NEWS_API_KEY}"
        response = requests.get(url).json()
        articles = response.get("articles", [])
        if not articles:
            articles = [{"title": f"Demo News from {location} {i+1}"} for i in range(5)]
        news_list = [f"{i+1}. {a['title']}" for i, a in enumerate(articles[:5])]
        return "Top News:\n" + "\n".join(news_list)
    except Exception as e:
        print("News error:", e)
        return "News data not available."

def get_time():
    return datetime.now().strftime("Time: %H:%M:%S")

# ---------------- Hand Detection Setup ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

def count_fingers(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if hand_landmarks.landmark[tips_ids[id]].y < hand_landmarks.landmark[tips_ids[id]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

# Calculate distance between thumb and index
def thumb_index_distance(hand_landmarks, frame_width, frame_height):
    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]
    x1, y1 = int(thumb.x * frame_width), int(thumb.y * frame_height)
    x2, y2 = int(index.x * frame_width), int(index.y * frame_height)
    distance = math.hypot(x2 - x1, y2 - y1)
    return distance

def detect_hand(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    city_name = None
    scale_factor = 1.0  # default
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            fingers_count = count_fingers(handLms)
            city_name = finger_city_map.get(fingers_count, None)
            # Zoom control based on thumb-index distance
            distance = thumb_index_distance(handLms, frame.shape[1], frame.shape[0])
            scale_factor = min(max(distance / 100, 0.5), 2.0)  # scale between 0.5x to 2x
    return frame, city_name, scale_factor

# ---------------- GUI Update ----------------
def search_location():
    location = location_entry.get().strip()
    if location:
        weather_label.config(text=get_weather(location))
        news_label.config(text=get_news(location))

def update_time_and_reminders():
    time_label.config(text=get_time())
    reminders_label.config(text="Reminders:\n" + "\n".join(REMINDERS))
    root.after(1000, update_time_and_reminders)

# --------------- OpenCV Camera ------------------
cap = cv2.VideoCapture(0)

def show_frame():
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame, city, scale = detect_hand(frame)
        if city:
            location_entry.delete(0, tk.END)
            location_entry.insert(0, city)
            search_location()
        # Resize frame based on scale_factor
        frame = cv2.resize(frame, None, fx=scale, fy=scale)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(10, show_frame)

# ----------------- GUI Setup -----------------
root = tk.Tk()
root.title("Weather & News Dashboard")
root.geometry("700x600")
root.config(bg="#d0f0fd")
root.attributes("-topmost", True)

location_frame = tk.Frame(root, bg="#d0f0fd")
location_frame.pack(pady=5)
tk.Label(location_frame, text="Enter Country/State/City:", bg="#d0f0fd", font=("Helvetica", 12)).pack(side="left")
location_entry = tk.Entry(location_frame, font=("Helvetica", 12), width=25)
location_entry.pack(side="left", padx=5)
location_entry.insert(0, "Delhi")
tk.Button(location_frame, text="Search", font=("Helvetica", 12), command=search_location).pack(side="left", padx=5)

time_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#d0f0fd")
time_label.pack(pady=5)
weather_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#d0f0fd", justify="left")
weather_label.pack(pady=5)
news_label = tk.Label(root, text="", font=("Helvetica", 10), bg="#d0f0fd", justify="left")
news_label.pack(pady=5)
reminders_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#d0f0fd", justify="left")
reminders_label.pack(pady=5)

camera_label = tk.Label(root)
camera_label.pack(pady=10)

update_time_and_reminders()
show_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
