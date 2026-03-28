import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import numpy as np
import sounddevice as sd
import librosa
import os
from PIL import Image
import RPi.GPIO as GPIO

TARGET_SR = 16000
MIC_SR = 44100
DURATION = 4
MIC_SAMPLES = MIC_SR * DURATION
BASE_DIR = os.path.expanduser("~/diploma")

LED_SHAHED = 17
LED_ROCKET = 27
LED_NOISE = 22

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup([LED_SHAHED, LED_ROCKET, LED_NOISE], GPIO.OUT, initial=GPIO.LOW)

def get_mic_id():
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if "USB PnP" in dev['name'] and dev['max_input_channels'] > 0:
            return i
    return None


MIC_ID = get_mic_id()
device = torch.device('cpu')

resnet_base = models.resnet18(weights=None)
resnet = nn.Sequential(*list(resnet_base.children())[:-1])

try:
    resnet.load_state_dict(torch.load(f"{BASE_DIR}/resnet_feature_extractor.pth", map_location=device))
    print("✓ ResNet готова до витягування ознак")
except Exception as e:
    print(f"! Помилка ваг ResNet: {e}")

resnet.eval()

svm = joblib.load(f"{BASE_DIR}/svm_resnet_features.pkl")
encoder = joblib.load(f"{BASE_DIR}/label_encoder.pkl")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def update_leds(label):
    GPIO.output([LED_SHAHED, LED_ROCKET, LED_NOISE], GPIO.LOW)
    l = label.lower()
    if "shahed" in l:
        GPIO.output(LED_SHAHED, GPIO.HIGH)
    elif "rocket" in l or "missile" in l:
        GPIO.output(LED_ROCKET, GPIO.HIGH)
    else:
        GPIO.output(LED_NOISE, GPIO.HIGH)


def callback(indata, frames, time, status):
    if status:
        print(f"Статус: {status}")
    try:
        audio = indata.flatten()
        audio_resampled = librosa.resample(audio, orig_sr=MIC_SR, target_sr=TARGET_SR)
        max_val = np.max(np.abs(audio_resampled)) + 1e-9
        mel = librosa.feature.melspectrogram(y=audio_resampled / max_val, sr=TARGET_SR, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9) * 255
        img = Image.fromarray(mel_norm.astype(np.uint8), mode='L')
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            features = resnet(tensor).view(1, -1).numpy()
            prediction = svm.predict(features)
            label = encoder.inverse_transform(prediction)[0]
        print(f"\n[ВИЯВЛЕНО]: {label.upper()}")
        update_leds(label)

    except Exception as e:
        print(f"Помилка в циклі: {e}")

if MIC_ID is None:
    print("! КРИТИЧНО: USB-мікрофон не знайдено.")
else:
    print(f"Слухаю через пристрій №{MIC_ID}. Для виходу натисніть Ctrl+C.")
    try:
        with sd.InputStream(device=MIC_ID, channels=1, samplerate=MIC_SR,
                            callback=callback, blocksize=MIC_SAMPLES):
            while True:
                sd.sleep(1000)
    except KeyboardInterrupt:
        print("\nПрограму зупинено користувачем.")
    finally:
        GPIO.cleanup()