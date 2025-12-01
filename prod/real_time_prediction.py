import threading
import queue
import cv2
import numpy as np
from src.utils.video import camera_reader
import mediapipe as mp
from src import Landmarks, nn_parser
from src.utils.tts import speak
import pandas as pd
import torch
from collections import deque
import time
from models.SimpleDetector import LitSimpleSignDetector

# =========================
# Setup de datos y modelo
# =========================

# Meta de signos (si la usás en otro lado)
signs = pd.read_csv("../data/LSA64/meta.csv")

# Landmarks compartidos
lm = Landmarks()

# Threading
capture_finished = threading.Event()   # Indica que se terminó la captura
tts_queue = queue.Queue(maxsize=3)             # Cola para TTS
tts_shutdown = threading.Event()      # Señal de apagado para TTS
window_queue = queue.Queue() # Cola de ventanas para el modelo

# Reconocimiento de handedness
#litModel = LitHandDetector.load_from_checkpoint(
#    "../models/TwoStageRNN/hand_detector/best_params.ckpt"
#)
#model = litModel.model
#model.eval()
# Reconocimiento de seña
litModel = LitSimpleSignDetector.load_from_checkpoint("../models/SimpleDetector/best_params.ckpt")
model = litModel.model
model.eval()

# =========================
# Configuración sliding window
# =========================

WINDOW_SIZE_SECONDS = 2.0
STRIDE_SECONDS = 0.1
FPS = 12

WINDOW_SIZE_FRAMES = int(WINDOW_SIZE_SECONDS * FPS)   # p.ej. 24
STRIDE_FRAMES = int(STRIDE_SECONDS * FPS)            # p.ej. 1–2


# =========================
# Hilos
# =========================

def tts_thread():
    """Thread dedicado para text-to-speech."""
    while not tts_shutdown.is_set() or not tts_queue.empty():
        try:
            text = tts_queue.get(timeout=0.1)
            speak(text)
            tts_queue.task_done()
        except queue.Empty:
            continue


def generator_thread():
    """Thread para captura de video y extracción de landmarks."""
    with mp.solutions.holistic.Holistic(
        model_complexity=2,
        static_image_mode=False
    ) as holistic:
        for frame in camera_reader(fps=FPS):
            # Procesar frame con Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hol_res = holistic.process(rgb)

            # Guardar landmarks en el buffer compartido
            lm.add(
                hol_res.pose_landmarks,
                hol_res.left_hand_landmarks,
                hol_res.right_hand_landmarks,
            )

            # Mostrar frame
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Señal de que ya no hay más frames
    capture_finished.set()
    cv2.destroyAllWindows()


def parser_thread():
    """
    Thread para procesar landmarks y construir ventanas.
    NO corre el modelo, solo arma la sliding window y la pone en window_queue.
    """
    print("Parser thread iniciado")

    frame_buffer = deque(maxlen=WINDOW_SIZE_FRAMES)
    frames_seen = 0

    # lm.get_landmarks(continuous=True) debería producir
    # (pose, left, right) continuamente mientras haya datos.
    for pose, left, right in lm.get_landmarks(continuous=True):
        # Parsear landmarks en vector de features
        x = nn_parser(pose, left, right)
        frame_buffer.append(x)
        frames_seen += 1

        # Esperar a tener suficientes frames para la primera ventana
        if frames_seen < WINDOW_SIZE_FRAMES:
            continue

        # Procesar cada STRIDE_FRAMES frames
        if (frames_seen - WINDOW_SIZE_FRAMES) % STRIDE_FRAMES != 0:
            continue

        # Construir ventana (solo la última)
        window_np = np.array(frame_buffer, dtype=np.float32)  # (window_size, features)
        last_window = torch.from_numpy(window_np).unsqueeze(0)  # (1, window_size, features)

        # Enviar ventana a la cola del modelo
        try:
            window_queue.put_nowait((last_window, WINDOW_SIZE_FRAMES))
        except queue.Full:
            # Si el modelo no da abasto, tiramos esta ventana y seguimos
            pass

        # Si ya se terminó la captura y no esperamos más landmarks, podemos ir saliendo.
        if capture_finished.is_set():
            # Le damos un pequeño tiempo a que se drenen landmarks remanentes (depende de implementación de lm)
            time.sleep(0.1)
            break

    print("Parser thread finalizado")


def model_thread():
    """
    Thread para correr el modelo sobre las ventanas que llegan por window_queue
    y mandar el resultado al TTS.
    """
    print("Model thread iniciado")
    last_prediction = None

    with torch.no_grad():
        while not (capture_finished.is_set() and window_queue.empty()):
            try:
                last_window, win_len = window_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            lengths = torch.tensor([win_len])

            # Forward pass de reconocimiento de señas
            y_pred = model.forward(last_window, lengths)

            # CORRECCIÓN: torch.max devuelve (valores, índices)
            # Necesitas extraer el índice correctamente
            probs = torch.softmax(y_pred, dim=1)  # dim=1 para batch
            max_prob, max_idx = torch.max(probs, dim=1)

            # Convertir a escalar
            max_idx_scalar = max_idx.item()
            max_prob_scalar = max_prob.item()

            pred = signs.iloc[max_idx_scalar]["Name"]
            print(f"Predicción: {pred} (confianza: {max_prob_scalar:.3f})")
            no_sign_prob = probs[0][64]

            # Evitar repetir el mismo texto
            if max_idx_scalar != last_prediction and no_sign_prob < 0.3:
                last_prediction = max_idx_scalar
                tts_queue.put(pred)

            window_queue.task_done()

    print("Model thread finalizado")


# =========================
# Main
# =========================

def main():
    """Función principal."""
    # Thread de TTS (daemon, así no bloquea salida forzada)
    tts_worker = threading.Thread(target=tts_thread, daemon=True)
    tts_worker.start()

    # Threads de captura, parser y modelo
    thread_gen = threading.Thread(target=generator_thread)
    thread_parser = threading.Thread(target=parser_thread)
    thread_model = threading.Thread(target=model_thread)

    thread_gen.start()
    thread_parser.start()
    thread_model.start()

    # Esperar a que termine la captura
    thread_gen.join()
    # En este punto capture_finished debería estar seteado dentro de generator_thread

    # Esperar a que el parser termine de consumir landmarks
    thread_parser.join()

    # Esperar a que el modelo procese todas las ventanas pendientes
    window_queue.join()
    # Asegurarnos de que el modelo vea la señal de finalización
    capture_finished.set()
    thread_model.join()

    # Vaciar cola de TTS
    tts_queue.join()
    tts_shutdown.set()
    tts_worker.join(timeout=2)

    print("Programa finalizado")


if __name__ == "__main__":
    main()
