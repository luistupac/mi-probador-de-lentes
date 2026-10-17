from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import mediapipe as mp
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialización estándar y segura para servidores
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

@app.post("/probar-lentes")
async def probar_lentes(
    foto_rostro: UploadFile = File(...), 
    foto_lentes: UploadFile = File(...)
):
    bytes_rostro = await foto_rostro.read()
    bytes_lentes = await foto_lentes.read()

    nparr_rostro = np.frombuffer(bytes_rostro, np.uint8)
    rostro = cv2.imdecode(nparr_rostro, cv2.IMREAD_COLOR)

    nparr_lentes = np.frombuffer(bytes_lentes, np.uint8)
    lentes = cv2.imdecode(nparr_lentes, cv2.IMREAD_UNCHANGED)

    if rostro is None or lentes is None:
        return Response(content=b"Error al leer las imagenes", media_type="text/plain", status_code=400)

    h_rostro, w_rostro, _ = rostro.shape
    rgb_rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
    
    resultados = face_mesh.process(rgb_rostro)

    if not resultados.multi_face_landmarks:
        _, im_png = cv2.imencode('.png', rostro)
        return Response(content=im_png.tobytes(), media_type="image/png")

    landmarks = resultados.multi_face_landmarks[0].landmark
    
    # Puntos fijos para las esquinas de los ojos (130 y 359)
    ojo_izq = np.array([landmarks[130].x * w_rostro, landmarks[130].y * h_rostro])
    ojo_der = np.array([landmarks[359].x * w_rostro, landmarks[359].y * h_rostro])

    centro_ojos = ((ojo_izq + ojo_der) / 2).astype(int)
    distancia_ojos = np.linalg.norm(ojo_der - ojo_izq)
    
    ancho_lentes = int(distancia_ojos * 2.1)
    escala = ancho_lentes / lentes.shape[1]
    alto_lentes = int(lentes.shape[0] * escala)

    lentes_resiz = cv2.resize(lentes, (ancho_lentes, alto_lentes))

    d_y = ojo_der[1] - ojo_izq[1]
    d_x = ojo_der[0] - ojo_izq[0]
    angulo = np.degrees(np.arctan2(d_y, d_x))

    M = cv2.getRotationMatrix2D((ancho_lentes // 2, alto_lentes // 2), -angulo, 1.0)
    lentes_rotados = cv2.warpAffine(lentes_resiz, M, (ancho_lentes, alto_lentes), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    x_offset = int(centro_ojos[0] - (ancho_lentes // 2))
    y_offset = int(centro_ojos[1] - (alto_lentes // 2))

    x1, x2 = max(0, x_offset), min(w_rostro, x_offset + ancho_lentes)
    y1, y2 = max(0, y_offset), min(h_rostro, y_offset + alto_lentes)

    lx1, lx2 = max(0, -x_offset), min(ancho_lentes, w_rostro - x_offset)
    y_top, y_bot = max(0, -y_offset), min(alto_lentes, h_rostro - y_offset)

    if x1 < x2 and y1 < y2:
        alpha = lentes_rotados[y_top:y_bot, lx1:lx2, 3] / 255.0
        for c in range(0, 3):
            color = lentes_rotados[y_top:y_bot, lx1:lx2, c]
            roi = rostro[y1:y2, x1:x2, c]
            rostro[y1:y2, x1:x2, c] = ((1.0 - alpha) * roi + alpha * color).astype(np.uint8)

    _, im_png = cv2.imencode('.png', rostro)
    return Response(content=im_png.tobytes(), media_type="image/png")
