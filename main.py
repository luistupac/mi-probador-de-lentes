from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import cv2
import mediapipe as mp
import numpy as np

# 1. Creamos nuestra "cocina virtual" llamada app
app = FastAPI()

# 2. El "Permiso de Entrada" (CORS)
# Esto le dice a la cocina: "Oye, deja que la página de Lovable te hable y te pida cosas"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite que cualquier página (como Lovable) le pida comida
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializamos el robot que detecta las caras
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 3. La "Ventana de Pedidos" (/probar-lentes)
# Aquí es donde el mesero de Lovable va a dejar las fotos
@app.post("/probar-lentes")
async def probar_lentes(
    foto_rostro: UploadFile = File(...), 
    foto_lentes: UploadFile = File(...)
):
    # --- PASO A: El chef recibe los ingredientes ---
    # Leemos la foto de la cara y la de los lentes desde el internet
    bytes_rostro = await foto_rostro.read()
    bytes_lentes = await foto_lentes.read()

    # Convertimos los archivos en imágenes que la computadora puede entender
    nparr_rostro = np.frombuffer(bytes_rostro, np.uint8)
    rostro = cv2.imdecode(nparr_rostro, cv2.IMREAD_COLOR)

    nparr_lentes = np.frombuffer(bytes_lentes, np.uint8)
    lentes = cv2.imdecode(nparr_lentes, cv2.IMREAD_UNCHANGED) # Mantiene la transparencia

    h_rostro, w_rostro, _ = rostro.shape
    rgb_rostro = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
    
    # --- PASO B: El robot busca los ojos ---
    resultados = face_mesh.process(rgb_rostro)

    # Si la persona se tapó la cara o no hay nadie, devolvemos la foto original sin lentes
    if not resultados.multi_face_landmarks:
        _, im_png = cv2.imencode('.png', rostro)
        return Response(content=im_png.tobytes(), media_type="image/png")

    # El robot encuentra los ojos usando "puntos mágicos" (130 y 359)
    landmarks = resultados.multi_face_landmarks[0].landmark
    ojo_izq = np.array([landmarks[130].x * w_rostro, landmarks[130].y * h_rostro])
    ojo_der = np.array([landmarks[359].x * w_rostro, landmarks[359].y * h_rostro])

    # --- PASO C: El chef acomoda los lentes ---
    # Calculamos el tamaño justo para que los lentes no se vean gigantes ni diminutos
    centro_ojos = ((ojo_izq + ojo_der) / 2).astype(int)
    distancia_ojos = np.linalg.norm(ojo_der - ojo_izq)
    
    ancho_lentes = int(distancia_ojos * 2.1)
    escala = ancho_lentes / lentes.shape[1]
    alto_lentes = int(lentes.shape[0] * escala)

    lentes_resiz = cv2.resize(lentes, (ancho_lentes, alto_lentes))

    # Si la cabeza está un poco chueca, giramos los lentes para que queden alineados
    d_y = ojo_der[1] - ojo_izq[1]
    d_x = ojo_der[0] - ojo_izq[0]
    angulo = np.degrees(np.arctan2(d_y, d_x))

    M = cv2.getRotationMatrix2D((ancho_lentes // 2, alto_lentes // 2), -angulo, 1.0)
    lentes_rotados = cv2.warpAffine(lentes_resiz, M, (ancho_lentes, alto_lentes), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

    # --- PASO D: El pegado mágico ---
    # Calculamos la posición exacta sobre la nariz
    x_offset = int(centro_ojos[0] - (ancho_lentes // 2))
    y_offset = int(centro_ojos[1] - (alto_lentes // 2))

    # Evitamos errores si el recorte se sale un milímetro de la foto
    x1, x2 = max(0, x_offset), min(w_rostro, x_offset + ancho_lentes)
    y1, y2 = max(0, y_offset), min(h_rostro, y_offset + alto_lentes)

    lx1, lx2 = max(0, -x_offset), min(ancho_lentes, w_rostro - x_offset)
    ly1, ly2 = max(0, -y_offset), min(alto_lentes, h_rostro - y_offset)

    # Superponemos los lentes usando su transparencia para que se vea real
    if x1 < x2 and y1 < y2:
        alpha = lentes_rotados[ly1:ly2, lx1:lx2, 3] / 255.0
        for c in range(0, 3):
            color = lentes_rotados[ly1:ly2, lx1:lx2, c]
            roi = rostro[y1:y2, x1:x2, c]
            rostro[y1:y2, x1:x2, c] = ((1.0 - alpha) * roi + alpha * color).astype(np.uint8)

    # --- PASO E: ¡Pedido listo! ---
    # Convertimos el resultado final en una imagen limpia y se la enviamos de vuelta a Lovable
    _, im_png = cv2.imencode('.png', rostro)
    return Response(content=im_png.tobytes(), media_type="image/png")
