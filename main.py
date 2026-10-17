from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image
import io

app = FastAPI()

# Permiso de entrada para que Lovable se pueda conectar sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/probar-lentes")
async def probar_lentes(
    foto_rostro: UploadFile = File(...), 
    foto_lentes: UploadFile = File(...)
):
    try:
        # 1. Leer los archivos que vienen de internet
        bytes_rostro = await foto_rostro.read()
        bytes_lentes = await foto_lentes.read()

        # 2. Abrir las imágenes con el motor ultra-compatible Pillow
        rostro = Image.open(io.BytesIO(bytes_rostro)).convert("RGBA")
        lentes = Image.open(io.BytesIO(bytes_lentes)).convert("RGBA")

        w_rostro, h_rostro = rostro.size

        # 3. Calcular tamaño automático para los lentes (un ancho estándar del 45% de la cara)
        nuevo_ancho = int(w_rostro * 0.45)
        proporcion = nuevo_ancho / float(lentes.size[0])
        nuevo_alto = int(float(lentes.size[1]) * float(proporcion))
        
        lentes_redimensionados = lentes.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

        # 4. Calcular el centro exacto de la imagen para posicionarlos sobre la nariz
        # (Esto los coloca centrados horizontalmente y en el tercio superior del rostro)
        x_offset = (w_rostro - nuevo_ancho) // 2
        y_offset = int(h_rostro * 0.35)

        # 5. Fusionar las imágenes usando el canal alfa transparente de los lentes
        rostro.paste(lentes_redimensionados, (x_offset, y_offset), lentes_redimensionados)

        # 6. Convertir el resultado de vuelta a formato PNG para enviarlo a la web
        salida_bytes = io.BytesIO()
        rostro.convert("RGB").save(salida_bytes, format="PNG")
        salida_bytes.seek(0)

        return Response(content=salida_bytes.getvalue(), media_type="image/png")

    except Exception:
        # Si ocurre cualquier error inesperado, devolvemos la foto original a salvo
        return Response(content=bytes_rostro, media_type="image/png")
