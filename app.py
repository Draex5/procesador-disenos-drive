import io
import os
import re
import uuid
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURACIÓN
# ============================================================

APP_VERSION = "1.8.0"

BASE_DIR = Path(__file__).resolve().parent

# Ajusta estos nombres si tus archivos se llaman diferente en Render
TEMPLATE_PATH = BASE_DIR / "plantilla.pdf"
WATERMARK_PATH = BASE_DIR / "marca_agua.pdf"

# Si en tu proyecto real los archivos tienen estos nombres:
if not TEMPLATE_PATH.exists():
    alt_template = BASE_DIR / "Medidas para Exportar Diseños Drive.pdf"
    if alt_template.exists():
        TEMPLATE_PATH = alt_template

if not WATERMARK_PATH.exists():
    alt_watermark = BASE_DIR / "Marca de Agua.pdf"
    if alt_watermark.exists():
        WATERMARK_PATH = alt_watermark


OUTPUT_DIR = Path("/tmp/processed_designs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DPI = 300
MAX_LONG_SIDE = 12000

# 0 = invisible, 255 = completamente opaca
WATERMARK_OPACITY = 180

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://procesador-disenos-drive.onrender.com"
).rstrip("/")

MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "").strip()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Procesador de Diseños Drive",
    version=APP_VERSION
)


# ============================================================
# MODELOS
# ============================================================

class DriveRequest(BaseModel):
    drive_url: str


class GPTDriveRequest(BaseModel):
    drive_url: str
    initials: str = Field(..., min_length=1, max_length=10)
    description: str = Field(..., min_length=1, max_length=180)


# ============================================================
# UTILIDADES DE TEXTO / NOMBRE
# ============================================================

def sanitize_initials(value: str) -> str:
    """
    Limpia las iniciales.
    Ej:
        ' dd ' -> 'DD'
        'D.D.' -> 'DD'
    """
    value = value.strip().upper()

    # Dejamos únicamente letras y números
    value = re.sub(r"[^A-Z0-9ÁÉÍÓÚÜÑ]", "", value)

    if not value:
        raise ValueError("Las iniciales no son válidas.")

    return value[:10]


def sanitize_description(value: str) -> str:
    """
    Limpia la descripción para que pueda formar parte del nombre de archivo.
    Conserva espacios, acentos, guiones y texto normal.
    """
    value = value.strip()

    # Caracteres prohibidos en nombres de archivo
    value = re.sub(r'[\\/:*?"<>|]', "-", value)

    # Espacios repetidos
    value = re.sub(r"\s+", " ", value)

    # Evita puntos o espacios al final
    value = value.strip(" .")

    if not value:
        raise ValueError("La descripción no es válida.")

    return value[:180]


# ============================================================
# GOOGLE DRIVE
# ============================================================

def extract_drive_file_id(url: str) -> str:
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
        r"/d/([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    raise ValueError("No se pudo obtener el ID del archivo de Google Drive.")


async def download_drive_pdf(drive_url: str) -> bytes:
    file_id = extract_drive_file_id(drive_url)

    download_url = (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&confirm=t"
    )

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=120.0
    ) as client:
        response = await client.get(download_url)

    if response.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo descargar el PDF de Google Drive. "
                   f"HTTP {response.status_code}"
        )

    content = response.content

    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo descargado no parece ser un PDF válido. "
                "Verifica que el archivo de Drive tenga acceso mediante enlace."
            )
        )

    return content


# ============================================================
# PDF / IMAGEN
# ============================================================

def render_pdf_page(
    pdf_source,
    page_number: int = 0,
    dpi: Optional[int] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Image.Image:

    if isinstance(pdf_source, (str, Path)):
        doc = fitz.open(str(pdf_source))
    else:
        doc = fitz.open(stream=pdf_source, filetype="pdf")

    try:
        page = doc[page_number]

        if target_width and target_height:
            rect = page.rect

            scale_x = target_width / rect.width
            scale_y = target_height / rect.height

            matrix = fitz.Matrix(scale_x, scale_y)

        else:
            render_dpi = dpi or OUTPUT_DPI
            scale = render_dpi / 72.0
            matrix = fitz.Matrix(scale, scale)

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=True
        )

        image = Image.frombytes(
            "RGBA",
            [pix.width, pix.height],
            pix.samples
        )

        return image

    finally:
        doc.close()


def limit_dimensions(width: int, height: int):
    longest = max(width, height)

    if longest <= MAX_LONG_SIDE:
        return width, height

    factor = MAX_LONG_SIDE / longest

    return (
        max(1, round(width * factor)),
        max(1, round(height * factor))
    )


def is_green_pixel(r: int, g: int, b: int) -> bool:
    """
    Detecta aproximadamente las líneas verdes de la plantilla.
    """
    return (
        g > 100
        and g > r * 1.15
        and g > b * 1.15
    )


def detect_green_rectangles(template: Image.Image):
    """
    Detecta los dos rectángulos verdes:
    - exterior = Display final
    - interior = área máxima del diseño
    """

    img = template.convert("RGB")
    width, height = img.size

    xs = []
    ys = []

    for y in range(height):
        for x in range(width):
            r, g, b = img.getpixel((x, y))

            if is_green_pixel(r, g, b):
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        raise RuntimeError(
            "No se pudieron detectar las líneas verdes de la plantilla."
        )

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    # Para encontrar el rectángulo interior, descartamos una zona próxima
    # al borde del rectángulo exterior.
    margin_x = max(5, int(width * 0.03))
    margin_y = max(5, int(height * 0.03))

    inner_xs = []
    inner_ys = []

    for y in range(height):
        for x in range(width):
            if (
                x <= min_x + margin_x
                or x >= max_x - margin_x
                or y <= min_y + margin_y
                or y >= max_y - margin_y
            ):
                continue

            r, g, b = img.getpixel((x, y))

            if is_green_pixel(r, g, b):
                inner_xs.append(x)
                inner_ys.append(y)

    if not inner_xs or not inner_ys:
        raise RuntimeError(
            "Se detectó el rectángulo exterior, "
            "pero no el área interior de diseño."
        )

    outer_rect = (
        min_x,
        min_y,
        max_x + 1,
        max_y + 1
    )

    inner_rect = (
        min(inner_xs),
        min(inner_ys),
        max(inner_xs) + 1,
        max(inner_ys) + 1
    )

    return outer_rect, inner_rect


def fit_inside(
    source_width: int,
    source_height: int,
    max_width: int,
    max_height: int
):
    """
    Ajusta conservando siempre la proporción original.
    """
    scale = min(
        max_width / source_width,
        max_height / source_height
    )

    return (
        max(1, round(source_width * scale)),
        max(1, round(source_height * scale))
    )


def apply_watermark(
    base_image: Image.Image,
    watermark_path: Path
) -> Image.Image:

    width, height = base_image.size

    watermark = render_pdf_page(
        watermark_path,
        target_width=width,
        target_height=height
    ).convert("RGBA")

    # Ajustamos opacidad global sin perder el alfa original
    alpha = watermark.getchannel("A")

    alpha = alpha.point(
        lambda p: int(p * (WATERMARK_OPACITY / 255.0))
    )

    watermark.putalpha(alpha)

    return Image.alpha_composite(
        base_image.convert("RGBA"),
        watermark
    )


# ============================================================
# PROCESAMIENTO PRINCIPAL
# ============================================================

def process_pdf(pdf_bytes: bytes) -> bytes:

    if not TEMPLATE_PATH.exists():
        raise RuntimeError(
            f"No existe la plantilla: {TEMPLATE_PATH}"
        )

    if not WATERMARK_PATH.exists():
        raise RuntimeError(
            f"No existe la marca de agua: {WATERMARK_PATH}"
        )

    # --------------------------------------------------------
    # 1. Renderizamos plantilla para detectar dimensiones
    # --------------------------------------------------------

    template_preview = render_pdf_page(
        TEMPLATE_PATH,
        dpi=150
    )

    outer_rect, inner_rect = detect_green_rectangles(
        template_preview
    )

    preview_width, preview_height = template_preview.size

    outer_left, outer_top, outer_right, outer_bottom = outer_rect
    inner_left, inner_top, inner_right, inner_bottom = inner_rect

    outer_width_preview = outer_right - outer_left
    outer_height_preview = outer_bottom - outer_top

    # --------------------------------------------------------
    # 2. Calculamos tamaño final a 300 DPI
    # --------------------------------------------------------

    scale_factor = OUTPUT_DPI / 150.0

    final_width = round(
        outer_width_preview * scale_factor
    )

    final_height = round(
        outer_height_preview * scale_factor
    )

    final_width, final_height = limit_dimensions(
        final_width,
        final_height
    )

    # Escala real luego del límite de tamaño
    scale_x = final_width / outer_width_preview
    scale_y = final_height / outer_height_preview

    inner_left_final = round(
        (inner_left - outer_left) * scale_x
    )

    inner_top_final = round(
        (inner_top - outer_top) * scale_y
    )

    inner_right_final = round(
        (inner_right - outer_left) * scale_x
    )

    inner_bottom_final = round(
        (inner_bottom - outer_top) * scale_y
    )

    inner_width_final = (
        inner_right_final - inner_left_final
    )

    inner_height_final = (
        inner_bottom_final - inner_top_final
    )

    # --------------------------------------------------------
    # 3. Obtenemos proporción ORIGINAL del PDF del cliente
    # --------------------------------------------------------

    design_doc = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    try:
        if design_doc.page_count < 1:
            raise RuntimeError(
                "El PDF del diseño no contiene páginas."
            )

        design_page = design_doc[0]

        source_width = design_page.rect.width
        source_height = design_page.rect.height

    finally:
        design_doc.close()

    target_width, target_height = fit_inside(
        source_width,
        source_height,
        inner_width_final,
        inner_height_final
    )

    # --------------------------------------------------------
    # 4. Render directo desde PDF al tamaño requerido
    # --------------------------------------------------------

    design_image = render_pdf_page(
        pdf_bytes,
        target_width=target_width,
        target_height=target_height
    ).convert("RGBA")

    # --------------------------------------------------------
    # 5. Canvas blanco = Display final
    # --------------------------------------------------------

    canvas = Image.new(
        "RGBA",
        (final_width, final_height),
        (255, 255, 255, 255)
    )

    design_x = (
        inner_left_final
        + (inner_width_final - target_width) // 2
    )

    design_y = (
        inner_top_final
        + (inner_height_final - target_height) // 2
    )

    canvas.alpha_composite(
        design_image,
        (design_x, design_y)
    )

    # --------------------------------------------------------
    # 6. Marca de agua
    # --------------------------------------------------------

    canvas = apply_watermark(
        canvas,
        WATERMARK_PATH
    )

    # --------------------------------------------------------
    # 7. PNG
    # --------------------------------------------------------

    output = io.BytesIO()

    canvas.convert("RGB").save(
        output,
        format="PNG",
        optimize=True,
        compress_level=6,
        dpi=(OUTPUT_DPI, OUTPUT_DPI)
    )

    return output.getvalue()


# ============================================================
# MAKE
# ============================================================

async def send_png_to_make(
    png_bytes: bytes,
    filename: str,
    source_drive_url: str,
    initials: str,
    description: str
):
    if not MAKE_WEBHOOK_URL:
        raise HTTPException(
            status_code=500,
            detail="MAKE_WEBHOOK_URL no está configurada en Render."
        )

    files = {
        "file": (
            filename,
            png_bytes,
            "image/png"
        )
    }

    data = {
        "filename": filename,
        "content_type": "image/png",
        "source_drive_url": source_drive_url,

        # NUEVOS CAMPOS
        "initials": initials,
        "description": description,
    }

    try:
        async with httpx.AsyncClient(
            timeout=120.0
        ) as client:

            response = await client.post(
                MAKE_WEBHOOK_URL,
                files=files,
                data=data
            )

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo conectar con Make: {exc}"
        )

    if response.status_code < 200 or response.status_code >= 300:
        raise HTTPException(
            status_code=502,
            detail=(
                f"Make rechazó el archivo. "
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )
        )

    return response.status_code


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "success": True,
        "service": "Procesador de Diseños Drive",
        "version": APP_VERSION
    }


@app.get("/health")
async def health():
    return {
        "success": True,
        "status": "ok",
        "version": APP_VERSION
    }


@app.post("/process")
async def process(request: DriveRequest):
    """
    Endpoint que devuelve directamente el PNG.
    """

    try:
        pdf_bytes = await download_drive_pdf(
            request.drive_url
        )

        png_bytes = process_pdf(pdf_bytes)

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition":
                    'inline; filename="diseno_procesado.png"'
            }
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.post("/process-gpt")
async def process_gpt(request: GPTDriveRequest):
    """
    Endpoint usado por el GPT.

    Recibe:
        drive_url
        initials
        description

    El consecutivo NO se genera aquí.
    Make lo obtendrá desde Google Sheets.
    """

    try:
        initials = sanitize_initials(
            request.initials
        )

        description = sanitize_description(
            request.description
        )

        # ----------------------------------------------------
        # Descargar PDF
        # ----------------------------------------------------

        pdf_bytes = await download_drive_pdf(
            request.drive_url
        )

        # ----------------------------------------------------
        # Procesar diseño
        # ----------------------------------------------------

        png_bytes = process_pdf(
            pdf_bytes
        )

        # ----------------------------------------------------
        # Nombre TEMPORAL
        #
        # Make cambiará el nombre definitivo después de
        # obtener el consecutivo de Google Sheets.
        # ----------------------------------------------------

        temp_id = uuid.uuid4().hex

        temp_filename = (
            f"diseno_temporal_{temp_id[:8]}.png"
        )

        temp_path = (
            OUTPUT_DIR / f"{temp_id}.png"
        )

        temp_path.write_bytes(
            png_bytes
        )

        download_url = (
            f"{PUBLIC_BASE_URL}/download/{temp_id}"
        )

        # ----------------------------------------------------
        # Enviar a Make
        # ----------------------------------------------------

        make_status = await send_png_to_make(
            png_bytes=png_bytes,
            filename=temp_filename,
            source_drive_url=request.drive_url,
            initials=initials,
            description=description
        )

        return {
            "success": True,

            "file_id": temp_id,

            # Este todavía NO es el nombre definitivo
            "filename": temp_filename,

            "content_type": "image/png",

            "download_url": download_url,

            "initials": initials,
            "description": description,

            "make_sent": True,
            "make_status": make_status,

            "message": (
                "PNG procesado y enviado a Make. "
                "Make asignará el consecutivo y el nombre final."
            )
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )


@app.get("/download/{file_id}")
async def download(file_id: str):

    if not re.fullmatch(
        r"[a-fA-F0-9]{32}",
        file_id
    ):
        raise HTTPException(
            status_code=400,
            detail="ID de archivo inválido."
        )

    path = OUTPUT_DIR / f"{file_id}.png"

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Archivo no encontrado."
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename="diseno_procesado.png"
    )


@app.get("/test-make")
async def test_make():

    if not MAKE_WEBHOOK_URL:
        raise HTTPException(
            status_code=500,
            detail="MAKE_WEBHOOK_URL no configurada."
        )

    test_image = Image.new(
        "RGB",
        (500, 500),
        "white"
    )

    output = io.BytesIO()

    test_image.save(
        output,
        format="PNG"
    )

    png_bytes = output.getvalue()

    status = await send_png_to_make(
        png_bytes=png_bytes,
        filename="prueba_make.png",
        source_drive_url="TEST",
        initials="DD",
        description="Prueba Make"
    )

    return {
        "success": True,
        "make_status": status
    }
