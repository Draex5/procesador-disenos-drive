# Desarrollado por Diego Damas + ChatGPT

import io
import re
import os

from pathlib import Path
from uuid import uuid4

import pymupdf
import httpx

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse
from pydantic import BaseModel


app = FastAPI(
    title="Procesador de Diseños",
    version="1.6.0"
)


TEMPLATE_PATH = "plantilla.pdf"
WATERMARK_PATH = "marca_agua.pdf"

# Calidad final
OUTPUT_DPI = 300

# Protección para la memoria de Render
MAX_LONG_SIDE = 12000

# Opacidad de la marca de agua
WATERMARK_OPACITY = 100


# =========================================================
# CONFIGURACIÓN RENDER / MAKE
# =========================================================

OUTPUT_DIR = Path(
    "/tmp/disenos_procesados"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://procesador-disenos-drive.onrender.com"
).rstrip("/")


# Webhook de Make.
#
# RECOMENDADO:
# Configurarlo en Render como variable de entorno:
#
# MAKE_WEBHOOK_URL
#
MAKE_WEBHOOK_URL = os.getenv(
    "MAKE_WEBHOOK_URL",
    ""
).strip()


class DriveRequest(BaseModel):
    drive_url: str


@app.get("/")
def inicio():
    return {
        "status": "ok",
        "message": (
            "API de procesamiento "
            "de diseños funcionando"
        ),
        "version": "1.6.0",
        "make_configured": bool(
            MAKE_WEBHOOK_URL
        )
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "make_configured": bool(
            MAKE_WEBHOOK_URL
        )
    }


# =========================================================
# GOOGLE DRIVE - DESCARGA DEL PDF ORIGINAL
# =========================================================

def get_google_drive_file_id(url):
    """
    Extrae el ID de un archivo desde distintos
    formatos de enlaces de Google Drive.
    """

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            url
        )

        if match:
            return match.group(1)

    raise HTTPException(
        status_code=400,
        detail=(
            "No pude obtener el ID del archivo "
            "de Google Drive."
        )
    )


async def download_pdf_from_drive(
    drive_url
):
    """
    Descarga un PDF público desde Google Drive.
    """

    file_id = get_google_drive_file_id(
        drive_url
    )

    download_url = (
        "https://drive.usercontent.google.com/"
        f"download?id={file_id}"
        "&export=download"
        "&confirm=t"
    )

    try:

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=60.0
        ) as client:

            response = await client.get(
                download_url
            )

    except httpx.RequestError as e:

        raise HTTPException(
            status_code=502,
            detail=(
                "No pude conectar con Google Drive: "
                f"{str(e)}"
            )
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=400,
            detail=(
                "Google Drive no permitió descargar "
                f"el archivo. HTTP "
                f"{response.status_code}"
            )
        )

    pdf_bytes = response.content

    if not pdf_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Google Drive devolvió "
                "un archivo vacío."
            )
        )

    if not pdf_bytes.startswith(
        b"%PDF"
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "El enlace de Google Drive "
                "no devolvió un PDF. "
                "Comprueba que el archivo esté "
                "compartido como "
                "'Cualquier persona con el enlace'."
            )
        )

    return pdf_bytes


# =========================================================
# RENDERIZADO PDF
# =========================================================

def render_pdf_page(
    pdf_bytes=None,
    pdf_path=None,
    alpha=False,
    dpi=OUTPUT_DPI,
    target_width=None,
    target_height=None
):
    """
    Renderiza la primera página de un PDF.

    Si se proporcionan target_width y target_height,
    rasteriza directamente desde el PDF al tamaño
    final requerido.
    """

    if pdf_bytes is not None:

        doc = pymupdf.open(
            stream=pdf_bytes,
            filetype="pdf"
        )

    elif pdf_path is not None:

        doc = pymupdf.open(
            pdf_path
        )

    else:

        raise ValueError(
            "Debe proporcionarse "
            "pdf_bytes o pdf_path."
        )

    try:

        if len(doc) == 0:

            raise ValueError(
                "El PDF no contiene páginas."
            )

        page = doc[0]

        page_width = page.rect.width
        page_height = page.rect.height

        if (
            page_width <= 0
            or page_height <= 0
        ):

            raise ValueError(
                "La página PDF tiene "
                "dimensiones inválidas."
            )

        if (
            target_width is not None
            and target_height is not None
        ):

            scale = min(
                target_width / page_width,
                target_height / page_height
            )

            scale = max(
                scale,
                0.01
            )

            matrix = pymupdf.Matrix(
                scale,
                scale
            )

            pix = page.get_pixmap(
                matrix=matrix,
                alpha=alpha
            )

        else:

            pix = page.get_pixmap(
                dpi=dpi,
                alpha=alpha
            )

        mode = (
            "RGBA"
            if alpha
            else "RGB"
        )

        image = Image.frombytes(
            mode,
            (
                pix.width,
                pix.height
            ),
            pix.samples
        )

        return image

    finally:

        doc.close()


# =========================================================
# CONTROL DE RESOLUCIÓN
# =========================================================

def limit_dimensions(
    width,
    height,
    max_long_side=MAX_LONG_SIDE
):
    """
    Limita las dimensiones máximas
    manteniendo proporción.
    """

    width = int(width)
    height = int(height)

    longest = max(
        width,
        height
    )

    if longest <= max_long_side:

        return (
            width,
            height,
            1.0
        )

    scale = (
        max_long_side
        / longest
    )

    new_width = max(
        1,
        round(
            width * scale
        )
    )

    new_height = max(
        1,
        round(
            height * scale
        )
    )

    return (
        new_width,
        new_height,
        scale
    )


# =========================================================
# DETECCIÓN DE PLANTILLA
# =========================================================

def is_green(pixel):
    """
    Detecta aproximadamente las líneas verdes
    de la plantilla oficial.
    """

    r, g, b = pixel[:3]

    return (
        g > 80
        and g > r * 1.25
        and g > b * 1.15
    )


def group_positions(
    values,
    tolerance=8
):
    """
    Agrupa posiciones consecutivas
    correspondientes a una misma línea.
    """

    if not values:
        return []

    groups = [
        [values[0]]
    ]

    for value in values[1:]:

        if (
            value
            - groups[-1][-1]
            <= tolerance
        ):

            groups[-1].append(
                value
            )

        else:

            groups.append(
                [value]
            )

    return [
        int(
            sum(group)
            / len(group)
        )
        for group
        in groups
    ]


def detect_rectangles(
    template
):
    """
    Detecta:

    outer_rect:
        Display final.

    inner_rect:
        Área máxima del diseño.
    """

    rgb = template.convert(
        "RGB"
    )

    width, height = rgb.size

    vertical_scores = [
        0
    ] * width

    horizontal_scores = [
        0
    ] * height

    pixels = rgb.load()

    step = 2

    for y in range(
        0,
        height,
        step
    ):

        for x in range(
            0,
            width,
            step
        ):

            if is_green(
                pixels[x, y]
            ):

                vertical_scores[x] += 1
                horizontal_scores[y] += 1

    vertical_candidates = [
        i
        for i, score
        in enumerate(
            vertical_scores
        )
        if (
            score
            > height
            * 0.08
            / step
        )
    ]

    horizontal_candidates = [
        i
        for i, score
        in enumerate(
            horizontal_scores
        )
        if (
            score
            > width
            * 0.08
            / step
        )
    ]

    xs = group_positions(
        vertical_candidates
    )

    ys = group_positions(
        horizontal_candidates
    )

    if (
        len(xs) < 4
        or len(ys) < 4
    ):

        raise ValueError(
            "No pude detectar correctamente "
            "los cuadros verdes "
            "de la plantilla."
        )

    xs = sorted(
        xs
    )

    ys = sorted(
        ys
    )

    outer_left = xs[0]
    outer_right = xs[-1]

    outer_top = ys[0]
    outer_bottom = ys[-1]

    inner_left = xs[1]
    inner_right = xs[-2]

    inner_top = ys[1]
    inner_bottom = ys[-2]

    if (
        outer_right <= outer_left
        or outer_bottom <= outer_top
    ):

        raise ValueError(
            "El cuadro exterior detectado "
            "no es válido."
        )

    if (
        inner_right <= inner_left
        or inner_bottom <= inner_top
    ):

        raise ValueError(
            "El cuadro interior detectado "
            "no es válido."
        )

    outer_rect = (
        outer_left,
        outer_top,
        outer_right,
        outer_bottom
    )

    inner_rect = (
        inner_left,
        inner_top,
        inner_right,
        inner_bottom
    )

    return (
        outer_rect,
        inner_rect
    )


# =========================================================
# AJUSTE DEL DISEÑO
# =========================================================

def fit_inside(
    image,
    max_width,
    max_height
):
    """
    Ajusta la imagen al área máxima
    sin recortar ni deformar.
    """

    width, height = image.size

    if (
        width <= 0
        or height <= 0
    ):

        raise ValueError(
            "El diseño tiene "
            "dimensiones inválidas."
        )

    scale = min(
        max_width / width,
        max_height / height
    )

    new_width = max(
        1,
        round(
            width * scale
        )
    )

    new_height = max(
        1,
        round(
            height * scale
        )
    )

    return image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )


# =========================================================
# MARCA DE AGUA
# =========================================================

def make_white_transparent(
    image,
    opacity=100
):
    """
    Convierte el blanco en transparente
    y aplica opacidad a la marca.
    """

    image = image.convert(
        "RGBA"
    )

    pixels = image.load()

    for y in range(
        image.height
    ):

        for x in range(
            image.width
        ):

            r, g, b, a = pixels[x, y]

            if (
                r > 245
                and g > 245
                and b > 245
            ):

                pixels[x, y] = (
                    255,
                    255,
                    255,
                    0
                )

            else:

                pixels[x, y] = (
                    r,
                    g,
                    b,
                    opacity
                )

    return image


def apply_watermark(
    canvas
):
    """
    Aplica la marca oficial
    sobre todo el Display.
    """

    watermark = render_pdf_page(
        pdf_path=WATERMARK_PATH,
        alpha=False,
        target_width=canvas.width,
        target_height=canvas.height
    )

    if watermark.size != canvas.size:

        watermark = watermark.resize(
            canvas.size,
            Image.Resampling.LANCZOS
        )

    watermark = make_white_transparent(
        watermark,
        opacity=WATERMARK_OPACITY
    )

    canvas.alpha_composite(
        watermark,
        (
            0,
            0
        )
    )


# =========================================================
# GENERACIÓN DEL PNG
# =========================================================

def generate_processed_png(
    pdf_bytes
):
    """
    Genera el PNG final.
    """

    template = render_pdf_page(
        pdf_path=TEMPLATE_PATH,
        alpha=False,
        dpi=OUTPUT_DPI
    )

    outer_rect, inner_rect = (
        detect_rectangles(
            template
        )
    )

    (
        outer_left,
        outer_top,
        outer_right,
        outer_bottom
    ) = outer_rect

    (
        inner_left,
        inner_top,
        inner_right,
        inner_bottom
    ) = inner_rect

    original_display_width = (
        outer_right
        - outer_left
    )

    original_display_height = (
        outer_bottom
        - outer_top
    )

    (
        display_width,
        display_height,
        output_scale
    ) = limit_dimensions(
        original_display_width,
        original_display_height
    )

    outer_left_scaled = (
        outer_left
        * output_scale
    )

    outer_top_scaled = (
        outer_top
        * output_scale
    )

    inner_left_scaled = (
        inner_left
        * output_scale
    )

    inner_top_scaled = (
        inner_top
        * output_scale
    )

    inner_right_scaled = (
        inner_right
        * output_scale
    )

    inner_bottom_scaled = (
        inner_bottom
        * output_scale
    )

    relative_inner_left = round(
        inner_left_scaled
        - outer_left_scaled
    )

    relative_inner_top = round(
        inner_top_scaled
        - outer_top_scaled
    )

    relative_inner_right = round(
        inner_right_scaled
        - outer_left_scaled
    )

    relative_inner_bottom = round(
        inner_bottom_scaled
        - outer_top_scaled
    )

    relative_inner_width = (
        relative_inner_right
        - relative_inner_left
    )

    relative_inner_height = (
        relative_inner_bottom
        - relative_inner_top
    )

    if (
        relative_inner_width <= 0
        or relative_inner_height <= 0
    ):

        raise ValueError(
            "El área interior calculada "
            "no es válida."
        )

    design = render_pdf_page(
        pdf_bytes=pdf_bytes,
        alpha=False,
        target_width=relative_inner_width,
        target_height=relative_inner_height
    )

    if (
        design.width > relative_inner_width
        or design.height > relative_inner_height
    ):

        design = fit_inside(
            design,
            relative_inner_width,
            relative_inner_height
        )

    canvas = Image.new(
        "RGBA",
        (
            display_width,
            display_height
        ),
        (
            255,
            255,
            255,
            255
        )
    )

    x = (
        relative_inner_left
        + (
            relative_inner_width
            - design.width
        ) // 2
    )

    y = (
        relative_inner_top
        + (
            relative_inner_height
            - design.height
        ) // 2
    )

    canvas.alpha_composite(
        design.convert(
            "RGBA"
        ),
        (
            x,
            y
        )
    )

    apply_watermark(
        canvas
    )

    output = io.BytesIO()

    final_image = canvas.convert(
        "RGB"
    )

    final_image.save(
        output,
        format="PNG",
        optimize=True,
        compress_level=6,
        dpi=(
            OUTPUT_DPI,
            OUTPUT_DPI
        )
    )

    output.seek(
        0
    )

    return output.getvalue()


# =========================================================
# MAKE.COM
# =========================================================

async def send_png_to_make(
    png_bytes,
    filename,
    source_drive_url=""
):
    """
    Envía el PNG generado a Make mediante
    multipart/form-data.

    Make recibirá:
        file
        filename
        content_type
        source_drive_url

    El campo "file" es el PNG real.
    """

    if not MAKE_WEBHOOK_URL:

        raise HTTPException(
            status_code=500,
            detail=(
                "MAKE_WEBHOOK_URL no está configurado "
                "en las variables de entorno de Render."
            )
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
        "source_drive_url": source_drive_url
    }

    try:

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=120.0
        ) as client:

            response = await client.post(
                MAKE_WEBHOOK_URL,
                data=data,
                files=files
            )

    except httpx.RequestError as e:

        raise HTTPException(
            status_code=502,
            detail=(
                "No pude conectar con Make: "
                f"{str(e)}"
            )
        )

    if (
        response.status_code < 200
        or response.status_code >= 300
    ):

        raise HTTPException(
            status_code=502,
            detail=(
                "Make rechazó el archivo. "
                f"HTTP {response.status_code}. "
                f"Respuesta: {response.text[:500]}"
            )
        )

    return {
        "success": True,
        "status_code": response.status_code,
        "response": response.text[:500]
    }


# =========================================================
# PRUEBA DE CONEXIÓN CON MAKE
# =========================================================

@app.get("/test-make")
async def test_make():
    """
    Prueba sencilla de comunicación
    Render -> Make.

    No envía un PNG real.
    """

    if not MAKE_WEBHOOK_URL:

        raise HTTPException(
            status_code=500,
            detail=(
                "MAKE_WEBHOOK_URL no está configurado."
            )
        )

    try:

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30.0
        ) as client:

            response = await client.post(
                MAKE_WEBHOOK_URL,
                data={
                    "test": "true",
                    "message": (
                        "Conexion Render -> Make funcionando"
                    )
                }
            )

        return {
            "success": True,
            "make_status": response.status_code,
            "make_response": response.text
        }

    except httpx.RequestError as e:

        raise HTTPException(
            status_code=502,
            detail=(
                "No pude conectar con Make: "
                f"{str(e)}"
            )
        )


# =========================================================
# ENDPOINT /PROCESS
# =========================================================

@app.post(
    "/process",
    response_class=Response
)
async def process_design(
    request: DriveRequest
):
    """
    Procesa el PDF y devuelve
    directamente el PNG.
    """

    try:

        pdf_bytes = (
            await download_pdf_from_drive(
                request.drive_url
            )
        )

        png_bytes = generate_processed_png(
            pdf_bytes
        )

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition":
                'attachment; filename="diseno_procesado.png"'
            }
        )

    except HTTPException:
        raise

    except pymupdf.FileDataError:

        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo descargado "
                "no pudo abrirse como PDF."
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# ENDPOINT /PROCESS-GPT
# =========================================================

@app.post("/process-gpt")
async def process_design_gpt(
    request: DriveRequest
):
    """
    Flujo principal para GPT.

    1. Descarga PDF desde Drive.
    2. Genera PNG.
    3. Guarda temporalmente en Render.
    4. Envía PNG real a Make.
    5. Devuelve URL temporal + estado Make.
    """

    try:

        # Descargar PDF
        pdf_bytes = (
            await download_pdf_from_drive(
                request.drive_url
            )
        )

        # Generar PNG
        png_bytes = generate_processed_png(
            pdf_bytes
        )

        # ID único
        file_id = uuid4().hex

        filename = (
            f"diseno_procesado_{file_id[:8]}.png"
        )

        # Guardar copia temporal en Render
        output_path = (
            OUTPUT_DIR
            / f"{file_id}.png"
        )

        output_path.write_bytes(
            png_bytes
        )

        download_url = (
            f"{PUBLIC_BASE_URL}"
            f"/download/{file_id}"
        )

        # Enviar PNG a Make
        make_result = (
            await send_png_to_make(
                png_bytes=png_bytes,
                filename=filename,
                source_drive_url=request.drive_url
            )
        )

        return {
            "success": True,
            "file_id": file_id,
            "filename": filename,
            "content_type": "image/png",
            "download_url": download_url,
            "make_sent": True,
            "make_status": (
                make_result[
                    "status_code"
                ]
            )
        }

    except HTTPException:
        raise

    except pymupdf.FileDataError:

        raise HTTPException(
            status_code=400,
            detail=(
                "El archivo descargado "
                "no pudo abrirse como PDF."
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =========================================================
# DESCARGA TEMPORAL
# =========================================================

@app.get("/download/{file_id}")
def download_processed_design(
    file_id: str
):
    """
    Descarga un PNG generado mediante
    /process-gpt.
    """

    if not re.fullmatch(
        r"[a-f0-9]{32}",
        file_id
    ):

        raise HTTPException(
            status_code=400,
            detail="ID de archivo inválido."
        )

    output_path = (
        OUTPUT_DIR
        / f"{file_id}.png"
    )

    if not output_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "El archivo no existe o "
                "ya no está disponible."
            )
        )

    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename="diseno_procesado.png"
    )
