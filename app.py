import io

import fitz
import httpx

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


app = FastAPI(
    title="Procesador de Diseños",
    version="1.1.0"
)

TEMPLATE_PATH = "plantilla.pdf"
WATERMARK_PATH = "marca_agua.pdf"

RENDER_SCALE = 2.0

# Ajustaremos estos valores después de la prueba final
WATERMARK_WIDTH_RATIO = 0.65
WATERMARK_OPACITY = 90


class ProcessRequest(BaseModel):
    file_url: str


@app.get("/")
def inicio():
    return {
        "status": "ok",
        "message": "API de procesamiento de diseños funcionando"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


def render_pdf_page(pdf_bytes=None, pdf_path=None, alpha=False):
    if pdf_bytes is not None:
        doc = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )
    else:
        doc = fitz.open(pdf_path)

    if len(doc) == 0:
        doc.close()
        raise ValueError("El PDF no contiene páginas")

    page = doc[0]

    matrix = fitz.Matrix(
        RENDER_SCALE,
        RENDER_SCALE
    )

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=alpha
    )

    mode = "RGBA" if alpha else "RGB"

    image = Image.frombytes(
        mode,
        [pix.width, pix.height],
        pix.samples
    )

    doc.close()

    return image


def is_green(pixel):
    r, g, b = pixel[:3]

    return (
        g > 80
        and g > r * 1.25
        and g > b * 1.15
    )


def detect_inner_rectangle(template):
    rgb = template.convert("RGB")

    width, height = rgb.size

    vertical_scores = [0] * width
    horizontal_scores = [0] * height

    pixels = rgb.load()

    step = 2

    for y in range(0, height, step):
        for x in range(0, width, step):
            if is_green(pixels[x, y]):
                vertical_scores[x] += 1
                horizontal_scores[y] += 1

    vertical_candidates = [
        i for i, score in enumerate(vertical_scores)
        if score > height * 0.08 / step
    ]

    horizontal_candidates = [
        i for i, score in enumerate(horizontal_scores)
        if score > width * 0.08 / step
    ]

    def group_positions(values, tolerance=8):
        if not values:
            return []

        groups = [[values[0]]]

        for value in values[1:]:
            if value - groups[-1][-1] <= tolerance:
                groups[-1].append(value)
            else:
                groups.append([value])

        return [
            int(sum(group) / len(group))
            for group in groups
        ]

    xs = group_positions(
        vertical_candidates
    )

    ys = group_positions(
        horizontal_candidates
    )

    if len(xs) < 4 or len(ys) < 4:
        raise ValueError(
            "No pude detectar correctamente los cuatro lados "
            "de los cuadros verdes de la plantilla."
        )

    xs = sorted(xs)
    ys = sorted(ys)

    # Cuadro interior
    left = xs[1]
    right = xs[-2]
    top = ys[1]
    bottom = ys[-2]

    if right <= left or bottom <= top:
        raise ValueError(
            "Las líneas verdes fueron detectadas, "
            "pero el área interior no es válida."
        )

    return left, top, right, bottom


def fit_inside(
    image,
    max_width,
    max_height
):
    width, height = image.size

    scale = min(
        max_width / width,
        max_height / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


def apply_watermark(
    canvas,
    inner_rect
):
    watermark = render_pdf_page(
        pdf_path=WATERMARK_PATH,
        alpha=False
    ).convert("RGBA")

    canvas_width, canvas_height = canvas.size

    # Ajustar la hoja completa de marca de agua
    # a todo el recuadro blanco del Display
    watermark = watermark.resize(
        (canvas_width, canvas_height),
        Image.Resampling.LANCZOS
    )

    pixels = watermark.load()

    for y in range(watermark.height):
        for x in range(watermark.width):
            r, g, b, a = pixels[x, y]

            # Eliminar fondo blanco
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
                    100
                )

    canvas.alpha_composite(
        watermark,
        (0, 0)
    )


@app.post("/process")
async def process_design(
    request: ProcessRequest
):
    try:
        # Descargar PDF desde la URL recibida
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=60.0
        ) as client:
            response = await client.get(
                request.file_url
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No se pudo descargar el PDF. "
                    f"HTTP {response.status_code}"
                )
            )

        pdf_bytes = response.content

        if not pdf_bytes:
            raise HTTPException(
                status_code=400,
                detail="El archivo descargado está vacío."
            )

        if not pdf_bytes.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "El archivo descargado "
                    "no parece ser un PDF válido."
                )
            )

        # Renderizar plantilla
        template = render_pdf_page(
            pdf_path=TEMPLATE_PATH
        )

        # Detectar área máxima permitida
        inner_rect = detect_inner_rectangle(
            template
        )

        left, top, right, bottom = inner_rect

        inner_width = right - left
        inner_height = bottom - top

        # Renderizar diseño recibido
        design = render_pdf_page(
            pdf_bytes=pdf_bytes,
            alpha=False
        )

        # Ajustar conservando proporción
        design = fit_inside(
            design,
            inner_width,
            inner_height
        )

        # Crear Display final sin guías verdes
        canvas = Image.new(
            "RGBA",
            template.size,
            (255, 255, 255, 255)
        )

        # Centrar diseño dentro del cuadro interior
        x = left + (
            inner_width - design.width
        ) // 2

        y = top + (
            inner_height - design.height
        ) // 2

        canvas.alpha_composite(
            design.convert("RGBA"),
            (x, y)
        )

        # Aplicar marca de agua oficial
        apply_watermark(
            canvas,
            inner_rect
        )

        # Exportar únicamente PNG
        output = io.BytesIO()

        canvas.convert("RGB").save(
            output,
            format="PNG",
            optimize=True
        )

        output.seek(0)

        return Response(
            content=output.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition":
                'attachment; filename="diseno_procesado.png"'
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
