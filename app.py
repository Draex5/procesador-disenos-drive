import io
import re

import pymupdf
import httpx

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel


app = FastAPI(
    title="Procesador de Diseños",
    version="1.3.1"
)

TEMPLATE_PATH = "plantilla.pdf"
WATERMARK_PATH = "marca_agua.pdf"

RENDER_SCALE = 2.0

# Opacidad de la marca de agua
WATERMARK_OPACITY = 100


class DriveRequest(BaseModel):
    drive_url: str


@app.get("/")
def inicio():
    return {
        "status": "ok",
        "message": "API de procesamiento de diseños funcionando"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


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
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    raise HTTPException(
        status_code=400,
        detail=(
            "No pude obtener el ID del archivo "
            "de Google Drive."
        )
    )


async def download_pdf_from_drive(drive_url):
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
                f"el archivo. HTTP {response.status_code}"
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


def render_pdf_page(
    pdf_bytes=None,
    pdf_path=None,
    alpha=False
):
    """
    Renderiza la primera página de un PDF
    como imagen Pillow.
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

        matrix = pymupdf.Matrix(
            RENDER_SCALE,
            RENDER_SCALE
        )

        pix = page.get_pixmap(
            matrix=matrix,
            alpha=alpha
        )

        mode = (
            "RGBA"
            if alpha
            else "RGB"
        )

        image = Image.frombytes(
            mode,
            [
                pix.width,
                pix.height
            ],
            pix.samples
        )

        return image

    finally:
        doc.close()


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
    Agrupa posiciones consecutivas detectadas
    como pertenecientes a una misma línea.
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


def detect_rectangles(template):
    """
    Detecta los cuadros verdes
    de la plantilla.

    Devuelve:

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

    # Cuadro exterior = Display final
    outer_left = xs[0]
    outer_right = xs[-1]
    outer_top = ys[0]
    outer_bottom = ys[-1]

    # Cuadro interior = área máxima del diseño
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


def fit_inside(
    image,
    max_width,
    max_height
):
    """
    Ajusta la imagen al área máxima disponible
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


def make_white_transparent(
    image,
    opacity=100
):
    """
    Convierte el fondo blanco de una imagen
    en transparente y aplica opacidad
    al resto de la marca de agua.
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
            r, g, b, a = (
                pixels[x, y]
            )

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


def apply_watermark(canvas):
    """
    Distribuye la hoja oficial de marca de agua
    sobre todo el recuadro blanco del Display.
    """

    watermark = render_pdf_page(
        pdf_path=WATERMARK_PATH,
        alpha=False
    )

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


@app.post(
    "/process",
    response_class=Response
)
async def process_design(
    request: DriveRequest
):
    try:
        # Descargar PDF desde Google Drive
        pdf_bytes = (
            await download_pdf_from_drive(
                request.drive_url
            )
        )

        # Renderizar plantilla oficial
        template = render_pdf_page(
            pdf_path=TEMPLATE_PATH,
            alpha=False
        )

        # Detectar:
        # - cuadro exterior = Display final
        # - cuadro interior = área máxima del diseño
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

        display_width = (
            outer_right
            - outer_left
        )

        display_height = (
            outer_bottom
            - outer_top
        )

        inner_width = (
            inner_right
            - inner_left
        )

        inner_height = (
            inner_bottom
            - inner_top
        )

        # Renderizar diseño descargado
        design = render_pdf_page(
            pdf_bytes=pdf_bytes,
            alpha=False
        )

        # Ajustar proporcionalmente
        # dentro del cuadro interior
        design = fit_inside(
            design,
            inner_width,
            inner_height
        )

        # Crear únicamente el Display final.
        # No copiamos la plantilla:
        # no salen líneas verdes ni guías.
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

        # Convertir coordenadas del área interior
        # desde la plantilla al nuevo canvas.
        relative_inner_left = (
            inner_left
            - outer_left
        )

        relative_inner_top = (
            inner_top
            - outer_top
        )

        relative_inner_right = (
            inner_right
            - outer_left
        )

        relative_inner_bottom = (
            inner_bottom
            - outer_top
        )

        relative_inner_width = (
            relative_inner_right
            - relative_inner_left
        )

        relative_inner_height = (
            relative_inner_bottom
            - relative_inner_top
        )

        # Centrar diseño dentro del cuadro interior
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

        # Aplicar marca de agua oficial
        # sobre todo el Display final
        apply_watermark(
            canvas
        )

        # Exportar únicamente PNG
        output = io.BytesIO()

        canvas.convert(
            "RGB"
        ).save(
            output,
            format="PNG",
            optimize=True
        )

        output.seek(
            0
        )

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
