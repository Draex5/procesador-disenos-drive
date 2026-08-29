from fastapi import FastAPI

app = FastAPI(
    title="Procesador de Diseños",
    version="1.0.0"
)

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
