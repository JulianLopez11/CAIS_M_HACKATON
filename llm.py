import os
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # lee el archivo .env en la carpeta actual, si existe

if not os.environ.get("GOOGLE_API_KEY"):
    raise SystemExit(
        "Falta GOOGLE_API_KEY.\n"
        "Crea un archivo .env en esta carpeta con la línea:\n"
        "  GOOGLE_API_KEY=tu-key-aqui\n"
        "o expórtala en tu terminal:\n"
        "  export GOOGLE_API_KEY=tu-key-aqui   (Mac/Linux)\n"
        "  set GOOGLE_API_KEY=tu-key-aqui      (Windows cmd)\n"
        "Consigue la key gratis en https://aistudio.google.com/apikey"
    )


llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)

# Esquema de salida estructurada para el nodo "clasificar" del grafo
class Clasificacion(BaseModel):
    categoria: Literal[
        "Facturación", "Acceso y cuentas", "Rendimiento",
        "Integraciones", "Datos y reportes", "Interfaz / uso",
        "Fuera de alcance",
    ] = Field(description="Categoría del ticket según la Tabla 1 del caso")
    severidad: Literal["Crítica", "Alta", "Media", "Baja", "Fuera de alcance"] = Field(
        description="Severidad del ticket según la Tabla 2 del caso"
    )
    justificacion: str = Field(description="Una frase explicando la decisión")

clasificador = llm.with_structured_output(Clasificacion)

SYSTEM_PROMPT = """Eres el clasificador de triage de soporte de Aurora, una
plataforma SaaS de gestión de proyectos. Dado el texto libre de un ticket,
asigna categoría y severidad EXACTAMENTE según estas reglas:

Categorías posibles: Facturación, Acceso y cuentas, Rendimiento,
Integraciones, Datos y reportes, Interfaz / uso, Fuera de alcance.

Severidades:
- Crítica: servicio caído o pérdida de datos para múltiples usuarios.
- Alta: función clave bloqueada, sin alternativa disponible.
- Media: función degradada, o bloqueada pero con alternativa.
- Baja: consulta, mejora o defecto menor.
- Fuera de alcance: no es un problema de la plataforma (p.ej. preguntas
  sobre facturación tributaria/DIAN de la empresa del cliente, no de Aurora).

Si el ticket no es sobre el funcionamiento de Aurora, usa categoría y
severidad "Fuera de alcance" en ambos campos."""


def activar():
    """Llamada mínima para confirmar que la conexión con Gemini funciona,
    antes de gastar tiempo corriendo todo el grafo de triage_agent.py."""
    respuesta = llm.invoke("Responde solo con la palabra: listo")
    print("Conexión con Gemini OK. Respuesta del modelo:", respuesta.content)


if __name__ == "__main__":
    activar()