# CAIS_M_HACKATON

Agente de triage para tickets de soporte de Aurora. El agente clasifica cada ticket por categoría y severidad, asigna el equipo responsable y calcula el SLA correspondiente.

## Requisitos

- Python 3.10 o superior
- Una API key de Google Gemini

## Instalación

Desde esta carpeta, ejecuta:

```powershell
python -m pip install -r requirements.txt
```

## Configurar la API key

Crea un archivo `.env` en la raíz del proyecto con este contenido:

```env
GOOGLE_API_KEY=tu_api_key_de_gemini
```

No compartas ni subas este archivo a GitHub.

## Probar la conexión

Para comprobar únicamente que Gemini responde:

```powershell
python llm.py
```

La salida esperada contiene:

```text
Conexión con Gemini OK
```

## Ejecutar los casos

El archivo `agent.py` incluye los seis casos de prueba (`T1` a `T6`). Ejecuta:

```powershell
python agent.py
```

Al terminar, se muestra el resultado de cada caso y un resumen similar a:

```text
=== 6/6 casos correctos ===
```

Cada caso imprime también su categoría, severidad, equipo, SLA y trazabilidad del grafo.

## Generar borradores de respuesta

La generación de un borrador para el cliente es opcional. En `agent.py`, cambia:

```python
correr_casos_de_prueba(incluir_borrador=False)
```

por:

```python
correr_casos_de_prueba(incluir_borrador=True)
```

Después ejecuta nuevamente:

```powershell
python agent.py
```

## Flujo del agente

1. Gemini clasifica el ticket.
2. El grafo verifica si está fuera de alcance.
3. Para tickets válidos, busca el equipo según la categoría.
4. Busca el SLA según la severidad.
5. Opcionalmente, genera un borrador de respuesta.

Los casos fuera de alcance se redirigen y reciben el SLA `Redirigir / No aplica`.

## Problemas frecuentes

### Error `503 UNAVAILABLE`

Significa que el modelo está temporalmente saturado. Espera unos segundos y ejecuta de nuevo:

```powershell
python agent.py
```

Los avisos sobre `temperature` o automatic function calling son advertencias de la librería y no indican que falte la API key.