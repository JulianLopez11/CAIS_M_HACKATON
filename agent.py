import os
from typing import TypedDict, Optional, List
from langgraph.graph import StateGraph, END
from llm import llm, clasificador, SYSTEM_PROMPT

# Data que tiene el reto 

CATEGORIAS = [
    "Facturación", "Acceso y cuentas", "Rendimiento",
    "Integraciones", "Datos y reportes", "Interfaz / uso",
    "Fuera de alcance",
]

SEVERIDADES = ["Crítica", "Alta", "Media", "Baja", "Fuera de alcance"]

EQUIPO_POR_CATEGORIA = {
    "Facturación": "Finanzas",
    "Acceso y cuentas": "Identity",
    "Rendimiento": "Plataforma",
    "Integraciones": "Integraciones",
    "Datos y reportes": "Datos",
    "Interfaz / uso": "Producto",
    "Fuera de alcance": "Ninguno (redirigir)",
}

SLA_POR_SEVERIDAD = {
    "Crítica": {"respuesta": "1 hora", "resolucion": "4 horas"},
    "Alta": {"respuesta": "4 horas", "resolucion": "24 horas"},
    "Media": {"respuesta": "8 horas", "resolucion": "72 horas"},
    "Baja": {"respuesta": "24 horas", "resolucion": "10 días hábiles"},
    "Fuera de alcance": {"respuesta": "Redirigir", "resolucion": "No aplica"},
}

# 2. Estado del grafo

class TicketState(TypedDict):
    ticket_text: str
    categoria: Optional[str]
    severidad: Optional[str]
    fuera_de_alcance: bool
    equipo: Optional[str]
    sla_respuesta: Optional[str]
    sla_resolucion: Optional[str]
    borrador_respuesta: Optional[str]
    trace: List[str]

# 3. Nodos (el LLM, el esquema de salida y el prompt vienen de llm.py)

def clasificar(state: TicketState) -> TicketState:
    resultado = clasificador.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", state["ticket_text"]),
    ])
    return {
        **state,
        "categoria": resultado.categoria,
        "severidad": resultado.severidad,
        "fuera_de_alcance": resultado.categoria == "Fuera de alcance",
        "trace": state["trace"] + [
            f"[LLM] categoria={resultado.categoria} severidad={resultado.severidad} "
            f"| {resultado.justificacion}"
        ],
    }

def asignar_equipo(state: TicketState) -> TicketState:
    equipo = EQUIPO_POR_CATEGORIA[state["categoria"]]
    return {**state, "equipo": equipo,
            "trace": state["trace"] + [f"[lookup Tabla 1] equipo={equipo}"]}

def asignar_sla(state: TicketState) -> TicketState:
    sla = SLA_POR_SEVERIDAD[state["severidad"]]
    return {
        **state,
        "sla_respuesta": sla["respuesta"],
        "sla_resolucion": sla["resolucion"],
        "trace": state["trace"] + [f"[lookup Tabla 2] sla={sla}"],
    }

def redirigir(state: TicketState) -> TicketState:
    sla = SLA_POR_SEVERIDAD["Fuera de alcance"]
    return {
        **state,
        "equipo": "Ninguno (redirigir)",
        "sla_respuesta": sla["respuesta"],
        "sla_resolucion": sla["resolucion"],
        "trace": state["trace"] + ["[regla] ticket fuera de alcance -> redirigir"],
    }

def borrador_respuesta(state: TicketState) -> TicketState:
    """Opcional (solo si sobra tiempo, según el enunciado)."""
    prompt = (
        f"Redacta un borrador breve (2-3 frases, tono profesional) de primera "
        f"respuesta al cliente para un ticket de categoría '{state['categoria']}' "
        f"y severidad '{state['severidad']}'. SLA de respuesta: "
        f"{state['sla_respuesta']}. No inventes detalles del ticket que no "
        f"estén en este texto: {state['ticket_text']}"
    )
    respuesta = llm.invoke(prompt).content
    return {**state, "borrador_respuesta": respuesta,
            "trace": state["trace"] + ["[LLM] borrador de respuesta generado"]}

def ruta_fuera_de_alcance(state: TicketState) -> str:
    return "redirigir" if state["fuera_de_alcance"] else "asignar_equipo"

# 5. Construcción del grafo

def build_graph(incluir_borrador: bool = False) -> StateGraph:
    graph = StateGraph(TicketState)

    graph.add_node("clasificar", clasificar)
    graph.add_node("asignar_equipo", asignar_equipo)
    graph.add_node("asignar_sla", asignar_sla)
    graph.add_node("redirigir", redirigir)

    graph.set_entry_point("clasificar")
    graph.add_conditional_edges(
        "clasificar", ruta_fuera_de_alcance,
        {"redirigir": "redirigir", "asignar_equipo": "asignar_equipo"},
    )
    graph.add_edge("asignar_equipo", "asignar_sla")

    if incluir_borrador:
        graph.add_node("borrador_respuesta", borrador_respuesta)
        graph.add_edge("asignar_sla", "borrador_respuesta")
        graph.add_edge("redirigir", "borrador_respuesta")
        graph.add_edge("borrador_respuesta", END)
    else:
        graph.add_edge("asignar_sla", END)
        graph.add_edge("redirigir", END)

    return graph.compile()

# 6. Los 6 casos de prueba de la Tabla 3, con la etiqueta esperada

CASOS_DE_PRUEBA = [
    dict(id="T1", texto="Desde esta mañana nadie de mi equipo puede iniciar sesión, "
                         "la página de login se queda cargando indefinidamente.",
         categoria_esp="Acceso y cuentas", severidad_esp="Crítica", equipo_esp="Identity"),
    dict(id="T2", texto="Los reportes de tiempo llevan dos días sin actualizarse, "
                         "seguimos viendo datos de la semana pasada.",
         categoria_esp="Datos y reportes", severidad_esp="Alta", equipo_esp="Datos"),

    dict(id="T3", texto="¿Podrían agregar la opción de exportar las tareas a PDF, "
                         "además de Excel?",
         categoria_esp="Interfaz / uso", severidad_esp="Baja", equipo_esp="Producto"),
    dict(id="T4", texto="La integración con Slack dejó de enviar notificaciones desde "
                         "ayer, ya no recibimos nada en el canal.",
         categoria_esp="Integraciones", severidad_esp="Alta", equipo_esp="Integraciones"),
]

def correr_casos_de_prueba(incluir_borrador: bool = False):
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("Falta GOOGLE_API_KEY en el entorno.")

    app = build_graph(incluir_borrador=incluir_borrador)
    resultados = []
    for caso in CASOS_DE_PRUEBA:
        estado_inicial: TicketState = {
            "ticket_text": caso["texto"], "categoria": None, "severidad": None,
            "fuera_de_alcance": False, "equipo": None, "sla_respuesta": None,
            "sla_resolucion": None, "borrador_respuesta": None, "trace": [],
        }
        salida = app.invoke(estado_inicial)
        ok_cat = salida["categoria"] == caso["categoria_esp"]
        ok_sev = salida["severidad"] == caso["severidad_esp"]
        ok_eq = salida["equipo"] == caso["equipo_esp"]
        resultados.append({**salida, "id": caso["id"],
                            "pasa": ok_cat and ok_sev and ok_eq})
        print(f"\n--- {caso['id']} ---")
        print(f"texto: {caso['texto']}")
        print(f"esperado: {caso['categoria_esp']} / {caso['severidad_esp']} / {caso['equipo_esp']}")
        print(f"obtenido: {salida['categoria']} / {salida['severidad']} / {salida['equipo']}")
        print(f"SLA: respuesta={salida['sla_respuesta']} resolucion={salida['sla_resolucion']}")
        print("PASA" if resultados[-1]["pasa"] else "FALLA")
        for linea in salida["trace"]:
            print("  ", linea)

    n_ok = sum(r["pasa"] for r in resultados)
    print(f"\n=== {n_ok}/{len(resultados)} casos correctos ===")
    return resultados


if __name__ == "__main__":
    # Cambia a True si les sobra tiempo para el borrador de respuesta (opcional
    correr_casos_de_prueba(incluir_borrador=True)