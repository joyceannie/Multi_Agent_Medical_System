# app/graph/graph_builder.py

from app.graph.types import State
from app.agents.icd10_agent import ICD10Agent
from app.agents.soap_generator_agent import SoapGeneratorAgent
from app.agents.image_analyzer_agent import ImageAnalyzerAgent
from app.agents.router_agent import RouterAgent
from langgraph.graph import START, END, StateGraph

def build_graph():
    graph = StateGraph(State)

    graph.add_node("router", RouterAgent().run)
    graph.add_node("icd10", ICD10Agent().run)
    graph.add_node("soap", SoapGeneratorAgent().run)
    graph.add_node("image_analysis", ImageAnalyzerAgent().run)

    graph.add_edge(START, "router")

    graph.add_conditional_edges("router", lambda s: s.type, {
        "icd10": "icd10",
        "soap": "soap",
        "image_analysis": "image_analysis"
    })

    graph.add_edge("icd10", END)
    graph.add_edge("soap", END)
    graph.add_edge("image_analysis", END)

    return graph.compile()
