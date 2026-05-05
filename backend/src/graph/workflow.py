from langgraph.graph import START,END,StateGraph
from backend.src.graph.state import VideoAuditState
from backend.src.graph.nodes import index_video_node,audio_content_node


def create_graph():
    
    graph = StateGraph(VideoAuditState)

    graph.add_node("indexer",index_video_node)
    graph.add_node("auditor",audio_content_node)

    graph.add_edge(START,"indexer")
    graph.add_edge("indexer","auditor")
    graph.add_edge("auditor",END)

    workflow = graph.compile()
    return workflow

workflow =create_graph()
