import json,os,logging
import regex as re
from typing import Dict,Any,List
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage,HumanMessage

from backend.src.graph.state import *
from backend.src.services.video_indexer import VideoIndexerService

logger= logging.getLogger("Brand Guardian")
logging.basicConfig(level=logging.INFO)
load_dotenv()



def index_video_node(state: VideoAuditState) ->Dict[str,Any]:
    """ Download video  """

    video_url = state.get("video_url")
    video_id_input = state.get("video_id","vid_demo")

    logger.info(f"[Node:Indexer] Processing: {video_url}")

    local_filename = 'temp_audit_video.mp4'

    try:
        vi_service=VideoIndexerService()

        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url,output_path=local_filename)

        else:
            raise Exception("PLease upload a valid youtube url")
        

        aws_video_id = vi_service.upload_video(local_path,video_name=video_id_input)
        logger.info(f"Upload Success . ID: {aws_video_id}")

        if os.path.exists(local_path):
            os.remove(local_path)
        
        raw_insights = vi_service.wait_for_processing(aws_video_id)
        clean_data =  vi_service.extract_data(raw_insights)

        return clean_data
    
    except Exception as e:
        logger.error(f"Video indexer failed : {e}")
        return {
            "error":[str(e)],
            "final_result":"FAIL",
            "transcript":"",
            "ocr_text":[]
        }

def audio_content_node(state:VideoAuditState) ->Dict[str,Any]:
    """ 
    RAG operation to audit the content
    """

    logger.info("[Node: Auditor] querying knowledge base")
    transcript = state.get("transcript","")
    if not transcript:
        logger.warning("No transcript available . Skipping audit")
        return {
            "final_status":"FAIL",
            "final_report":"Audit Skipped , video processing failed(No transcript)"
        }
    
    llm = ChatMistralAI()
    embeddings = MistralAIEmbeddings()

    vector_store=PineconeVectorStore(
        index=os.environ.get("PINECONE_INDEX_NAME"),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY"),
        embedding=embeddings
    )

    ocr_text=state.get("ocr_text",[])
    query_text=f"{transcript} {''.join(ocr_text)}"
    docs=vector_store.similarity_search(query_text,k=4)
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
        You are a senior brand compliance auditor.
        OFFICAL REGULATORY RULES:
        {retrieved_rules}

        INSTRUCTIONS:
        1.Analyze the Transcript and OCT text below.
        2.Identify any violation of the rules.
        3.Return strictly json in the following format.
        {{
            "compliance_results: [
             {{
             "category":'Claim Validation',
             "severity":"CRITICAL",
             "description":"Explanation of the violation"
             }}
            ],
            "status":"Fail",
            "final_report";"Summary of the findings"
        }}
        """

    user_message = f"""
        VIDEO_METADATA: {state.get('video_metadata',{})},
        ON_SCREEN_TEXT:{ocr_text}
        """
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])

        content = response.content
        if "```" in content:
            content = re.search(r"```(?:json)?(.?)```",content,re.DOTALL).group(1)
        audit_data=json.loads(content.strip())
        return {
            "compliance_result":audit_data.get("compliance_result",[]),
            "final_result":audit_data.get("status","FAIL"),
            "final_report":audit_data.get("final_report","No report generated")
        }
    
    except Exception as e:
        logger.error(f"System Error in Auditor Node : {str(e)}")
        logger.error(f"Raw LLM Response : {response.content if 'response' in locals() else ''}")
        return {
            "error":[str(e)],
            "final_status":"Fail"
        }
