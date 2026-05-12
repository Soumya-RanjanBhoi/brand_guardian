import json, os, logging
import re
from pinecone import Pinecone,ServerlessSpec
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage

from backend.src.graph.state import *
from backend.src.services.video_indexer import VideoIndexerService

logger = logging.getLogger("Brand Guardian")
logging.basicConfig(level=logging.INFO)
load_dotenv()



def index_video_node(state: VideoAuditState) -> Dict[str, Any]:
    """Download, upload and process video through AWS Transcribe + Rekognition."""

    video_url = state.get("video_url")
    video_id_input = state.get("video_id", "vid_demo")

    logger.info(f"[Node:Indexer] Processing: {video_url}")
    local_filename = 'temp_audit_video.mp4'

    try:
        vi_service = VideoIndexerService()

        if "youtube.com" in video_url or "youtu.be" in video_url:
            local_path = vi_service.download_youtube_video(video_url, output_path=local_filename)
        else:
            raise Exception("Please upload a valid YouTube URL")

        
        s3_uri = vi_service.upload_video(local_path, video_name=video_id_input)
        logger.info(f"Upload Success. S3 URI: {s3_uri}")

        
        vi_service.wait_for_processing(video_id_input)

        
        if os.path.exists(local_path):
            os.remove(local_path)
            logger.info("Local temp file removed")

        
        transcribe_job_name = vi_service.start_transcription(video_id_input)
        label_job_id, text_job_id = vi_service.start_rekognition(video_id_input)

        transcript = vi_service.wait_for_transcription(job_name=transcribe_job_name)
        label_response = vi_service.wait_for_rekognition(job_id=label_job_id, job_type="label")
        text_response = vi_service.wait_for_rekognition(job_id=text_job_id, job_type="text")

        
        clean_data = vi_service.extract_data(transcript, label_response, text_response)
        return clean_data

    except Exception as e:
        logger.error(f"Video indexer failed: {e}")
        return {
            "error": [str(e)],
            "final_result": "FAIL",
            "transcript": "",
            "ocr_text": []
        }


def audio_content_node(state: VideoAuditState) -> Dict[str, Any]:
    """
    RAG operation to audit the video content against compliance rules.
    """

    logger.info("[Node: Auditor] Querying knowledge base")

    transcript = state.get("transcript", "")
    if not transcript:
        logger.warning("No transcript available. Skipping audit")
        return {
            "final_result": "FAIL",
            "final_report": "Audit skipped — video processing failed (no transcript)"
        }

    llm = ChatMistralAI(model_name="mistral-large-latest")

    embeddings = MistralAIEmbeddings(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-embed"
    )

    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

    index_name = os.environ.get("PINECONE_INDEX_NAME")

    pc.create_index(
        name="compliance-rules",dimension=1024,metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    
    if index_name is None:
        raise RuntimeError("PINECONE_INDEX_NAME environment variable is required")
    
    print("index created successfully")
    index = pc.Index(index_name)

    vector_store = PineconeVectorStore(
        index=index,
        embedding=embeddings
    )

    ocr_text = state.get("ocr_text", [])
    query_text = f"{transcript} {' '.join(ocr_text)}"
    docs = vector_store.similarity_search(query_text, k=4)
    retrieved_rules = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
        You are a senior brand compliance auditor.
        OFFICIAL REGULATORY RULES:
        {retrieved_rules}

        INSTRUCTIONS:
        1. Analyze the transcript and OCR text below.
        2. Identify any violations of the rules.
        3. Return ONLY strict JSON in this exact format with no extra text:
        {{
            "compliance_results": [
             {{
                "category": "Claim Validation",
                "severity": "CRITICAL",
                "description": "Explanation of the violation"
             }}
            ],
            "status": "FAIL",
            "final_report": "Summary of the findings"
        }}
    """

    user_message = f"""
        VIDEO_METADATA: {state.get('video_metadata', {})}
        TRANSCRIPT: {transcript}
        ON_SCREEN_TEXT: {ocr_text}
    """

    response = None
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])

        content = response.content

        if "```" in content:
            match = re.search(r"```(?:json)?(.*?)```", content, re.DOTALL)
            if match:
                content = match.group(1)

        audit_data = json.loads(content.strip())

        return {
            "compliance_result": audit_data.get("compliance_results", []),
            "final_result": audit_data.get("status", "FAIL"),
            "final_report": audit_data.get("final_report", "No report generated")
        }

    except Exception as e:
        logger.error(f"System error in Auditor Node: {str(e)}")
        logger.error(f"Raw LLM response: {response.content if response else 'N/A'}")
        return {
            "error": [str(e)],
            "final_result": "FAIL",
            "final_report": "Audit failed due to system error"
        }