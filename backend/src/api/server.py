import uuid,logging
from fastapi import FastAPI,HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import  CORSMiddleware
from pydantic import BaseModel
from typing import List,Optional
from dotenv import load_dotenv
from backend.src.api.telemetry import * 

load_dotenv()

from backend.src.graph.workflow import workflow as graph 

logging.basicConfig(level=logging.INFO)

logger=logging.getLogger("api-server")

app=FastAPI(
    title="Brand Guardian"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuditRequest(BaseModel):
    video_url : str 

class ComplianceIssue(BaseModel):
    category: str 
    severity:str 
    description:str 

class AuditResponse(BaseModel):
    session_id: str
    video_id: str
    status: str
    final_report: str
    compliance_results: List[ComplianceIssue] = []


@app.post("/health")
def health_check():
    return JSONResponse(
        status_code=200,
        content={"status":"healthy","service":"Brand Guardian"}
    )

@app.post("/audit",response_model=AuditResponse)
async def audit_video(request:AuditRequest):
    ''' 
    Start the Audit workflow 
    '''

    session_id=str(uuid.uuid4())
    video_id=session_id[:8]
    logger.info(f"Received the Audit request : {request.video_url} , Session :{session_id}")

    initial_inputs = {
        "video_url":request.video_url,
        "video_id":video_id,
        "compliance_result":[],
        "error":[]
    }

    try:
        final_state=graph.invoke(initial_inputs)
        return AuditResponse(
            session_id=session_id,
            video_id=final_state.get("video_id",""),
            status=final_state.get("final_result",'FAIL'),
            final_report=final_state.get("final_report",'No report Generated'),
            compliance_results=final_state.get("compliance_result",[])
        )
    
    except Exception as e:
        logger.error(f"Audit failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Workflow Execution Failed: {str(e)}"
        )
    



