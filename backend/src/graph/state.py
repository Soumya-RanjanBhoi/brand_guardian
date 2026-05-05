import operator
from typing import Annotated,List,Dict,Optional,Any,TypedDict

#Report
class ComplianceIssue(TypedDict):
    category: str 
    description : str
    severity : str
    timestamp : Optional[str]



class VideoAuditState(TypedDict):

    video_url:str
    video_id:str

    local_file_path : Optional[str]
    video_metadata: Dict[str,Any]
    transcript: Optional[str]
    ocr_text: List[str]

    compliance_result : Annotated[List[ComplianceIssue],operator.add]

    final_result : str
    final_report : str

    error : Annotated[List[str],operator.add]  #system error

