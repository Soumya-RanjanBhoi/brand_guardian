import uuid ,json,logging 
from pprint import pprint
from backend.src.graph.workflow import workflow
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger=logging.getLogger("brand-guardian-runner")

def main():
    session_id=str(uuid.uuid4())

    logger.info(f"Starting Audit session: {session_id}")

    initial_inputs={
        "video_url":"https://youtu.be/dT7S75eYhcQ",
        "video_id": f"vid_{session_id[:8]}",
        "compliance_result":[],
        "error":[]
    }

    print('initilazing workflow....')
    print(f"Input payload:{json.dumps(initial_inputs,indent=2)}")

    try:

        final_state = workflow.invoke(initial_inputs)
    
        print("\n--- 2. WORKFLOW EXECUTION COMPLETE ---")    
        print("\n=== COMPLIANCE AUDIT REPORT ===")
        print(f"Video ID:    {final_state.get('video_id')}")
        print(f"Status:      {final_state.get('final_status')}")
        print("\n[ VIOLATIONS DETECTED ]")
        
        results = final_state.get('compliance_results', [])
        
        if results:
            for issue in results:
                print(f"- [{issue.get('severity')}] {issue.get('category')}: {issue.get('description')}")
        else:
            print("No violations found.")

        print("\n[ FINAL SUMMARY ]")
        print(final_state.get('final_report'))

    except Exception as e:
        logger.error(f"Workflow Execution Failed: {str(e)}")
        
        raise e



if __name__ == "__main__":
    main()