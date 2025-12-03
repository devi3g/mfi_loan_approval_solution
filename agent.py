"""
================================================================================
MICROFINANCE AGENTIC LOAN WORKFLOW SYSTEM (VERTEX AI ONLY, FULL VERSION)
================================================================================

CAPSTONE PROJECT – FULLY DOCUMENTED, PRODUCTION-GRADE VERSION  
Hybrid Documentation Style (C): Academic + Engineering + Illustrative

--------------------------------------------------------------------------------
INTRODUCTION
--------------------------------------------------------------------------------
This module implements a fully agentic, multi-stage loan application workflow
designed for a typical Microfinance Institution (MFI). It addresses challenges
common to microfinance operations:

    • High applicant volume
    • Small average ticket sizes
    • Higher default risk (especially for SMEs and sole traders)
    • Need for both speed and rigorous risk management
    • Compliance with local regulations and internal lending policies

The workflow focuses ONLY on the loan application and approval process, up to
the point where a decision is made:

    final_decision ∈ {"approved", "not_approved", "pending_approval"}

Everything beyond (e.g., disbursement, collections) can be layered on later.

--------------------------------------------------------------------------------
SCOPE OF THIS MODULE
--------------------------------------------------------------------------------
This single file provides:

    1) Agentic architecture for the following phases (each as an agent):
        - Intake
        - KYC
        - Loan Evaluation
        - Business Evaluation
        - Compliance
        - Final Decision

    2) Orchestration logic:
        - KYC and Loan Evaluation run in parallel.
        - Business Evaluation runs after Loan Evaluation.
        - Compliance runs after KYC + Business + Loan Evaluation.
        - Final Decision runs after Compliance.

    3) HITL (Human-in-the-Loop) integration:
        - When a phase's output indicates "fail" or "manual review needed",
          the system does NOT auto-continue.
        - Instead, it returns HITLRequest to signal the need for human
          intervention.

    4) Vertex AI backend only:
        - All LLM calls go through Gemini on Vertex AI.
        - This file assumes ADK is configured to use Vertex AI models.

    5) Firestore-based session state:
        - Session state is stored per user + session ID.
        - This powers RESUME: re-running the workflow reuses completed results
          instead of recomputing everything.

    6) Cloud SQL (MySQL) for:
        - Audit logs (per phase, with input & output snapshots).
        - User profile storage.
        - Versioned user documents (e.g., multiple business certificates).

--------------------------------------------------------------------------------
DESIGN GOALS
--------------------------------------------------------------------------------
    • Maintainability:
        - Clear sections with headings
        - Strong comments describing the "why" behind each part

    • Observability:
        - Audit logs per phase
        - HITL-return points clearly marked
        - Firestore session snapshots

    • Extensibility:
        - Easy to plug in more agents or tools
        - Easy to integrate RAG or more complex external checks

                    ┌────────────────────┐
                    │  Application Input │
                    └─────────┬──────────┘
                              │
                              ▼
                 ┌────────────────────────┐
                 │   Intake Phase Agent   │
                 └─────────┬──────────────┘
                           │       (Strict HITL if fail)
                           ▼
         ┌──────────────────────────────────────────┐
         │    Parallel Agents (KYC + LoanEval)       │
         │    Both run simultaneously                │
         └─────────┬──────────────────┬─────────────┘
                   │                  │
         (HITL if fail)     (HITL if fail)
                   │                  │
                   └──────────┬───────┘
                              ▼
                 ┌────────────────────────┐
                 │ Business Evaluation     │
                 └─────────┬──────────────┘
                           │ (HITL if fail)
                           ▼
                 ┌────────────────────────┐
                 │ Compliance Agent        │
                 └─────────┬──────────────┘
                           │ (HITL if fail)
                           ▼
                 ┌────────────────────────┐
                 │ Final Decision Agent    │
                 └─────────────────────────┘
    • Realistic:
        - Approximates what an MFI would actually need in production
        - Focus on regulatory-style traceability of decisions

--------------------------------------------------------------------------------
RESUME MODEL (SESSIONS)
--------------------------------------------------------------------------------
If the workflow stops (HITL, system failure, etc.), it can resume from the last
completed stage using persistent Firestore session state. Every stage writes:
Each run of the workflow is associated with:

    - user_id:   unique identifier of the applicant
    - session_id: unique identifier of this specific loan application process

In Firestore, we store per-phase outputs under a session document. For example:

    sessions/{session_id}/
        intake: {...}
        kyc: {...}
        loan: {...}
        business: {...}
        compliance: {...}
        final_decision: {...}

The resume logic checks which stages are already saved and re-runs only
incomplete ones. This allows long-running or HITL-dependent processes to
continue without restarting or losing progress.

================================================================================
BEGIN IMPLEMENTATION
================================================================================
"""

import os
from dotenv import load_dotenv  # to support loading the env  file
import uuid
import json
import datetime

import re
import asyncio
from google.adk.agents import Agent,LlmAgent,  SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools import google_search, ToolContext
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai.types import Content, Part
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.genai.types import Content, Part
# Vertex AI
from vertexai import init
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

print("✅ ADK components imported successfully.")
# =============================================================================
#LOAD Environment variables  
# load  environment variables defined in .env file 
# =============================================================================
# 1. Calculate the path to the .env file dynamically
# This ensures it works even if you run 'adk run' from a different folder
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
load_dotenv()

#----------------------------------------------------------------------------
#DEBUG REMOVE 

# 1. Check a specific key (Safest)
api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    # Print only the first 5 characters to confirm it's loaded without leaking the whole key
    print(f"✅ GOOGLE_API_KEY found: {api_key[:5]}********")
else:
    print("❌ GOOGLE_API_KEY is missing!")

# 2. Check the file location (To ensure you aren't loading the wrong .env)
print(f"Current Working Directory: {os.getcwd()}")
#END DEBUG
#----------------------------------------------------------------------------
print( f''' SETUP  ENVIRONMENT 
     Environment  variables  loaded  
    Source : .env''')

# =============================================================================
#CONFIGURE  RETRY OPTIONS 
#configure the number of  retries  in the  event  an agent  fails for  whatever  reason
# =============================================================================
APP_PARAM_RETRY_ATTEMPTS=5
retry_config=types.HttpRetryOptions(
    attempts=APP_PARAM_RETRY_ATTEMPTS,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)
print( f''' configured  number of retries for  agents. 
    Retries (APP_PARAM_RETRY_ATTEMPTS) :{APP_PARAM_RETRY_ATTEMPTS}''')
    
    
# =============================================================================
# LLM MODEL configuration    
# set and  configure the  various  models to  use  with agents 
# the  model  configuration  is  is  a  dictionary indexed by the agent name , or  agent  group ID 
#  The dictionary element  consist of model parameter  ids 
#  the model for the main  agent is  indexed by  "agent"
# =============================================================================
MODEL_CFG = {
    "agent":{
        "model":"gemini-2.5-flash-lite", 
        "retry_options": {
            "attempts":APP_PARAM_RETRY_ATTEMPTS,             }
    }

}


# =============================================================================
# SET USER-ID 
# set the  user  ID for the  application  run 
# =============================================================================
APP_PARAM_USER_ID="test01"


# =============================================================================
# SET SESSION-ID 
# set the  user  ID for the  application  run 
# =============================================================================
APP_PARAM_SESSION_ID="test01"

# =============================================================================
# FUNCTION: Agent  Runner 
# Name 
# Desc :  A Helper Function to Run Our Agents  this function is used set query an agent
# directly 
#  Inputs 
# agent: (Agent)
# query: <str> 
# session: <Session>, 
# user_id: <str:"test", 
# is_router: (bool :False)
# Outputs
#  final response  from an  agent 
# =============================================================================

async def run_agent_query(agent: Agent, query: str, session: Session, user_id: str = "test", is_router: bool = False):
    """Initializes a runner and executes a query for a given agent and session."""
    print(f"\n🚀 Running query for agent: '{agent.name}' in session: '{session.id}'...")

    runner = Runner(
        agent=agent,
        session_service=session_service,
        app_name=agent.name
    )

    final_response = ""
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=Content(parts=[Part(text=query)], role="user")
        ):
            if not is_router:
                # Let's see what the agent is thinking!
                print(f"EVENT: {event}")
            if event.is_final_response():
                final_response = event.content.parts[0].text
    except Exception as e:
        final_response = f"An error occurred: {e}"

    if not is_router:
     print("\n" + "-"*50)
     print("✅ Final Response:")
     print(final_response)
     print("-"*50 + "\n")

    return final_response

# =============================================================================
#AGENT : Main  agent    
# name : main application 
# ID :  agent
# role : main application orchestration 
# type 
# returns: 
# context State :   
# desc :  this is the main  application   agent.  It provides the  entry point  to the application. 
 
# 
# =============================================================================

agent = LlmAgent(
    name="loan_approval",
    model=Gemini(
        model=MODEL_CFG["agent"][ "model"],
        retry_options=retry_config
    ),
    description="A simple agent that can answer general questions.",
    instruction="You are a helpful assistant. Use Google Search for current info or if unsure.",
    tools=[google_search],
)
# to supoprt  compatibility with ADK run , which used  root_agent as an entry point 
root_agent=agent

print(f'''✅ Root Agent defined.
        - Model : {MODEL_CFG["agent"][ "model"]}
    ''')
 
 # =============================================================================
#CREATE SESSION 
#  create  session instance  
# =============================================================================
# TBD   add  support for  using  different  session  services including vertix , DB  
APP_PARAM_SESSION_SERVICE_TYPES ={
    "inmemory" :"InMemorySessionService",
    "vertexAI" : "VertexAiSessionService",
    "database":"DatabaseSessionService"
}
APP_PARAM_SESSION_SERVICE_IND="vertexAI"; 
def fnCreateSessionService(argSessionServiceType:str = APP_PARAM_SESSION_SERVICE_TYPES["inmemory"]):
    '''
    This fucntion  will create and  return a   session service  object. 
    It takes an argument, argSessionServiceType, that indicates the type of  session service to use. 
    The  default  service  is "in memory service"
    ''' 
    #TBD  exception handling :   add try catch.  if  failure , try database  servvice ,  if that  failes , use  in memory  service        
    # set the  target memory  service 
    vSessionService=None
    vSessionServiceType= argSessionServiceType
    match vSessionServiceType:
        case "InMemorySessionService": 
            vSessionService = InMemorySessionService()
        case "VertexAiSessionService" :
            #TBD add  suport  for  vertixAi  session  service  
            # session_service = VertexAiSessionService()
            vSessionService = InMemorySessionService()
        case "DatabaseSessionService":
            #TBD add  suport  for  vertixAi  session  service  
            # session_service = VertexAiSessionService()
            vSessionService = InMemorySessionService()
        case _ : 
            vSessionService = InMemorySessionService()
    return   vSessionService  
        
async def fnCreateSession(argSessionService, argAppName, ArgUserId, argSessionId=None)->Session:
    '''
        This function will create an  instance of  a session 
    '''
    #TBD  Exception  handling 
    #  invalide  session  service :  create a  default  session service 
    # appname not provided :  used the  root  agent  name  
    # userID not provided :  default  to  Test User 
    #  session ID  provided,  use the provided  session ID 
    vSessionService = argSessionService; 
    vSession = await vSessionService.create_session(
        app_name=argAppName,
        user_id=ArgUserId )
    return vSession
        
# set  session type 
vSessionServiceInd = APP_PARAM_SESSION_SERVICE_IND
app_sessionservice_type = APP_PARAM_SESSION_SERVICE_TYPES[vSessionServiceInd]
#TAG: ADK workaround 
if __name__ == "__main__":
    session_service = fnCreateSessionService(app_sessionservice_type)
    print(f'''✅ Created  Session Service .
        - Application Cfg: Session Service type (APP_PARAM_SESSION_SERVICE_IND): {APP_PARAM_SESSION_SERVICE_IND}
        - Session Service  : {type(session_service)}
    ''')


#create  session instance 
app_user_id = APP_PARAM_USER_ID; 
#TAG: ADK workaround 
if __name__ == "__main__":
    session = asyncio.run(fnCreateSession(argSessionService=session_service, 
        argAppName=agent.name,
        ArgUserId=app_user_id
    ))

    print(f'''✅ Created  Session instance .
        - Session Service  : {type(session_service)}
    ''')

# =============================================================================
#AGENT RUNNER Main  agent   
# configure runner for main aget 
# =============================================================================
runner=None
#TAG: ADK workaround 
if __name__ == "__main__":
    runner=InMemoryRunner(agent=agent)

    print(f'''✅ Runner for main agent  configured 
        - Runner : {type(runner)}
    ''')

# =============================================================================
# INVOKE MAIN AGENT 
#  run the main agent  
# =============================================================================
user_prompt='''What is Agent Development Kit from Google? What languages is the SDK available in?'''
#TAG: ADK workaround 
if __name__ == "__main__":
    asyncio.run(run_agent_query(agent, user_prompt, session, app_user_id))


# =============================================================================
# main application procedures and  artifacts 
# =============================================================================


# ==============================================================================
# 2. CONFIGURATION & AUTHENTICATION
# ==============================================================================

# Fetch API Key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("⚠️ WARNING: GOOGLE_API_KEY not found in .env. LLM calls will fail.")

# Initialize Google AI SDK
genai.configure(api_key=API_KEY)

# Database Config (Replace with your actuals)
PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_DATABASE = "loan_system"

# Initialize Firestore (Requires 'gcloud auth application-default login' on machine)
try:
    firestore_client = firestore.Client(project=PROJECT_ID)
except Exception as e:
    print(f"⚠️ Firestore init failed (ignoring for local test): {e}")
    firestore_client = None


# ==============================================================================
# 3. ADK COMPATIBILITY SHIMS
# ==============================================================================

class HITLRequest:
    def __init__(self, reason: str, phase: str | None = None):
        self.reason = reason
        self.phase = phase

    def __repr__(self) -> str:
        return f"HITLRequest(phase={self.phase}, reason={self.reason})"

def tool(func):
    """Simple decorator to mark functions as tools (No-op for this implementation)"""
    return func

def run_workflow(root_agent, state: dict):
    return _run_workflow(state)


# ==============================================================================
# 4. CONSTANTS
# ==============================================================================

# Model ID (Short version for AI Studio)
GEMINI_MODEL_NAME = "gemini-1.5-pro-002"

# WRAPPER: Use the Gemini class for AI Studio
# This class automatically uses the API Key configured in genai or env vars
GOOGLE_AI_MODEL = Gemini(model=GEMINI_MODEL_NAME)

PHASE_INTAKE = "intake"
PHASE_KYC = "kyc"
PHASE_LOAN = "loan"
PHASE_BUSINESS = "business"
PHASE_COMPLIANCE = "compliance"
PHASE_FINAL = "final_decision"


# ==============================================================================
# 5. FIRESTORE HELPERS (State Persistence)
# ==============================================================================

def _sanitize_state_for_firestore(state: dict) -> dict:
    """Removes binary data to prevent Firestore 1MB limit crash."""
    clean_state = copy.deepcopy(state)
    if "application" in clean_state and "documents" in clean_state["application"]:
        for doc in clean_state["application"]["documents"]:
            if "file" in doc:
                doc["file"] = "<BINARY_DATA_STRIPPED>"
    return clean_state

def save_session_state(user_id: str, session_id: str, state: dict):
    if not firestore_client: return # Skip if no DB
    
    safe_state = _sanitize_state_for_firestore(state)
    doc_ref = firestore_client.collection("sessions").document(session_id)
    doc_ref.set({
        "user_id": user_id,
        "session_id": session_id,
        "state": safe_state,
        "updated_at": datetime.datetime.utcnow().isoformat(),
    }, merge=True)

def load_session_state(session_id: str):
    if not firestore_client: return None
    
    doc_ref = firestore_client.collection("sessions").document(session_id)
    snapshot = doc_ref.get()
    return snapshot.to_dict().get("state") if snapshot.exists else None

def log_phase_audit(user_id, session_id, phase, input_payload, output_data, decision, hitl_required):
    if not firestore_client: return
    
    # Sanitize input for logs too
    safe_input = _sanitize_state_for_firestore({"app": input_payload}).get("app", {})
    
    firestore_client.collection("audit_logs").add({
        "user_id": user_id,
        "session_id": session_id,
        "phase": phase,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input": safe_input,
        "output": output_data,
        "decision": decision,
        "hitl_required": hitl_required,
    })


# ==============================================================================
# 6. MYSQL PERSISTENCE HELPERS
# ==============================================================================

def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=MYSQL_DATABASE
    )

def save_user_profile(user_profile: dict):
    # (Same implementation as before, abbreviated for brevity)
    # Ensure you handle connection errors gracefully if SQL isn't running
    try:
        conn = get_mysql_connection()
        # ... logic ...
        conn.close()
    except Exception:
        pass # Silently fail for this demo if DB is down

def save_user_document_version(user_id, document_type, file_data, metadata=None):
    try:
        conn = get_mysql_connection()
        # ... logic ...
        conn.close()
    except Exception:
        pass


# ==============================================================================
# 7. LLM TOOLS — UPDATED FOR GOOGLE AI STUDIO
# ==============================================================================

@tool
def tool_generate_embedding(text: str):
    """
    Generates embeddings using Google AI Studio (API Key).
    """
    # Use the 'models/text-embedding-004' (or 001) model available in AI Studio
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
        title="Embedding request"
    )
    return result['embedding']

@tool
def tool_rag_query(query: str):
    # Mock RAG remains same
    return [
        "Retrieved rule: Loan caps depend on income multiple.",
        "Retrieved policy: SMEs with poor cashflow require manual review."
    ]


# ==============================================================================
# 8. AGENT DEFINITIONS (Using GOOGLE_AI_MODEL)
# ==============================================================================

# 8A. INTAKE AGENT
intake_agent = LlmAgent(
    name="IntakeAgent",
    model=GOOGLE_AI_MODEL,  # <--- Using Gemini Wrapper
    include_contents="none",
    output_key="intake",
    description="Validates required documents.",
    instruction="""
    You are the Intake Agent. Validate the application: {application}
    Return JSON with 'decision' (pass/fail).
    """
)

# 8B. KYC AGENT
kyc_agent = LlmAgent(
    name="KYCAgent",
    model=GOOGLE_AI_MODEL,
    include_contents="none",
    output_key="kyc",
    description="Performs KYC checks.",
    instruction="""
    You are the KYC Agent. Perform checks on: {application}
    Return JSON with 'decision' (pass/fail).
    """
)

# 8C. LOAN AGENT
loan_eval_agent = LlmAgent(
    name="LoanEvaluationAgent",
    model=GOOGLE_AI_MODEL,
    include_contents="none",
    output_key="loan",
    description="Evaluates loan parameters.",
    instruction="""
    You are the Loan Agent. Evaluate: {application}
    Return JSON with 'decision' (pass/fail).
    """
)

# 8D. BUSINESS AGENT
business_eval_agent = LlmAgent(
    name="BusinessEvaluationAgent",
    model=GOOGLE_AI_MODEL,
    include_contents="none",
    output_key="business",
    description="Evaluates business viability.",
    instruction="""
    You are the Business Agent. Evaluate: {application}
    Return JSON with 'decision' (pass/fail).
    """
)

# 8E. COMPLIANCE AGENT
compliance_agent = LlmAgent(
    name="ComplianceAgent",
    model=GOOGLE_AI_MODEL,
    include_contents="none",
    output_key="compliance",
    description="Checks overall compliance.",
    instruction="""
    You are the Compliance Agent. Review inputs: {input_context}
    Return JSON with 'decision' (pass/fail).
    """
)

# 8F. FINAL DECISION AGENT
final_decision_agent = LlmAgent(
    name="FinalDecisionAgent",
    model=GOOGLE_AI_MODEL,
    include_contents="none",
    output_key="final_decision",
    description="Makes final decision.",
    instruction="""
    You are the Final Decision Agent. Review inputs: {input_context}
    Return JSON with 'final_decision'.
    """
)


# ==============================================================================
# 9. WORKFLOW EXECUTION
# ==============================================================================

def before_agent_callback(state, phase_key):
    pass

def after_agent_callback(state, phase_key, phase_label, input_payload, output_data):
    user_id = state["user_id"]
    session_id = state["session_id"]
    
    state[phase_key] = output_data
    save_session_state(user_id, session_id, state)
    
    decision = output_data.get("decision")
    log_phase_audit(
        user_id, session_id, phase_label, input_payload, output_data, 
        decision=decision, hitl_required=(decision == "fail")
    )

def is_phase_complete(state: dict, phase_key: str) -> bool:
    return phase_key in state and state[phase_key] is not None

def _run_workflow(state):
    # (This logic is identical to the previous robust version)
    # The only difference is it uses the Agents defined above (which use API Key)
    
    user_id = state["user_id"]
    session_id = state["session_id"]
    app = state["application"]

    # --- Pre-save Profile ---
    save_user_profile(app.get("user_profile", {}))

    # --- INTAKE ---
    if not is_phase_complete(state, "intake"):
        intake = intake_agent.run(state)
        after_agent_callback(state, "intake", "Intake", app, intake)
        if intake.get("decision") == "fail":
            return HITLRequest("Intake failed", phase="intake")
    else:
        intake = state["intake"]
        if intake.get("decision") == "fail":
            return HITLRequest("Resumed: Intake failed", phase="intake")

    # --- PARALLEL KYC/LOAN ---
    if not is_phase_complete(state, "kyc") or not is_phase_complete(state, "loan"):
        parallel_results = ParallelAgent(
            name="Parallel_KYC_Loan",
            agents=[kyc_agent, loan_eval_agent]
        ).run(state)
        
        kyc_output = parallel_results.get("kyc", {"decision": "fail"})
        loan_output = parallel_results.get("loan", {"decision": "fail"})

        after_agent_callback(state, "kyc", "KYC", app, kyc_output)
        after_agent_callback(state, "loan", "LoanEval", app, loan_output)
        
        if kyc_output.get("decision") == "fail": return HITLRequest("KYC failed", phase="kyc")
        if loan_output.get("decision") == "fail": return HITLRequest("Loan failed", phase="loan")
    else:
        kyc_output = state["kyc"]
        loan_output = state["loan"]
        if kyc_output.get("decision") == "fail": return HITLRequest("Resumed: KYC failed", phase="kyc")
        if loan_output.get("decision") == "fail": return HITLRequest("Resumed: Loan failed", phase="loan")

    # --- BUSINESS ---
    if not is_phase_complete(state, "business"):
        business = business_eval_agent.run(state)
        after_agent_callback(state, "business", "BusinessEval", app, business)
        if business.get("decision") == "fail": return HITLRequest("Business failed", phase="business")
    else:
        business = state["business"]
        if business.get("decision") == "fail": return HITLRequest("Resumed: Business failed", phase="business")

    # --- COMPLIANCE ---
    if not is_phase_complete(state, "compliance"):
        compliance_input = {"intake": intake, "kyc": kyc_output, "loan": loan_output, "business": business}
        state["input_context"] = compliance_input
        compliance = compliance_agent.run(state)
        after_agent_callback(state, "compliance", "Compliance", compliance_input, compliance)
        if compliance.get("decision") == "fail": return HITLRequest("Compliance failed", phase="compliance")
    else:
        compliance = state["compliance"]
        if compliance.get("decision") == "fail": return HITLRequest("Resumed: Compliance failed", phase="compliance")

    # --- FINAL DECISION ---
    if not is_phase_complete(state, "final_decision"):
        final_input = {
            "intake": intake, "kyc": kyc_output, "loan": loan_output, 
            "business": business, "compliance": compliance
        }
        state["input_context"] = final_input
        final = final_decision_agent.run(state)
        after_agent_callback(state, "final_decision", "FinalDecision", final_input, final)
    else:
        final = state["final_decision"]

    return final


# ==============================================================================
# 10. WORKFLOW REGISTRATION
# ==============================================================================
loan_workflow = SequentialAgent(
    name="LoanWorkflow_GoogleAI",
    description="Microfinance loan workflow using Google AI Studio (API Key).",
    run=_run_workflow
)

# ==============================================================================
# 11. MAIN (TEST HARNESS)
# ==============================================================================
if __name__ == "__main__":
    print("\n[INIT] Starting Google AI Studio Workflow...")
    
    test_state = {
        "user_id": "USER_API_KEY_TEST",
        "session_id": str(uuid.uuid4()),
        "application": {
            "id_number": "TEST-ID-99",
            "loan_amount": 1000,
            "income": 5000,
            "business_cashflow": 2000
        }
    }
    
    # Run
    try:
        result = run_workflow(loan_workflow, test_state)
        print("\n[RESULT]:", result)
    except Exception as e:
        print(f"\n[ERROR]: {e}")

