# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBasic, HTTPBasicCredentials
# from pydantic import BaseModel, Field
# import json
# import os
# import traceback
# from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator import Orchestrator

# app = FastAPI()
# security = HTTPBasic()

# class OrchestratorManager:
#     def __init__(self):
#         self.orchestrator = None

#     def initialize(self, file_path: str):
#         """Initializes the orchestrator with the given file path."""
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#         self.orchestrator = Orchestrator(file_path)

#     def run_computation(self, input_value: float, model_to_run: str):
#         """Runs the computation using the initialized orchestrator."""
#         if self.orchestrator is None:
#             raise RuntimeError("Orchestrator not initialized")
#         self.orchestrator.run(input_value, model_to_run)

#     def get_json_content(self, file_path: str):
#         """Reads and returns JSON content from the specified file path."""
#         if not os.path.exists(file_path):
#             raise FileNotFoundError(f"File not found: {file_path}")
#         with open(file_path, 'r') as file:
#             return json.load(file)

# orchestrator_manager = OrchestratorManager()

# class Parameters(BaseModel):
#     E: float
#     model_to_run: str = Field(..., alias='model_to_run')

#     class Config:
#         protected_namespaces = ()

# def authenticate_user(credentials: HTTPBasicCredentials = Depends(security)):
#     if credentials.username != "user" or credentials.password != "password":
#         raise HTTPException(status_code=401, detail="Invalid credentials")

# @app.post("/initialize_orchestrator")
# async def initialize_orchestrator(credentials: HTTPBasicCredentials = Depends(authenticate_user)):
#     file_path = "/home/msoenmez/Desktop/NibelungenbrueckeDemonstrator/use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
#     try:
#         orchestrator_manager.initialize(file_path)
#         return {"message": "Orchestrator initialized successfully"}
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         raise HTTPException(status_code=500, detail=f"Initialization error: {str(e)}\n{error_trace}")

# @app.post("/run_computation")
# async def run_computation(params: Parameters, credentials: HTTPBasicCredentials = Depends(authenticate_user)):
#     try:
#         orchestrator_manager.run_computation(params.E, params.model_to_run)
#         json_file_path = "/home/msoenmez/Desktop/NibelungenbrueckeDemonstrator/nibelungenbruecke/scripts/digital_twin_orchestrator/output_data.json"
#         json_content = orchestrator_manager.get_json_content(json_file_path)
#         return {"json_content": json_content}
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         raise HTTPException(status_code=500, detail=f"Computation error: {str(e)}\n{error_trace}")

#%%

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
import traceback
from nibelungenbruecke.scripts.digital_twin_orchestrator.orchestrator import Orchestrator

app = FastAPI()

class OrchestratorManager:
    def __init__(self):
        self.orchestrator = None

    def initialize(self, file_path: str):
        """Initializes the orchestrator with the given file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        self.orchestrator = Orchestrator(file_path)

    def run_computation(self, input_value: float, model_to_run: str):
        """Runs the computation using the initialized orchestrator."""
        if self.orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")
        self.orchestrator.run(input_value, model_to_run)

    def get_json_content(self, file_path: str):
        """Reads and returns JSON content from the specified file path."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        with open(file_path, 'r') as file:
            return json.load(file)

orchestrator_manager = OrchestratorManager()

class Parameters(BaseModel):
    E: float
    model_to_run: str = Field(..., alias='model_to_run')

    class Config:
        protected_namespaces = ()

@app.post("/initialize_orchestrator")
async def initialize_orchestrator():
    file_path = "/home/msoenmez/Desktop/NibelungenbrueckeDemonstrator/use_cases/nibelungenbruecke_demonstrator_self_weight_fenicsxconcrete/input/settings/digital_twin_default_parameters.json"
    try:
        orchestrator_manager.initialize(file_path)
        return {"message": "Orchestrator initialized successfully"}
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Initialization error: {str(e)}\n{error_trace}")

@app.post("/run_computation")
async def run_computation(params: Parameters):
    try:
        orchestrator_manager.run_computation(params.E, params.model_to_run)
        json_file_path = "/home/msoenmez/Desktop/NibelungenbrueckeDemonstrator/nibelungenbruecke/scripts/digital_twin_orchestrator/output_data.json"
        json_content = orchestrator_manager.get_json_content(json_file_path)
        return {"json_content": json_content}
    except Exception as e:
        error_trace = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Computation error: {str(e)}\n{error_trace}")
