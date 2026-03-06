import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io

# Import our engines
from data_cleaner_api import run_cleaning_pipeline
from eda_engine import EDAEngine
from ai_engine import AIEngine

app = FastAPI(title="DataLens AI Platform")

# Ensure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("eda_charts", exist_ok=True)
os.makedirs("charts", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Expose chart directories to frontend
app.mount("/eda_charts_img", StaticFiles(directory="eda_charts"), name="eda_charts_img")
app.mount("/charts_img", StaticFiles(directory="charts"), name="charts_img")

# Serve the main page
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# In-memory session store (development only)
SESSIONS = {}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files supported.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Original shapes
        orig_rows, orig_cols = df.shape
        raw_preview = df.head(5).fillna("NaN").to_dict(orient="records")
        
        # Run generic data cleaner
        cleaned_df = run_cleaning_pipeline(df)
        clean_rows, clean_cols = cleaned_df.shape
        clean_preview = cleaned_df.head(5).fillna("NaN").to_dict(orient="records")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Store df and init tools
        SESSIONS[session_id] = {
            "df": cleaned_df,
            "filename": file.filename,
            "eda_insights": None
        }
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "summary": {
                "original_rows": orig_rows,
                "original_cols": orig_cols,
                "cleaned_rows": clean_rows,
                "cleaned_cols": clean_cols
            },
            "columns": list(cleaned_df.columns),
            "dtypes": {k: str(v) for k, v in cleaned_df.dtypes.items()},
            "raw_preview": raw_preview,
            "clean_preview": clean_preview
        }
    except Exception as e:
        raise HTTPException(500, f"Upload error: {str(e)}")

@app.get("/api/eda/{session_id}")
async def get_eda(session_id: str):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
        
    session = SESSIONS[session_id]
    
    # Return from cache if we already ran it
    if session["eda_insights"]:
        return session["eda_insights"]
        
    try:
        engine = EDAEngine(session["df"])
        insights = engine.run_full_eda()
        
        # Cache it for reuse (like AI Engine)
        session["eda_insights"] = insights
        return insights
    except Exception as e:
        raise HTTPException(500, f"EDA generated error: {str(e)}")

@app.get("/api/eda_charts/{session_id}")
async def get_eda_charts(session_id: str):
    import eda_engine
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
        
    try:
        # Clear previous charts list in module
        eda_engine.saved_charts = []
        
        engine = EDAEngine(SESSIONS[session_id]["df"])
        engine.run_dashboard()
        
        return {"charts": eda_engine.saved_charts.copy()}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Dashboard error: {str(e)}")

@app.post("/api/ask/{session_id}")
async def ask_question(session_id: str, question: str = Form(...)):
    if session_id not in SESSIONS:
        raise HTTPException(404, "Session not found")
        
    session = SESSIONS[session_id]
    try:
        engine = AIEngine(session["df"], eda_insights=session["eda_insights"])
        result = engine.ask(question)
        
        # Make the chart path relative for frontend serving
        chart_url = None
        if result.get("chart"):
            chart_filename = os.path.basename(result["chart"])
            chart_url = f"/charts_img/{chart_filename}"
            
        return {
            "answer": result["answer"],
            "chart": chart_url,
            "tool_used": result["tool_used"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"AI error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
