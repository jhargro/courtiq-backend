from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[
        "https://courtiq.app",
        "https://www.courtiq.app", 
        "https://court-iq-2-0-iqk3.vercel.app",
        "http://localhost:3000"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "CourtIQ API is working!", "port": os.environ.get("PORT", "not set")}

@app.get("/api/")
async def api_root():
    return {"message": "CourtIQ Basketball AI - Test Mode", "status": "working"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
