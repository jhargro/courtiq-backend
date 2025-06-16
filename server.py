from fastapi import FastAPI
import os
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "CourtIQ API is working!", "port": os.environ.get("PORT", "not set")}

@app.get("/api/")
async def api_root():
    return {"message": "CourtIQ Basketball AI - Test Mode", "status": "working"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
