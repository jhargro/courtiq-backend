from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import base64
import tempfile
import subprocess
from pathlib import Path
from openai import OpenAI

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# OpenAI client
openai_client = None
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        print("✅ OpenAI GPT-4V client initialized")
except Exception as e:
    print(f"❌ OpenAI init error: {e}")

def extract_video_frames(video_path: str) -> list:
    """Extract frames from video using FFmpeg"""
    try:
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp()
        output_pattern = os.path.join(temp_dir, "frame_%03d.jpg")
        
        # FFmpeg command to extract 3 frames
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", "fps=1/10,scale=640:360",  # 1 frame every 10 seconds, smaller size
            "-frames:v", "3",  # Only 3 frames
            "-y",  # Overwrite
            output_pattern
        ]
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return []
        
        # Read frames and convert to base64
        frames = []
        for i in range(1, 4):  # frames 001, 002, 003
            frame_path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
            if os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    frame_data = base64.b64encode(f.read()).decode()
                    frames.append(frame_data)
                # Clean up frame file
                os.remove(frame_path)
        
        # Clean up temp directory
        os.rmdir(temp_dir)
        
        return frames
        
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return []

async def analyze_with_vision(frames: list) -> str:
    """Send actual video frames to OpenAI GPT-4V for basketball analysis"""
    
    if not openai_client:
        return "OpenAI GPT-4V not available - visual analysis cannot be performed"
    
    if not frames:
        return "No video frames extracted - cannot perform visual analysis"
    
    try:
        # Prepare content with actual images
        content = [
            {
                "type": "text",
                "text": """You are analyzing ACTUAL basketball video frames. Look at these specific images and provide detailed basketball shooting analysis:

🏀 ANALYZE WHAT YOU ACTUALLY SEE:
- Shooting form and mechanics
- Hand placement and grip
- Release point and angle
- Footwork and balance
- Follow-through

📐 ASSESS RELEASE ANGLE:
- Is the shot arc too flat, optimal (45-50°), or too high?
- What adjustments would improve accuracy?

🎯 SPECIFIC CORRECTIONS:
- What exactly needs to be fixed based on what you observe?
- Provide actionable drills for improvement

Be specific about what you see in these frames."""
            }
        ]
        
        # Add actual video frames
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame}"
                }
            })
        
        # Call OpenAI GPT-4V with actual images
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # GPT-4V model that can see images
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Vision analysis error: {str(e)}"

@app.get("/")
async def root():
    return {"message": "CourtIQ Basketball AI Ready", "status": "working", "cors": "enabled"}

@app.get("/api/")
async def api_root():
    openai_status = "✅ GPT-4V Ready" if openai_client else "❌ Not configured"
    return {
        "message": "CourtIQ Basketball AI - REAL Video Analysis Platform",
        "status": {
            "openai_gpt4v": openai_status,
            "video_processing": "✅ FFmpeg frame extraction",
            "visual_analysis": "✅ Real image analysis",
            "cors": "✅ Enabled"
        },
        "features": [
            "🏀 REAL basketball video analysis",
            "🧠 OpenAI GPT-4V visual processing",
            "📐 Actual release angle assessment",
            "🎯 Visual shooting form corrections",
            "👁️ Frame-by-frame analysis"
        ]
    }

@app.post("/api/analyze-video")
async def analyze_basketball_video(file: UploadFile = File(...)):
    """REAL basketball video analysis with actual frame processing"""
    try:
        if not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Please upload a video file")
        
        # Read and save video file
        file_content = await file.read()
        file_size = len(file_content)
        
        # Size limit
        if file_size > 50 * 1024 * 1024:  # 50MB
            raise HTTPException(status_code=413, detail="File too large. Please use videos under 50MB.")
        
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(file_content)
            temp_video_path = temp_file.name
        
        try:
            # Extract actual frames from video
            print("🎬 Extracting frames from video...")
            frames = extract_video_frames(temp_video_path)
            
            if frames:
                print(f"✅ Extracted {len(frames)} frames for analysis")
                
                # Analyze with OpenAI GPT-4V using actual images
                analysis = await analyze_with_vision(frames)
                
                # Format the analysis
                formatted_analysis = f"""🏀 BASKETBALL SHOT ANALYSIS (OpenAI GPT-4V - REAL VISION)

{analysis}

✅ Analysis based on ACTUAL video frames
🎬 Processed {len(frames)} frames from your video
👁️ Visual analysis completed with OpenAI GPT-4V"""
                
                recommendations = [
                    "Practice the specific corrections identified in your video",
                    "Focus on the release angle adjustments mentioned",
                    "Work on the shooting form issues observed"
                ]
                
            else:
                formatted_analysis = """🏀 BASKETBALL SHOT ANALYSIS

❌ TECHNICAL ISSUE: Could not extract frames from video
- Video format may not be supported
- Try uploading MP4, MOV, or AVI format
- Ensure video contains clear basketball footage

🔧 TROUBLESHOOTING:
- Re-upload in MP4 format
- Ensure file is under 50MB
- Video should show clear shooting motion

❌ Visual analysis not possible without extractable frames"""
                
                recommendations = [
                    "Try re-uploading video in MP4 format",
                    "Ensure video shows clear basketball shooting",
                    "Contact support if issue persists"
                ]
        
        finally:
            # Clean up temp video file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
        return JSONResponse({
            "success": True,
            "analysis_id": "visual-analysis",
            "filename": file.filename,
            "analysis": formatted_analysis,
            "recommendations": recommendations,
            "file_size": file_size,
            "frames_processed": len(frames) if frames else 0,
            "message": f"Video analysis complete - {len(frames) if frames else 0} frames processed"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
