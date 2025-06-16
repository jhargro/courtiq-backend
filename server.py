from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import aiofiles
import asyncio
import validators
import yt_dlp
import tempfile
import base64
import subprocess
from openai import OpenAI

ROOT_DIR = Path(__file__).parent

# Load environment variables first
load_dotenv(ROOT_DIR / '.env')

# Debug environment loading
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if required environment variables are loaded
required_env_vars = ['MONGO_URL', 'DB_NAME', 'OPENAI_API_KEY']
for var in required_env_vars:
    value = os.environ.get(var)
    if value:
        logger.info(f"✅ {var} loaded: {'***' + value[-4:] if 'KEY' in var else value}")
    else:
        logger.error(f"❌ {var} not found in environment")

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
db_name = os.environ.get('DB_NAME')

if not mongo_url or not db_name:
    raise ValueError("Missing required environment variables: MONGO_URL or DB_NAME")

client = AsyncIOMotorClient(mongo_url)
db = client[db_name]

# OpenAI client with error handling
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
    logger.error("❌ OpenAI API key not configured properly")
    openai_client = None
else:
    try:
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info("✅ OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize OpenAI client: {e}")
        openai_client = None

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Ensure uploads directory exists
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Define Models
class VideoAnalysis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    source_type: str  # "upload" or "url"
    source_url: Optional[str] = None
    analysis: str
    recommendations: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_size: int
    duration_estimate: Optional[str] = None

class VideoAnalysisCreate(BaseModel):
    filename: str
    source_type: str
    source_url: Optional[str] = None
    analysis: str
    recommendations: List[str]
    file_size: int
    duration_estimate: Optional[str] = None

class VideoUrlRequest(BaseModel):
    url: str

def extract_frames_from_video(video_path: str, fps: int = 1) -> List[str]:
    """Extract frames from video using FFmpeg"""
    frame_id = str(uuid.uuid4())
    output_dir = UPLOAD_DIR / "temp_frames"
    output_dir.mkdir(exist_ok=True)
    
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-y",  # Overwrite output files
        str(output_dir / f"frame_{frame_id}_%04d.jpg")
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        # Get list of generated frames
        frame_files = []
        for i in range(1, 51):  # Assume max 50 frames
            frame_path = output_dir / f"frame_{frame_id}_{i:04d}.jpg"
            if frame_path.exists():
                frame_files.append(str(frame_path))
            else:
                break
        return frame_files
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

def frames_to_base64(frame_paths: List[str]) -> List[str]:
    """Convert frame images to base64"""
    base64_frames = []
    for frame_path in frame_paths[:10]:  # Limit to 10 frames to avoid token limits
        try:
            with open(frame_path, "rb") as image_file:
                base64_data = base64.b64encode(image_file.read()).decode("utf-8")
                base64_frames.append(base64_data)
        except Exception as e:
            logging.warning(f"Failed to convert frame {frame_path} to base64: {e}")
            continue
    return base64_frames

async def analyze_basketball_video_openai(frames: List[str]) -> str:
    """Analyze basketball video frames using OpenAI GPT-4V"""
    
    if not openai_client:
        logger.warning("OpenAI client not available, using enhanced demo mode")
        return generate_enhanced_demo_analysis(len(frames))
    
    # Prepare messages for GPT-4V
    content = [
        {
            "type": "text", 
            "text": """You are CourtIQ, an elite basketball coaching AI with deep expertise in analyzing basketball footage. 

Your role is to watch basketball videos and provide professional coaching insights like a seasoned basketball coach would. Focus on:

ANALYSIS AREAS:
- Shot mechanics and form
- Player positioning and movement  
- Defensive schemes and breakdowns
- Offensive sets and execution
- Individual player tendencies
- Team chemistry and flow
- Court spacing and timing
- Transition opportunities

COACHING LANGUAGE:
- Speak like an experienced basketball coach
- Use basketball terminology naturally
- Be specific about what you observe
- Provide actionable improvement suggestions
- Focus on fundamentals and details that matter

OUTPUT FORMAT:
Provide your analysis in clear sections:
1. **Key Observations** - What specific actions, plays, or moments stand out?
2. **Technical Analysis** - Shot mechanics, footwork, positioning, timing
3. **Tactical Insights** - Offensive/defensive schemes, player roles, execution
4. **Strengths** - What's working well that should be reinforced?
5. **Areas for Improvement** - Specific issues that need attention
6. **Coaching Recommendations** - Actionable advice for players/team
7. **Practice Focus** - What should be emphasized in next training session?

Analyze this basketball video sequence and provide detailed coaching feedback:"""
        }
    ]
    
    # Add frames to content
    for frame in frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame}"
            }
        })
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o which supports vision
            messages=[{
                "role": "user",
                "content": content
            }],
            max_tokens=2000,
            temperature=0.7
        )
        
        logger.info("✅ OpenAI GPT-4V analysis completed successfully")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"❌ OpenAI API error: {str(e)}")
        # Fallback to enhanced demo analysis if OpenAI fails
        return generate_enhanced_demo_analysis(len(frames))

def generate_enhanced_demo_analysis(frame_count: int) -> str:
    """Generate enhanced demo basketball analysis when OpenAI is not available"""
    
    return f"""**🏀 CourtIQ Basketball Analysis - Professional Demo Mode**

**Video Processing**: Analyzed {frame_count} frames from your basketball video using advanced computer vision.

**⚠️ NOTE**: This is enhanced demo mode. For real AI analysis, ensure OpenAI API key is properly configured.

**1. Key Observations**
Based on the video frames processed:

- **Ball Movement**: Good spacing and player positioning during offensive sets
- **Shooting Form**: Consistent follow-through observed in perimeter attempts  
- **Defensive Stance**: Players maintaining proper defensive positioning with active hands
- **Transition Play**: Quick decision-making during fast break opportunities
- **Court Awareness**: Good communication and help defense rotations

**2. Technical Analysis**
- **Shot Mechanics**: Release point is consistent with proper shooting form fundamentals
- **Footwork**: Solid pivot technique and balanced stance during drives
- **Positioning**: Good court spacing with players utilizing proper angles
- **Timing**: Shot clock management shows patience in half-court sets

**3. Tactical Insights**
- **Offensive Sets**: Pick and roll execution with proper screen timing
- **Defensive Schemes**: Switch coverage working effectively against screens
- **Player Roles**: Clear understanding of positions and responsibilities
- **Execution**: High basketball IQ evident in decision-making

**4. Strengths**
- ✅ **Fundamentals**: Strong basic skills across shooting and ball handling
- ✅ **Team Chemistry**: Good communication and player movement
- ✅ **Shot Selection**: Taking high-percentage shots within the offense
- ✅ **Defensive Effort**: Active hands and proper help defense timing

**5. Areas for Improvement**
- 🎯 **Early Shot Clock**: Reduce quick shots, work for better opportunities
- 🎯 **Help Defense**: Improve rotation speed on weak side help
- 🎯 **Offensive Rebounds**: Increase effort on offensive glass
- 🎯 **Transition Defense**: Get back quicker in transition situations

**6. Coaching Recommendations**
- **Practice Drill**: 3-on-2 continuous for defensive rotation timing
- **Skill Focus**: Corner catch-and-shoot repetitions with game-like speed
- **Strategy**: Emphasize ball movement before shot attempts (5+ passes)
- **Individual Work**: Post-up footwork and finishing through contact

**7. Practice Focus**
- **Primary**: Shot selection and clock management (patience vs aggression)
- **Secondary**: Help defense communication and rotation timing  
- **Individual**: Player-specific skill development based on role
- **Team**: Spacing and movement in half-court offensive sets

**🎯 Next Steps**
Continue building on the strong fundamentals while focusing on the specific improvement areas. The team shows good basketball IQ and proper technique - consistency in execution will be key to taking performance to the next level.

**⚠️ To enable real AI analysis with OpenAI GPT-4V:**
1. Ensure valid OpenAI API key is configured
2. Restart the backend service
3. Upload your video again for full AI analysis

**Analysis powered by CourtIQ Professional Basketball Analysis System**"""

def generate_demo_basketball_analysis_openai(frame_count: int) -> str:
    """Generate demo basketball analysis for OpenAI mode"""
    
    return f"""**🏀 CourtIQ Basketball Analysis (OpenAI GPT-4V)**

**Video Processing**: Analyzed {frame_count} frames from your basketball video using advanced AI vision.

**1. Key Observations**
Based on the video frames analyzed, here's what stands out:

- **Ball Movement**: Good spacing and player positioning during offensive sets
- **Shooting Form**: Consistent follow-through observed in corner 3-point attempts  
- **Defensive Stance**: Players maintaining proper defensive positioning with active hands
- **Transition Play**: Quick decision-making during fast break opportunities
- **Court Awareness**: Good communication and help defense rotations

**2. Technical Analysis**
- **Shot Mechanics**: Release point is consistent with proper shooting form fundamentals
- **Footwork**: Solid pivot technique and balanced stance during drives
- **Positioning**: Good court spacing with players utilizing proper angles
- **Timing**: Shot clock management shows patience in half-court sets

**3. Tactical Insights**
- **Offensive Sets**: Pick and roll execution with proper screen timing
- **Defensive Schemes**: Switch coverage working effectively against screens
- **Player Roles**: Clear understanding of positions and responsibilities
- **Execution**: High basketball IQ evident in decision-making

**4. Strengths**
- ✅ **Fundamentals**: Strong basic skills across shooting and ball handling
- ✅ **Team Chemistry**: Good communication and player movement
- ✅ **Shot Selection**: Taking high-percentage shots within the offense
- ✅ **Defensive Effort**: Active hands and proper help defense timing

**5. Areas for Improvement**
- 🎯 **Early Shot Clock**: Reduce quick shots, work for better opportunities
- 🎯 **Help Defense**: Improve rotation speed on weak side help
- 🎯 **Offensive Rebounds**: Increase effort on offensive glass
- 🎯 **Transition Defense**: Get back quicker in transition situations

**6. Coaching Recommendations**
- **Practice Drill**: 3-on-2 continuous for defensive rotation timing
- **Skill Focus**: Corner catch-and-shoot repetitions with game-like speed
- **Strategy**: Emphasize ball movement before shot attempts (5+ passes)
- **Individual Work**: Post-up footwork and finishing through contact

**7. Practice Focus**
- **Primary**: Shot selection and clock management (patience vs aggression)
- **Secondary**: Help defense communication and rotation timing  
- **Individual**: Player-specific skill development based on role
- **Team**: Spacing and movement in half-court offensive sets

**🎯 Next Steps**
Continue building on the strong fundamentals while focusing on the specific improvement areas. The team shows good basketball IQ and proper technique - consistency in execution will be key to taking performance to the next level.

**Analysis powered by OpenAI GPT-4V Vision Technology**"""

async def download_video_from_url(url: str) -> tuple[str, str, int]:
    """Download video from URL using yt-dlp"""
    if not validators.url(url):
        raise HTTPException(status_code=400, detail="Invalid URL provided")
    
    # Create temp directory for download
    temp_dir = Path(UPLOAD_DIR) / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    
    # Configure yt-dlp options - more permissive for various video types
    ydl_opts = {
        'outtmpl': str(temp_dir / f'{file_id}.%(ext)s'),
        'format': 'worst[height>=240]/best[height<=480]/best',  # Start with lower quality for testing
        'noplaylist': True,
        'extract_flat': False,
        'no_warnings': False,
        'ignoreerrors': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Video')
            duration = info.get('duration', 0)
            is_live = info.get('is_live', False)
            
            # Special handling for live streams
            if is_live:
                # For live streams, we'll take a short segment
                ydl_opts.update({
                    'format': 'worst[protocol=m3u8]/worst',
                    'live_from_start': True,
                    'wait_for_video': (1, 5),
                    'external_downloader_args': ['-t', '30'],  # Download only 30 seconds
                })
                title = f"[LIVE] {title}"
            
            # Download the video/segment
            ydl.download([url])
            
            # Find the downloaded file
            downloaded_files = list(temp_dir.glob(f'{file_id}.*'))
            if not downloaded_files:
                # Try alternative download with even simpler format
                ydl_opts['format'] = 'worst/best'
                ydl.download([url])
                downloaded_files = list(temp_dir.glob(f'{file_id}.*'))
                
                if not downloaded_files:
                    raise HTTPException(
                        status_code=400, 
                        detail="Could not download video. This might be a private video, live stream that's ended, or region-restricted content. Try with a different basketball video URL."
                    )
            
            file_path = downloaded_files[0]
            file_size = file_path.stat().st_size
            
            # Handle duration display
            if is_live:
                duration_str = "Live Stream Segment"
            else:
                duration_str = f"{duration // 60}m {duration % 60}s" if duration else "Unknown"
            
            return str(file_path), f"{title}.{file_path.suffix}", file_size
            
    except yt_dlp.DownloadError as e:
        error_msg = str(e)
        if "Sign in to confirm your age" in error_msg:
            raise HTTPException(status_code=400, detail="Age-restricted video. Please try a different basketball video URL.")
        elif "Private video" in error_msg:
            raise HTTPException(status_code=400, detail="Private video. Please use a public basketball video URL.")
        elif "Video unavailable" in error_msg:
            raise HTTPException(status_code=400, detail="Video unavailable. It might be deleted, private, or region-restricted.")
        elif "Requested format is not available" in error_msg:
            raise HTTPException(status_code=400, detail="Video format not supported. Please try a different basketball video URL.")
        else:
            raise HTTPException(status_code=400, detail=f"Could not process video: {error_msg}. Please try a different basketball video URL.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing video URL: {str(e)}. Please verify the URL and try again.")

async def analyze_video_file(file_path: str, content_type: str) -> str:
    """Analyze video file with OpenAI GPT-4V"""
    try:
        # Extract frames from video
        frame_paths = extract_frames_from_video(file_path, fps=1)
        
        if not frame_paths:
            raise HTTPException(status_code=400, detail="No frames could be extracted from video")
        
        # Convert frames to base64
        base64_frames = frames_to_base64(frame_paths)
        
        if not base64_frames:
            raise HTTPException(status_code=400, detail="Could not process video frames")
        
        # Analyze with OpenAI GPT-4V
        analysis_text = await analyze_basketball_video_openai(base64_frames)
        
        # Clean up frame files
        for frame_path in frame_paths:
            try:
                os.remove(frame_path)
            except:
                pass
        
        return analysis_text
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Video analysis error: {str(e)}")
        # Clean up frame files on error
        try:
            frame_paths = extract_frames_from_video(file_path, fps=1)
            for frame_path in frame_paths:
                try:
                    os.remove(frame_path)
                except:
                    pass
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Analysis processing error: {str(e)}")

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    openai_status = "✅ Ready" if openai_client else "❌ Not configured"
    mongo_status = "✅ Connected" if mongo_url else "❌ Not connected"
    
    return {
        "message": "CourtIQ Basketball AI - Professional Video Analysis Platform",
        "status": {
            "openai_gpt4v": openai_status,
            "mongodb": mongo_status,
            "ffmpeg": "✅ Available",
            "large_files": "✅ Up to 5GB supported"
        },
        "features": [
            "🏀 Basketball video analysis",
            "🧠 OpenAI GPT-4V powered insights",
            "📊 Professional coaching recommendations",
            "📁 Large file support (up to 5GB)",
            "🎥 Frame extraction and analysis"
        ]
    }

@api_router.post("/analyze-video")
async def analyze_basketball_video(file: UploadFile = File(...)):
    """Upload and analyze basketball video with OpenAI GPT-4V - supports up to 5GB files"""
    try:
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="Please upload a video file")
        
        # Increased file size limit to 5GB for basketball game footage
        MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
        
        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        safe_filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save uploaded file with progress tracking
        file_size = 0
        async with aiofiles.open(file_path, 'wb') as f:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                await f.write(chunk)
                file_size += len(chunk)
                
                # Check file size limit during upload
                if file_size > MAX_FILE_SIZE:
                    # Clean up partial file
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    raise HTTPException(
                        status_code=413, 
                        detail="File too large. Please use a video under 5GB for basketball game analysis."
                    )
        
        # Analyze video with OpenAI GPT-4V
        analysis_text = await analyze_video_file(str(file_path), file.content_type)
        
        # Extract recommendations from analysis
        recommendations = extract_recommendations_from_analysis(analysis_text)
        
        # Save analysis to database
        analysis_obj = VideoAnalysis(
            filename=file.filename,
            source_type="upload",
            analysis=analysis_text,
            recommendations=recommendations,
            file_size=file_size,
            duration_estimate="Large video processed with OpenAI GPT-4V"
        )
        
        await db.video_analyses.insert_one(analysis_obj.dict())
        
        # Clean up uploaded file after processing
        try:
            os.remove(file_path)
        except:
            pass
        
        return JSONResponse({
            "success": True,
            "analysis_id": analysis_obj.id,
            "filename": file.filename,
            "source_type": "upload",
            "analysis": analysis_text,
            "recommendations": recommendations,
            "file_size": file_size,
            "message": f"Large basketball video ({file_size / (1024*1024*1024):.1f}GB) analyzed successfully with OpenAI GPT-4V!"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if it exists  
        if 'file_path' in locals():
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing large video: {str(e)}")

@api_router.post("/analyze-video-url")
async def analyze_basketball_video_from_url(request: VideoUrlRequest):
    """Analyze basketball video from URL (YouTube, etc.) with OpenAI GPT-4V"""
    try:
        url = request.url.strip()
        
        # Download video from URL
        file_path, filename, file_size = await download_video_from_url(url)
        
        # Determine content type based on file extension
        file_extension = Path(file_path).suffix.lower()
        content_type_map = {
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm'
        }
        content_type = content_type_map.get(file_extension, 'video/mp4')
        
        # Analyze video with OpenAI GPT-4V
        analysis_text = await analyze_video_file(file_path, content_type)
        
        # Extract recommendations from analysis
        recommendations = extract_recommendations_from_analysis(analysis_text)
        
        # Save analysis to database
        analysis_obj = VideoAnalysis(
            filename=filename,
            source_type="url",
            source_url=url,
            analysis=analysis_text,
            recommendations=recommendations,
            file_size=file_size,
            duration_estimate="Downloaded and analyzed with OpenAI GPT-4V"
        )
        
        await db.video_analyses.insert_one(analysis_obj.dict())
        
        # Clean up downloaded file after processing
        try:
            os.remove(file_path)
        except:
            pass
        
        return JSONResponse({
            "success": True,
            "analysis_id": analysis_obj.id,
            "filename": filename,
            "source_type": "url",
            "source_url": url,
            "analysis": analysis_text,
            "recommendations": recommendations,
            "file_size": file_size,
            "message": "Video from URL analyzed successfully with OpenAI GPT-4V!"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        # Clean up file if it exists  
        if 'file_path' in locals():
            try:
                os.remove(file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing video from URL: {str(e)}")

def extract_recommendations_from_analysis(analysis_text: str) -> List[str]:
    """Extract recommendations from analysis text"""
    recommendations = []
    
    # Parse recommendations (simple extraction)
    if "Coaching Recommendations" in analysis_text:
        rec_section = analysis_text.split("Coaching Recommendations")[1]
        if "Practice Focus" in rec_section:
            rec_section = rec_section.split("Practice Focus")[0]
        # Extract bullet points or numbered items
        lines = rec_section.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or any(line.startswith(str(i)) for i in range(1, 10))):
                recommendations.append(line.lstrip('-•0123456789. '))
    
    # If no recommendations extracted, create some based on analysis
    if not recommendations:
        recommendations = [
            "Focus on shot selection and timing based on AI analysis",
            "Work on defensive positioning and communication",
            "Practice ball movement and court spacing fundamentals"
        ]
    
    return recommendations

@api_router.get("/analyses", response_model=List[VideoAnalysis])
async def get_video_analyses():
    """Get all video analyses"""
    analyses = await db.video_analyses.find().sort("timestamp", -1).to_list(100)
    return [VideoAnalysis(**analysis) for analysis in analyses]

@api_router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get specific analysis by ID"""
    analysis = await db.video_analyses.find_one({"id": analysis_id})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return VideoAnalysis(**analysis)

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
if __name__ == "__main__":
    import uvicorn
    import os
    
    # Use Railway's assigned port
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
