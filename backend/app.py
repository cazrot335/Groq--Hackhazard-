import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import uvicorn
from paddleocr import PaddleOCR
import speech_recognition as sr
from gtts import gTTS
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import tempfile

# Load environment variables
load_dotenv()

app = FastAPI()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
ocr = PaddleOCR(use_angle_cls=True, lang='en')
recognizer = sr.Recognizer()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Replace "*" with specific origins if needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# File Storage Directory
WORKSPACE_DIR = "workspace"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# -------------------- MODELS --------------------

class CodeAnalysisRequest(BaseModel):
    code: str
    model: str = "llama-3.3-70b-versatile"
    filename: str

class ContentGenerationRequest(BaseModel):
    prompt: str
    content_type: str = "file"

class DocumentationRequest(BaseModel):
    code: str

class ChatRequest(BaseModel):
    message: str

class CreateFileRequest(BaseModel):
    filename: str
    content: str = ""

class CreateFolderRequest(BaseModel):
    folderName: str

class UpdateFileRequest(BaseModel):
    filename: str
    content: str

class ValidateCodeRequest(BaseModel):
    filename: str
    content: str

# -------------------- ENDPOINTS --------------------

@app.get("/files")
def list_files():
    """
    List all files in the workspace directory.
    """
    files = os.listdir(WORKSPACE_DIR)
    return {"files": files}

@app.post("/create-file")
def create_file(request: CreateFileRequest):
    """
    Create a new file in the workspace directory.
    """
    file_path = os.path.join(WORKSPACE_DIR, request.filename)
    if os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="File already exists.")
    with open(file_path, "w") as f:
        f.write(request.content)
    return {"message": f"{request.filename} created successfully."}

@app.post("/create-folder")
def create_folder(request: CreateFolderRequest):
    """
    Create a new folder in the workspace directory.
    """
    folder_path = os.path.join(WORKSPACE_DIR, request.folderName)
    if os.path.exists(folder_path):
        raise HTTPException(status_code=400, detail="Folder already exists.")
    os.makedirs(folder_path)
    return {"message": f"Folder '{request.folderName}' created successfully."}

@app.get("/get-file")
def get_file(filename: str):
    """
    Retrieve the content of a file from the workspace directory.
    """
    file_path = os.path.join(WORKSPACE_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    with open(file_path, "r") as f:
        content = f.read()
    return {"filename": filename, "content": content}

@app.put("/update-file")
def update_file(request: UpdateFileRequest):
    """
    Update the content of a file in the workspace directory.
    """
    file_path = os.path.join(WORKSPACE_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    with open(file_path, "w") as f:
        f.write(request.content)
    return {"message": f"Content of '{request.filename}' updated successfully."}

@app.post("/validate")
def validate_code(request: ValidateCodeRequest):
    """
    Validate the syntax of a code file based on its file extension.
    """
    if request.filename.endswith(".py"):
        # Python syntax validation
        try:
            compile(request.content, request.filename, 'exec')
            return {"status": "valid", "output": "No syntax errors found."}
        except SyntaxError as e:
            return {
                "status": "invalid",
                "output": f"File \"{request.filename}\", line {e.lineno}\n    {e.text.strip()}\n{e.msg}: {e.offset}"
            }

    elif request.filename.endswith(".js"):
        # JavaScript syntax validation using Node.js
        try:
            with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as temp_file:
                temp_file.write(request.content.encode())
                temp_file_path = temp_file.name

            result = subprocess.run(
                ["node", "--check", temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                return {"status": "invalid", "output": result.stderr}
            return {"status": "valid", "output": "No syntax errors found."}
        except Exception as e:
            return {"status": "error", "output": str(e)}
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    elif request.filename.endswith(".cpp"):
        # C++ syntax validation using g++
        try:
            with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as temp_file:
                temp_file.write(request.content.encode())
                temp_file_path = temp_file.name

            result = subprocess.run(
                ["g++", "-fsyntax-only", temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                return {"status": "invalid", "output": result.stderr}
            return {"status": "valid", "output": "No syntax errors found."}
        except Exception as e:
            return {"status": "error", "output": str(e)}
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    elif request.filename.endswith(".java"):
        # Java syntax validation using javac
        try:
            with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as temp_file:
                temp_file.write(request.content.encode())
                temp_file_path = temp_file.name

            result = subprocess.run(
                ["javac", temp_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                return {"status": "invalid", "output": result.stderr}
            return {"status": "valid", "output": "No syntax errors found."}
        except Exception as e:
            return {"status": "error", "output": str(e)}
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    return {"status": "unknown file type", "output": "Unsupported file type for validation."}

@app.post("/code-analysis")
def analyze_code(request: CodeAnalysisRequest):
    """
    Analyze the provided code and return insights or suggestions.
    If the code compiles successfully, execute it and return the output.
    Otherwise, return the compilation/runtime errors.
    Supports Python, JavaScript, C++, and Java.
    """
    try:
        code = request.code
        filename = request.filename
        simulated_input = "madam\n"  # Example input for user input simulation
        output = None
        errors = None

        # Python Code Execution
        if filename.endswith(".py"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
                    temp_file.write(code.encode())
                    temp_file_path = temp_file.name

                result = subprocess.run(
                    ["python", temp_file_path],
                    input=simulated_input,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode != 0:
                    errors = result.stderr
                else:
                    output = result.stdout
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        # JavaScript Code Execution
        elif filename.endswith(".js"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as temp_file:
                    temp_file.write(code.encode())
                    temp_file_path = temp_file.name

                result = subprocess.run(
                    ["node", temp_file_path],
                    input=simulated_input,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if result.returncode != 0:
                    errors = result.stderr
                else:
                    output = result.stdout
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        # C++ Code Compilation and Execution
        elif filename.endswith(".cpp"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as temp_file:
                    temp_file.write(code.encode())
                    temp_file_path = temp_file.name
                    executable_path = temp_file_path.replace(".cpp", "")

                # Compile the C++ code
                compile_result = subprocess.run(
                    ["g++", temp_file_path, "-o", executable_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if compile_result.returncode != 0:
                    errors = compile_result.stderr
                else:
                    # Run the compiled executable
                    run_result = subprocess.run(
                        [executable_path],
                        input=simulated_input,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if run_result.returncode != 0:
                        errors = run_result.stderr
                    else:
                        output = run_result.stdout
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                if os.exists(executable_path):
                    os.remove(executable_path)

        # Java Code Compilation and Execution
        elif filename.endswith(".java"):
            try:
                with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as temp_file:
                    temp_file.write(code.encode())
                    temp_file_path = temp_file.name
                    class_name = os.path.basename(temp_file_path).replace(".java", "")

                # Compile the Java code
                compile_result = subprocess.run(
                    ["javac", temp_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                if compile_result.returncode != 0:
                    errors = compile_result.stderr
                else:
                    # Run the compiled Java class
                    run_result = subprocess.run(
                        ["java", "-cp", os.path.dirname(temp_file_path), class_name],
                        input=simulated_input,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    if run_result.returncode != 0:
                        errors = run_result.stderr
                    else:
                        output = run_result.stdout
            finally:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                class_file_path = temp_file_path.replace(".java", ".class")
                if os.path.exists(class_file_path):
                    os.remove(class_file_path)

        # Unsupported File Type
        else:
            return {"status": "error", "message": "Unsupported file type for code analysis."}

        # Return the results
        if errors:
            return {"status": "error", "filename": filename, "errors": errors}
        else:
            return {"status": "success", "filename": filename, "output": output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-content")
def generate_content(request: ContentGenerationRequest):
    try:
        completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Generate a {request.content_type} based on the following prompt:\n\n{request.prompt}"
            }],
            model="llama-3.3-70b-versatile"
        )
        return {"content": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/image-analysis")
async def analyze_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("temp_img.png", "wb") as f:
            f.write(contents)
        result = ocr.ocr("temp_img.png", cls=True)
        extracted_text = " ".join([line[1][0] for block in result for line in block])
        
        completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Here is some code extracted from an image:\n\n{extracted_text}\n\nProvide insights or suggestions."
            }],
            model="llama-3.3-70b-versatile"
        )
        return {
            "extracted_text": extracted_text,
            "ai_suggestions": completion.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return {"transcription": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/text-to-speech")
def text_to_speech(text: str):
    try:
        tts = gTTS(text)
        file_path = "response.mp3"
        tts.save(file_path)
        return {"audio_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-docs")
def generate_docs(request: DocumentationRequest):
    try:
        completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": f"Generate docstrings for this code:\n\n{request.code}"
            }],
            model="llama-3.3-70b-versatile"
        )
        return {"docstrings": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-assistant")
def chat_assistant(request: ChatRequest):
    try:
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": request.message}],
            model="llama-3.3-70b-versatile"
        )
        return {"reply": completion.choices[0].message.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- MAIN --------------------

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
