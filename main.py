from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
from pydub import AudioSegment, silence
import whisper
import os
import uuid

app = FastAPI()

# 🗂️ Diretórios
UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

# 🎨 Templates visuais de legenda
TEMPLATES = {
    "classic": {"fontsize": 70, "color": "white", "stroke_color": "black", "position": "bottom"},
    "highlight": {"fontsize": 90, "color": "yellow", "stroke_color": "black", "position": "center"},
    "bold_red": {"fontsize": 80, "color": "red", "stroke_color": "white", "position": "bottom"},
    "shadow": {"fontsize": 70, "color": "white", "stroke_color": "black", "position": "bottom"},
    "upper_box": {"fontsize": 60, "color": "white", "stroke_color": "black", "position": "top"},
    "big_center": {"fontsize": 100, "color": "cyan", "stroke_color": "black", "position": "center"},
}

# 🏠 Página inicial
@app.get("/")
def read_root():
    content = """
    <html>
    <head><title>MicroSaas Editor</title></head>
    <body>
    <h1>🚀 MicroSaas Editor funcionando!</h1>
    <form action="/upload/" enctype="multipart/form-data" method="post">
    <input name="file" type="file">
    <input type="submit" value="Upload">
    </form>
    </body>
    </html>
    """
    return HTMLResponse(content=content)


# 📤 Upload
@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"message": "Upload feito com sucesso!", "file_path": filename}


# ✂️ Remove silêncio
@app.post("/remove_silence/")
def remove_silence(filename: str):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return {"error": "Arquivo não encontrado"}

    temp_audio = os.path.join(UPLOAD_FOLDER, "temp.wav")

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio)

    audio = AudioSegment.from_wav(temp_audio)

    chunks = silence.detect_nonsilent(audio, min_silence_len=500, silence_thresh=audio.dBFS-30)

    if not chunks:
        return {"error": "Não encontrou áudio suficiente"}

    clips = []
    for start_ms, end_ms in chunks:
        start = start_ms / 1000
        end = end_ms / 1000
        clips.append(video.subclip(start, end))

    final = concatenate_videoclips(clips)
    output_filename = f"nosilence_{filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    os.remove(temp_audio)

    return {"message": "Silêncio removido", "output_file": output_filename}


# 📝 Gera legenda
@app.post("/generate_subtitles/")
def generate_subtitles(filename: str):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return {"error": "Arquivo não encontrado"}

    temp_audio = os.path.join(UPLOAD_FOLDER, "temp.wav")

    video = VideoFileClip(video_path)
    video.audio.write_audiofile(temp_audio)

    model = whisper.load_model("base")
    result = model.transcribe(temp_audio)

    srt_file = os.path.join(UPLOAD_FOLDER, f"{filename}.srt")
    with open(srt_file, "w") as f:
        for i, segment in enumerate(result["segments"]):
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip().replace("\n", " ")
            f.write(f"{i+1}\n")
            f.write(f"{start:.2f} --> {end:.2f}\n")
            f.write(f"{text}\n\n")

    os.remove(temp_audio)

    return {"message": "Legenda gerada", "subtitle_file": f"{filename}.srt"}


# 🎨 Lista templates
@app.get("/list_templates/")
def list_templates():
    return {"templates": list(TEMPLATES.keys())}


# 🎥 Renderiza vídeo com legenda
@app.post("/render_with_subtitles/")
def render_with_subtitles(
    filename: str,
    template: str = Query("classic")
):
    if template not in TEMPLATES:
        return {"error": "Template não encontrado"}

    subtitle_path = os.path.join(UPLOAD_FOLDER, f"{filename}.srt")
    video_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(video_path) or not os.path.exists(subtitle_path):
        return {"error": "Arquivo de vídeo ou legenda não encontrado"}

    video = VideoFileClip(video_path)

    # Lê legendas
    subtitles = []
    with open(subtitle_path, "r") as f:
        blocks = f.read().strip().split("\n\n")
        for block in blocks:
            lines = block.strip().split("\n")
            if len(lines) >= 3:
                times = lines[1].split(" --> ")
                start = float(times[0])
                end = float(times[1])
                text = lines[2]
                subtitles.append((start, end, text))

    # Template config
    cfg = TEMPLATES[template]

    # Cria clips de legenda
    subtitle_clips = []
    for start, end, text in subtitles:
        txt = TextClip(
            text,
            fontsize=cfg["fontsize"],
            color=cfg["color"],
            stroke_color=cfg["stroke_color"],
            stroke_width=2,
            font="Arial-Bold"
        ).set_start(start).set_end(end)

        if cfg["position"] == "bottom":
            txt = txt.set_position(("center", video.h - 150))
        elif cfg["position"] == "top":
            txt = txt.set_position(("center", 50))
        elif cfg["position"] == "center":
            txt = txt.set_position("center")

        subtitle_clips.append(txt)

    final = CompositeVideoClip([video] + subtitle_clips)
    output_filename = f"final_{filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return {"message": "Renderização completa", "output_file": output_filename}
