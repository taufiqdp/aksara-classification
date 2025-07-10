import io

import torch
import torch.nn.functional as F
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.model import load_model
from app.utils import upload_to_s3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model, transforms = load_model("hf_hub:taufiqdp/arutala-sunda-v1")
labels = model.default_cfg["label_names"]

model.eval()


@app.get("/")
async def root():
    return {"last_updated": "2025-02-05", "status": "running"}


@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode == "RGBA":
            image = image.convert("RGB")

        image_tensor = transforms(image).unsqueeze(dim=0)

        with torch.inference_mode():
            logits = model(image_tensor)

        probs = F.softmax(logits, dim=-1)

        pred_idx = logits.argmax(dim=-1)
        pred = str(labels[pred_idx])
        prob = probs[0][pred_idx].item()

        background_tasks.add_task(
            upload_to_s3,
            image_data=contents,
            filename=file.filename,
            prediction=pred,
            probability=prob,
        )

        return {"prediction": pred, "probability": prob}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing file {file.filename}: {e}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
