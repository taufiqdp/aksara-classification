import io
import torch
import time
import torch.nn.functional as F
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from app.model import load_model


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model, transforms = load_model("hf_hub:taufiqdp/convnext_tiny-arutala")
labels = model.default_cfg["label_names"]

model.eval()


@app.get("/")
async def root():
    return {"message": "last updated: 2025-02-05", "status": "running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
        pred = labels[pred_idx]
        prob = probs[0][pred_idx].item()

        return {"prediction": pred, "probability": prob}

    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing file {file.filename}: {e}"
        )


@app.post("/batch-predict")
async def predict(files: List[UploadFile] = File(...)):

    predictions = []
    probabilities = []
    image_tensors = []

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image_tensor = transforms(image).unsqueeze(dim=0)
            image_tensors.append(image_tensor)

        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Error processing file {file.filename}: {e}"
            )

    if image_tensors:
        batch_tensor = torch.cat(image_tensors, dim=0)
        with torch.inference_mode():
            logits = model(batch_tensor)

        predicted_indices = logits.argmax(dim=-1).tolist()
        predicted_labels = [labels[idx] for idx in predicted_indices]
        predictions.extend(predicted_labels)
        
        top_probs, _ = torch.topk(F.softmax(logits, dim=-1), k=1)
        probabilities = top_probs.squeeze().tolist()

    return {
        "predictions": predictions,
        "probabilities": probabilities,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
