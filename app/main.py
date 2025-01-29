import io
import torch

from fastapi import FastAPI, UploadFile, File
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
    return {"message": "Arutala....."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_tensor = transforms(image).unsqueeze(dim=0)

    with torch.inference_mode():
        logits = model(image_tensor)

    pred = labels[logits.argmax(dim=-1)]

    return {"prediction": pred}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
