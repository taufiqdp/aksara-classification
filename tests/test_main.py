import io
import os
import tempfile

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def create_test_image(color="RGB"):
    """Creates a simple test image in memory."""
    image = Image.new(color, (100, 100), (255, 0, 0))  # Red image
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


@pytest.fixture(scope="module")
def test_image_file():
    """Creates a temporary image file for testing and yields its path.
    Deletes the file after the test.
    """
    img_byte_arr = create_test_image()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_file.write(img_byte_arr)
        tmp_file_path = tmp_file.name

    yield tmp_file_path

    os.remove(tmp_file_path)  # Clean up the temporary file


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_success(test_image_file):
    """Tests a successful prediction with a valid image file."""

    with open(test_image_file, "rb") as f:
        response = client.post(
            "/predict", files={"file": ("test_image.png", f, "image/png")}
        )

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
    assert isinstance(response.json()["prediction"], str)
    assert isinstance(response.json()["probability"], float)


def test_predict_invalid_file():
    """Tests the prediction endpoint with an invalid file type."""
    response = client.post(
        "/predict",
        files={
            "file": ("test.txt", io.BytesIO(b"some initial text data"), "text/plain")
        },
    )
    assert response.status_code == 400
    assert "Error processing file" in response.json()["detail"]


def test_predict_no_file():
    """Tests the prediction endpoint with no file provided."""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable Entity - missing file


def test_batch_predict_success(test_image_file):
    """Tests a successful batch prediction with valid image files."""
    files = []
    for i in range(2):
        files.append(("files", open(test_image_file, "rb")))

    response = client.post("/batch-predict", files=files)

    # Close the files after the request is made
    for name, file in files:
        file.close()

    assert response.status_code == 200
    assert "predictions" in response.json()
    assert "probabilities" in response.json()
    assert isinstance(response.json()["predictions"], list)
    assert isinstance(response.json()["probabilities"], list)
    assert len(response.json()["predictions"]) == 2
    assert len(response.json()["probabilities"]) == 2
    assert all(isinstance(p, str) for p in response.json()["predictions"])
    assert all(isinstance(p, float) for p in response.json()["probabilities"])


def test_batch_predict_empty_list():
    """Tests a batch prediction with an empty list of files."""
    response = client.post("/batch-predict", files=[])
    assert response.status_code == 422


def test_batch_predict_no_files():
    response = client.post("/batch-predict")
    assert response.status_code == 422
