# LeafLens — working AI plant-leaf disease screening website

This project keeps the LeafLens single-page design and adds a Flask API plus a pretrained PlantVillage image classifier. It is a student prototype, not a substitute for an agricultural diagnosis.

## Run on a computer

1. Install Python 3.10 or 3.11 (64-bit).
2. Extract this ZIP.
3. Open a terminal in the extracted `LeafLens_Working_Website` folder.
4. Run `python -m venv .venv`, then activate it:
   - Windows: `.venv\\Scripts\\activate`
   - macOS/Linux: `source .venv/bin/activate`
5. Run `pip install -r requirements.txt`.
6. Run `python app.py`.
7. Open `http://127.0.0.1:5000` in your browser.

On first use, the backend downloads the model (~39 MB) and its class-name list from Hugging Face. Keep internet access available for the first launch. Later runs use the downloaded files in `model/`.

On Windows, `run_windows.bat` automates setup. On macOS/Linux, use `bash run_mac_linux.sh`.

## Put it online

This project includes `render.yaml` and `Procfile` as deployment starting points for a Python web host such as Render. Push the extracted folder to a Git repository, create a web service from it, and allow the build and first model download to finish. Free-host limits, memory, build time, and outbound network restrictions can affect deployment. The model download happens at first prediction, not during the build.

## What it does

- JPG, PNG and WEBP upload (10 MB limit)
- Image preview
- Real model inference through `/predict`
- Predicted class and model confidence score
- General cautious guidance and low-confidence warning
- Responsive phone-friendly layout

## Important limitations

- Model classes are from the PlantVillage dataset (38 classes). The model is not guaranteed to recognize every plant or disease.
- Training images are often cleaner than real farm photos. A high confidence score is not proof of correctness.
- Guidance is general and deliberately avoids prescribing a pesticide from an image alone.
- Use a local agricultural expert to verify before treatment.

## Model source

Model and class names are downloaded at first run from:
https://huggingface.co/Nefflymicn/PlantVillage-plant-disease-detection

Check the model card and license before redistributing or using this project commercially.
