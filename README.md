# 🎭 Multi-Modal Sarcasm Detection System (With Batch Ingestion Engine)

An advanced, production-grade **Flask Web Server Pipeline** designed to identify and classify sarcasm across social media micro-blogs. This framework features a custom dual-stream pipeline that runs predictive text analytics on **RoBERTa-Base** while extracting and tracking deep contextual cues using an optimized **Twitter-RoBERTa-Emoji-BERT** model.

---

## 🌌 Core Features & Architecture

* **Multi-Modal Tensor Merging:** Extracts dual $L_2$-normalized feature vectors (768-dimensions each) from standard text layers and structural emojis to form a unified 1536-dimensional inference matrix.
* **On-the-Fly Normalization:** Cleans out heavy tracking URLs and system characters automatically while safely keeping `#hashtags` and `@mentions` intact.
* **Memory-Optimized Processing:** Streamlines file reading operations entirely inside the server's memory, running batch predictions on text datasets without cluttering local disk storage.
* **Smart Device Mapping:** Auto-detects configuration properties to map heavy calculations onto **NVIDIA CUDA** cores whenever available, seamlessly shifting back to local CPU pipelines if needed.

```text
               ┌──► Data Sanitization ──► RoBERTa Base ──► Text Tensor (768d) ──┐
[Raw Text Line]┤                                                                ├──► [1536d Merged Vector] ──► 3-Layer DNN ──► Sarcastic?
               └──► Emoji Extraction  ──► Emoji-BERT  ──► Emoji Tensor (768d) ──┘

sarcasm-detection-hub/
│
├── app.py                         # Unified Flask server engine code
├── sarcasm_detection_model.pth    # Trained custom DNN classification weights file
├── README.md                      # Repository production manual docs
│
└── templates/
    └── index.html                 # UI Dashboard featuring text & bulk upload fields

    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install flask transformers scikit-learn
python app.py
* Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
 * Debug mode: on
 curl -X POST [http://127.0.0.1:5000/predict](http://127.0.0.1:5000/predict) \
     -H "Content-Type: application/json" \
     -d '{"text": "Oh great, another server crash right before deployment launch 🙃"}'
