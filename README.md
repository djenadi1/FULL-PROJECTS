# FULL-PROJECTS

My machine learning projects, each going from raw data to a trained model, plus one research project. Every project has its own README with implementation details, methodology, and results.

| Project | What it does | Demo |
|---|---|---|
| [TikTok AI Content Detector](Tiktok/README.md) | Predicts whether a TikTok post is AI-generated using metadata and captions only. Trained on a sample of 99 million posts and evaluated at the real 5.8% AI rate. Includes a human-review setup that catches approximately 70% of AI-generated posts. | — |
| [Steam Game Genre Predictor](Steam/README.md) | Predicts a game's genres and store categories from its statistics, name, and description using text embeddings and a multi-label neural network. | [Streamlit app](https://full-projects-8iblvu4aktnyyaairiralk.streamlit.app/) |
| [Superstore Discount Optimizer](Superstore/README.md) | Cleans retail data using SQL, compares a PyTorch network with XGBoost, which predicts an order's profit class with 89.3% test accuracy, and recommends discount ranges for a target profit level. | [Streamlit app](https://djenadi1-full-projects-superstoreapp-nrbv0q.streamlit.app/) |
| [Satellite Image Classifier](Satellite/README.md) | Final project for the Programming in Python II course. Uses a CNN to classify satellite images into 10 land-use categories, achieving 97.3% validation accuracy. Includes a small Streamlit app. The code retains the student-style implementation of the original submission. | [Streamlit app](https://full-projects-st.streamlit.app/) |
| [PCB Defect Detector](PCBD/README.md) | Detects defects on high-resolution PCB images by splitting them into overlapping 320 px tiles, running a fine-tuned YOLO26 detector on each tile, and drawing a bounding box around every defect on the full image. Trained on DsPCBSD+ and tiles from PKU-Market-PCB (10 defect types), reaching 0.90 mAP50. | [Streamlit app](https://full-projects-fwgz2ytdquksf6mexeldqg.streamlit.app/) |
| [Alpine Snow Depth Estimation](ASDE/README.md) | Depth model for snow with a finished dataset: 1,556 Unreal Engine 5 snow images on real Swiss terrain, with exact depth, LiDAR and GPS/IMU. The current refiner cuts Depth Anything 3's AbsRel from 0.977 to 0.387 on rendered validation frames and from 0.962 to 0.854 on real mountain photos, but DA3 with a simple LiDAR scale fix still beats it on real roads. A simulated drone uses its depth to cross a valley with no terrain map. | [Hugging Face dataset](https://huggingface.co/datasets/Ab1231/alpine-snow-depth), [drone video](ASDE/videos/drone_showcase.mp4) |
| [Vehicle Speed Estimation](SPEEDESTIM/README.md) | Work in progress. Measures each car's speed from a fixed traffic camera using YOLO26-seg masks, ByteTrack and a small transformer over one second before and after each frame. The dataset is finished: 150 Unreal Engine 5 clips of freeway traffic from 150 different cameras, with masks, track IDs and the exact speed of 5,281 cars on every frame. YOLO26m-seg trained on it reaches 0.884 mask mAP50 on roads it never saw. | [Hugging Face dataset](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed), [4K demo](https://huggingface.co/datasets/Ab1231/city-sample-freeway-speed/resolve/main/showcase/showcase_4k.mp4) |

**Tech stack:** Python, PyTorch, Ultralytics YOLO, OpenCV, pandas, scikit-learn, SQL, and Streamlit.

**Mohammed Djenadi**  
BSc Artificial Intelligence, Johannes Kepler University Linz (JKU)
