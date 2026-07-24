# Myanmar Handwritting Classification
This project was developed as part of a Digital Image Processing course to demonstrate the application of image processing and deep learning techniques in recognizing Myanmar handwritten characters using an EfficientNetB0-based classification model.

# project structure
```
Myanmar-Handwriting-OCR/
│
├── app.py                                  # Streamlit web application
├── requirements.txt                        # Required Python packages
├── README.md                               # Project documentation
├── 00_Data/                                # Original handwritten image dataset
│   ├── image_1.png
│   ├── image_2.png
│   └── ...
├── 01_Preprocessing/                       # Image preprocessing scripts
│   ├── bw_convert.py                       # Convert images into black & white
│   ├── batch_process_lines.py              # use batch_process_line.py or use batch_process_web.py for cropping
│   ├── batch_process_web.py
│   ├── cropped_images/                     # Cropped single character images
│       ├── image_1_r0_c0/
│       ├── image_1_r0_c1/
│       └── ...
│   ├── group_croppedImages.py              # groupping by row & column 
│   ├── groups_cropped_images/              # groups cropped images     
│       ├── r0_c0/
│       ├── r0_c1/
│       └── ...
├── 02_Model_Training/                      
│   ├── train_model.ipynb                    # Model training notebook
│   ├── phase1_training.ipynb                # Frozen layer training
│   ├── phase2_finetuning.ipynb              # Fine-tuning process
│   │
│   ├── phase1_model.keras                   # model of phase1_trainging
│   ├── phase2_model.keras                   # Final phase2 trained model
│   └── label_encoder.pkl                    # Character label mapping
├── 03_Testing/   
│   ├── test_unseen_images                   # test for unseen images
│   ├── figures/                             # test result by unseen images
│   │   ├── prediction_result.xlsx
│   │   └── correct_predictions_bar_chart.png
├── .venv/                                  # Python virtual environment
└── models/                                 # Deployment models
    ├── phase2_model.keras
    └── label_encoder.pkl
```


# Convert the image to Black & White
    python bw_convert.py 


# Crop the 33_character_image to single_character_image 
### Run the batch processor
    python batch_process_lines.py --input ../00_Data --output ../cropped_cells --rows 7 --cols 5
### Margin-based adjustment 
    python batch_process_web.py 

### How to Use the Web Interface

    1. **Start the tool** - Run the command above
    2. **Browser opens automatically** - Shows first image with grid overlay
    3. **Adjust grid lines:**
     - Green lines = Horizontal (rows)
     - Blue lines = Vertical (columns)
      - Click ▲/▼ to move horizontal lines up/down
      - Click ◄/► to move vertical lines left/right
    4. **Save or Skip:**
     - Press **Enter** or click "Save & Next" to save cells and move to next image
     - Press **Space** or click "Skip" to skip current image
    5. **Progress tracking** - Progress bar shows how many images processed
    6. **Stop processing** - Press **Ctrl+C** in terminal when done


# Train the Model
```
    Raw Handwritten Images
          ↓
    Image Preprocessing
    (RGB Conversion + Resize)
          ↓
    Label Encoding
          ↓
    Train / Validation Split
          ↓
    Data Augmentation
          ↓
    EfficientNetB0 Transfer Learning
          ↓
    Phase 1 Training
    (Frozen Layers)
          ↓
    Phase 2 Fine-tuning
    (Unfreeze Last 40 Layers)
          ↓
    Model Evaluation
          ↓
    Saved Model + Label Encoder
          ↓
    Handwriting Recognition System(as_phase2_model.keras)
```

# Streamlit_UI

### Create virtual environment
    python -m venv .venv
    .venv\Scripts\activate

### Install requirements
    pip install -r requirements.txt

### Requirements
    streamlit==1.46.1
    tensorflow==2.19.0
    opencv-python==4.11.0.86
    numpy==2.1.3
    pillow==11.1.0
    scikit-learn==1.6.1
    pandas==2.2.3
    matplotlib==3.10.1
    streamlit-drawable-canvas==0.9.3

### Run the application
    streamlit run app.py

### How It Works
    1. User writes Myanmar handwritten characters on the drawing canvas.
    2. The input image is captured from the canvas.
    3. Image preprocessing is applied:
        - crop each character.
        - Convert image into RGB format.
        - Resize image to 224 × 224 pixels.
        - Apply EfficientNet preprocessing.
    4. The processed image is passed into the trained EfficientNetB0 deep learning model.
    5. The model predicts the handwritten character classes.
    6. The predicted class index is converted back into the original Myanmar character using Label Encoder.
    7. The recognized character is displayed in the Streamlit interface.

# License
Educational use for course project.