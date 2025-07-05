# Air Quality Monitoring System

This pipeline estimates PM2.5 and PM10 concentrations across India using satellite data.

## Workflow

1. **Satellite Preprocessing**: `ml_pipeline/preprocess_satellite.py`
   - Converts radiance to reflectance
   - Generates cloud mask

2. **Data Alignment**: `ml_pipeline/align_data.py`
   - Matches satellite data with ground measurements

3. **Model Training**: `ml_pipeline/train_model.py`
   - Trains Random Forest models for PM prediction

4. **Prediction**: `ml_pipeline/predict_pm.py`
   - Generates India-wide PM maps


## Setup
1. Create virtual environment:
```bash
python -m venv aq-env
source aq-env/bin/activate  # Linux/Mac
.\aq-env\Scripts\activate   # Windows


2. Install dependencies:
```bash
pip install -r requirements.txt


3. Create data folders:
mkdir -p data/satellite data/ground


''' Handoff Email
Subject: Air Quality ML Pipeline Ready for Integration '''

Hi [backend],

I've completed the ML pipeline for air quality monitoring. Here's what you'll find in the attachment:

- Complete pipeline code in `ml_pipeline/` directory
- Requirements file for dependencies
- Detailed README with setup/running instructions
- Sample data for testing

Key components:
1. Satellite data preprocessing (radiance → reflectance + cloud masking)
2. Ground data alignment
3. PM prediction model training
4. India-wide PM map generation

To test:
1. Unzip the attachment
2. Follow setup instructions in README.md
3. Run the 4 Python scripts in order

Let's schedule a quick 15-minute call this week to walk through it together. What time works for you?

Best regards,
[ml]