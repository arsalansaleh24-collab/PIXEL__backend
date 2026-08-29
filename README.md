# AI Deepfake Detector

## setup (one time)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## get the dataset

need a kaggle api key first:
1. go to kaggle.com/settings
2. scroll to API -> Create New Token (downloads kaggle.json)
3. `mkdir ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/`
4. `chmod 600 ~/.kaggle/kaggle.json`

then run:
```bash
python data/download_dataset.py
```

## train
```bash
python data/preprocess.py
python data/train.py
```

when its done you'll get `deepfake_detector_b4.pth` in the project root
