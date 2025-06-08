# Anti-Spoofing Detection With MobileNetV2

[**Streamlit Cloud**](-)

## Local Installation

### Option 1: Automated Setup with Conda (Recommended)

#### **Windows:**

```cmd
# Double-click deploy.bat atau run via command prompt:
deploy.bat
```

#### **Linux/macOS:**

```bash
# Make executable and run
chmod +x deploy.sh
./deploy.sh
```

### **Option 2: Manual Installation (conda)**

#### **All Platforms:**

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate anti-spoofing

# Run application
streamlit run app.py
```

#### **Alternative: Pip Installation (venv):**

```bash
# Create virtual environment
python -m venv anti-spoofing
source anti-spoofing/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

## Model Metrics

- **Accuracy**: -
- **APCER**: -
- **BPCER**: -
- **ACER**: -
