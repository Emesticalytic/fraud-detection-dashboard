# Streamlit Cloud Deployment Guide
## Deploy Your Fraud Detection Dashboard in 5 Minutes

---

## 🚀 Quick Deployment Steps

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Add fraud detection dashboard with Streamlit"

# Create GitHub repo and push
# Go to github.com and create a new repository named "fraud-detection-dashboard"

# Add remote and push
git remote add origin https://github.com/YOUR_USERNAME/fraud-detection-dashboard.git
git branch -M main
git push -u origin main
```

---

### Step 2: Deploy to Streamlit Cloud

1. **Visit:** https://streamlit.io/cloud
2. **Sign in** with your GitHub account
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/fraud-detection-dashboard`
5. **Main file path:** `streamlit_app.py`
6. Click **"Deploy"**

**That's it!** Your app will be live at: `https://YOUR-APP-NAME.streamlit.app`

---

## ⚠️ Important: Handling Data Files

Since the dataset (creditcard.csv) is too large for GitHub, you have 3 options:

### Option 1: Use Sample Data (Recommended for Demo)

Create a sample dataset generator that Streamlit Cloud will use:

```python
# Add to streamlit_app.py at the top
import os
if not os.path.exists('data/creditcard.csv'):
    st.warning("Using demo mode with sample data")
    # Generate sample data or use cached predictions
```

### Option 2: Download from Kaggle on Deployment

Add this to `streamlit_app.py`:

```python
import os
if not os.path.exists('data/creditcard.csv'):
    os.system('kaggle datasets download -d mlg-ulb/creditcardfraud')
    os.system('unzip creditcardfraud.zip -d data/')
```

Then add Kaggle API credentials to Streamlit Cloud secrets.

### Option 3: Use Pre-computed Results Only

The dashboard can run using only the `.pkl` files (y_test, y_pred, y_prob, feature_importance) which are small enough for GitHub.

---

## 📦 Files Needed for Deployment

✅ **Must Have:**
- `streamlit_app.py` - Main app
- `requirements.txt` - Dependencies
- `README.md` - Project description
- `.gitignore` - Exclude large files

✅ **Optional:**
- `prepare_dashboard_data.py` - Data prep script
- `data/*.pkl` - Pre-computed results (small, can commit)
- `models/*.pkl` - Trained model (if <100MB)

❌ **Exclude (too large):**
- `data/creditcard.csv` - 150MB dataset
- `.venv/` - Virtual environment
- Large Jupyter notebooks

---

## 🔒 Security: Adding Secrets

If you need API keys (like Kaggle), add them to Streamlit Cloud:

1. Go to your app settings
2. Click **"Secrets"**
3. Add your secrets in TOML format:

```toml
[kaggle]
username = "your_username"
key = "your_api_key"
```

Access in your app:
```python
import streamlit as st
kaggle_user = st.secrets["kaggle"]["username"]
```

---

## 🎨 Customization Before Deploy

### Update README.md

Make sure your README.md has a good description for GitHub visitors.

### Add App Icon

Create `streamlit_app.py` config at the top:
```python
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution:** Make sure all imports are in `requirements.txt`

### Issue: "Data file not found"
**Solution:** Use one of the 3 options above for handling large data

### Issue: "App keeps crashing"
**Solution:** Check Streamlit Cloud logs for error messages

---

## 🚀 After Deployment

Once deployed, you'll get:

✅ **Free public URL** (e.g., `fraud-detection.streamlit.app`)
✅ **Auto-updates** when you push to GitHub
✅ **Free hosting** (no credit card required)
✅ **SSL/HTTPS** automatic
✅ **Share with anyone** via URL

---

## 📊 Alternative: Use Demo Mode

If you want to deploy immediately without the full dataset, I can modify the app to use sample/demo data. This will:

- Generate synthetic fraud patterns
- Use pre-computed predictions
- Show full dashboard functionality
- Deploy in 2 minutes

Would you like me to create the demo mode version?

---

*Deployment Guide - January 2026*
