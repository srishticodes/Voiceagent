c# Deployment Guide for Streamlit Cloud

## Prerequisites

1. **GitHub Repository**: Push your code to GitHub
2. **Streamlit Cloud Account**: Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Google Cloud Account** (optional): For TTS functionality

## Files Required for Deployment

### Essential Files
- `streamlit_app.py` - Main Streamlit application
- `walmart_assistant.py` - Core assistant logic
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

### Data Files
- `users.csv` - User accounts
- `user_addresses.csv` - User addresses
- `user_payment_methods.csv` - Payment methods
- `walmart_inventory.csv` - Product catalog
- `user_orders.csv` - Order history (optional, created automatically)
- `user_carts.csv` - Shopping carts (optional, created automatically)

### Configuration Files
- `.streamlit/secrets.toml` - Streamlit secrets configuration
- `.gitignore` - Git ignore rules

## Step-by-Step Deployment

### 1. Prepare Your Repository

```bash
# Ensure all files are committed
git add .
git commit -m "Prepare for Streamlit deployment"
git push origin main
```

### 2. Configure Secrets (Optional - for TTS)

If using Google Cloud TTS, add your credentials to `.streamlit/secrets.toml`:

```toml
[google_cloud]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
```

### 3. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Set the following:
   - **Repository**: `your-username/Voice-AI-Agent`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click "Deploy!"

### 4. Configure Secrets in Streamlit Cloud

1. In your Streamlit Cloud dashboard, click on your app
2. Go to "Settings" → "Secrets"
3. Paste your secrets configuration:

```toml
[google_cloud]
type = "service_account"
project_id = "your-project-id"
# ... rest of your credentials
```

## Troubleshooting

### Common Issues

**App won't start**
- Check `requirements.txt` for correct dependencies
- Ensure all required CSV files are present
- Check Streamlit logs for specific errors

**TTS not working**
- Verify Google Cloud credentials in secrets
- Check that the service account has TTS permissions
- Fallback TTS (pyttsx3) will be used if Google Cloud fails

**Models not loading**
- Ollama models may not work on Streamlit Cloud
- Consider using alternative embedding models
- You may need to modify the code for cloud-compatible models

**Large file issues**
- `walmart_inventory.csv` is 233KB - should be fine
- If vector database is too large, consider reducing product count
- Use `.gitignore` to exclude large model files

### Performance Optimization

1. **Reduce model size**: Use smaller AI models
2. **Cache data**: Use `@st.cache_data` for CSV loading
3. **Optimize dependencies**: Remove unused packages from requirements.txt
4. **Database size**: Consider reducing product inventory for demo

## Alternative Deployment Options

### Local Development
```bash
streamlit run streamlit_app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Heroku Deployment
1. Add `Procfile`:
```
web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

## Post-Deployment

### Testing
1. Test user login (U001, U002)
2. Test product search
3. Test cart functionality
4. Test order placement
5. Verify TTS functionality

### Monitoring
- Check Streamlit Cloud logs for errors
- Monitor app performance
- Track user interactions

### Updates
```bash
git add .
git commit -m "Update features"
git push origin main
# Streamlit Cloud will auto-deploy
```

## Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Create issues in your repository 

## The Problem: **Python 3.10+ Required**

**Found in `walmart_assistant.py` line 1181:**
```python
target_order: pd.DataFrame | None = None
```

This line uses **pipe union type annotation syntax** (`|`) which was introduced in **Python 3.10**. This is incompatible with older Python versions.

## Package Version Issues

Additionally, some packages in `requirements.txt` have high version requirements:

- `streamlit==1.28.1` - requires Python 3.8+
- `pandas==2.0.3` - requires Python 3.8+
- `transformers==4.35.0` - may require Python 3.8+
- `torch==2.0.1` - requires Python 3.8+

## The Fix

**Option 1: Change the type annotation (Recommended for compatibility)**
```python
<code_block_to_apply_changes_from>
from typing import Optional, Union
target_order: Optional[pd.DataFrame] = None
# OR
target_order: Union[pd.DataFrame, None] = None
```

**Option 2: Update project requirements**
Update the documentation to specify Python 3.10+ minimum requirement.

## Current Streamlit Cloud Compatibility

Streamlit Cloud typically runs Python 3.9 by default, which **does not support** the pipe union syntax (`|`). This is why you're getting deployment errors.

## Recommended Action

I recommend **Option 1** - updating the type annotation to be compatible with Python 3.8+ since:
1. It maintains broader compatibility
2. Works with current Streamlit Cloud infrastructure
3. Doesn't require users to upgrade Python versions

The documentation says "Python 3.8+" but the actual code requires **Python 3.10+** due to that one line using modern union type syntax. 