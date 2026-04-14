# Render.com Setup Instructions

## Important: Setting Python Version in Render Dashboard

Render might be ignoring `runtime.txt`. You need to manually set the Python version in the Render dashboard:

### Option 1: Use Python 3.10 (Recommended for TensorFlow 2.18.0)

1. Go to your Render dashboard: https://dashboard.render.com
2. Click on your service: **blood-group-detection**
3. Go to **Settings** tab
4. Scroll down to **Environment** section
5. Find **Python Version** field
6. Enter: `3.10.12` (or select from dropdown if available)
7. Click **Save Changes**
8. Go to **Manual Deploy** → **Deploy latest commit**

### Option 2: Use Python 3.13 with TensorFlow 2.20.0 (Current Setup)

The `requirements.txt` has been updated to use TensorFlow 2.20.0 which supports Python 3.13.

**Current Status**: The latest commit uses TensorFlow 2.20.0, so it should work with Python 3.13.4.

## If Build Still Fails

1. **Check Build Logs**: Look for specific error messages
2. **Verify Python Version**: In Render dashboard → Settings → Environment
3. **Clear Build Cache**: Sometimes Render caches old builds
   - Go to Settings → Clear Build Cache → Save
   - Then redeploy

## Manual Configuration in Render Dashboard

When creating/editing your service, make sure:

- **Environment**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`
- **Python Version**: Set to `3.10.12` (or use 3.13 with updated TensorFlow)

## Current Configuration Files

- ✅ `requirements.txt` - Updated to TensorFlow 2.20.0
- ✅ `runtime.txt` - Specifies Python 3.10.12
- ✅ `Procfile` - Contains start command
- ✅ `render.yaml` - Render configuration

## Next Steps

1. **If using Python 3.10**: Set it manually in Render dashboard
2. **If using Python 3.13**: The updated requirements.txt should work
3. **Redeploy**: Click "Manual Deploy" → "Deploy latest commit"

---

**Note**: TensorFlow 2.20.0 is compatible with Python 3.13, so the current setup should work. If you prefer Python 3.10, manually set it in the dashboard.


