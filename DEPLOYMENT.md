# Deployment Guide - Blood Group Detection App

This guide will help you deploy your Flask application to Render.com (free tier) so you can share it with anyone.

## Prerequisites

1. A GitHub account
2. A Render.com account (free at https://render.com)
3. Git installed on your computer

## Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial commit - ready for deployment"
   ```

2. **Create a GitHub repository**:
   - Go to https://github.com/new
   - Create a new repository (e.g., "blood-group-detection")
   - **DO NOT** initialize with README, .gitignore, or license

3. **Push your code to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/blood-group-detection.git
   git branch -M main
   git push -u origin main
   ```

## Step 2: Deploy to Render.com

1. **Sign up/Login to Render**:
   - Go to https://render.com
   - Sign up with your GitHub account (recommended)

2. **Create a New Web Service**:
   - Click "New +" button
   - Select "Web Service"
   - Connect your GitHub account if not already connected
   - Select your repository: `blood-group-detection`

3. **Configure the Service**:
   - **Name**: `blood-group-detection` (or any name you prefer)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Select "Free" (or upgrade if needed)

4. **Environment Variables** (Optional):
   - You can add environment variables if needed in the "Environment" section

5. **Deploy**:
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Wait for the build to complete (5-10 minutes for first deployment)

6. **Get Your Live URL**:
   - Once deployed, you'll get a URL like: `https://blood-group-detection.onrender.com`
   - Share this URL with anyone!

## Important Notes

### Model Files
- Make sure `blood_group_model_vgg16.keras` and `class_indices.pkl` are committed to your repository
- These files are essential for the app to work

### File Size Limits
- Render free tier has some limitations
- If model files are too large (>100MB), consider:
  - Using Git LFS (Large File Storage)
  - Or uploading models to cloud storage (S3, etc.) and loading them at runtime

### Free Tier Limitations
- Render free tier services spin down after 15 minutes of inactivity
- First request after spin-down may take 30-60 seconds
- For production use, consider upgrading to a paid plan

## Alternative Deployment Options

### Option 2: Railway.app
1. Go to https://railway.app
2. Sign up with GitHub
3. Create new project from GitHub repo
4. Railway auto-detects Flask and deploys

### Option 3: Fly.io
1. Install flyctl: `https://fly.io/docs/getting-started/installing-flyctl/`
2. Run: `fly launch`
3. Follow the prompts

### Option 4: PythonAnywhere
1. Go to https://www.pythonanywhere.com
2. Upload your files
3. Configure WSGI file
4. Reload web app

## Troubleshooting

### Build Fails
- Check build logs in Render dashboard
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

### App Crashes
- Check runtime logs
- Ensure model files are present
- Verify file paths are correct (use relative paths)

### Slow First Request
- This is normal on free tier (cold start)
- Consider upgrading for better performance

## Testing Locally Before Deployment

Test your app locally with production settings:
```bash
pip install -r requirements.txt
gunicorn app:app
```

Then visit: http://localhost:8000

## Support

If you encounter issues:
1. Check Render logs in the dashboard
2. Verify all files are committed to Git
3. Ensure model files are included in repository

---

**Your app will be live at**: `https://YOUR-APP-NAME.onrender.com`

Happy deploying! 🚀


