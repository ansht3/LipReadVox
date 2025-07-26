# GitHub Push Instructions

Follow these steps to successfully push your LipReadVox repository to GitHub:

## Step 1: Clean up the repository

```bash
# Navigate to your project directory
cd /Users/anshtandon/projects-fullstack/LipReadVox

# Remove virtual environment from git tracking
git rm -r --cached lipread_env/

# Remove large model files from git tracking
git rm --cached model/shape_predictor_68_face_landmarks.dat
git rm --cached model/lip_reader_3dcnn.h5
git rm --cached model/labels.npy
git rm --cached model/*.png

# Remove data directories from git tracking
git rm -r --cached data/
git rm -r --cached processed_data/
```

## Step 2: Add the new files

```bash
# Add .gitignore and documentation
git add .gitignore
git add README.md
git add DOWNLOAD_MODELS.md
git add PUSH_INSTRUCTIONS.md

# Add source code files
git add src/
git add requirements.txt
git add demo_collection.py
git add quick_demo.py
git add run_lip_reader.py
git add SETUP_GUIDE.md
```

## Step 3: Create model directory structure

```bash
# Create model directory without large files
mkdir -p model
touch model/.gitkeep
git add model/.gitkeep
```

## Step 4: Commit and push

```bash
# Commit the changes
git commit -m "Add .gitignore and documentation, remove large files

- Added comprehensive .gitignore to exclude virtual environment and large model files
- Updated README with setup instructions and troubleshooting
- Added DOWNLOAD_MODELS.md with instructions for downloading required files
- Removed large files that exceed GitHub's size limits
- Added proper documentation for setup and usage"

# Push to GitHub
git push origin main
```

## Step 5: Verify the push

Check your GitHub repository to ensure:

- ✅ Source code is present
- ✅ Documentation files are present
- ✅ Large files are NOT present
- ✅ .gitignore is working correctly

## Alternative: Use the cleanup script

If you prefer, you can run the cleanup script:

```bash
# Make the script executable
chmod +x cleanup_and_push.sh

# Run the cleanup script
./cleanup_and_push.sh
```

## What happens after pushing

1. **Repository will be clean** - No large files that exceed GitHub limits
2. **Documentation is complete** - Users can follow setup instructions
3. **Model files are documented** - DOWNLOAD_MODELS.md explains how to get required files
4. **Setup is automated** - quick_demo.py handles the complete pipeline

## For new users cloning the repository

After cloning, users will need to:

1. Set up Python environment
2. Install dependencies
3. Download the dlib shape predictor (95MB)
4. Run `python quick_demo.py` to generate trained model
5. Test with `python src/predict.py`

All of these steps are documented in the README.md and DOWNLOAD_MODELS.md files.
