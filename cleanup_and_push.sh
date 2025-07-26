#!/bin/bash

# LipReadVox Repository Cleanup and Push Script
# This script will clean up large files and push the repository to GitHub

echo "🧹 Cleaning up LipReadVox repository for GitHub push..."

# Step 1: Remove virtual environment from git tracking
echo "📁 Removing virtual environment from git tracking..."
git rm -r --cached lipread_env/ 2>/dev/null || echo "Virtual environment not tracked"

# Step 2: Remove large model files from git tracking
echo "🗂️  Removing large model files from git tracking..."
git rm --cached model/shape_predictor_68_face_landmarks.dat 2>/dev/null || echo "Shape predictor not tracked"
git rm --cached model/lip_reader_3dcnn.h5 2>/dev/null || echo "Trained model not tracked"
git rm --cached model/labels.npy 2>/dev/null || echo "Labels not tracked"
git rm --cached model/*.png 2>/dev/null || echo "PNG files not tracked"

# Step 3: Remove data directories from git tracking
echo "📊 Removing data directories from git tracking..."
git rm -r --cached data/ 2>/dev/null || echo "Data directory not tracked"
git rm -r --cached processed_data/ 2>/dev/null || echo "Processed data directory not tracked"

# Step 4: Add .gitignore and new files
echo "📝 Adding .gitignore and documentation files..."
git add .gitignore
git add README.md
git add DOWNLOAD_MODELS.md
git add cleanup_and_push.sh

# Step 5: Add source code files
echo "💻 Adding source code files..."
git add src/
git add requirements.txt
git add demo_collection.py
git add quick_demo.py
git add run_lip_reader.py
git add SETUP_GUIDE.md

# Step 6: Create model directory structure (without large files)
echo "📁 Creating model directory structure..."
mkdir -p model
touch model/.gitkeep

# Step 7: Commit changes
echo "💾 Committing changes..."
git commit -m "Add .gitignore and documentation, remove large files

- Added comprehensive .gitignore to exclude virtual environment and large model files
- Updated README with setup instructions and troubleshooting
- Added DOWNLOAD_MODELS.md with instructions for downloading required files
- Added cleanup script for repository management
- Removed large files that exceed GitHub's size limits
- Added proper documentation for setup and usage"

# Step 8: Push to GitHub
echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Repository cleanup and push completed!"
echo ""
echo "📋 Next steps:"
echo "1. Clone the repository on another machine"
echo "2. Follow the setup instructions in README.md"
echo "3. Download required model files using DOWNLOAD_MODELS.md"
echo "4. Run: python quick_demo.py"
echo "5. Test: python src/predict.py" 