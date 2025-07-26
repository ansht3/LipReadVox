cd /Users/anshtandon/projects-fullstack/LipReadVox
source lipread_env/bin/activate
python quick_demo.py

python src/predict.py

python demo_collection.py # Collect your own data
python src/preprocess_training.py # Process data
python src/model_training.py # Train model
python src/predict.py # Test real-time
