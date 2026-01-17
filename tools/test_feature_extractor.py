import sys
import os

# Add the parent directory to the Python path to allow importing from tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.feature_extractor import extract_features_from_c_source

def main():
    c_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'training_programs', '01_insertion_sort.c'))
    
    print(f"Testing feature extraction for: {c_file}")
    
    try:
        features = extract_features_from_c_source(c_file, target_arch='native')
        print("Feature extraction successful!")
        print("Extracted features:")
        import json
        print(json.dumps(features, indent=2))
    except Exception as e:
        print(f"Feature extraction failed: {e}")

if __name__ == '__main__':
    main()
