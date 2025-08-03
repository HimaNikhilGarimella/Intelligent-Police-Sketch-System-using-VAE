# Directory structure setup script (save as setup.py)
import os

def setup_project():
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Create necessary directories
    directories = [
        'models',
        'data',
        'src/static',
        'src/templates',
        'logs'
    ]
    
    for directory in directories:
        dir_path = os.path.join(project_root, directory)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    
    # Create empty __init__.py files
    init_locations = [
        'src',
        'src/static',
        'src/templates'
    ]
    
    for location in init_locations:
        init_file = os.path.join(project_root, location, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass
            print(f"Created __init__.py in: {location}")
    
    print("Project structure setup complete!")

if __name__ == "__main__":
    setup_project()