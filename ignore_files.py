import os 
import shutil


# Ignore filetypes
ignore_filetypes = ['.pyc','ipynb','pkl']

ignore_folders = ['__pycache__', '.git']
# List all files in the current directory
files = os.listdir('.')
print("Files in the current directory:")

cases = []

for folder in files:
    if folder in ignore_folders:
        continue
    # Check if the folder is a directory
    if os.path.isdir(folder):
        print(f"Directory: {folder}")
        for root, dirs, files in os.walk(folder):
            for filename in files:
                if not any(filename.endswith(ext) for ext in ignore_filetypes):
                    print(f"\tFile: {filename}")
                else:
                    cases.append(os.path.join(root, filename))
    else:
        print(f"File: {folder}")

print("Files to be ignored:")
sorted(cases)
with open('.gitignore', 'w') as f:
    for case in cases:
        f.write(case + '\n')
        print(case)