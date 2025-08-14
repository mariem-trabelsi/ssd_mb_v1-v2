import os
import shutil
import glob

res = input("\r\nPlease enter model name: ")
dirLst = os.listdir("data") if os.path.exists("data") else []
if res in dirLst:
    print("\r\nError. Model directory already exists. Please choose some other name or delete the current directory\r\n")
    sys.exit() 

print("Making directory structure")
os.makedirs("data/{}/Annotations".format(res))
os.makedirs("data/{}/ImageSets/Main".format(res))
os.makedirs("data/{}/JPEGImages".format(res)) 

def process_directory(source_dir, img_counter_start):
    """Process a directory and copy images and annotations"""
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} not found, skipping...")
        return [], img_counter_start
    
    processed_files = []
    img_counter = img_counter_start
    
    # Get all image files
    image_files = glob.glob(f"{source_dir}/*.jpg") + glob.glob(f"{source_dir}/*.png")
    
    for img_path in image_files:
        img_counter += 1
        img_name = os.path.basename(img_path)
        img_base = os.path.splitext(img_name)[0]
        
        # New filename
        new_name = f"file{img_counter}"
        new_img_name = f"{new_name}.jpg"
        new_xml_name = f"{new_name}.xml"
        
        # Copy image
        shutil.copy2(img_path, f"data/{res}/JPEGImages/{new_img_name}")
        
        # Find and copy corresponding XML file
        xml_path = f"{source_dir}/{img_base}.xml"
        if os.path.exists(xml_path):
            shutil.copy2(xml_path, f"data/{res}/Annotations/{new_xml_name}")
            print(f"Copied: {img_name} and {img_base}.xml -> {new_img_name} and {new_xml_name}")
        else:
            print(f"Warning: No XML file found for {img_name}")
        
        processed_files.append(new_name)
    
    return processed_files, img_counter

# Check if people directory exists
if not os.path.exists("people"):
    print("Error: 'people' directory not found. Please make sure you're in the correct directory.")
    sys.exit(0)

print("Processing Roboflow dataset...")

# Process each directory
train_files, counter = process_directory("people/train", 0)
test_files, counter = process_directory("people/test", counter)
valid_files, counter = process_directory("people/valid", counter)

# Combine all files
all_files = train_files + test_files + valid_files

if len(all_files) == 0:
    print("No files processed. Please check your dataset structure.")
    sys.exit(0)

print(f"Total files processed: {len(all_files)}")
print(f"Train files: {len(train_files)}")
print(f"Test files: {len(test_files)}")
print(f"Valid files: {len(valid_files)}")

# Create the split files
# Use original train files for training
if len(train_files) > 0:
    with open(f"data/{res}/ImageSets/Main/train.txt", 'w') as f:
        for filename in train_files:
            f.write(filename + "\n")
    
    with open(f"data/{res}/ImageSets/Main/trainval.txt", 'w') as f:
        for filename in train_files:
            f.write(filename + "\n")

# Use original test files for testing
if len(test_files) > 0:
    with open(f"data/{res}/ImageSets/Main/test.txt", 'w') as f:
        for filename in test_files:
            f.write(filename + "\n")
else:
    # If no test files, use 10% of train files
    if len(train_files) > 0:
        test_count = max(1, len(train_files) // 10)
        test_subset = train_files[:test_count]
        with open(f"data/{res}/ImageSets/Main/test.txt", 'w') as f:
            for filename in test_subset:
                f.write(filename + "\n")

# Use valid files for validation, or test files if no valid files
if len(valid_files) > 0:
    with open(f"data/{res}/ImageSets/Main/val.txt", 'w') as f:
        for filename in valid_files:
            f.write(filename + "\n")
elif len(test_files) > 0:
    with open(f"data/{res}/ImageSets/Main/val.txt", 'w') as f:
        for filename in test_files:
            f.write(filename + "\n")
else:
    # Use a subset of train files
    if len(train_files) > 0:
        val_count = max(1, len(train_files) // 10)
        val_subset = train_files[-val_count:]
        with open(f"data/{res}/ImageSets/Main/val.txt", 'w') as f:
            for filename in val_subset:
                f.write(filename + "\n")

print("Dataset conversion completed!")
print(f"Files are now organized in: data/{res}/")
print("- Images: JPEGImages/")
print("- Annotations: Annotations/")
print("- Split files: ImageSets/Main/")
