import os
import cv2
from shutil import move

def dhash(image, hash_size=16): # Modify the hash_size according to your task
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def find_and_move_similar_images(folder_path, threshold=4):
    # Modify the threshold according to your task, usually ranges from 1 to 10. The larger the value,
    # the greater the tolerance for differences. The specific value to be set depends on your task
    similar_folder = os.path.join(folder_path, 'similar')
    record_file_path = os.path.join(similar_folder, 'record.txt')

    # Create "similar" folder
    os.makedirs(similar_folder, exist_ok=True)

    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path)
                   if os.path.isfile(os.path.join(folder_path, file))
                   and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    hash_dict = {}

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"can't read image：{path}")
            continue

        # caculate hash value
        h = dhash(image)
        hash_dict[path] = h

    checked_images = set()

    similar_images_record = set()

    for path1, hash1 in hash_dict.items():
        for path2, hash2 in hash_dict.items():
            if path1 != path2 and path1 not in checked_images and path2 not in checked_images:
                hamming_distance = bin(hash1 ^ hash2).count('1')

                # If the Hamming distance is below the threshold, the images are considered similar
                if hamming_distance < threshold:
                    print(f"found similar images：{path1} and {path2}")
                    similar_images_record.add(path1)
                    similar_images_record.add(path2)
                    checked_images.add(path1)
                    checked_images.add(path2)

    # Record the names of similar images to the 'record.txt' file
    with open(record_file_path, 'w') as record_file:
        record_file.write('\n'.join([os.path.basename(path) for path in similar_images_record]))

    # Move similar images to the 'similar' folder
    for path in similar_images_record:
        try:
            move(path, os.path.join(similar_folder, os.path.basename(path)))
        except FileNotFoundError as e:
            print(f"file not exist：{path}")

    print("done")

folder_path = 'path/to/your/folder'  # Modify the images folder path
find_and_move_similar_images(folder_path)
