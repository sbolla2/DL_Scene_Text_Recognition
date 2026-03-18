import os
import numpy as np
import cv2
import hashlib

np.random.seed(42)

DATASETS = {
    "TotalText_train": {
        "clean_directory": "Total-Text/Train",
        "noisy_directory": "Total-Text/Train_noisy",
    },
    "TotalText_test": {
        "clean_directory": "Total-Text/Test",
        "noisy_low_directory": "Total-Text/Test_noisy_low",
        "noisy_medium_directory": "Total-Text/Test_noisy_medium",
        "noisy_high_directory": "Total-Text/Test_noisy_high",
    },
}

def get_image_paths(dataset_name):
    dataset = DATASETS[dataset_name]
    clean_directory = dataset["clean_directory"]

    if dataset_name.endswith("test"):
        return sorted([os.path.join(clean_directory, f) for f in os.listdir(clean_directory) if f.lower().endswith('.jpg')]), dataset["noisy_low_directory"], dataset["noisy_medium_directory"], dataset["noisy_high_directory"]

    noisy_directory = dataset["noisy_directory"]
    return sorted([os.path.join(clean_directory, f) for f in os.listdir(clean_directory) if f.lower().endswith('.jpg')]), noisy_directory

def add_noise_to_image(image_path, kernel_size, gaussian_sigma, gaussian_noise_std, blur_type, motion_blur_length, motion_angle):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    if blur_type == "gaussian":
        blurred = cv2.GaussianBlur(image, kernel_size, gaussian_sigma)
    elif blur_type == "motion":
        M = cv2.getRotationMatrix2D((motion_blur_length / 2, motion_blur_length / 2), motion_angle, 1)

        kernel = np.zeros((motion_blur_length, motion_blur_length))
        kernel[int((motion_blur_length - 1) / 2), :] = np.ones(motion_blur_length)
        kernel = cv2.warpAffine(kernel, M, (motion_blur_length, motion_blur_length))

        if np.sum(kernel) == 0:
            kernel = np.ones((motion_blur_length, motion_blur_length)) / (motion_blur_length * motion_blur_length)
        else:
            kernel /= np.sum(kernel)

        blurred = cv2.filter2D(image, -1, kernel)
    else:
        blurred = cv2.GaussianBlur(image, kernel_size, gaussian_sigma)
        M = cv2.getRotationMatrix2D((motion_blur_length / 2, motion_blur_length / 2), motion_angle, 1)

        kernel = np.zeros((motion_blur_length, motion_blur_length))
        kernel[int((motion_blur_length - 1) / 2), :] = np.ones(motion_blur_length)
        kernel = cv2.warpAffine(kernel, M, (motion_blur_length, motion_blur_length))

        if np.sum(kernel) == 0:
            kernel = np.ones((motion_blur_length, motion_blur_length)) / (motion_blur_length * motion_blur_length)
        else:
            kernel /= np.sum(kernel)
        
        blurred = cv2.filter2D(blurred, -1, kernel)

    h, w = image.shape[:2]
    lr = cv2.resize(blurred, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)

    key = image_path.encode("utf-8")
    digest = hashlib.sha256(key).digest()

    seed = int.from_bytes(digest[:4], byteorder="big", signed=False)
    rng = np.random.default_rng(seed)
    
    noise = rng.normal(0.0, gaussian_noise_std, lr.shape).astype(np.float32)
    noisy_image = lr.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image

def motion_blur_angle(image_path):
    key = image_path.encode("utf-8")
    digest = hashlib.sha256(key).digest()

    value = int.from_bytes(digest[:8], byteorder="big", signed=False)

    angle = (value / float(2**64)) * 180.0
    return angle

def medium_motion_or_gaussian_blur(image_path):
    key = image_path.encode("utf-8")
    digest = hashlib.sha256(key).digest()

    value = int.from_bytes(digest[:8], byteorder="big", signed=False)

    if value % 2 == 0:
        return "motion"
    else:
        return "gaussian"

def process_dataset(clean_paths, noisy_directory, set_type, overwrite=True):
    os.makedirs(noisy_directory, exist_ok=True)

    for clean_path in clean_paths:
        base_name = os.path.splitext(os.path.basename(clean_path))[0]
        noisy_filename = f"{base_name}_noisy.jpg"
        noisy_path = os.path.join(noisy_directory, noisy_filename)

        if not overwrite and os.path.exists(noisy_path):
            print(f"Skipping existing file: {noisy_path}")
            continue
        
        motion_blur_length = 3
        motion_angle = motion_blur_angle(clean_path)
        kernel_size = (3, 3)
        gaussian_sigma = 1.2
        gaussian_noise_std = 2.0
        jpeg_quality = 70
        blur = "both"

        if set_type == "low":
            gaussian_sigma = 0.7
            gaussian_noise_std = 2.8
            jpeg_quality = 80
            blur = "both"
        elif set_type == "high":
            motion_blur_length = 5
            kernel_size = (5, 5)
            gaussian_sigma = 1.7
            gaussian_noise_std = 3.8
            jpeg_quality = 60
            blur = "both"
        elif set_type == "train":
            # randomize parameters for training set
            motion_blur_length = np.random.choice([3, 5])
            size = np.random.choice([3, 5])
            gaussian_sigma = np.random.uniform(0.7, 1.7)
            kernel_size = (size, size)
            gaussian_noise_std = np.random.uniform(2.0, 4.0)
            jpeg_quality = np.random.randint(60, 81)
            which_blur = np.random.randint(0, 100)

            if which_blur < 35:
                blur = "motion"
            elif which_blur < 70:
                blur = "gaussian"
            else:
                blur = "both"

        noisy_image = add_noise_to_image(clean_path, kernel_size, gaussian_sigma, gaussian_noise_std, blur, motion_blur_length, motion_angle)
        if noisy_image is not None:
            success = cv2.imwrite(noisy_path, noisy_image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            if not success:
                print(f"Failed to save noisy image: {noisy_path}")
        else:
            print(f"Failed to process image: {clean_path}")

if __name__ == "__main__":
    totaltext_train_clean_paths, totaltext_train_noisy_directory = get_image_paths("TotalText_train")
    totaltext_test_clean_paths, totaltext_test_noisy_low_directory, totaltext_test_noisy_medium_directory, totaltext_test_noisy_high_directory = get_image_paths("TotalText_test")

    print("Processing TotalText Train Set...")
    process_dataset(totaltext_train_clean_paths, totaltext_train_noisy_directory, set_type="train")

    print("Processing TotalText Test Set...")
    process_dataset(totaltext_test_clean_paths, totaltext_test_noisy_low_directory, set_type="low")
    process_dataset(totaltext_test_clean_paths, totaltext_test_noisy_medium_directory, set_type="medium")
    process_dataset(totaltext_test_clean_paths, totaltext_test_noisy_high_directory, set_type="high")
    
    print("\nNumber of Clean TotalText Train Images:", len(totaltext_train_clean_paths))
    print("Number of Noisy TotalText Train Images:", len(os.listdir(totaltext_train_noisy_directory)))
    print("\nNumber of Clean TotalText Test Images:", len(totaltext_test_clean_paths))
    print("Number of Noisy TotalText Test Images of Low Noise:", len(os.listdir(totaltext_test_noisy_low_directory)))
    print("Number of Noisy TotalText Test Images of Medium Noise:", len(os.listdir(totaltext_test_noisy_medium_directory)))
    print("Number of Noisy TotalText Test Images of High Noise:", len(os.listdir(totaltext_test_noisy_high_directory)))