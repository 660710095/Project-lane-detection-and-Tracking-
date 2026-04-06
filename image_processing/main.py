import cv2
import matplotlib.pyplot as plt
import os
from utils import load_image, load_video, save_video, get_kitti_image
from edge_detection import full_pipeline

def test_single_image(img_path):
    img = load_image(img_path)

    # ============= dowlaoded  from path ============
    result, edges,roi = full_pipeline(img)
    # ============= run All pipeline util result 3 sections =======
    fig, axes = plt.subplots(1,4,figsize=(20,5))
    #chanel 1  : original image tranformed to BGR --> RGB
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')

    #===========================================================
    #chanel 2 : edge detection result (black and white image)
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Edge Detection')

#===========================================================
    #chanel 3 : region of interest (ROI) mask applied to the original image
    axes[2].imshow(roi)
    axes[2].set_title('Region of Interest')

#===========================================================
    #chanel 4 : final lane detection result with detected lanes highlighted
    # green Lines 
    axes[3].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Lane Detection Result')

    plt.tight_layout()
    plt.show()

def test_video(video_path, output_path):
    cap = load_video(video_path)
    # open video
    
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #count frames in Video


    while cap.isOpened():
        ret, frame = cap.read()
        #Read frame by frame
        # ret =  True if Read Frames successfully, False if no more frames or error
        # frame = 1 Picture --> 1 Frame

        if not ret:
            break
        # run pipiline for each frame
        # _ =  ignore edge and roi
        result, _, _ = full_pipeline(frame)
        frames.append(result)

        frames.append(result)
        count += 1
        if count % 30 == 0:
            print(f'Processed {count}/{total} frames')
        # show progress every 30 frames
    cap.release()
    save_video(output_path, frames)

def test_kitti_batch(kitti_dir):
    path = get_kitti_image(kitti_dir)
    # Pull all images from kitti dataset

    for i ,path in enumerate(path[:5]):
        # loop for 5 pictures in kitti dataset
        img = load_image(path)
        result, _, _= full_pipeline(img)
        cv2.imwrite(f"result/kitti_result_{i:003}.png", result)
        # บันทึกผลลัพธ์เป็นไฟล์รูป
        # i:03d แปลว่าตัวเลข 3 หลัก เช่น 000, 001, 002

if __name__ == "__main__":
    os.makedirs("result", exist_ok=True)
    # create folder result if not exist

    KITTI_DIR = "data/kitti"
    # path to dataset

    test_single_image(f"{KITTI_DIR}/training/image_2/um_000000.png")
    #run mode1 : test single image from kitti dataset

    
    # test_video("data/videos/test.mp4")     ← comment ไว้ก่อน
    # test_kitti_batch(KITTI_DIR)            ← comment ไว้ก่อน