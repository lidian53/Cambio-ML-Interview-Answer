import argparse
import cv2
import insightface
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image_path", type=str, required=False, help="Path to the image file")
parser.add_argument("-d", "--folder_path", type=str, required=False, help="Path to the folder containing images")
args = parser.parse_args()
detector = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

detector.prepare(ctx_id=0, det_size=(640, 640))


if args.folder_path:
    for img_path in os.listdir(args.folder_path):
        if img_path.endswith(".jpg") or img_path.endswith("jpeg") or img_path.endswith("png"):
            img = cv2.imread(os.path.join(args.folder_path, img_path))
            faces = detector.get(img)
            print(f"{img_path} {len(faces)}")
else:
    image_path = args.image_path
    img = cv2.imread(image_path)
    faces = detector.get(img)
    print(f"{image_path} {len(faces)}")
    rimg = detector.draw_on(img, faces)
    cv2.imwrite("./t3_output.jpg", rimg)
