import os
import sys
import argparse

import cv2
from tqdm import tqdm
from ssds.ssds import SSDDetector

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def plot_one_box(img, x, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def demo_image(model, image_path, display):
    # 1. prepare image
    image = cv2.imread(image_path)
    image = cv2.resize(image, model.image_size)

    # 2. model infer
    scores, boxes, classes = model(image)

    # 3. draw bounding box on the image
    for score, box, labels in zip(scores, boxes, classes):
        plot_one_box(image, box, COLORS[labels % 3], '{label}: {score:.3f}'.format(label=labels, score=score))
    
    # 4. visualize result
    if display:
        cv2.imshow('result', image)
        cv2.waitKey(0)
    else:
        path, _ = os.path.splitext(image_path)
        cv2.imwrite(path + '_result.jpg', image)
        print("output file save at '{}'".format(path + '_result.jpg'))

def demo_video(model, video_path, display):
    # 0. prepare video
    cap    = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cap.isOpened() and (not display): 
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        writer = cv2.VideoWriter(video_path+"_output.mp4", fourcc, fps, (width,height))
    
    for fid in tqdm(range(frames)):
        # 1. prepare image
        flag, image = cap.read()
        image = cv2.resize(image, model.image_size)

        # 2. model infer
        scores, boxes, classes = model(image)

        # 3. draw bounding box on the image
        for score, box, labels in zip(scores, boxes, classes):
            plot_one_box(image, box, COLORS[labels % 3], '{label}: {score:.3f}'.format(label=labels, score=score))

        image = cv2.resize(image, (width,height))

        # 4. visualize result
        if display:
            cv2.imshow("Image", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            writer.write(image)
        
    # 5. release the video resources
    cap.release()
    if display:
        cv2.destroyAllWindows()
    else:
        writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo a ssds.pytorch network')
    parser.add_argument('-cfg', '--confg-file',
            help='the address of optional config file', default=None, type=str, required=True)
    parser.add_argument('-i', '--demo-file',
            help='the address of the demo file', default=None, type=str, required=True)
    parser.add_argument('-t', '--type', 
            default='image', choices=['image', 'video'])
    parser.add_argument('-d', '--display', 
            help='whether display the detection result', action="store_true")
    parser.add_argument('-s', '--shift', action="store_true")  

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    model = SSDDetector(args.confg_file, args.shift)
    getattr(sys.modules[__name__], "demo_"+args.type)(model, args.demo_file, args.display)