import cv2
import numpy as np
def work():
    # Use a breakpoint in the code line below to debug your script.
    labels = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'
    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)


    cap = cv2.VideoCapture(0)
    writer = None



    while cap.isOpened():
        ret, image = cap.read()

        if not ret:
            break

        (H, W) = image.shape[:2]

        classindex, confidence, bbox = net.detect(image, confThreshold=0.6)

        if len(classindex) != 0:
            for index, conf, box in zip(classindex.flatten(), confidence.flatten(), bbox):
                if index <= 80:
                    color = [int(c) for c in colors[index]]
                    cv2.rectangle(image, box, color, 2)
                    text = "{}: {:.4f}".format(labels[index - 1], conf)
                    cv2.putText(image, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"DIVX")
            writer = cv2.VideoWriter("OutputVideo.mp4", fourcc, 15,
                                     (W, H), True)

        cv2.imshow("Video", image)
        #     writer.write(image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    writer.release()
    cap.release()
    cv2.destroyAllWindows()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    work()

