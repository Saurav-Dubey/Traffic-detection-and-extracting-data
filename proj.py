import cv2
import csv

#we are using TensorFlow
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' #configuration file
frozen_model = 'frozen_inference_graph.pb' #pretrained graph file

model = cv2.dnn_DetectionModel(config_file,frozen_model)#using deep nueral network


fields=['Object_ID','Classification','Confidence','Box','Time']
labels= []
file_name = "Labels.txt"

with open(file_name,'rt') as fpt:
    labels=fpt.read().rstrip('\n').split('\n')

    file = open("Data extracted.csv", "w")

    writer = csv.writer(file)
    writer.writerow(fields)

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture("sample1.mp4")# reading frame by frame
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN


while True:
    ret,frame = cap.read()
    ClassIndex , confidece ,bbox =model.detect(frame,confThreshold=0.50)

    print(bbox)
    frame_count=frame_count+1
    time = float(frame_count) / fps
    if len(ClassIndex)!=0:
        for ClassIndex , conf ,boxes in zip(ClassIndex.flatten() , confidece.flatten() ,bbox):
            if ClassIndex<80:
                cv2.rectangle(frame,boxes,(255,0,0) , 2)
                cv2.putText(frame, labels[ClassIndex-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0))
    cv2.imshow('Object detection tutorial',frame)
    for w in range(len(confidece)):
        writer.writerow([ClassIndex,labels[ClassIndex-1],confidece[w]*100,bbox,time])


    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
file.close()