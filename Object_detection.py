import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

yolo=cv2.dnn.readNet("2yolov3.weights","2yolov3.cfg")

classes=[]
with open("coco.names","r") as f:
    classes=f.read().splitlines()
print(classes)

img=cv2.imread("test5.png")
# img=cv2.resize(img,(224,224))
height,width, _ =img.shape
print(height,width)
blob= cv2.dnn.blobFromImage(img,1/255, (320,320), (0,0,0),swapRB=True,crop=False)
print(blob.shape)
plt.imshow(img)

yolo.setInput(blob)
output_layers_name=yolo.getUnconnectedOutLayersNames()
layeroutput =yolo.forward(output_layers_name)

boxs=[]
confidences=[]
class_ids=[]
for output in layeroutput:
    for detection in output:
        score=detection[5:]
        class_id=np.argmax(score)
        confidence=score[class_id]
        if confidence>0.7:
            center_X=int(detection[0]*width)
            center_y=int(detection[0]*height)
            w=int(detection[0]*width)
            h=int(detection[0]*height)
            x=int(center_X-w/2)
            y=int(center_y-h/2)
            boxs.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexs=cv2.dnn.NMSBoxes(boxs,confidences,.5,.4)

font=cv2.FONT_HERSHEY_PLAIN
colors=np.random.uniform(0,255,size=(len(boxs),3))

for i in indexs.flatten():
    x,y,w,h =boxs[i]
    label=str(classes[class_ids[i]])
    confi=str(round(confidences[i],2))
    color=colors[i]
    print(boxs[i])
    cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
    cv2.putText(img,label+" "+confi,(x,y+20),font,2,(255,0,0),2)

plt.imshow(img)
plt.show()
# cv2.imwrite("./img.jpg",img)