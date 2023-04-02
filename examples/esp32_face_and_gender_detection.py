import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
 
url='http://10.0.0.133/cam-hi.jpg'
frame=None

        
def run1():
    #cv2.namedWindow("Face and Gender Detection", cv2.WINDOW_AUTOSIZE)
    padding = 20
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp,-1)
 
        # apply face detection
        face, confidence = cv.detect_face(frame)

        print(face)
        print(confidence)

        # loop through detected faces
        for idx, f in enumerate(face):
            
            (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
            (endX,endY) = min(frame.shape[1]-1, f[2]+padding), min(frame.shape[0]-1, f[3]+padding)
        
            # draw rectangle over face
            cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

            face_crop = np.copy(frame[startY:endY, startX:endX])

            # apply face detection    
            (label, confidence) = cv.detect_gender(face_crop)

            print(confidence)
            print(label)

            idx = np.argmax(confidence)
            label = label[idx]

            label = "{}: {:.2f}%".format(label, confidence[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write detected gender and confidence percentage on top of face rectangle
            cv2.putText(frame, label, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0,255,0), 2)

        # display output
        frames = cv2.resize(frame, (820,460))
        cv2.imshow("Gender Detection", frames)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
 
 
 
if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
            f1= executer.submit(run1)