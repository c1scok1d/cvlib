import cv2
#import matplotlib.pyplot as plt
import cvlib as cv
import urllib.request
import numpy as np
from cvlib.object_detection import draw_bbox
import concurrent.futures
 
url='http://10.0.0.13/cam-hi.jpg' # front room cam
#url='http://10.0.0.96/cam-hi.jpg' # lab window cam

im=None

        
def object_detection():
    padding = 20
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp,-1)
 
        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        print(bbox, label, conf)

        # draw bounding box over detected objectsP
        out = draw_bbox(frame, bbox, label, conf)
        
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

            # apply gender detection    
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
        cv2.imshow("Object Detection", out)
        cv2.imshow("Object Detection", frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
 
def lab_window():
    padding = 20
    cv2.namedWindow("Lab Window", cv2.WINDOW_NORMAL)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp,-1)
 
        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        print(bbox, label, conf)

        # draw bounding box over detected objectsP
        out = draw_bbox(frame, bbox, label, conf)
        
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

            # apply gender detection    
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
        #frames = cv2.resize(out, (820,460))
        cv2.imshow("Lab Window", out)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows() 

def run3():
    cv2.namedWindow("Lab", cv2.WINDOW_NORMAL)
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp,-1)
 
        # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        print(bbox, label, conf)

        # draw bounding box over detected objectsP
        out = draw_bbox(frame, bbox, label, conf)

        # display output
        #frames = cv2.resize(out, (820,460))
        cv2.imshow("Lab", out)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows() 
    
def gender_detection():
    cv2.namedWindow("Gender Detection", cv2.WINDOW_NORMAL)
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
        #frames = cv2.resize(frame, (820,460))
        cv2.imshow("Gender Detection", frame)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executer:
            f1= executer.submit(object_detection)
            #f2= executer.submit(gender_detection)
            #f3= executer.submit(run3)