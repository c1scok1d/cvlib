# Import Libraries
import cv2
import cvlib as cv
import numpy as np
import urllib.request
import concurrent.futures
from cvlib.object_detection import draw_bbox
from deepface import DeepFace

# https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
FACE_PROTO = "examples/gender_age/weights/deploy.prototxt.txt"
# https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
FACE_MODEL = "examples/gender_age/weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
# The gender model architecture
# https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
GENDER_MODEL = 'examples/gender_age/weights/deploy_gender.prototxt'
# The gender model pre-trained weights
# https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
GENDER_PROTO = 'examples/gender_age/weights/gender_net.caffemodel'
# Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
# substraction to eliminate the effect of illunination changes
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# Represent the gender classes
GENDER_LIST = ['Male', 'Female']
# The model architecture
# download from: https://drive.google.com/open?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW
AGE_MODEL = 'examples/gender_age/weights/deploy_age.prototxt'
# The model pre-trained weights
# download from: https://drive.google.com/open?id=1kWv0AjxGSN0g31OeJa02eBGM0R_jcjIl
AGE_PROTO = 'examples/gender_age/weights/age_net.caffemodel'
# Represent the 8 age classes of this CNN probability layer
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
# Initialize frame size
frame_width = 1280
frame_height = 720
# load face Caffe model
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
# Load age prediction model
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
# Load gender prediction model
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def get_faces(frame, confidence_threshold=0.5):
    # convert the frame into a blob to be ready for NN input
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
    # set the image as input to the NN
    face_net.setInput(blob)
    # perform inference and get predictions
    output = np.squeeze(face_net.forward())
    # initialize the result list
    faces = []
    # Loop over the faces detected
    for i in range(output.shape[0]):
        confidence = output[i, 2]
        if confidence > confidence_threshold:
            box = output[i, 3:7] * \
                np.array([frame.shape[1], frame.shape[0],
                         frame.shape[1], frame.shape[0]])
            # convert to integers
            start_x, start_y, end_x, end_y = box.astype(np.int)
            # widen the box a little
            start_x, start_y, end_x, end_y = start_x - \
                10, start_y - 10, end_x + 10, end_y + 10
            start_x = 0 if start_x < 0 else start_x
            start_y = 0 if start_y < 0 else start_y
            end_x = 0 if end_x < 0 else end_x
            end_y = 0 if end_y < 0 else end_y
            # append to our list
            faces.append((start_x, start_y, end_x, end_y))
    return faces


# from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    return cv2.resize(image, dim, interpolation = inter)


def get_gender_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False, crop=False
    )
    gender_net.setInput(blob)
    return gender_net.forward()


def get_age_predictions(face_img):
    blob = cv2.dnn.blobFromImage(
        image=face_img, scalefactor=1.0, size=(227, 227),
        mean=MODEL_MEAN_VALUES, swapRB=False
    )
    age_net.setInput(blob)
    return age_net.forward()

def get_eyes(face_img):
    # Load eye detection model
    face_cascade = cv2.CascadeClassifier('examples/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('examples/haarcascade_eye.xml')
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(face_img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = face_img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return face_img

def front_cam():
    """Predict the gender of the faces showing in the image"""
    cv2.namedWindow("Front", cv2.WINDOW_NORMAL)
    # create a new cam object
    #cap = cv2.VideoCapture(0)
    url='http://10.0.0.236/cam-hi.jpg' # front room cam

    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp,-1)
        
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        # predict the faces
        faces = get_faces(frame)
        # Loop over the faces detected
        # for idx, face in enumerate(faces):
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            # run facial analysis on image in frame
            result = DeepFace.analyze(face_img, enforce_detection=False, actions=['gender', 'age', 'race', 'emotion'])
            label = f"Gender: {result[0]['dominant_gender'].upper()}\nAge: {result[0]['age']}\nRace: {result[0]['dominant_race'].upper()}\nEmotion: {result[0]['dominant_emotion'].upper()}"
            print(label)
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            box_color = (255, 0, 0) if result[0]['dominant_gender'] == "Man" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            
            # Label processed image
            for line in label.splitlines():
                yPos -=15
                cv2.putText(frame, line, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, .5, box_color, 1, lineType=cv2.LINE_AA,)
        
         # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        print(bbox, label, conf)

        # draw bounding box over detected objectsP
        out = draw_bbox(frame, bbox, label, conf)

        # display output
        #frames = cv2.resize(out, (820,460))
        cv2.imshow("Front", out)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows() 

   
    
def attached_cam(): # for attached camera
    """Predict the gender of the faces showing in the image"""
    cv2.namedWindow("Attached", cv2.WINDOW_NORMAL)
    # create a new cam object
    cap = cv2.VideoCapture(0)
    #url='http://10.0.0.96/cam-hi.jpg' #lab window
    while True:
        #img_resp=urllib.request.urlopen(url)
        #imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        #frame = cv2.imdecode(imgnp,-1)
        
        _, img = cap.read()
        # Take a copy of the initial image and resize it
        frame = img.copy()
        # resize if higher than frame_width
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        # predict the faces
        faces = get_faces(frame)
        # Loop over the faces detected
        # for idx, face in enumerate(faces):
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            # run facial analysis on image in frame
            result = DeepFace.analyze(face_img, enforce_detection=False, actions=['gender', 'age', 'race', 'emotion'])
            label = f"Gender: {result[0]['dominant_gender'].upper()}\nAge: {result[0]['age']}\nRace: {result[0]['dominant_race'].upper()}\nEmotion: {result[0]['dominant_emotion'].upper()}"
            print(label)
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            box_color = (255, 0, 0) if result[0]['dominant_gender'] == "Man" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            
            # Label processed image
            for line in label.splitlines():
                yPos -=15
                cv2.putText(frame, line, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, .5, box_color, 1, lineType=cv2.LINE_AA,)
        
         # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        print(bbox, label, conf)

        # draw bounding box over detected objectsP
        out = draw_bbox(frame, bbox, label, conf)

        # display output
        #frames = cv2.resize(out, (820,460))
        cv2.imshow("Attached", out)
        
            # Display processed image
        cv2.imshow("Attached", frame)
        if cv2.waitKey(1) == ord("q"):
            break
        # uncomment if you want to save the image
        # cv2.imwrite("output.jpg", frame)
    # Cleanup
    cv2.destroyAllWindows()    

def lab_win_cam():
    cv2.namedWindow("Lab Window", cv2.WINDOW_NORMAL)
    url='http://10.0.0.96/cam-hi.jpg' # lab window cam
    while True:
        img_resp=urllib.request.urlopen(url)
        imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
        frame = cv2.imdecode(imgnp,-1)
 
        #frame = img.copy()
        # resize if higher than frame_width
        if frame.shape[1] > frame_width:
            frame = image_resize(frame, width=frame_width)
        # predict the faces
        faces = get_faces(frame)
        # Loop over the faces detected
        # for idx, face in enumerate(faces):
        for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
            face_img = frame[start_y: end_y, start_x: end_x]
            # run facial analysis on image in frame
            result = DeepFace.analyze(face_img, enforce_detection=False, actions=['gender', 'age', 'race', 'emotion'])
            label = f"Gender: {result[0]['dominant_gender'].upper()}\nAge: {result[0]['age']}\nRace: {result[0]['dominant_race'].upper()}\nEmotion: {result[0]['dominant_emotion'].upper()}"
            print(label)
            yPos = start_y - 15
            while yPos < 15:
                yPos += 15
            box_color = (255, 0, 0) if result[0]['dominant_gender'] == "Man" else (147, 20, 255)
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
            
            # Label processed image
            for line in label.splitlines():
                yPos -=15
                cv2.putText(frame, line, (start_x, yPos), cv2.FONT_HERSHEY_SIMPLEX, .5, box_color, 1, lineType=cv2.LINE_AA,)
        
         # apply object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        print(bbox, label, conf)

        # draw bounding box over detected objectsP
        out = draw_bbox(frame, bbox, label, conf)

        # display output
        #frames = cv2.resize(out, (820,460))
        cv2.imshow("Lab Window", out)

        # press "Q" to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    #predict_age_and_gender()
    #object_detection()
    with concurrent.futures.ProcessPoolExecutor() as executer:
            f1= executer.submit(attached_cam)
            f2= executer.submit(lab_win_cam)
            #f3= executer.submit(front_cam)