from ultralytics import YOLO
import cv2
import math
# from roboflow import Roboflow
# rf = Roboflow(api_key="mBHuXaIJY0K50A4nkxUH")
# project = rf.workspace("lampard-guardiola-qpvyr").project("pests_single-izgnq")
# dataset = project.version(2).download("yolov8")
# CLASSES = sorted(project.classes.keys())

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))
    
    model=YOLO('best_m.pt')
    classNames = ['Anoplophora_chinensis', 'Apriona_germari(Hope)', 'Cerambycidae_larvae', 'Chalcophora_japonica', 'Clostera_anachoreta', 'Cnidocampa_flavescens(Walker_pupa)', 'Cnidocampa_flavescens(Walker）', 'Drosicha_contrahens_female', 'Drosicha_contrahens_male', 'Erthesina_fullo', 'Erthesina_fullo_nymph', 'Erthesina_fullo_nymph-2', 'Hyphantria_cunea', 'Hyphantria_cunea_larvae', 'Hyphantria_cunea_pupa', 'Latoia_consocia_Walker', 'Latoia_consocia_Walker_larvae', 'Micromelalopha_troglodyta(Graeser)', 'Micromelalopha_troglodyta(Graeser)_larvae', 'Monochamus_alternatus', 'Plagiodera_versicolora(Laicharting)', 'Plagiodera_versicolora(Laicharting)_larvae', 'Plagiodera_versicolora(Laicharting)_ovum', 'Psacothea_hilaris(Pascoe)', 'Psilogramma_menephron', 'Psilogramma_menephron_larvae', 'Sericinus_montela', 'Sericinus_montela_larvae', 'Spilarctia_subcarnea(Walker)_larvae', 'Spilarctia_subcarnea(Walker)_larvae-2', 'Spilarctia_subcarnea(Walker）']
    print(classNames)
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == 'Dust Mask':
                    color=(0, 204, 255)
                elif class_name == "Glove":
                    color = (222, 82, 175)
                elif class_name == "Protective Helmet":
                    color = (0, 149, 255)
                else:
                    color = (85,45,255)
                if conf>0.5:
                    cv2.rectangle(img, (x1,y1), (x2,y2), color,3)
                    cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)  # filled
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)

        yield img
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()