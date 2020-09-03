import time
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np


##############
framework='tf'
weights='./checkpoints/yolov4-416'
size=416
tiny=True
model='yolov4'
video='E:\\New folder (2)\\tolls_for_labelling\\number_plates\\np_vd_10.ts'#5
iou=0.45
score=0.15


c=0
a_v=0
ccc=0
def cal(l,frame):
    global c
    global a_v
    global ccc
    x_c=(l[2]+l[0])/2
    y_c=(l[3]+l[1])/2
    if y_c>=260 and y_c<=360:
        lo=1
        if a_v!=lo and ccc==0:
            a_v=1
            c=c+1
            crop_img=frame[l[1]:l[3],l[0]:l[2]]
            cv2.imwrite("X:\\parameswar's\\number_plates\\np_vd_11\\"+str(c)+".jpg",crop_img)
            ccc=15
    else:
        lo=0
        if a_v!=lo:
            a_v=0
        if ccc>0:
            ccc=ccc-1
        else:
            ccc=0

    print("ccc:",ccc)
    return c
input_size = size
video_path = video
#21-26 ----6 frames in ROI
print("Video from: ", video_path )

vid = cv2.VideoCapture(video_path)
# width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
# height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
# size = (width, height)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('./data/output_np_vd_2.ts', fourcc, 20.0, size)

   
saved_model_loaded = tf.saved_model.load(weights, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']
empty_f=1
labelled=1
t_c=0
while True:
    return_value, frame = vid.read()
    if return_value:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)
        # print(image.size)
    else:
        raise ValueError("No image! Try with another video format")
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    save_img=cv2.resize(frame,(608,608))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    #print(image_data.shape)
    prev_time = time.time()
    batch_data = tf.constant(image_data)
    
    pred_bbox = infer(batch_data)

    #print(pred_bbox)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        #print("boxes:",(boxes))
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=iou,
        score_threshold=score
    )
    pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
    
    op=int(pred_bbox[3])
    if op==0:
        t_c=t_c+1
        if t_c%6==0:

            # print(0)
            cv2.imwrite(f"X:\\parameswar's\\number_plates_for_labelling\\Empty_frames\\np_vd_11\\{str(empty_f)}.jpg",frame)
            empty_f=empty_f+1
        cv2.line(frame,(0,260),(720,260),(0,0,255),5)
        cv2.line(frame,(0,360),(720,360),(0,0,255),5)
        cv2.rectangle(frame,(60,30), (250,70), (0,0,0), -1)
        cv2.putText(frame, "count:"+str(int(c)), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (70,11,219), 2)
        frame = np.asarray(frame)
        cv2.namedWindow("np_vd_11", cv2.WINDOW_AUTOSIZE)
        # out.write(frame)
        cv2.imshow("np_vd_11", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    else:
        # print(1)
        num_classes = len(classes)
        image_h, image_w, _ = frame.shape
        #print(image_h)
        #print(image_w)
        out_boxes, out_scores, out_classes, num_boxes = pred_bbox
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            t_c=t_c+1
            if t_c%6==0:
                x=coor[1]+(coor[3]-coor[1])/2
                y=coor[0]+(coor[2]-coor[0])/2
                width=coor[3]-coor[1]
                height=coor[2]-coor[0]
                cla=0
                file1=open("X:\\parameswar's\\number_plates_for_labelling\\labelled_nps\\np_vd_11\\"+str(labelled)+".txt","w")
                s=f"{str(cla)} {str(x)} {str(y)} {str(width)} {str(height)}"
                file1.write(s)
                cv2.imwrite("X:\\parameswar's\\number_plates_for_labelling\\labelled_nps\\np_vd_11\\"+str(labelled)+".jpg",save_img)
                labelled=labelled+1
            #print(coor)
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)
            #print(f"{coor[1]}:{coor[0]}:{coor[3]}:{coor[2]}")
            x_min,y_min,x_max,y_max=int(coor[1]),int(coor[0]),int(coor[3]),int(coor[2])
            #y_ce=(y_max+y_min)/2
            
           
            c=cal([x_min,y_min,x_max,y_max],frame)

            # print(out_scores)
            cnf="{:.2f}".format(out_scores[0][0])


            
            cv2.rectangle(frame, (x_min,y_min),(x_max,y_max), (51,51,255), 3)
            cv2.line(frame,(0,260),(720,260),(0,0,255),5)
            cv2.line(frame,(0,360),(720,360),(0,0,255),5)
            cv2.rectangle(frame,(60,30), (250,70), (0,0,0), -1)
            cv2.putText(frame, "count:"+str(int(c)), (60,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (70,11,219), 2)
            cv2.putText(frame,cnf,(x_min,y_min-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(frame,"*",((x_max+x_min)//2,(y_max+y_min)//2),cv2.FONT_HERSHEY_SIMPLEX, 1, (51,51,255), 2)
            frame = np.asarray(frame)
            cv2.namedWindow("np_vd_11", cv2.WINDOW_AUTOSIZE)
            # out.write(frame)
            cv2.imshow("np_vd_11", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    curr_time = time.time()
    exec_time = curr_time - prev_time
    info = "time: %.2f ms" %(1000*exec_time)
    print(info)