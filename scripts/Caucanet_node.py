#!/usr/bin/env python3
# encoding: utf-8
# license removed for brevity

import rospy
from PIL import Image
import cv2 as cv
import numpy as np
import time
import os
import datetime


from keras.models import load_model

from orca_msgs.msg import Object
from orca_msgs.msg import ObjectArray

from orca_msgs.msg import box2d
from orca_msgs.msg import box2d
from orca_msgs.msg import box2ds

import scipy.misc




# !!!!!!!!!!!!!!!在ros下运行的话，文件目录是catkin_ws,而不是.py文件所在的文件目录
path="home/catkin_ws/src/Simple-cnn-node-for-barrel-colour-classification/scripts/"
model=load_model(path+"baseline.h5")
mean_image=np.load(path+"meanimage.npy")
size=(64,64)

boxes_listened=box2ds()

def predict(frame):
    global boxes_listened
    global path
    global model
    global size 
    global mean_image
    
    num = 0

    for i in boxes_listened.boxes:
        x_max=i.x_max
        x_min=i.x_min
        y_min=i.y_min
        y_max=i.y_max
        # if x_max > 1900 or x_min <=1 or y_max > 1010 or x_min <=1 :
        #     continue
        try:
            if y_min + 25 > y_max or x_min + 15 > x_max :
                continue 
            np.array( Image.fromarray( frame[y_min:y_max,x_min:x_max,:]).resize(size,Image.ANTIALIAS) )
            num = num + 1  
        except:
            continue
        

    res_obj = []

    #网络输入
    predictX = np.empty((num,64,64,3))
    
    #画框
    cnt=0
    #把图片用框分割成小图,并放入输入中
    for i in boxes_listened.boxes:
        x_max=i.x_max
        x_min=i.x_min
        y_min=i.y_min
        y_max=i.y_max
        # print(y_min,"      ",y_max,"          ",x_min,"      ",x_max)
        # if x_max > 1900 or x_min <=1 or y_max > 1010 or x_min <=1 :
        #     continue
        
        try:
            if y_min + 25 > y_max or x_min + 15 > x_max :
                continue 

            cp = frame[y_min:y_max,x_min:x_max,:]
            cp = cp[...,::-1]
            X=np.array( Image.fromarray( cp ).resize(size,Image.ANTIALIAS) )
            predictX[cnt]=X
            res_obj.append(i)
            cnt=cnt+1
        except:
            continue
    
    #输入网络
    predictX=predictX.astype('float32')
    predictX -= mean_image
    predictX /= 128.
    
    predict_start=time.time()
    predictY = model.predict(predictX)
    predict_end=time.time()
    
    #输出结果
    print("-----Success-------",time.time())
    print("predict %d barrel(s) :",cnt)
    print("Time cost for predict :{:.5f}".format(predict_end-predict_start))
    
    for i in range(0,cnt):
        print(i,"     {:.2f},   {:.2f},  {:.2f},  {:.2f}".format(predictY[i][0],predictY[i][1],predictY[i][2],predictY[i][3]))
        print()
    print(" done ",str(datetime.datetime.now()))
    print()
    
    return predictY,res_obj
    


def listen_callback(data):
    global boxes_listened
    boxes_listened=data


def talker():
    
    global boxes


    rospy.init_node("resnet", anonymous=True)
    pub = rospy.Publisher("barrels", ObjectArray, queue_size=1)
    rospy.Subscriber("box2ds", box2ds, listen_callback)
    
    cap = cv.VideoCapture(0)  # 打开摄像头
    cap.set(cv.CAP_PROP_FRAME_WIDTH,1920)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,1024)


    camera_matrix=np.array([[971.0575,0,947.9791],
                            [0,974.7013,542.2467],
                            [0,0,1]])

    distort_param=np.array([-0.313640,0.079529,-0.001175,0.000956])

    w , h = ( 1920 , 1024 )

    newcameramtx , roi = cv.getOptimalNewCameraMatrix(camera_matrix,distort_param,(w,h),1,(w,h))
    mapx , mapy = cv.initUndistortRectifyMap(camera_matrix,distort_param,None,newcameramtx,(w,h),5)

    aaaa = 0
    while (True):
        hx, frame = cap.read()  # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像        

        print(" node get image ",str(datetime.datetime.now()))
        if hx is False:
            print('read video error')
            exit()
        
        
        
        arr_object = ObjectArray()

        #如果boxes不为空,则预测
        if boxes_listened.boxes :
            
            predict_result , res_obj = predict(frame)
            header=rospy.Header()
            header.stamp=rospy.Time.now()
            header.frame_id = "hrt"
            arr_object.header=boxes_listened.header
            
            #遍历预测结果
            for cnt in range(0,len(predict_result)):
        
                # 结果画在图片上
                cv.rectangle(frame,(res_obj[cnt].x_min,res_obj[cnt].y_max),(res_obj[cnt].x_max,res_obj[cnt].y_min),(255,0,0),3)            
                cv.putText(frame,"{:.2f},{:.2f},{:.2f},{:.2f}".format( predict_result[cnt][0],
                                                                       predict_result[cnt][1],
                                                                       predict_result[cnt][2],
                                                                       predict_result[cnt][3]),
                                                                       (res_obj[cnt].x_min,res_obj[cnt].y_min),
                                                                       cv.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

                #封装预测结果
                if predict_result[cnt][3] == max(predict_result[cnt]):
                    cnt+=1    
                    continue
                
                a_object = Object()
                a_object.dimensions.x=res_obj[cnt].boundingbox.pose.position.x
                a_object.dimensions.y=res_obj[cnt].boundingbox.pose.position.y
                a_object.dimensions.z=res_obj[cnt].boundingbox.pose.position.z
                class_=max(predict_result[cnt])
            
                # net_output : 红蓝黄无
                if predict_result[cnt][0] == class_:	
                    a_object.class_id = 0
                if predict_result[cnt][1] == class_:
                    a_object.class_id = 1
                if predict_result[cnt][2] == class_:
                    a_object.class_id = 2
                arr_object.objects.append(a_object)
                cnt+=1
                print( "cnt+1,桶" )
        
            boxes_listened.boxes.clear()        
            pub.publish(arr_object)

                
        
        
        
        # frame = cv.remap(frame , mapx , mapy , cv.INTER_LINEAR)
        cv.imshow('frame', frame )
        
                
        if cv.waitKey(1) & 0xFF == ord('z'):       # 按q退出
            break
	
    cap.release()   # 释放摄像头
    cv.destroyAllWindows() 


if __name__ == '__main__':
    
    try:
        talker()
        _#thread.start_new_thread( talker, () )
        
    except rospy.ROSInterruptException:
        pass