#!/usr/bin/env python3
# encoding: utf-8
# license removed for brevity

import rospy

import cv2 as cv
import numpy as np
from keras.models import load_model
from PIL import Image
import time
import os

from net.msg import barrel
from net.msg import barrels
from net.msg import BoundingBox
from net.msg import BoundingBoxArray





g_count = 0

# !!!!!!!!!!!!!!!在ros下运行的话，文件目录是catkin_ws,而不是.py文件所在的文件目录
path = "src/net/scripts/" 
model=load_model(path+"resnet5_barrel.h5")
size=(64,64)
mean_image=np.load(path+"meanimage.npy")
boxes_listened=BoundingBoxArray()



def transform( object_ : BoundingBoxArray ):
    boxes_2d_=[]
    for i in object_.boxes:
        # x_image_min,x_image_max,y_image_min,y_image_max
        boxes_2d_.append([200,300,200,300])
    return boxes_2d_
    

def predict(frame,boxes_2d ):
    
    global boxes_listened
    global path
    global model
    global size
    global mean_image
    
    #网络输入
    predictX=np.empty((int(len(boxes_listened.boxes)),64,64,3))
    
    #画框
    cnt=0
    #把图片用框分割成小图,并放入输入中  
    for i in boxes_listened.boxes:
        cv.rectangle(frame,(boxes_2d[cnt][0],boxes_2d[cnt][3]),(boxes_2d[cnt][1],boxes_2d[cnt][2]),(255,0,0),3)
        X=np.array(Image.fromarray(frame[boxes_2d[cnt][2]:boxes_2d[cnt][3],boxes_2d[cnt][0]:boxes_2d[cnt][1],:]).resize(size,Image.ANTIALIAS))
        predictX[cnt]=X
        cnt=cnt+1
    
    
    
    
    #输入网络
    predictX=predictX.astype('float32')
    predictX-=mean_image
    predictX/=128.
    
    predict_start=time.time()
    predictY=model.predict(predictX)
    predict_end=time.time()
    
    #输出结果
    print("-----Success-------")
    print("predict %d barrel(s) :",cnt)
    print("Time cost for predict :{:.5f}".format(predict_end-predict_start))
    
    for i in range(0,cnt):
        print(i)
        #print(i+1,"   ",boxes_2d[cnt][0],"  ",boxes_2d[cnt][1],"  ",boxes_2d[cnt][2],"  ",boxes_2d[cnt][3],end="")
        print("     {:.2f},   {:.2f},  {:.2f},  {:.2f}".format(predictY[i][0],predictY[i][1],predictY[i][2],predictY[i][3]))
        print()
        
    print()
    
    return predictY
    

def listen_callback(data):
    global  g_count
    g_count += 1
    print("new data")
    global boxes_listened
    boxes_listened=data


#在队列中搜索时间戳与点云输入到欧式聚类节点的时间最匹配的图片
def search_image(image_queue,header_begin):
    print("queue size is ",len(image_queue[0]))
    
    if len(image_queue[0])==1:
        return image_queue[0][0][0]
    
    min_=header_begin.stamp.to_sec()    
    print(min_)
    o=0
    
    for i in image_queue[0]:
        if min_ > abs(i[1].to_sec()-header_begin.stamp.to_sec()):
            min_=abs(i[1].to_sec()-header_begin.stamp.to_sec())
        else:
            image_queue[0]=image_queue[0][o:]
            return i[0]
        o=o+1
        
    return image_queue[0][-1][0]
    

def talker():
    global g_count
    global boxes
    
    rospy.init_node("resnet", anonymous=True)
    pub = rospy.Publisher("barrels", barrels, queue_size=10)
    rospy.Subscriber("detected_bounding_boxs", BoundingBoxArray, listen_callback)
    
    #存储图片的队列
    image_queue=[]
    image_queue_=[image_queue]
    
    
    cap = cv.VideoCapture(0)  # 打开摄像头
    while (True):
        hx, frame = cap.read()  # 开始用摄像头读数据，返回hx为true则表示读成功，frame为读的图像
        
        image_queue_[0].append((frame,rospy.Time.now()))
        
        if hx is False:
            print('read video error')
            exit()
	
    
        arr_barrel = barrels()
        #如果boxes不为空,则预测
        if boxes_listened.boxes :
            boxes_2d=transform( boxes_listened )
            image=search_image(image_queue_,boxes_listened.header)
            predict_result=predict(image,boxes_2d)
            header=rospy.Header()
            header.stamp=rospy.Time.now()
            header.frame_id = "hrt"
            arr_barrel.header=boxes_listened.header
            boxes_=boxes_listened
	
        #把预测结果画在frame上
        num=0
        if 1 == g_count:
            num=len(boxes_.boxes)
            print("b")
        print("aa")
        for i in range(0,num):
            print( i ,"  " , len(boxes_.boxes), "   ",len(boxes_2d) )
            print("c c")
            cv.rectangle(image,(boxes_2d[i][0],boxes_2d[i][3]),(boxes_2d[i][1],boxes_2d[i][2]),(255,0,0),3)
            cv.putText(image,"{:.2f},{:.2f},{:.2f},{:.2f}".format(predict_result[i][0],predict_result[i][1],predict_result[i][2],predict_result[i][3]),(boxes_2d[i][0],boxes_2d[i][2]),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
            a_barrel = barrel()
            a_barrel.x=boxes_.boxes[i].pose.position.x
            a_barrel.y=boxes_.boxes[i].pose.position.x
            a_barrel.z=boxes_.boxes[i].pose.position.x
            a_barrel.red=predict_result[i][0]
            a_barrel.yellow=predict_result[i][1]
            a_barrel.blue=predict_result[i][2]
            a_barrel.not_object=predict_result[i][3]
            arr_barrel.barrels.append(a_barrel)
            num=len(boxes_.boxes)
            
        
	
        
        pub.publish(arr_barrel)
	
	
        cv.namedWindow('frame', cv.WINDOW_AUTOSIZE)     # 窗口设置为自动调节大小
        cv.imshow('frame', frame )
        
        
        
        if boxes_listened.boxes:
            cv.namedWindow('object', cv.WINDOW_AUTOSIZE)     # 窗口设置为自动调节大小
            cv.imshow('object', image if boxes_listened.boxes else frame )
        
        if cv.waitKey(1) & 0xFF == ord('z'):       # 按q退出
            break
	
    cap.release()   # 释放摄像头
    cv.destroyAllWindows() 


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
