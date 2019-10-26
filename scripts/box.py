#!/usr/bin/env python
# license removed for brevity
import rospy
import numpy as npy

from std_msgs.msg import String
from std_msgs.msg import Header 


from orca_msgs.msg import BoundingBox
from orca_msgs.msg import BoundingBoxArray
from orca_msgs.msg import box2d
from orca_msgs.msg import box2ds


import math



pub=0

camera_matrix = None
distort_param = None
rotate_transform_matrix = None

# [def transform( i  ):
    
#     global pub, camera_matrix, distort_param , rotate_transform_matrix
    
#     #x_min y_min
#     XYZ1=npy.mat([[i.pose.position.x-i.dimensions.x/2],
#                   [i.pose.position.y+i.dimensions.y/2],
#                   [i.pose.position.z-i.dimensions.z/2],
#                   [1]])
    
#     #x_max y_max
#     XYZ2=npy.mat([[i.pose.position.x-i.dimensions.x/2],
#                   [i.pose.position.y-i.dimensions.y/2],
#                   [i.pose.position.z+i.dimensions.z/2],
#                   [1]])
    
    

#     XYZ2 = rotate_transform_matrix * XYZ2 
#     XYZ1 = rotate_transform_matrix * XYZ1     

#     def undistort(k1,k2,p1,p2,x,y):
#         r2 = x*x + y*y
#         aaa = 1 + k1*r2 + k2*r2*r2
#         return x*aaa + 2*p1*x*y + p2*(r2+2*x*x) , y*aaa + p1*(r2+2*y*y) + 2*p2*x*y


#     u_1,v_1 = undistort(distort_param[0],distort_param[1],distort_param[2],distort_param[3], 
#                         XYZ1[1,0]/XYZ1[0,0] , XYZ1[2,0]/XYZ1[0,0] )
    
#     u_2,v_2 = undistort(distort_param[0],distort_param[1],distort_param[2],distort_param[3], 
#                         XYZ2[1,0]/XYZ2[0,0] , XYZ2[2,0]/XYZ2[0,0]) 
    
#     def projection(u,v,projection_matrix):
#         return projection_matrix[0,0]*u + projection_matrix[0,2] , projection_matrix[1,1]*v + projection_matrix[1,2]

#     u_1,v_1 = projection(u_1,v_1,camera_matrix)
#     u_2,v_2 = projection(u_2,v_2,camera_matrix)

#     def in_image(x,y):
#         return x>=1 and x<=1919 and y>=1 and y<=1023
#     if( not in_image(u_1,v_1) or not in_image(u_2,v_2)  ):
#         return None

#     return int(u_1),int(u_2),int(v_1),int(v_2)

def transform( i  ):
    
    global pub, camera_matrix, distort_param , rotate_transform_matrix
    
    #x_min y_min
    XYZ1=npy.mat([[i.pose.position.x],
                  [i.pose.position.y],
                  [i.pose.position.z],
                  [1]])
    
    XYZ1 = rotate_transform_matrix * XYZ1     

    def undistort(k1,k2,p1,p2,x,y):
        r2 = x*x + y*y
        aaa = 1 + k1*r2 + k2*r2*r2
        return x*aaa + 2*p1*x*y + p2*(r2+2*x*x) , y*aaa + p1*(r2+2*y*y) + 2*p2*x*y


    u_1,v_1 = undistort(distort_param[0],distort_param[1],distort_param[2],distort_param[3], 
                        XYZ1[1,0]/XYZ1[0,0] , XYZ1[2,0]/XYZ1[0,0] )
    
    
    def projection(u,v,projection_matrix):
        return projection_matrix[0,0]*u + projection_matrix[0,2] , projection_matrix[1,1]*v + projection_matrix[1,2]

    u_1,v_1 = projection(u_1,v_1,camera_matrix)
    

    def in_image(x,y):
        return x>=1 and x<=1919 and y>=1 and y<=1023
    if not in_image(u_1,v_1) :
        return None



    x_par = abs( i.pose.position.x ) / 15;
    y_par = ( i.pose.position.y ) / 6;

    dis = math.sqrt(i.pose.position.x*i.pose.position.x + i.pose.position.y*i.pose.position.y)    
    coeffi =  (20 - dis )/20 
    

    try:    
        h_ = int ( 90 *  math.pow(coeffi ,2  ) )
        w_ = int ( 40 *  math.pow(coeffi ,1.7  ) )
    

        u_1 = u_1 + y_par * 5
        v_1 = v_1 + x_par * 20

        return int(u_1) - w_ , int(u_1) + w_ , int(v_1) - h_ , int(v_1) + h_
    except:
        return None
 

    
def call_back(box):
    
    global pub

    box2ds_=box2ds()
    box2ds_.header = box.header
    for i in box.boxes:
        one_box=box2d()
        one_box.boundingbox = i
        res_box = transform(i)
        if not res_box:
            continue
        one_box.x_min,one_box.x_max,one_box.y_min,one_box.y_max = res_box
        box2ds_.boxes.append(one_box)
    
    pub.publish(box2ds_)

        
def talker():

    global pub, camera_matrix, distort_param ,rotate_transform_matrix



    x__ , y__ , z__ , O =0.05   , 0.00,     -0.13     ,-0.038



    rotate_transform_matrix = npy.mat( [[npy.cos(O),-npy.sin(O),0,x__],
                                       [npy.sin(O),npy.cos(O),0,y__],
                                       [0,0,1,z__],
                                       [0,0,0,1]]  )

    camera_matrix=npy.mat([[-971.0575,0,947.9791],
                            [0,-974.7013,542.2467],
                            [0,0,1]])

    distort_param=[-0.313640,0.079529,-0.001175,0.000956]

    rospy.init_node("box_2d", anonymous=False)
    rospy.Subscriber("/detected_bounding_box", BoundingBoxArray, call_back)
    pub = rospy.Publisher("box2ds", box2ds, queue_size=1)

    rospy.spin()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
