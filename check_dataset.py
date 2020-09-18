
import os,cv2,glob
l_pics=glob.glob("X:\\parameswar's\\toll_project\\toll_dataset_collection\\np_charcter_labelled\\np_vd_11\\*.jpg")
l=glob.glob("X:\\parameswar's\\toll_project\\toll_dataset_collection\\np_charcter_labelled\\np_vd_11\\*")
# print(len(l))

print(len(l_pics))
# print(l_pics[0])
lll=[]
for i in range(len(l_pics)):
    if i%100==0:
        print(i)
    txt_num=(l_pics[i].split(".")[0])
    # print(txt_num)
    txt_path=txt_num+".txt"
    # print(l_pics[i])
    # print(txt_path)
    if txt_path  not in l:
        print(l_pics[i])
        # lll.append(txt_path)
        os.remove(l_pics[i])
    else:
        file2=open(txt_path,"a+")
        for line in file2:
            
            if (line[0])!='0':
                print(txt_path)
                print(int(line[0]))
                lll.append(txt_path)
                print(txt_path)
                # file2=open(txt_path,"w")
                # line[0]='0'
                # file2.write(line)
                
print(lll)
print(len(lll))

import cv2,glob
from random import shuffle
c=1
for i in range(1,12):
    img_files=f"X:\parameswar's\\number_plates_for_labelling\\final_labelling_need_to_freelancing\\np_vd_{str(i)}\\*.jpg"
    i_l=glob.glob(img_files)
    shuffle(i_l)
    for j in range(len(i_l)):
        img=cv2.imread(i_l[j])
        cv2.imwrite(f"X:\parameswar's\\number_plates_for_labelling\\empty_final_labelled_nps\\E_{str(c)}.jpg",img)
        txt_file_path=i_l[j].split(".jpg")[0]+".txt"
        # print(txt_file_path)
        file1 = open(txt_file_path,"r")
        file2 = open(f"X:\parameswar's\\number_plates_for_labelling\\empty_final_labelled_nps\\E_{str(c)}.txt","w") 
        L=file1.readlines()
        file2.writelines(L)
        file1.close()
        file2.close()
        c=c+1
        print(c)


import cv2,glob,shutil
s="X:\\parameswar's\\toll_project\\toll_dataset_collection\\all_np_labelling_collection\\obj\\"
d="X:\\parameswar's\\toll_project\\toll_dataset_collection\\all_np_labelling_collection\\test\\"
c=150001
for i in range(150001,160001):
    s_path=s+str(i)+".jpg"
    s_path_txt=s+str(i)+".txt"
    dest1 = shutil.move(s_path, d)
    dest2 = shutil.move(s_path_txt, d)


import cv2,glob
for j in range(12):
    img_files=f"X:\\parameswar's\\toll_project\\toll_dataset_collection\\np_charcter_labelled\\np_vd_{str(j)}\\*.jpg"
    des_path="X:\\parameswar's\\toll_project\\toll_dataset_collection\\np_charcter_labelled\\final_dataset_for_np_c_label\\"
    img_list=glob.glob(img_files)
    print(j)
    print(len(img_list))
    for i in range(len(img_list)):
        img=cv2.imread(img_list[i])
        cv2.imwrite(des_path+f"np_vd_{str(j)}_{str(i)}.jpg",img)
        txt_file_path=img_list[i].split(".jpg")[0]+".txt"
        # print(txt_file_path)
        file1 = open(txt_file_path,"r")
        file2 = open(des_path+f"np_vd_{str(j)}_{str(i)}.txt","w") 
        L=file1.readlines()
        file2.writelines(L)
        file1.close()
        file2.close()

import cv2,glob,os
img_files="X:\\parameswar's\\toll_project\\toll_dataset_collection\\np_charcter_labelled\\test\\*.jpg"
img_list=glob.glob(img_files)
print(len(img_list))
print(img_list[0])
for i in range(len(img_list)):
    img=cv2.imread(img_list[i])
    img=cv2.resize(img,(416,416))
    cv2.imwrite(img_list[i],img)


import os,cv2,glob

l_pics=glob.glob(f"X:\\parameswar's\\toll_project\\toll_dataset_collection\\all_np_labelling_collection\\obj\\*.jpg")
l=glob.glob(f"X:\\parameswar's\\toll_project\\toll_dataset_collection\\all_np_labelling_collection\\obj\\*")
# print(len(l))

print(len(l_pics))
# print(l_pics[0])
lll=[]
for i in range(len(l_pics)):
    if i%100==0:
        print(i)
    txt_num=(l_pics[i].split(".")[0])
    # print(txt_num)
    txt_path=txt_num+".txt"
    # print(l_pics[i])
    # print(txt_path)
    if txt_path  not in l:
        print(l_pics[i])
        # lll.append(txt_path)
        os.remove(l_pics[i])
    else:
        file2=open(txt_path,"r")
        for line in file2:
            
            if (line[0])!='0':
                print(txt_path)
                print(int(line[0]))
                lll.append(txt_path)
                print(txt_path)
                ##
                file1 = open(txt_path,"r")
                
                L=file1.readlines()
                E=[("0"+L[0][1:])]
                # print(L[0][1:])
                print(E)
                print(L)
                file2 = open(txt_path,"w") 
                file2.writelines(E)
                file1.close()
                file2.close()
                ##

                # file2=open(txt_path,"w")
                # line[0]='0'
                # file2.write(line)
                
print(lll)
print(len(lll))

X:\parameswar's\toll_project\toll_dataset_collection\need_to_automate_to_check\check_dataset.py
