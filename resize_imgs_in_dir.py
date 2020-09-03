import cv2,glob,os
l=os.listdir("X:\\parameswar's\\number_plates\\np_vd_4")
print((l[0:5]))
c=568
for i in range(len(l)):
    img_path="X:\\parameswar's\\number_plates\\np_vd_4\\"+l[i]
    img=cv2.imread(img_path)
    im=cv2.resize(img,(416,416))
    cv2.imwrite(f"X:\\parameswar's\\number_plates\\to_freelancer\\{(str(c))}.jpg",im)
    c=c+1
    print(c)
