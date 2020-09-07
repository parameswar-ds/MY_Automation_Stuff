
import os,cv2,glob
l_pics=glob.glob("X:\\parameswar's\\ocr_nps\\to_freelancer\\*.jpg")
l=glob.glob("X:\\parameswar's\\ocr_nps\\to_freelancer\\*")
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
                line[0]='0'
                file2.write(line)
                
print(lll)
print(len(lll))

