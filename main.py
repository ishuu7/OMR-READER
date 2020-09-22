import cv2
import numpy as np
import argparse
import imutils 

user1=cv2.imread(r"C:\Users\hp\Pictures\user1.png")
answerkey=cv2.imread(r"C:\Users\hp\Pictures\answerkey.png")


user1=imutils.resize(user1,height=800)
answerkey=imutils.resize(answerkey,height=800)

user1=user1[232:726,59:489]
answerkey=answerkey[232:726,59:489]

user1=imutils.resize(user1,height=800)
answerkey=imutils.resize(answerkey,height=800)


user1_1=cv2.cvtColor(user1,cv2.COLOR_BGR2GRAY)
answerkey_1=cv2.cvtColor(answerkey,cv2.COLOR_BGR2GRAY)


user1=cv2.GaussianBlur(user1_1,(13,13),10)
answerkey=cv2.GaussianBlur(answerkey_1,(13,13),10)


(T1,user1_2)=cv2.threshold(user1,135,255,cv2.THRESH_BINARY_INV)
cv2.imshow("threshold1",user1_2)
(T2,answerkey_2)=cv2.threshold(answerkey,135,255,cv2.THRESH_BINARY)
cv2.imshow("threshold2",answerkey_2)

user1_3=cv2.Canny(user1_2,75,150)
answerkey_3=cv2.Canny(answerkey_2,75,150)

(cnts2,_)=cv2.findContours(answerkey_3.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("Total no. of questions={}".format(len(cnts2)))
total_marks=len(cnts2)*4
n=answerkey_1.copy()
cv2.drawContours(n,cnts2,-1,(0,255,0),2)


(cnts1,_)=cv2.findContours(user1_3.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("No. of questions attempted by student = {}".format(len(cnts1)))
attempt=len(cnts1)
m=user1_1.copy()
cv2.drawContours(m,cnts1,-1,(0,255,0),2)




incorrect=cv2.bitwise_and(user1_2,answerkey_2)

incorrect=cv2.medianBlur(incorrect,5)

incorrect=cv2.GaussianBlur(incorrect,(9,9),0)

incorrect1=cv2.Canny(incorrect,75,200)
cv2.imshow("Canny incorrect",incorrect1)


(cnts3,_)=cv2.findContours(incorrect1.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print("No. of incorrect answers are {}".format(len(cnts3)))
wrong=len(cnts3)
p=incorrect.copy()

cv2.drawContours(p,cnts3,-1,(0,0,255),1)
cv2.imshow("Wrong",p)

marks_obt=(attempt-wrong)*4-wrong

print("Total marks obtained",marks_obt)

cv2.waitKey(0)







