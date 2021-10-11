#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
from pandas.io.pytables import DataIndexableCol
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.metrics import  confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# file=pd.read_csv("final.csv")
file=pd.read_csv("final.csv", encoding = 'latin-1')

ChildBodyIndexes=["sex","agemons","weight","height","BMI","zbmi","zwaz","zwhz","TTDD_WHO_BMI"]

InfantIndexes = ["Stress_khimangthai","So_tuan_mang_thai","Hinhthuc_de",
"Me_tangcan_khi_mangthai","Can_so_sinh","Bu_sua_me",
"An_sua_bot_6_thang_dau","Thang_cai_sua","Thang_an_dam","HauAn_khi_an_dam"]

ChildEatingHabit = ["So_bua_sang", "An_theo_y_thich", "Uong_sua_truoc_ngu",
"Hau_an", "Thoi_gian_an", "Luong_an", "Thich_ngot", "Thich_beo", "Thich_nac",
"Thich_trung", "Thich_rau_hoaqua", "So_lan_an_ngucoc", "So_lan_an_dam", "So_lan_an_Sua", 
"So_lan_an_banh_keo", "So_lan_uong_nuoc_ngot"]

ChildExerciseHabits = [ "Vandong_o_truong", "Time_van_dong_manh", "Thich_van_dong",
"Time_hoatdongtinhtai_o_truong", "ThoigianxemTV_dienthoai", "Homnay_chau_gap_kho_khan_khi_di_lai",
"Homnay_chau_gap_kho_khan_cham_soc_ban_than", "Homnay_chau_gap_kho_khan_trong_cong_viec_hang_ngay", 
"Homnay_chau_dau_don_kho_chiu", "Homnay_chau_lo_au_buon_phien", "Hieudong", "Tron_hoc", "Nhut_nhat",
"Thuong_lam_hong_do", "Danh_nhau_voi_ban", "Khong_duoc_ban_quy_men", "Hay_lo_lang", "Thich_mot_minh",
"Hay_cau_kinh", "Hay_buon_chan", "Thuong_co_vet_muc_tren_nguoi", "Thuong_mut_ngon_tay", "Thuong_can_mong_tay", 
"Thuong_Nghi_hoc", "Thuong_khong_vang_loi", "Mat_tap_trung", "So_hai", "Bon_chon", "Noi_doi", "An_trom", "Tu_lam_ban_minh",
"Keu_dau", "Khoc_khi_di_hoc", "Noi_lap", "Kho_noi_chuyen", "Bat_nat_ban", "Gio_di_ngu", "Gio_thuc_day", "Time_ngu_toi"]

ParentIndexes = ["Tuoi_CHA","BMI_CHA","Tuoi_ME","BMI_ME","Tinhtrang_honnhan",]

ParentRoutine = ["Bome_Hutthuocla_Truockethon","Bome_sodieuthuochut1ngay_truockethon",
"Bome_dudinhcaithuoc_Truockethon","Hut_thuoc_khi_mang_thai","So_dieu_thuoc_hut_khi_mang_thai"
,"Du_dinh_cai_thuoc_khi_mang_thai","Hut_thuoc_hien_tai","Sodieuthuochutngay_hientai","Dudinhcaithuoc_hientai"
,"Solanuongruou_truockethon","Luongruouuong_truockethon","Uongtren6chenruou_truockethon","Solanuongruou_khimangthai"
,"Luongruouuong_khimangthai","Uongtren6chenruou_Khimangthai","Solanuongruou_hientai","Luongruouuong_hientai","Uongtren6chenruou_hientai",]



cate_of_BMI=[] #Categories of BMI
for i in range(len(file["BMI"])):
    if file["BMI"][i]>=40:
        cate_of_BMI.append(0)#"Béo phì độ III"
    elif 35<=file["BMI"][i]<40:
        cate_of_BMI.append(1)#"Béo phì độ II"
    elif 30<=file["BMI"][i]<35:
        cate_of_BMI.append(2)#"Béo phì độ I"
    elif 25 <= file["BMI"][i] < 30 :
        cate_of_BMI.append(3)#"Tiền béo phì"
    elif 18.5<=file["BMI"][i]<25:
        cate_of_BMI.append(4)#"Bình thường"
    else:
        cate_of_BMI.append(5)#"Thiếu năng lượng trường diễn"
file["Categories of BMI"]=cate_of_BMI


Y=file["Categories of BMI"]
# reduant_columns=["TTDD_WHO_BMI","zbmi","CODE",
# "CoDe_check","So_kg_muon_giam","HOTEN","LOP",
# "TRUONG","NGAYtraloi","BMI","Categories of BMI","Shape_real","zwaz",
# "sex","agemons","weight","height","qhc","boy_shape_identification","thich_hinh_dang_con_trai","girl_shape_identification"
# ,"thich_hinh_dang_con_gai","nhan_dinh_hinh_dang"
# , "Nhanthuc_hinhdang_conTrai", "Thich_hinhdang_conGai", "Nhanthuc_hinhdang_conGai", "So_lan_an_dam", "So_lan_an_Sua", 
#  "Thich_hinhdang_conTrai","Danh_gia_shape_tre","Me_tu_danh_gia_TTDDcon"]
    #columns don't have any benifit



reduant_columns=["Quan_huyen","Nghe_CHA","Nghe_ME","TTDD_WHO_BMI","zbmi","CODE","zwhz","BMI","Categories of BMI","height","weight","zwaz","zhaz","agemons","sex"]

action_col = ["Homnay_chau_gap_kho_khan_khi_di_lai", "Homnay_chau_gap_kho_khan_cham_soc_ban_than",
"Homnay_chau_gap_kho_khan_trong_cong_viec_hang_ngay", "Homnay_chau_dau_don_kho_chiu",
"Homnay_chau_lo_au_buon_phien", "Hieudong", "Tron_hoc", "Nhut_nhat", "Thuong_lam_hong_do",
"Danh_nhau_voi_ban", "Khong_duoc_ban_quy_men", "Hay_lo_lang", "Thich_mot_minh", "Hay_cau_kinh", 
"Hay_buon_chan", "Thuong_co_vet_muc_tren_nguoi", "Thuong_mut_ngon_tay", "Thuong_can_mong_tay", 
"Thuong_Nghi_hoc", "Thuong_khong_vang_loi", "Mat_tap_trung", "So_hai", "Bon_chon", "Noi_doi", "An_trom",
"Tu_lam_ban_minh", "Keu_dau", "Khoc_khi_di_hoc", "Noi_lap", "Kho_noi_chuyen", "Bat_nat_ban"]
cols = (list(file.columns))
count = 0
for col in cols:
    print("feature "+str(count)+": "+ col)
    count = count + 1

file = file.drop(columns=reduant_columns,axis=1)
file = file.drop(columns=action_col,axis=1)
# file = file[ParentRoutine]
mycol = file.columns

file = file.replace(" ", 0)
# test = file["Tuoi_CHA"].values

# for row in test:
#     try:
#         float(row)
#     except:
#         if row == " ":
#             print("thue")
#         print(row)

# print(file)
file = file.astype(float)


## simple heatmap of correlations (without values)
corr = file.corr()
plt.figure(figsize = (15, 10))

# Heatmap of correlations
# sns.heatmap(corr, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
# plt.title('Correlation Heatmap');
# plt.savefig("corr.png")


from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
onc=OneHotEncoder()
MMC=MinMaxScaler(feature_range=(0,1))

num_in_file=[name for name in file.columns if file[name].dtype=="int64"or file[name].dtype=="float64"]

# quit()
X=file[num_in_file]
cols = (list(X.columns))
count = 0
for col in cols:
    print("feature "+str(count)+": "+ col)
    count = count + 1


from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)
    #separate data into train set and test set
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

# print(X_train)

print("Random forest")
RFC=RandomForestClassifier()
RFC.fit(X_train,Y_train)
print(RFC.score(X_test,Y_test))

print(classification_report(Y_test.values,RFC.predict(X_test),labels=[0,1,2,3,4,5],digits = 4))

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

print("training set")
print(mean_squared_error(Y_train.values, RFC.predict(X_train), squared=False))
print(accuracy_score(Y_train.values, RFC.predict(X_train)))


print("testing set")
print(mean_squared_error(Y_test.values, RFC.predict(X_test), squared=False))
print(accuracy_score(Y_test.values, RFC.predict(X_test)))



print("Decision tree") 
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
clf = clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)

print(classification_report(Y_test.values,clf.predict(X_test),labels=[0,1,2,3,4,5],digits = 4))

print("training set")
print(mean_squared_error(Y_train.values, clf.predict(X_train), squared=False))
print(accuracy_score(Y_train.values, clf.predict(X_train)))


print("testing set")
print(mean_squared_error(Y_test.values, clf.predict(X_test), squared=False))
print(accuracy_score(Y_test.values, clf.predict(X_test)))





text_representation = tree.export_text(clf)
# print(text_representation)

#draw tree graph
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  feature_names=X_train.columns,
                filled=True, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')



sorted_idx = clf.feature_importances_.argsort()
print("????")
# print(clf.feature_importances_)
col_name = [mycol[index] for index in sorted_idx]
plt.barh(col_name, clf.feature_importances_[sorted_idx])
plt.subplots_adjust(bottom=0.15, left=0.3)
plt.xlabel("Entropy")
plt.ylabel("Feature Importance")
plt.savefig("Importance.png")


from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification

clf = AdaBoostClassifier(n_estimators=100, random_state=0)
clf = clf.fit(X_train,Y_train)

print(classification_report(Y_test.values,clf.predict(X_test),labels=[0,1,2,3,4,5],digits = 4))

print("training set")
print(mean_squared_error(Y_train.values, clf.predict(X_train), squared=False))
print(accuracy_score(Y_train.values, clf.predict(X_train)))


print("testing set")
print(mean_squared_error(Y_test.values, clf.predict(X_test), squared=False))
print(accuracy_score(Y_test.values, clf.predict(X_test)))


print("XGB ")
import xgboost as xgb
# fit model no training data
XGB = xgb.XGBClassifier(objective='binary:logistic',booster = 'gbtree', colsample_bylever = 0.8, colsample_bytree=1,learning_rate = 0.025, 
                        max_depth = 7, min_child_weight = 11, scale_pos_weight=1,n_estimators = 150, random_state = 1, seed = 1,  eta=0.3, subsample= 0.8)
XGB.fit(X_train,Y_train)


print(classification_report(Y_test.values,XGB.predict(X_test),labels=[0,1,2,3,4,5],digits = 4))


print("training set")
print(mean_squared_error(Y_train.values, XGB.predict(X_train), squared=False))
print(accuracy_score(Y_train.values, XGB.predict(X_train)))


print("testing set")
print(mean_squared_error(Y_test.values, XGB.predict(X_test), squared=False))
print(accuracy_score(Y_test.values, XGB.predict(X_test)))


print("RNN")
print(len(X_train.values[0]))
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(24, input_dim=len(X_train.values[0]), activation='relu'))
model.add(Dense(12, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(1, activation='linear'))


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])



model.fit(X_train, Y_train, epochs=3, batch_size=1)


raw_output = model.predict(X_test)

predict = []
for y in raw_output:
    print(y[0])
    predict.append(round(y[0]))


# print(predict)

print(classification_report(Y_test.values,predict,labels=[0,1,2,3,4,5],digits = 4))

quit()
#0.9825783972125436
print("LR")
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(X_train,Y_train)
print(LR.score(X_test,Y_test))
print(classification_report(Y_test.values,LR.predict(X_test),labels=[0,1,2,3,4,5],digits = 4))
#0.9790940766550522

print("SVC")
from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
print(svc.score(X_test,Y_test))

print(classification_report(Y_test.values,svc.predict(X_test),labels=[0,1,2,3,4,5],digits = 4))
#0.975609756097561

print("naive_bayes")
from sklearn.naive_bayes import GaussianNB
GB=GaussianNB()
GB.fit(X_train,Y_train)
print(GB.score(X_test,Y_test))
# 0.818815331010453
print(classification_report(Y_test.values,GB.predict(X_test),labels=[0,1,2,3,4,5],digits = 4))
#0.9094076655052264

print("KNeighborsClassifier")
from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier(n_neighbors=6)
KNN.fit(X_train,Y_train)
print(KNN.score(X_test,Y_test))
print(classification_report(Y_test.values,KNN.predict(X_test),labels=[0,1,2,3,4,5],digits = 4))
#0.9094076655052264

