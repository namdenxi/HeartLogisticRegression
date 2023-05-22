from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('./home/media/files/data_heart.csv')
result = ['Bạn không mắc bệnh tim','Bạn có nguy cơ mắc bệnh tim']

le = LabelEncoder()
data['ChestPainType'] = le.fit_transform(data['ChestPainType'])
transform_list = ['Sex','RestingECG','ExerciseAngina','ST_Slope']
data[transform_list] = data[transform_list].apply(le.fit_transform)

X = np.asarray(data.drop(['HeartDisease'] , axis = 1))
y = np.asarray(data['HeartDisease']).reshape(-1,1)
X_train , X_test , y_train , y_test = train_test_split(X,y , test_size=0.2 , train_size=0.8 , random_state = 42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pred_score = accuracy_score(y_test, y_pred)*100

# Create your views here.
def Heart(request):
    text_warning = ''
    show = ''
    content = ''
    if request.method == 'POST':
        Age = request.POST.get('Age') 
        Sex = request.POST.get('Sex') 
        ChestPainType = request.POST.get('ChestPainType')
        RestingBP = request.POST.get('RestingBP')
        Cholesterol = request.POST.get('Cholesterol')
        FastingBS = request.POST.get('FastingBS')
        RestingECG = request.POST.get('RestingECG')
        MaxHR = request.POST.get('MaxHR')
        ExerciseAngina = request.POST.get('ExerciseAngina')
        Oldpeak = request.POST.get('Oldpeak')
        ST_Slope = request.POST.get('ST_Slope')

        if Age == '' or Sex == '' or ChestPainType == '' or RestingBP == '' or Cholesterol == '' or FastingBS == '' or RestingECG == '' or MaxHR == '' or ExerciseAngina == '' or Oldpeak == '' or ST_Slope == '':
            text_warning = 'Hãy nhập đầy đủ thông tin trước khi gửi đi'
        else:
            try:
                input_data = (int(Age), int(Sex), int(ChestPainType), int(RestingBP), int(Cholesterol), int(FastingBS), int(RestingECG), int(MaxHR), int(ExerciseAngina), float(Oldpeak), int(ST_Slope))
                input_data_as_numpy_array = np.asarray(input_data)
                input_data_reshaped = input_data_as_numpy_array.reshape(-1, 1).T
                prediction = model.predict(input_data_reshaped)
                show = 'show'
                content = f'Kết quả dự đoán: {result[int(prediction)]}. Độ chính xác: {round(pred_score, 2)}%'
            except:
                text_warning = 'Hãy nhập đúng giá trị'
    return render(request, 'heart.html', {'text_warning':text_warning, 'show': show, 'content': content})
