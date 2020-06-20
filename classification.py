import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
if __name__ == '__main__':
    dataset_path = os.path.join("~/s3data/dataset", 'citrus.csv')
    df = pd.read_csv(dataset_path)
    classes=df.get('name').values
    label=[]
    features=df.drop('name', axis=1).values
    for item in classes:
        if(item=="orange"):{
            label.append(1)
        }
        else: {
            label.append(0)
        }
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.33, random_state=0)
    #  构建SGD分类器进行训练  
    sgdClassifier = SGDClassifier(random_state=42)
    sgdClassifier.fit(X_train, y_train)
    #  y作为label已经是0,1形式，不需进一步处理  
    #  使用训练好的SGD分类器对陌生数据进行分类  

    test_predicted = sgdClassifier.predict(X_test)
    y_test=np.array(y_test).reshape(3300,1)
    test_predicted=np.array(test_predicted).reshape(3300,1)
    X_test=np.array(X_test)
    print(accuracy_score(y_test,test_predicted))
    result=np.hstack((X_test,y_test))
    result=np.hstack((result,test_predicted))
    np.savetxt('result.csv', result, delimiter=',')
    print(result)
    print("写入成功")




