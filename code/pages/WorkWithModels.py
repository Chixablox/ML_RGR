import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler

def LearnWithTeacher(arr):
    st.title("Обучение с учителем")
    
    models = ["KNeighborsClassifier", "SVM", "DecisionTreeClassifier"]
    models_type = st.selectbox("Выберите модель", models)
    
    if models_type is not None:
        if models_type == "SVM":
            st.header("SVM")
            with open('data/Models/SVM.pkl', 'rb') as file:
                model = pickle.load(file)
            Pred(model, arr)
            
        elif models_type == "KNeighborsClassifier":
            st.header("KNeighborsClassifier")
            with open('data/Models/KNeighborsClassifier.pkl', 'rb') as file:
                model = pickle.load(file)
            Pred(model, arr)
            
        elif models_type == "DecisionTreeClassifier":
            st.header("DecisionTreeClassifier")
            with open('data/Models/DecisionTreeClassifier.pkl', 'rb') as file:
                model = pickle.load(file)
            Pred(model, arr)

def Ensembles(arr):
    st.title("Ансамбли")
    
    models = ["BaggingClassifier", "GradientBoostingClassifier", "StackingClassifier"]
    models_type = st.selectbox("Выберите модель", models)
    
    if models_type is not None:
        if models_type == "BaggingClassifier":
            st.header("BaggingClassifier")
            with open('data/Models/BaggingClassifier.pkl', 'rb') as file:
                model = pickle.load(file)
            Pred(model, arr)
            
        elif models_type == "GradientBoostingClassifier":
            st.header("GradientBoostingClassifier")
            with open('data/Models/GradientBoostingClassifier.pkl', 'rb') as file:
                model = pickle.load(file)
            Pred(model, arr)
            
        elif models_type == "StackingClassifier":
            st.header("StackingClassifier")
            with open('data/Models/StackingClassifier.pkl', 'rb') as file:
                model = pickle.load(file)
            Pred(model, arr)
            
def LearnWithoutTeacher(arr):
    st.title("Обучение без учителя")
    st.header("KMeans")
    with open('data/Models/KMeans.pkl', 'rb') as file:
            model = pickle.load(file)
    Pred(model, arr)

def DNN(arr):
    st.title("Нейронные сети")
    st.header("DNN")
    model = load_model('data/Models/DNNClass.h5')
    PredForDNN(model, arr)
    

def Pred(model, arr):
    pred = model.predict(arr)
    pred = pred.flatten()
    if (len(pred) > 1):
        pred_df = pd.DataFrame(data = pred, columns=["hazardous"])
        col1, col2, col3 = st.columns(3)
        with col2:
            st.write(pred_df)
    else:
        if(np.max(pred) == 1):
            st.write(f"Объект может представлять опасность, так как его класс {np.max(pred)}")
        else:
            st.write(f"Объект опасности не представляет, так как его класс {np.max(pred)}")

def PredForDNN(model, arr):
    pred = model.predict(arr)
    if (len(pred) == 1):
        pred = pred.flatten()
        if(np.argmax(pred) == 1):
            st.write(f"Объект может представлять опасность, так как его класс {np.argmax(pred)}")
        else:
            st.write(f"Объект опасности не представляет, так как его класс {np.argmax(pred)}")
    else:
        pred = [np.argmax(p) for p in pred]
        pred_df = pd.DataFrame(data = pred, columns=["hazardous"])
        col1, col2, col3 = st.columns(3)
        with col2:
            st.write(pred_df)
    

            
def WorkWithOneObject():  
    
    st.header("Введите данные")
    
    est_diameter_min = st.number_input("Минимальный диаметр:", min_value=0.001, max_value=0.5, value=0.05)
    
    est_diameter_max = st.number_input("Максимальный диаметр:", min_value=0.001, max_value=1.0, value=0.1)
    
    relative_velocity = st.number_input("Скорость объекта:", min_value=10.0, max_value=150000.0, value=8424.2)
    
    miss_distance = st.number_input("Пропущенное расстояние:", min_value=0.0, max_value=100000000.0, value=3800000.2)
    
    absolute_magnitude = st.slider('Абсолютная светимость:', 15.0, 35.0, 25.0, step = 0.1)
    
    arr = np.array([est_diameter_min, est_diameter_max, relative_velocity, miss_distance, absolute_magnitude])
    
    checkboxbox = st.checkbox("Выбрать модель")
    
    if checkboxbox:
        
        arr = arr.reshape(1, -1)
        arr = sc.transform(arr)
        
        models = ["Обучение с учителем", "Ансамбли", "Обучение без учителя", "DNN"]
    
        models_type = st.selectbox("Выберите тип модели", models)
    
        if models_type is not None:
            if models_type == "Обучение с учителем":
                LearnWithTeacher(arr)
            elif models_type == "Ансамбли":
                Ensembles(arr)    
            elif models_type == "Обучение без учителя":
                LearnWithoutTeacher(arr)
            elif models_type == "DNN":
                DNN(arr)

def WorkWithDataset():
    
    st.header("Загрузите датасет")
    load_df = st.file_uploader("Загрузите файл типа *.csv", type="csv")

    if load_df is not None:
        df = pd.read_csv(load_df)
    
        df  = df[["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "absolute_magnitude"]]
        st.write(df[:5])
        df_sc = sc.transform(df)
        checkboxbox = st.checkbox("Выбрать модель")
    
        if checkboxbox:
        
            models = ["Обучение с учителем", "Ансамбли", "Обучение без учителя", "DNN"]
    
            models_type = st.selectbox("Выберите тип модели", models)
    
            if models_type is not None:
                if models_type == "Обучение с учителем":
                    LearnWithTeacher(df_sc)
                elif models_type == "Ансамбли":
                    Ensembles(df_sc)    
                elif models_type == "Обучение без учителя":
                    LearnWithoutTeacher(df_sc)
                elif models_type == "DNN":
                    DNN(df_sc)
    


st.title("Работа с моделями классификации")

st.header("Выбирите тип предсказания")
types_of_predict = ["Предсказание для одного объекта", "Предсказание для датасета"]

pred_type = st.selectbox("Выберите тип предсказаний", types_of_predict)

if pred_type is not None:
    if pred_type == "Предсказание для одного объекта":
        with open('data/Models/StandardScaler.pkl', 'rb') as file:
            sc = pickle.load(file)
        WorkWithOneObject()
    elif pred_type == "Предсказание для датасета":
        with open('data/Models/StandardScaler.pkl', 'rb') as file:
            sc = pickle.load(file)
        WorkWithDataset()  
    
