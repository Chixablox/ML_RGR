import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

def LearnWithTeacher(arr):
    st.title("Обучение с учителем")
    
    models = ["KNeighborsClassifier", "DecisionTreeClassifier", "SVM"]
    models_type = st.selectbox("Выберите модель", models)
    
    if models_type is not None:
        if models_type == "SVM":
            st.header("SVM")
            with open('data/Models/SVM.pkl', 'rb') as file:
                svm_model = pickle.load(file)
            Pred(svm_model, arr)
            
        elif models_type == "KNeighborsClassifier":
            st.header("KNeighborsClassifier")
            with open('data/Models/KNeighborsClassifier.pkl', 'rb') as file:
                knn_model = pickle.load(file)
            Pred(knn_model, arr)
            
        elif models_type == "DecisionTreeClassifier":
            st.header("DecisionTreeClassifier")
            with open('data/Models/DecisionTreeClassifier.pkl', 'rb') as file:
                tree_model = pickle.load(file)
            Pred(tree_model, arr)

def Ensembles(arr):
    st.title("Ансамбли")
    
    models = ["BaggingClassifier", "GradientBoostingClassifier", "StackingClassifier"]
    models_type = st.selectbox("Выберите модель", models)
    
    if models_type is not None:
        if models_type == "BaggingClassifier":
            st.header("BaggingClassifier")
            with open('data/Models/BaggingClassifier.pkl', 'rb') as file:
                bag_model = pickle.load(file)
            Pred(bag_model, arr)
            
        elif models_type == "GradientBoostingClassifier":
            st.header("GradientBoostingClassifier")
            with open('data/Models/GradientBoostingClassifier.pkl', 'rb') as file:
                grad_model = pickle.load(file)
            Pred(grad_model, arr)
            
        elif models_type == "StackingClassifier":
            st.header("StackingClassifier")
            with open('data/Models/StackingClassifier.pkl', 'rb') as file:
                stack_model = pickle.load(file)
            Pred(stack_model, arr)
            
def LearnWithoutTeacher(arr):
    st.title("Обучение без учителя")
    st.header("KMeans")
    with open('data/Models/KMeans.pkl', 'rb') as file:
            km_model = pickle.load(file)
    Pred(km_model, arr)

def DNN(arr):
    st.title("Нейронные сети")
    st.header("DNN")
    dnn_model = load_model('data/Models/DNNClass.h5')
    Pred(dnn_model, arr)
    

def Pred(model, arr):
    pred = model.predict(arr)
    pred = pred.flatten()
    if (len(pred) > 1):
        if(np.argmax(pred) == 1):
            st.write(f"Объект может представлять опасность, так как его класс {np.argmax(pred)}")
        else:
            st.write(f"Объект опасности не представляет, так как его класс {np.argmax(pred)}")
    else:
        if(np.max(pred) == 1):
            st.write(f"Объект может представлять опасность, так как его класс {np.max(pred)}")
        else:
            st.write(f"Объект опасности не представляет, так как его класс {np.max(pred)}")


st.title("Работа с моделями классификации")


est_diameter_min = st.number_input("Минимальный диаметр:", min_value=0.001, max_value=100.0, value=0.084)
    
est_diameter_max = st.number_input("Максимальный диаметр:", min_value=0.001, max_value=100.0, value=0.18)
    
relative_velocity = st.number_input("Скорость объекта:", min_value=0.01, max_value=10000000.01, value=79519.078)
    
miss_distance = st.number_input("Пропущенное расстояние:", min_value=0.01, max_value=100000000.01, value=30000000.1)
    
absolute_magnitude = st.number_input("Абсолютная светимость:", min_value=0.01, max_value=100.0, value=22.9)
    
arr = np.array([est_diameter_min, est_diameter_max, relative_velocity, miss_distance, absolute_magnitude])

    
checkboxbox = st.checkbox("Выбрать модель")
    
if checkboxbox:
    
    arr = arr.reshape(1, -1)
    st.write(arr)
        
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
