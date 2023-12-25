import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    
    data = pd.read_csv(path)
    
    return data


def HeatMap():
    st.title("Тепловая карта")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Тепловая карта с корреляцией')
    st.pyplot(plt)
        
        
def BoxPlot():
    st.title("Ящик с усами")    
    features = ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance" , "absolute_magnitude"]
    ft = st.selectbox("Выберите признак", features)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x = df[ft])
    plt.title(f'Диаграмма "ящик с усами" для признака {ft}')
    plt.xlabel('Значение')
    st.pyplot(plt)


def Hist():
    st.title("Гистограмма")    
    features = ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance" , "absolute_magnitude", "hazardous"]
    ft = st.selectbox("Выберите первый признак", features, key='selectbox1')
    plt.figure(figsize=(10, 6))
    sns.histplot(df[ft], bins=100)
    plt.title(f'Гистограмма для признака{ft}')
    st.pyplot(plt)


def LmlPlot():
    st.title("Диаграмма рассеивания для первой 1000 строк")    
    features = ["est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance" , "absolute_magnitude", "hazardous"]
    ft1 = st.selectbox("Выберите первый признак", features, key='selectbox1')
    ft2 = st.selectbox("Выберите второй признак", features, key='selectbox2')
    plt.figure(figsize=(10, 6))
    plt.scatter (df[ft1][:1000], df[ft2][:1000], s= 60 , c='purple')
    plt.title(f'Диаграмма рассеивания для признаков {ft1} и {ft2}')
    st.pyplot(plt)


df = load_data("data/Dataset/prepared_neo_task.csv")
df  = df.drop(["Unnamed: 0"], axis=1)


st.title("Визуализация")

st.write("Датасет, данные которого мы будем визуализировать:")
st.write(df[:5])

st.write("Выберите тип визуализации, который вы хотите увидеть:")

vis_types = ['Тепловая карта', 'Ящик с усами', 'Гистограмма', 'Диаграмма рассеивания']

vis_type = st.selectbox("Выберите тип визуализации", vis_types)

if vis_type is not None:
    if vis_type == "Тепловая карта":
        HeatMap()
    elif vis_type == "Ящик с усами":
        BoxPlot()    
    elif vis_type == "Гистограмма":
        Hist()
    elif vis_type == "Диаграмма рассеивания":
        LmlPlot()

