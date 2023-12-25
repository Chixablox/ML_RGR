import streamlit as st
from PIL import Image

st.header("Web-приложение для вывода моделей ML и анализа данных")

img = Image.open("D:/MachineLearning/ML_RGR/data/Photo/I.jpg")

st.header("Обо мне")
st.subheader("ФИО")
st.write("Мрдак Александр Браниславович")
st.subheader("Группа")
st.write("МО-221")
st.subheader("Фото")
st.image(img, width=200)

