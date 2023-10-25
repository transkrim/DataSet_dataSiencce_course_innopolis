import streamlit as st
import pandas as pd
import seaborn as sns
import json
import joblib

st.header('House prices in Russia!')

PATH_DATA = "data/moscow3.csv"
PATH_UNIQUE_VALUES = "data/un.json"
PATH_MODEL = "data/lr.sav"

# загрузка данных
@st.cache_data
def load_data(path):
    """Load data from path"""
    data = pd.read_csv(path)
    # для демонстрации
    data = data.sample(5000)
    return data

# загрузка модели
@st.cache_data
def load_model(path):
    """Load model from path"""
    model = joblib.load(PATH_MODEL)
    return model

# Функция для распределения данных по цветам и цене
@st.cache_data
def transform(data):
    """Transform data"""
    colors = sns.color_palette("coolwarm").as_hex() # get palette color SEABORN
    n_colors = len(colors)

    data = data.reset_index(drop=True)
    data["norm_price"] = data["price"] / data["area"] # поиск цены за квадратный метр

    data["label_colors"] = pd.qcut(data["norm_price"], n_colors, labels=colors) # color + price
    data["label_colors"] = data["label_colors"].astype("str")
    return data

df = load_data(PATH_DATA)
df = transform(df)
st.write(df[:4])

st.map(data=df, latitude="geo_lat", longitude="geo_lon", color='label_colors') # отображение данных на карте

with open(PATH_UNIQUE_VALUES) as file: # подгрузка уникальных данных (нужно для создания определённых ограничений)
    dict_unique = json.load(file)

# метрики
building_type = st.sidebar.selectbox('Building type', (dict_unique['building_type']))
object_type = st.sidebar.selectbox("Object type", (dict_unique["object_type"]))
level = st.sidebar.slider("Level",
                          min_value=min(dict_unique["level"]),
                          max_value=max(dict_unique["level"]))
levels = st.sidebar.slider(
    "Levels", min_value=min(dict_unique["levels"]),max_value=max(dict_unique["levels"]))
rooms = st.sidebar.selectbox("Rooms", (dict_unique["rooms"]))
area = st.sidebar.slider("Area",
                         min_value=min(dict_unique["area"]),
                         max_value=max(dict_unique["area"]))
kitchen_area = st.sidebar.slider("Kitchen_area",
                                 min_value=min(dict_unique["kitchen_area"]),
                                 max_value=max(dict_unique["kitchen_area"]))

# мапинг данных для датасета
dict_data = {
    "building_type": building_type,
    "object_type": object_type,
    "level": level,
    "levels": levels,
    "rooms": rooms,
    "area": area,
    "kitchen_area": kitchen_area,
}

data_predict = pd.DataFrame([dict_data])
model = load_model(PATH_MODEL)

button = st.button("Predict")
if button:
    model.predict(data_predict)
    st.write(model.predict(data_predict)[0])

st.markdown(
    """
    ### Описание полей
        - building_type - Тип фасада - Другое. 1 – Панель. 2 – Монолитный. 3 – Кирпич. 4 – Блочный. 5 - Деревянный
        - object_type - Тип квартиры. 1 – Вторичный рынок недвижимости; 2 - Новостройка
        - level - Этаж квартиры
        - rooms - количество жилых комнат. Если значение «-1», то это означает «квартира-студия».
        - area - общая площадь квартиры
        - kitchen_area - площадь кухни
        - price - Цена в рублях 
"""
)