import streamlit as st
from sklearn import linear_model
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu

import plotly.express as px

st.set_page_config(layout="wide")

select= option_menu(
    menu_title=None,
    options=["HOME","PREDICT PRICE"],
    orientation="horizontal"

)

df=pd.read_csv("houseprice.csv")

if select=="HOME":

    st.title("HOUSE PRICE PREDICTION")

    st.dataframe(df)

    st.subheader("CHART VISUALIZATION")

    location_wise_sales= df.groupby("location")["price_pkr"].sum().reset_index()

    fig = px.bar(
        location_wise_sales,
        x="location",
        y="price_pkr",
        title="Location Wise Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    sales_by_sqft = df.groupby("area_sqft")["price_pkr"].sum().reset_index()

    fig = px.scatter(
        sales_by_sqft,
        x="area_sqft",
        y="price_pkr",
        title="Sales Over SQFT"
    )

    st.plotly_chart(fig, use_container_width=True)

    col1,col2,col3,col4=st.columns(4)

    with col1:
        st.metric("Total Rooms", df["rooms"].sum())

    with col2:
        st.metric("Total Locations", df["location"].nunique())

    with col3:
        st.metric("Approximate Price", df["price_pkr"].mean())

    price_wise_rooms= df.groupby("rooms")["price_pkr"].sum().reset_index()

    fig = px.pie(
        price_wise_rooms,
        names="rooms",
        labels="price_pkr",
        title="PRICE WISE ROOMS"
    )
    st.plotly_chart(fig, use_container_width=True)

if select=="PREDICT PRICE":
    st.title("PREDICT HOUSE PRICE")

    st.image("housie.png", use_container_width=True)

    le=LabelEncoder()

    df['location']=le.fit_transform(df['location'])

    x=df[["area_sqft","location","rooms"]]
    y=df.price_pkr

    model=linear_model.LinearRegression()
    model.fit(x,y)

    col4,col5=st.columns(2)

    with col4:
        st.info("By Selecting SQFT, Rooms & Location")

        rooms=int(st.number_input("Enter No.Of Rooms:",min_value=1))
        sales_by_sqft=int(st.number_input("Enter SQFT's",min_value=100))
        location=st.selectbox("Select Location",le.classes_)
        pred=st.button("Predict")

    with col5:
        loc_ch=le.transform([location])[0] #color change
        predicted_price=model.predict([[loc_ch,sales_by_sqft,rooms]])
        score=model.score(x,y)         #pehle hi upr accuracy batata rahyga
        accuracy=int(score*100)
        st.metric("Accuracy Of Current Model As Per Recent Data",accuracy,"%")
        if pred:
            if location:
                loc_ch=le.transform([location])[0] #color change
                predicted_price=model.predict([[loc_ch,sales_by_sqft,rooms]])  #yahan predict krny k bd b wahi cheez hgi price k sth
                st.balloons()
                score=model.score(x,y)
                accuracy=int(score*100)

                st.subheader("How Accurate The Predicted Price Is!")
                st.info(f"{accuracy}%")
                st.subheader("Here Is Your Predicted Price For House")
                predic=int(predicted_price[0])   #2 k decimal me round krwa k predic k variable me dal dia or phr nechy print krwadia
                st.info(predic)









    


    



