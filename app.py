import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import helper



st.set_page_config(page_title='Welcome 🖤', page_icon='👻', layout="wide", initial_sidebar_state="collapsed", menu_items=None)


from PIL import Image
image = Image.open('images/cars41.jpg')


st.title("Project: Carbon Dioxide Emissiion.🚦")
st.image(image)
st.markdown('- ### **Business Objective:''The fundamental goal here is to model the CO2 emissions as a function of several car engine features.**')
next =st.radio("Select:",['Data Analysis','Predict using model'])
st.markdown('***')

df = helper.get_Data('Data/co2_emissions (1).csv')

if next =='Data Analysis':


    st.title("Analysis of Carbon Dioxide Emission from Cars.")



    st.markdown("- Analyzing the data which has around `7k` data points. The goal is to get some insights about the data for model Building. ")




    #dropping duplicates
    df=df.drop_duplicates()


    st.header("A small portion of the data")
    st.dataframe(df.head())

    st.markdown("___")

    st.write(''' ### Feature Details:  
        ※  
        1. make ⇨ car brand under study.     
        2. model ⇨ the specific model of the car. 
        3. vehicle_class ⇨ car body type of the car.
        4. engine_size ⇨ size of the car engine, in Liters.    
        5. cylinders ⇨ number of cylinders.
        6. transmission ⇨ "A" for`Automatic', "AM" for ``Automated manual', "AS" for 'Automatic with select shift', "AV" for 'Continuously variable', "M" for 'Manual'.
        7. fuel_type ⇨ "X" for 'Regular gasoline', "Z" for 'Premium gasoline', "D" for 'Diesel', "E" for 'Ethanol (E85)', "N" for 'Natural gas'.
        8. fuel_consumption_city ⇨ City fuel consumption ratings, in liters per 100 kilometers.
        9. fuel_consumption_hwy ⇨ Highway fuel consumption ratings, in liters per 100 kilometers.
        10. fuel_consumption_comb(l/100km) ⇨ the combined fuel consumption rating (55% city, 45% highway), in L/100 km.
        11. fuel_consumption_comb(mpg) ⇨ the combined fuel consumption rating (55% city, 45% highway), in miles per gallon (mpg).
        12. co2_emissions ⇨ the tailpipe emissions of carbon dioxide for combined city and highway driving, in grams per kilometer.
    ''')
    st.markdown("___")
    st.write('''####  Basic info about the data: 
                 #   Column                          Non-Null Count  Dtype  
        ---  ------                          --------------  -----  
         0   make                            7385 non-null   object 
         1   model                           7385 non-null   object 
         2   vehicle_class                   7385 non-null   object 
         3   engine_size                     7385 non-null   float64
         4   cylinders                       7385 non-null   int64  
         5   transmission                    7385 non-null   object 
         6   fuel_type                       7385 non-null   object 
         7   fuel_consumption_city           7385 non-null   float64
         8   fuel_consumption_hwy            7385 non-null   float64
         9   fuel_consumption_comb(l/100km)  7385 non-null   float64
         10  fuel_consumption_comb(mpg)      7385 non-null   int64  
         11  co2_emissions                   7385 non-null   int64  
         dtypes: float64(4), int64(3), object(5)''')





    st.header("Histogram For Getting to know about frequency.")
    columns =['make', 'vehicle_class', 'engine_size', 'cylinders',
                          'transmission', 'fuel_type', 'fuel_consumption_city',
                          'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)',
                          'fuel_consumption_comb(mpg)', 'co2_emissions']
    selected_feature=st.selectbox('Choose the feature',columns)
    for column in columns:
           if selected_feature == column:
                  fig = px.histogram(df, x=column,color_discrete_sequence=['#8B1A1A'])
                  fig.update_layout(
                      autosize=True,
                      width=1300,
                      height=650,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4,
                      ),
                      paper_bgcolor="white",
                  )
                  st.plotly_chart(fig)
           else:
                  pass


    st.header("Scatterplot with respect to Co2 emission.")
    columns_for_scatterplot =['engine_size', 'cylinders',
                          'transmission', 'fuel_type', 'fuel_consumption_city',
                          'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)',
                          'fuel_consumption_comb(mpg)']
    selected_feature_for_scatterplot=st.selectbox('Choose the feature',columns_for_scatterplot)
    for c in columns_for_scatterplot:
           if selected_feature_for_scatterplot== c:
                  fig = px.scatter(data_frame=df,x=c,y='co2_emissions',color_discrete_sequence=['#8B1A1A'])
                  fig.update_layout(
                      autosize=True,
                      width=1300,
                      height=650,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4,
                      ),
                      paper_bgcolor="white",
                  )
                  st.plotly_chart(fig)
           else:
                  pass








    st.header("Boxplots with respect to co2 Emission.")
    columns_for_boxplot =['make', 'vehicle_class', 'cylinders',
                          'transmission', 'fuel_type',
                          ]
    selected_feature_for_boxplot=st.selectbox('Choose the feature',columns_for_boxplot)
    for column in columns_for_boxplot:
           if selected_feature_for_boxplot == column:
                  fig = px.box(df, x=column,y='co2_emissions',color_discrete_sequence=['#8B1A1A'])
                  fig.update_layout(
                      autosize=True,
                      width=1300,
                      height=650,
                      margin=dict(
                          l=50,
                          r=50,
                          b=100,
                          t=100,
                          pad=4,
                      ),
                      paper_bgcolor="white",
                  )
                  st.plotly_chart(fig)
           else:
                  pass

    st.header("Boxplots ")
    columns_for_new = ['make', 'vehicle_class', 'engine_size', 'cylinders',
                'transmission', 'fuel_type', 'fuel_consumption_city',
                'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)',
                'fuel_consumption_comb(mpg)', 'co2_emissions'
                ]
    new_features = st.selectbox('select a feature',columns_for_new)
    for column in columns_for_new:
        if new_features == column:
            fig = px.box(df, x=column, color_discrete_sequence=['#8B1A1A'])
            fig.update_layout(
                autosize=True,
                width=1300,
                height=650,
                margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=100,
                    pad=4,
                ),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig)
        else:
            pass



    #label Encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['make'] = le.fit_transform(df['make'])
    df['model']  = le.fit_transform(df['model'])
    df['vehicle_class']  = le.fit_transform(df['vehicle_class'])
    df['transmission']  =le.fit_transform(df['transmission'])
    df['fuel_type']  = le.fit_transform(df['fuel_type'])


    st.header("Heatmap for knowing the correlation.")
    fig, ax = plt.subplots(figsize=(4,2))
    sns.set_context('paper',font_scale=0.4)
    sns.heatmap(df.corr(), ax=ax,annot=True)
    st.write(fig)



    final_df=helper.get_Data1('Data/final_df.csv')
    #st.write(final_df)
    st.header("Regression Plots.")
    import statsmodels.api as sm
    new_cols = ['engine_size', 'cylinders',
                'transmission', 'fuel_type', 'fuel_consumption_city',
                'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)',
                'fuel_consumption_comb(mpg)']
    col = st.selectbox('select a feature.',new_cols)
    for n_col in new_cols:
        if n_col == col:
            fig = px.scatter(final_df, x=n_col, y='co2_emissions', opacity=0.65,trendline='ols',
                             trendline_color_override='#030303',color_discrete_sequence=['#8B1A1A'])
            fig.update_layout(
                autosize=True,
                width=1300,
                height=650,
                margin=dict(
                    l=50,
                    r=50,
                    b=100,
                    t=100,
                    pad=4,
                ),
                paper_bgcolor="white",
            )
            st.plotly_chart(fig)

    st.markdown("---")

    st.markdown('## Some insights from the data:')
    st.markdown('- #### Most of the features are `right skewed`.\n- #### Data has `several outliers`.''\n'
                '- #### There is multicollinearity between `fuel_consumption_city` `fuel_consumption_hwy` '
                '`fuel_consumption_comb(l/100km)`.')

    st.markdown("---")




if next =='Predict using model':
    from sklearn.preprocessing import StandardScaler ,LabelEncoder
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Model
    from tensorflow.keras.models import load_model
    st.header("Predict the amount of Co2 released from your car by entering the following aspects.")

    model=st.radio("Select Method : " ,['Method1','Method2'])
    st.markdown("---")


    if model == 'Method2':

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        col7,col8 = st.columns([1,3])

        #### Taking inputs ####
        with col1:
            a = df.make.unique()
            a.sort()
            i1 = st.selectbox("Select Company :",a)
        #-------
        with col2:
            dfn = df[(df['make'] == i1)]
            b = dfn.model.unique()
            b.sort()
            i2 = st.selectbox("Select model :",b)
        #-----
        with col3:
            dfn1 = dfn[dfn['model'] == i2]
            c = dfn1['vehicle_class'].unique()
            i3=st.selectbox("Select veicle Class :" ,c)
        #-----
        with col4:
            dfn2 = dfn1[dfn1['vehicle_class'] == i3]
            d = dfn2['engine_size'].unique()
            i4 = st.selectbox("Engine size: ",d)
        #-----
        with col5:
            dfn3 = dfn2[dfn2['engine_size'] == i4]
            e = dfn2['cylinders'].unique()
            e.sort()
            i5 = st.selectbox("Cylinders :",e)
        #----
        with col6:
            dfn4 = dfn3[dfn3['cylinders'] == i5]
            f = dfn3['transmission'].unique()
            i6 = st.selectbox("Transmission :",f,help=("A = Automatic , AM = Automated manual , AV = Continuously variable , AS = Automatic with select shift , M = Manual. "))
        #----
        with col7:
            dfn5 = dfn4[dfn4['transmission'] == i6]
            i = dfn4['fuel_type'].unique()
            i7 = st.selectbox("Fuel Type :", i,help=("X = Regular gasoline, Z = Premium gasoline, D = Diesel, E= Ethanol(E85), N = Natural gas."))
        #------
        with col8 :
            i8 = st.number_input("Enter the combined fuel consumption rating (55% city, 45% highway), in L/100 km.",
                                 min_value=float(0),max_value=float(50),step=float(1.0))
        #data preprocessing

        df = helper.preprocessing1(df, 'co2_emissions')
        df = helper.preprocessing1(df, 'fuel_consumption_comb(l/100km)')
                #
        df = df.sample(frac=1, random_state=4)
                #
        le_make = LabelEncoder()
        le_model = LabelEncoder()
        le_veh = LabelEncoder()
        le_tran = LabelEncoder()
        le_fuel = LabelEncoder()
                #
        # label Encoding
        df['make'] = le_make.fit_transform(df['make'])
        df['model'] = le_model.fit_transform(df['model'])
        df['vehicle_class'] = le_veh.fit_transform(df['vehicle_class'])
        df['transmission'] = le_tran.fit_transform(df['transmission'])
        df['fuel_type'] = le_fuel.fit_transform(df['fuel_type'])
                #
        Xi = df[['make', 'model', 'vehicle_class', 'engine_size', 'cylinders', 'transmission', 'fuel_type',
                 'fuel_consumption_comb(l/100km)']]
        Y = df['co2_emissions']
                #
        scaler = StandardScaler()
        X = scaler.fit_transform(Xi)
                #
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
                #
        #Loading the trained model
        st.cache()
        T_model = savedModel=load_model('Models/DL_1.h5')
        ########
        #Predictions
        input_data = pd.DataFrame({'make': i1, 'model': i2, 'vehicle_class': i3,
                                   'engine_size': i4, 'cylinders': i5, 'transmission': i6, 'fuel_type': i7,
                                   'fuel_consumption_comb(l/100km)': i8}, index=[1])
        input_data['make'] = le_make.fit_transform(input_data['make'])
        input_data['model'] = le_model.fit_transform(input_data['model'])
        input_data['vehicle_class'] = le_veh.fit_transform(input_data['vehicle_class'])
        input_data['transmission'] = le_tran.fit_transform(input_data['transmission'])
        input_data['fuel_type'] = le_fuel.fit_transform(input_data['fuel_type'])
        input_data = scaler.transform(input_data)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            pass
        with col2:
            pass
        with col4:
            pass
        with col5:
            pass
        with col3:
            button=st.button("Predict 🦾")
        if button:
            pred = T_model.predict(input_data)
            pred = float(pred)
            pred = round(pred, 2)
            if pred > 150:
                st.error(f"The Co2 emmitted from your car is {pred} grams per kilometer.")
            else:
                st.success(f"The Co2 emmitted from your car is {pred} grams per kilometer.")
                st.balloons()


    if model == 'Method1':
        col1, col2 = st.columns([1.5,3.5])
        col3, col4 = st.columns(2)
        col5, col6 = st.columns([2,3])
        with col1:
            a = df.vehicle_class.unique()
            a.sort()
            i1= st.selectbox("Select Vehicle type : ",a)
        with col2:
            i2=st.slider("Select Engine Size :", min_value=0.0, max_value=10.0, value=2.5, step=0.5,help =("Slide to select.."))
        with col3:
            i3 = st.number_input("Enter Number of Cylinders : ",min_value=2,max_value=18,step=1,format="%u")
        with col4:
            b = df.transmission.unique()
            b.sort()
            i4 = st.selectbox("Select the Transmission Type :",b,help=("A = Automatic , AM = Automated manual , AV = Continuously variable , AS = Automatic with select shift , M = Manual. "))
        with col5 :
            c = df.fuel_type.unique()
            c.sort()
            i5 = st.selectbox("Select Fuel Type :" , c,help=("X = Regular gasoline, Z = Premium gasoline, D = Diesel, E= Ethanol(E85), N = Natural gas."))
        with col6 :
            i6 = st.number_input("Enter the combined fuel consumption rating (55% city, 45% highway), in L/100 km.",
                                 min_value=float(0),max_value=float(50),step=float(1.0))


        #### Data preprocessing
        df = helper.preprocessing1(df, 'co2_emissions')
        df = helper.preprocessing1(df, 'fuel_consumption_comb(l/100km)')

        le_veh = LabelEncoder()
        le_tran = LabelEncoder()
        le_fuel = LabelEncoder()
        df['vehicle_class'] = le_veh.fit_transform(df['vehicle_class'])
        df['transmission'] = le_tran.fit_transform(df['transmission'])
        df['fuel_type'] = le_fuel.fit_transform(df['fuel_type'])

        Xi = df[['vehicle_class', 'engine_size', 'cylinders', 'transmission', 'fuel_type',
                 'fuel_consumption_comb(l/100km)']]
        Y = df['co2_emissions']

        scaler = StandardScaler()
        X = scaler.fit_transform(Xi)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)

        # Loading the trained model
        st.cache()
        T_model2 = savedModel = load_model('Models/DL_2.h5')
        # Predictions
        # Predictions
        input_data = pd.DataFrame({'vehicle_class': i1,'engine_size': i2, 'cylinders': i3,
                                   'transmission': i4, 'fuel_type': i5,
                                   'fuel_consumption_comb(l/100km)': i6}, index=[1])
        input_data['vehicle_class'] = le_veh.fit_transform(input_data['vehicle_class'])
        input_data['transmission'] = le_tran.fit_transform(input_data['transmission'])
        input_data['fuel_type'] = le_fuel.fit_transform(input_data['fuel_type'])
        input_data = scaler.transform(input_data)
        col1, col2, col3,col4,col5 = st.columns(5)

        with col1:
            pass
        with col2:
            pass
        with col4:
            pass
        with col5:
            pass
        with col3:
            button = st.button("Predict 🦾")
        if button:
            pred = T_model2.predict(input_data)
            pred = float(pred)
            pred = round(pred, 2)
            if pred > 150 :
                st.error(f"The Co2 emmitted from your car is {pred} grams per kilometer.")
            else:
                st.success(f"The Co2 emmitted from your car is {pred} grams per kilometer.")
                st.balloons()

st.markdown("***")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
st.markdown("\n")





#st.markdown("---")
st.markdown(" Developed by `SKY`.   ⇨[github ](https://github.com/suraj4502), [Linkedin](https://www.linkedin.com/in/surajkumar-yadav-6ab2011a4/),[Ig](https://www.instagram.com/suraj452/).")
#st.markdown("---")