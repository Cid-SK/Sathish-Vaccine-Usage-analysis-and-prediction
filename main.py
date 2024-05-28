import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

df=pd.read_csv("G:\\SK\\Ds\\vaccine\\encoded_vaccine.csv")

#streamlit part
st.set_page_config(page_title="Vaccine Prediction",layout="wide")
st.title("***:red[Vaccine Usage analysis and prediction]***")
st.markdown("<style>div.block-container{padding-top:1rem;}</style>",unsafe_allow_html=True)

tab1,tab2,tab3=st.tabs(["***Home***","***Analysis***","***Prediction***"])

with tab1:
       st.subheader(":blue[***Welcome to the Vaccine Usage Analysis and Prediction project homepage!***]")

       st.write('''This project aims to predict the likelihood of people taking an H1N1 flu vaccine 
                using machine learning techniques and provide valuable insights for healthcare
                professionals and policymakers.''')
       
       st.write("""

    ### :blue[***Project Tasks:***]
                
    - **Data Engineering**: Understand and preprocess the dataset.
    - **Dashboard Development**: Create interactive visualizations.
    - **Model Development**: Train a predictive model.
    - **Model Serving API**: Develop an application for prediction.

     :red[Use the tabs above to navigate through different sections of the project.]

    """)
       
with tab2:     
       data_df=pd.read_csv("G:\\SK\\Ds\\vaccine\\vaccine.csv")
       data=data_df.copy()

       data['h1n1_awareness'] = data['h1n1_awareness'].replace({0:"No knowledge",1:"little knowledge",2:"good knowledge"})
       data['h1n1_vaccine'] = data['h1n1_vaccine'].replace({0:"Not Vaccinated",1:"Vaccinated"})
       data['is_h1n1_risky']=data['is_h1n1_risky'].replace({1:"Thinks it is not very low risk", 2:"Thinks it is somewhat low risk", 3:"don’t know if it is risky or not", 4:"Thinks it is a somewhat high risk",5:"Thinks it is very highly risky"})
       
       st.subheader(":red[***Overview***]")
       c1,c2=st.columns(2)
       with c1:
              with st.container(border=True):
                     #Gender Distribution
                     vaccine_distribution = data['h1n1_vaccine'].value_counts().reset_index()
                     vaccine_distribution.columns = ['H1N1 Vaccine', 'Count']
                     fig_pie = px.pie(vaccine_distribution, values='Count', names='H1N1 Vaccine',
                                          width=450,height=450,
                                          title='H1N1 Vaccine Distribution',hole=0.5)
                     st.plotly_chart(fig_pie)
       with c2:
              with st.container(border=True):
                     #Gender Distribution
                     gender_distribution = data['sex'].value_counts().reset_index()
                     fig_pie = px.pie(gender_distribution, values='count', names='sex',
                                      width=450,height=450,
                                      title='Gender Distribution',hole=0.5)
                     st.plotly_chart(fig_pie)

       with st.container(border=True):
              st.subheader(":red[Age Based Analysis]")
              age_filter=st.selectbox("Select the Age Bracket",data['age_bracket'].unique(),key="age",index=2)
              age_df=data[data['age_bracket']==age_filter]

              c3,c4=st.columns([3.8,4.2])
              with c3:
                     # H1N1 Awareness Distribution
                     h1n1_awareness_distribution = age_df['h1n1_awareness'].value_counts().reset_index()
                     h1n1_awareness_distribution.columns = ['Awareness Level', 'Count']

                     h1n1_awareness_distribution['Vaccine'] = h1n1_awareness_distribution['Awareness Level'].apply(lambda x: "\n".join(f"{vaccine}: {count}" for vaccine, count in age_df[age_df['h1n1_awareness'] == x]['h1n1_vaccine'].value_counts().items()))
                     
                     fig_bar_aw = px.bar(h1n1_awareness_distribution, y='Count', x='Awareness Level',
                                   hover_data={'Count', 'Awareness Level', 'Vaccine'},
                                   width=450,
                                   color='Count',color_continuous_scale=px.colors.sequential.Redor_r,
                                   title='H1N1 Awareness Distribution')
                     st.plotly_chart(fig_bar_aw)
              with c4:
                     # H1N1 Risky (respondent think)
                     h1n1_risky_distribution = age_df['is_h1n1_risky'].value_counts().reset_index()
                     h1n1_risky_distribution.columns = ['Respondent Think', 'Count']

                     h1n1_risky_distribution['Vaccine'] = h1n1_risky_distribution['Respondent Think'].apply(lambda x: "\n".join(f"{vaccine}: {count}" for vaccine, count in age_df[age_df['h1n1_awareness'] == x]['h1n1_vaccine'].value_counts().items()))
                     
                     fig_bar_risk = px.bar(h1n1_risky_distribution, x='Count', y='Respondent Think',
                                   hover_data={'Count', 'Respondent Think', 'Vaccine'},
                                   orientation='h',width=500,
                                   color='Count',color_continuous_scale=px.colors.sequential.Redor_r,
                                   title='H1N1 Risky Distribution')
                     st.plotly_chart(fig_bar_risk)

       
       n_data=data.copy()
       n_data['Contact Avoidance']=n_data['contact_avoidance'].replace({0:"No",1:"Yes"})
       n_data['Wash Hands Frequently']=n_data['wash_hands_frequently'].replace({0:"No",1:"Yes"})
       n_data['Avoid Large Gatherings']=n_data['avoid_large_gatherings'].replace({0:"No",1:"Yes"})
       n_data['Avoid Touch Face']=n_data['avoid_touch_face'].replace({0:"No",1:"Yes"})
       n_data['Reduced Outside Home Contact']=n_data['reduced_outside_home_cont'].replace({0:"No",1:"Yes"})

       be_analysis=['Contact Avoidance','Wash Hands Frequently','Avoid Large Gatherings','Avoid Touch Face','Reduced Outside Home Contact']
       st.subheader(":red[Behavioural Analysis]")
       option=st.selectbox("Select the Option",be_analysis)
       def distribution(option):
              #Distribution
              distribution = n_data[option].value_counts().reset_index()
              fig_pie = px.pie(distribution, values='count', names=option,
                                   width=450,height=450,
                                   title=f'{option}',hole=0.5)
              st.plotly_chart(fig_pie)
       distribution(option)

with tab3:
       c1,c2=st.columns(2)
       with c1:
              h1n1_worry=st.selectbox("H1n1 Worry",df['h1n1_worry'].unique(),key="h1n1_worry")
              h1n1_awareness=st.selectbox("H1n1 Awareness",df['h1n1_awareness'].unique(),key="h1n1_awareness")
              antiviral_medication=st.selectbox("Antiviral Medication",df['antiviral_medication'].unique(),key="antiviral_medication")
              contact_avoidance=st.selectbox("Contact Avoidance",df['contact_avoidance'].unique(),key="contact_avoidance")
              bought_face_mask=st.selectbox("Bought Face Mask",df['bought_face_mask'].unique(),key="bought_face_mask")
              wash_hands_frequently=st.selectbox("Wash Hands Frequently",df['wash_hands_frequently'].unique(),key="wash_hands_frequently")
              avoid_large_gatherings=st.selectbox("Avoid Large Gatherings",df['avoid_large_gatherings'].unique(),key="avoid_large_gatherings")
              reduced_outside_home_cont=st.selectbox("Reduced Outside Home Cont",df['reduced_outside_home_cont'].unique(),key="reduced_outside_home_cont")
              avoid_touch_face=st.selectbox("Avoid Touch Face",df['avoid_touch_face'].unique(),key="avoid_touch_face")
              dr_recc_h1n1_vacc=st.selectbox("Dr recc h1n1 vacc",df['dr_recc_h1n1_vacc'].unique(),key="dr_recc_h1n1_vacc")
              dr_recc_seasonal_vacc=st.selectbox("Dr recc seasonal vacc",df['dr_recc_seasonal_vacc'].unique(),key="dr_recc_seasonal_vacc")
              chronic_medic_condition=st.selectbox("Chronic medic condition",df['chronic_medic_condition'].unique(),key="chronic_medic_condition")
              cont_child_undr_6_mnths=st.selectbox("Cont child undr 6 mnths",df['cont_child_undr_6_mnths'].unique(),key="cont_child_undr_6_mnths")
              is_health_worker=st.selectbox("is health worker",df['is_health_worker'].unique(),key="is_health_worker")
              is_h1n1_vacc_effective=st.selectbox("is h1n1 vacc effective",df['is_h1n1_vacc_effective'].unique(),key="is_h1n1_vacc_effective")
              is_h1n1_risky=st.selectbox("is h1n1 risky",df['is_h1n1_risky'].unique(),key="is_h1n1_risky")
       with c2:
              sick_from_h1n1_vacc=st.selectbox("Sick from h1n1 vacc",df['sick_from_h1n1_vacc'].unique(),key="sick_from_h1n1_vacc")
              is_seas_vacc_effective=st.selectbox("is seas vacc effective",df['is_seas_vacc_effective'].unique(),key="is_seas_vacc_effective")
              is_seas_risky=st.selectbox("is seas risky",df['is_seas_risky'].unique(),key="is_seas_risky")
              sick_from_seas_vacc=st.selectbox("Sick from seas vacc",df['sick_from_seas_vacc'].unique(),key="sick_from_seas_vacc")
              age_bracket=st.selectbox("Age Bracket",df['age_bracket'].unique(),key="age_bracket")
              qualification=st.selectbox("Qualification",df['qualification'].unique(),key="qualification")
              race=st.selectbox("Race",df['race'].unique(),key="race")
              sex=st.selectbox("Sex",df['sex'].unique(),key="sex")
              income_level=st.selectbox("Income Level",df['income_level'].unique(),key="income_level")
              marital_status=st.selectbox("Marital Status",df['marital_status'].unique(),key="marital_status")
              housing_status=st.selectbox("Housing Status",df['housing_status'].unique(),key="housing_status")
              employment=st.selectbox("Employment",df['employment'].unique(),key="employment")
              census_msa=st.selectbox("Census msa",df['census_msa'].unique(),key="census_msa")
              no_of_adults=st.selectbox("No. of Adults",df['no_of_adults'].unique(),key="no_of_adults")
              no_of_children=st.selectbox("No. of Children",df['no_of_children'].unique(),key="no_of_children")



       submit=st.button("***:blue[Submit]***")

       with open('G:\\SK\\Ds\\vaccine\\vaccine_prediction.pkl', 'rb') as file:   
          model = pickle.load(file)
       
       if submit:
              prediction=model.predict([[h1n1_worry,h1n1_awareness,antiviral_medication,contact_avoidance,bought_face_mask,
                            wash_hands_frequently,avoid_large_gatherings,reduced_outside_home_cont,avoid_touch_face,
                            dr_recc_h1n1_vacc,dr_recc_seasonal_vacc,chronic_medic_condition,cont_child_undr_6_mnths,
                            is_health_worker,is_h1n1_vacc_effective,is_h1n1_risky,sick_from_h1n1_vacc,
                            is_seas_vacc_effective,is_seas_risky,sick_from_seas_vacc,age_bracket,qualification,
                            race,sex,income_level,marital_status,housing_status,employment,census_msa,no_of_adults,
                            no_of_children]])
              
              if prediction == 0:
                 st.warning("Not Vaccinated")
              elif prediction ==1:
                 st.success("Vaccinated")
