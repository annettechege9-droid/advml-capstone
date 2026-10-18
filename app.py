import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="Kenya Motor Insurance ML", layout="wide")
CSV = "Kenyan_Motor_Insurance_2023_2024.csv"
NUM = ['Customer_Age', 'Vehicle_Age', 'Vehicle_Value_KES', 'Vehicle_Engine_Capacity',
       'Annual_Premium_KES', 'No_Claim_Bonus_%', 'Policy_Term_Months',
       'Previous_Claims_Count', 'Driver_Experience_Years']
CAT = ['Gender', 'Region', 'Vehicle_Type', 'Use_Purpose', 'Third_Party_Only']
CLUSTER_FEATS = ['Customer_Age', 'Vehicle_Age', 'Vehicle_Value_KES', 'Vehicle_Engine_Capacity',
                 'Annual_Premium_KES', 'No_Claim_Bonus_%', 'Previous_Claims_Count',
                 'Driver_Experience_Years']
TARGET = 'Total_Claim_Amount_KES'


@st.cache_data
def load():
    df = pd.read_csv(CSV)
    df['Accident_Cause'] = df['Accident_Cause'].fillna('No Claim')
    return df


@st.cache_resource
def train(df):
    # Same split and tuned Random Forest settings as the notebook (champion model)
    X, y = df[NUM + CAT], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('num', StandardScaler(), NUM),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), CAT)])),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=4,
                                        max_features=0.8, random_state=42, n_jobs=-1))])
    pipe.fit(Xtr, ytr)
    p = pipe.predict(Xte)
    return pipe, r2_score(yte, p), mean_absolute_error(yte, p)


@st.cache_data
def segments(df):
    Xs = StandardScaler().fit_transform(df[CLUSTER_FEATS])
    out = df.copy()
    out['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Xs)
    pcs = PCA(n_components=2, random_state=42).fit_transform(Xs)
    out['PC1'], out['PC2'] = pcs[:, 0], pcs[:, 1]
    # name clusters from their profile, so names stay right if labels shuffle
    prof = out.groupby('Cluster')[['Customer_Age', 'Vehicle_Value_KES']].mean()
    hv = prof['Vehicle_Value_KES'].idxmax()
    rest = prof.drop(hv)
    names = {hv: 'High-value Commercial & PSV fleet',
             rest['Customer_Age'].idxmin(): 'Young / Entry-level drivers',
             rest['Customer_Age'].idxmax(): 'Experienced / Established drivers'}
    out['Segment'] = out['Cluster'].map(names)
    return out


df = load()
model, r2, mae = train(df)

st.title("Motor Insurance Claim Prediction & Policyholder Profiling (Kenya)")
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Segments", "Predict a claim", "Model comparison"])

with tab1:
    c = st.columns(4)
    c[0].metric("Policies", f"{len(df):,}")
    c[1].metric("Zero-claim share", f"{(df[TARGET] == 0).mean():.1%}")
    c[2].metric("Avg claim (claimants)", f"{df.loc[df[TARGET] > 0, TARGET].mean():,.0f} KES")
    c[3].metric("Regions", df['Region'].nunique())
    st.subheader("Average claim by vehicle type (KES)")
    st.bar_chart(df.groupby('Vehicle_Type')[TARGET].mean())
    st.dataframe(df.head(20))

with tab2:
    seg = segments(df)
    st.subheader("K-Means (k=3) shown on the first 2 principal components")
    st.scatter_chart(seg, x='PC1', y='PC2', color='Segment')
    st.dataframe(seg.groupby('Segment')[CLUSTER_FEATS + [TARGET]].mean().round(0))

with tab3:
    st.caption(f"Champion: tuned Random Forest · test R² {r2:.3f} · MAE {mae:,.0f} KES. "
               "Uses underwriting-time features only (no leakage).")
    a, b, c = st.columns(3)
    age = a.slider("Customer age", 18, 80, 40)
    exp = a.slider("Driving experience (yrs)", 0, 60, 15)
    veh_age = a.slider("Vehicle age (yrs)", 0, 25, 8)
    value = b.number_input("Vehicle value (KES)", 100000, 10000000, 1500000, step=50000)
    premium = b.number_input("Annual premium (KES)", 5000, 500000, 80000, step=5000)
    ncb = b.select_slider("No-claim bonus %", [0, 10, 20, 30, 40, 50, 60], value=20)
    vtype = c.selectbox("Vehicle type", sorted(df['Vehicle_Type'].unique()))
    use = c.selectbox("Use purpose", sorted(df['Use_Purpose'].unique()))
    region = c.selectbox("Region", sorted(df['Region'].unique()))
    gender = a.selectbox("Gender", sorted(df['Gender'].unique()))
    tpo = b.selectbox("Third party only", ['Yes', 'No'])
    prev = c.number_input("Previous claims", 0, 10, 0)
    row = pd.DataFrame([{
        'Customer_Age': age, 'Vehicle_Age': veh_age, 'Vehicle_Value_KES': value,
        'Vehicle_Engine_Capacity': df['Vehicle_Engine_Capacity'].median(),
        'Annual_Premium_KES': premium, 'No_Claim_Bonus_%': ncb, 'Policy_Term_Months': 12,
        'Previous_Claims_Count': prev, 'Driver_Experience_Years': exp, 'Gender': gender,
        'Region': region, 'Vehicle_Type': vtype, 'Use_Purpose': use, 'Third_Party_Only': tpo}])
    pred = float(model.predict(row)[0])
    st.metric("Predicted claim amount", f"{pred:,.0f} KES")
    st.info("Expected value across all possible outcomes, so it blends 'no claim' and 'big claim'.")

with tab4:
    res = pd.DataFrame({
        'Model': ['Gradient Boosting', 'Random Forest (tuned)', 'Random Forest', 'Gradient Boosting (tuned)',
                  'XGBoost', 'Neural Network', 'Linear Regression', 'Baseline (mean)'],
        'R2': [0.725, 0.724, 0.717, 0.711, 0.701, 0.676, 0.353, -0.002],
        'MAE_KES': [104769, 93800, 95474, 104034, 110984, 135391, 243767, 293597]})
    st.dataframe(res, hide_index=True)
    st.bar_chart(res.set_index('Model')['R2'])
