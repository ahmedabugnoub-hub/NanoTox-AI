import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import joblib
import os

# =============================
# GOOGLE SHEETS
# =============================
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("NanoTox_Data").sheet1

# =============================
# LOAD DATA
# =============================
def load_data():
    data = sheet.get_all_records()
    if len(data) == 0:
        df = pd.DataFrame({
            "Pesticide":["Emamectin","Lambda","Imidacloprid"],
            "AI":[10,10,12],
            "Surfactant":[15,15,20],
            "Solvent":[5,5,6],
            "Sonication":[10,10,15],
            "DLS":[150,220,120],
            "Zeta":[-25,-10,-30],
            "logP":[5,7,0.57],
            "Solubility":[0.02,0.005,610],
            "MW":[886,449,255],
            "LC50":[0.25,0.55,0.18]
        })
        return df
    return pd.DataFrame(data)

# =============================
# SAVE DATA
# =============================
def save_data(df):
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

# =============================
# MODEL
# =============================
MODEL_FILE = "model.pkl"

def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return GradientBoostingRegressor()

def save_model(model):
    joblib.dump(model, MODEL_FILE)

# =============================
# UI
# =============================
st.set_page_config(page_title="NanoTox AI", layout="wide")

st.title("NanoTox AI – Smart AI Platform")
st.caption("Adaptive AI + Optimization + Nano Analysis")

# =============================
# DATA
# =============================
if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

st.subheader("📊 Dataset")
st.dataframe(df, use_container_width=True)

# =============================
# ADD DATA
# =============================
st.sidebar.header("➕ Add Data")

pest_input = st.sidebar.text_input("New Pesticide Name")

ai = st.sidebar.slider("AI (%)",1.0,20.0,10.0)
surf = st.sidebar.slider("Surfactant (%)",5.0,30.0,15.0)
solv = st.sidebar.slider("Solvent (%)",1.0,10.0,5.0)
sonic = st.sidebar.slider("Sonication (min)",1,30,10)
dls = st.sidebar.slider("DLS (nm)",50,300,150)
zeta = st.sidebar.slider("Zeta (mV)",-60,60,-25)
logp = st.sidebar.number_input("logP",value=5.0)
solub = st.sidebar.number_input("Solubility",value=0.02)
mw = st.sidebar.number_input("MW",value=800.0)
lc50 = st.sidebar.number_input("LC50",value=0.2)

if st.sidebar.button("Save Data"):
    new = pd.DataFrame([[pest_input,ai,surf,solv,sonic,dls,zeta,logp,solub,mw,lc50]],
                       columns=df.columns)
    df = pd.concat([df,new],ignore_index=True)
    st.session_state.df = df
    save_data(df)
    st.success("Saved ✔")

# =============================
# MODEL
# =============================
if len(df) > 1:

    df_encoded = pd.get_dummies(df, columns=["Pesticide"])

    X = df_encoded.drop(columns=["LC50"])
    y = df_encoded["LC50"]

    model = load_model()
    model.fit(X, y)
    save_model(model)

    # =============================
    # PREDICTION
    # =============================
    st.sidebar.header("⚙️ Prediction")

    pest = st.sidebar.selectbox("Select Pesticide", df["Pesticide"].unique())

    input_dict = {
        "AI": ai,
        "Surfactant": surf,
        "Solvent": solv,
        "Sonication": sonic,
        "DLS": dls,
        "Zeta": zeta,
        "logP": logp,
        "Solubility": solub,
        "MW": mw
    }

    for col in X.columns:
        if "Pesticide_" in col:
            input_dict[col] = 1 if col == f"Pesticide_{pest}" else 0

    input_df = pd.DataFrame([input_dict])

    pred = model.predict(input_df)[0]
    lc90 = pred * 2.5

    st.subheader("📊 Results")
    c1, c2 = st.columns(2)
    c1.metric("LC50", round(pred,4))
    c2.metric("LC90", round(lc90,4))

    # =============================
    # 🎯 OPTIMIZATION
    # =============================
    st.subheader("🎯 Optimal Formulation")

    if st.button("Find Best Formulation"):

        best_lc50 = 999
        best = None

        for _ in range(300):

            test = {
                "AI": np.random.uniform(5,10),
                "Surfactant": np.random.uniform(12,20),
                "Solvent": np.random.uniform(3,8),
                "Sonication": np.random.uniform(5,30),
                "DLS": np.random.uniform(50,150),
                "Zeta": np.random.uniform(-45,-20),
                "logP": logp,
                "Solubility": solub,
                "MW": mw
            }

            for col in X.columns:
                if "Pesticide_" in col:
                    test[col] = 1 if col == f"Pesticide_{pest}" else 0

            val = model.predict(pd.DataFrame([test]))[0]

            if val < best_lc50:
                best_lc50 = val
                best = test

        st.json(best)
        st.success(f"Best LC50 = {best_lc50:.4f}")

    # =============================
    # ⚖️ COMPARISON
    # =============================
    st.subheader("⚖️ Comparison")

    if st.button("Compare Pesticides"):

        res = []

        for p in df["Pesticide"].unique():

            test = {
                "AI": ai,
                "Surfactant": surf,
                "Solvent": solv,
                "Sonication": sonic,
                "DLS": dls,
                "Zeta": zeta,
                "logP": logp,
                "Solubility": solub,
                "MW": mw
            }

            for col in X.columns:
                if "Pesticide_" in col:
                    test[col] = 1 if col == f"Pesticide_{p}" else 0

            val = model.predict(pd.DataFrame([test]))[0]

            res.append({"Pesticide": p, "LC50": val})

        res_df = pd.DataFrame(res).sort_values("LC50")

        st.dataframe(res_df)
        st.bar_chart(res_df.set_index("Pesticide"))

        st.success(f"Best: {res_df.iloc[0]['Pesticide']}")

    # =============================
    # 📊 DLS
    # =============================
    st.subheader("📊 DLS Distribution")

    data = np.random.normal(dls, dls*0.08, 800)

    fig, ax = plt.subplots()
    count,bins,_ = ax.hist(data, bins=20)

    mu, sigma = norm.fit(data)
    x = np.linspace(min(bins), max(bins), 200)
    y = norm.pdf(x, mu, sigma)

    ax.plot(x, y * max(count)/max(y), 'r')
    st.pyplot(fig)

    # =============================
    # ⚡ ZETA
    # =============================
    st.subheader("⚡ Zeta Potential")

    x = np.linspace(-150,150,2000)
    y = np.exp(-(x - zeta)**2/(2*5**2))

    fig2, ax2 = plt.subplots()
    ax2.plot(x, y, 'r')
    st.pyplot(fig2)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<center><b>Ahmed Abdulhakim</b><br>"
    "ahmed.abdulhakim_a015@agr.kfs.edu.eg</center>",
    unsafe_allow_html=True
)
