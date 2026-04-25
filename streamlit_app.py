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
# CONFIG
# =============================
st.set_page_config(page_title="NanoTox AI", layout="wide")
st.title("NanoTox AI – Smart Platform")
st.caption("Adaptive AI + Cloud/Local Mode")

# =============================
# GOOGLE SHEETS (SMART MODE)
# =============================
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

use_cloud = True

try:
    creds_dict = st.secrets["gcp_service_account"]

    # fix private_key
    if "\\n" in creds_dict["private_key"]:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("NanoTox_Data").sheet1

except:
    use_cloud = False
    st.warning("⚠️ Running in LOCAL mode (No Google Sheets)")

# =============================
# LOCAL STORAGE
# =============================
LOCAL_FILE = "data.csv"

# =============================
# LOAD DATA
# =============================
def load_data():

    # Cloud
    if use_cloud:
        try:
            data = sheet.get_all_records()
            if len(data) > 0:
                return pd.DataFrame(data)
        except:
            pass

    # Local
    if os.path.exists(LOCAL_FILE):
        return pd.read_csv(LOCAL_FILE)

    # Default
    return pd.DataFrame({
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

# =============================
# SAVE DATA
# =============================
def save_data(df):

    # Local save
    df.to_csv(LOCAL_FILE, index=False)

    # Cloud save
    if use_cloud:
        try:
            sheet.clear()
            sheet.update([df.columns.values.tolist()] + df.values.tolist())
        except:
            st.error("❌ Failed to save to Google Sheets")

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
# INIT DATA
# =============================
if "df" not in st.session_state:
    st.session_state.df = load_data()

df = st.session_state.df

# =============================
# DATA VIEW
# =============================
st.subheader("📊 Dataset")
st.dataframe(df, use_container_width=True)

# =============================
# ADD DATA
# =============================
st.sidebar.header("➕ Add Data")

pest_input = st.sidebar.text_input("Pesticide")

ai = st.sidebar.slider("AI (%)",1.0,20.0,10.0)
surf = st.sidebar.slider("Surfactant (%)",5.0,30.0,15.0)
solv = st.sidebar.slider("Solvent (%)",1.0,10.0,5.0)
sonic = st.sidebar.slider("Sonication",1,30,10)
dls = st.sidebar.slider("DLS",50,300,150)
zeta = st.sidebar.slider("Zeta",-60,60,-25)
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
    # DLS
    # =============================
    st.subheader("📊 DLS Distribution")

    data = np.random.normal(dls, dls*0.08, 800)

    fig, ax = plt.subplots()
    count,bins,_ = ax.hist(data, bins=20)

    mu, sigma = norm.fit(data)
    x = np.linspace(min(bins), max(bins), 200)
    y_curve = norm.pdf(x, mu, sigma)

    ax.plot(x, y_curve * max(count)/max(y_curve), 'r')
    ax.set_xlabel("Size (nm)")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    # =============================
    # ZETA
    # =============================
    st.subheader("⚡ Zeta Potential")

    x = np.linspace(-150,150,2000)
    y_curve = np.exp(-(x - zeta)**2/(2*5**2))

    fig2, ax2 = plt.subplots()
    ax2.plot(x, y_curve, 'r')
    ax2.set_xlabel("Zeta (mV)")
    ax2.set_ylabel("Stability")

    st.pyplot(fig2)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<center><b>Ahmed Abdulhakim</b><br>"
    "NanoTox AI Platform</center>",
    unsafe_allow_html=True
)
