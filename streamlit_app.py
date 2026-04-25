import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(page_title="NanoTox AI", layout="wide")

st.title("NanoTox AI – Multi Pesticides")
st.caption("Manual Data Entry + Auto Training Model")

# =============================
# DATASET
# =============================
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame({
        "Pesticide":["Emamectin benzoate","Lambda-cyhalothrin","Imidacloprid"],
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

df = st.session_state.df

# =============================
# ADD DATA
# =============================
st.sidebar.subheader("➕ Add New Data")

pest = st.sidebar.text_input("Pesticide Name")

ai_in = st.sidebar.number_input("AI (%)",0.0,100.0,10.0)
surf_in = st.sidebar.number_input("Surfactant (%)",0.0,100.0,15.0)
solv_in = st.sidebar.number_input("Solvent (%)",0.0,100.0,5.0)
sonic_in = st.sidebar.number_input("Sonication (min)",0,60,10)
dls_in = st.sidebar.number_input("DLS (nm)",0.0,500.0,150.0)
zeta_in = st.sidebar.number_input("Zeta (mV)",-100.0,100.0,-25.0)
logp_in = st.sidebar.number_input("logP",0.0,10.0,5.0)
solub_in = st.sidebar.number_input("Solubility (mg/L)",0.0,1000.0,0.02)
mw_in = st.sidebar.number_input("MW (g/mol)",0.0,2000.0,800.0)
lc50_in = st.sidebar.number_input("LC50 (ppm)",0.0,10.0,0.2)

if st.sidebar.button("Add Data"):
    if pest.strip() == "":
        st.sidebar.error("Enter pesticide name")
    else:
        new_row = pd.DataFrame([{
            "Pesticide":pest,
            "AI":ai_in,
            "Surfactant":surf_in,
            "Solvent":solv_in,
            "Sonication":sonic_in,
            "DLS":dls_in,
            "Zeta":zeta_in,
            "logP":logp_in,
            "Solubility":solub_in,
            "MW":mw_in,
            "LC50":lc50_in
        }])
        st.session_state.df = pd.concat([df,new_row],ignore_index=True)
        st.success("Data Added")
        st.rerun()

# =============================
# DATA VIEW + ID
# =============================
st.subheader("📊 Dataset")

df_display = df.copy()
df_display.insert(0, "ID", range(1, len(df_display) + 1))

st.dataframe(
    df_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "ID": st.column_config.NumberColumn("ID", width="small")
    }
)

# =============================
# DELETE DATA
# =============================
st.subheader("🗑️ Delete Data")

if len(df) > 0:
    row_to_delete = st.selectbox(
        "Select row",
        df.index,
        format_func=lambda x: f"{x+1} - {df.loc[x,'Pesticide']}"
    )

    if st.button("Delete Selected Row"):
        st.session_state.df = df.drop(row_to_delete).reset_index(drop=True)
        st.success("Deleted")
        st.rerun()

# =============================
# EDIT DATA
# =============================
st.subheader("✏️ Edit Data")

if len(df) > 0:
    row_to_edit = st.selectbox(
        "Select row to edit",
        df.index,
        format_func=lambda x: f"{x+1} - {df.loc[x,'Pesticide']}"
    )

    selected_row = df.loc[row_to_edit]

    new_name = st.text_input("Pesticide Name", value=selected_row["Pesticide"])
    new_ai = st.number_input("AI (%)", value=float(selected_row["AI"]))
    new_surf = st.number_input("Surfactant (%)", value=float(selected_row["Surfactant"]))
    new_solv = st.number_input("Solvent (%)", value=float(selected_row["Solvent"]))
    new_sonic = st.number_input("Sonication (min)", value=int(selected_row["Sonication"]))
    new_dls = st.number_input("DLS (nm)", value=float(selected_row["DLS"]))
    new_zeta = st.number_input("Zeta (mV)", value=float(selected_row["Zeta"]))
    new_logp = st.number_input("logP", value=float(selected_row["logP"]))
    new_solub = st.number_input("Solubility", value=float(selected_row["Solubility"]))
    new_mw = st.number_input("MW", value=float(selected_row["MW"]))
    new_lc50 = st.number_input("LC50", value=float(selected_row["LC50"]))

    if st.button("Update Row"):
        st.session_state.df.loc[row_to_edit] = [
            new_name,new_ai,new_surf,new_solv,new_sonic,
            new_dls,new_zeta,new_logp,new_solub,new_mw,new_lc50
        ]
        st.success("Updated Successfully")
        st.rerun()

# =============================
# MODEL
# =============================
if len(df) < 1:
    st.info("Dataset is empty. Please add data.")
else:
    X = df[[
        "AI","Surfactant","Solvent","Sonication",
        "DLS","Zeta","logP","Solubility","MW"
    ]]
    y = df["LC50"]

    model = GradientBoostingRegressor()
    model.fit(X,y)

    # =============================
    # INPUT PARAMETERS
    # =============================
    st.sidebar.subheader("⚙️ Input Parameters")

    ai = st.sidebar.slider("AI (%)",1.0,20.0,10.0)
    surf = st.sidebar.slider("Surfactant (%)",5.0,30.0,15.0)
    solv = st.sidebar.slider("Solvent (%)",1.0,10.0,5.0)
    sonic = st.sidebar.slider("Sonication (min)",1,30,10)
    dls = st.sidebar.slider("DLS (nm)",50,300,150)
    zeta = st.sidebar.slider("Zeta (mV)",-60,60,-25)
    logp = st.sidebar.number_input("logP",value=5.0)
    solub = st.sidebar.number_input("Solubility (mg/L)",value=0.02)
    mw = st.sidebar.number_input("MW (g/mol)",value=800.0)

    input_data = [[ai,surf,solv,sonic,dls,zeta,logp,solub,mw]]

    pred = model.predict(input_data)[0]
    lc90 = pred * 2.5

    # =============================
    # RESULTS
    # =============================
    st.subheader("📊 Results")

    c1,c2 = st.columns(2)
    c1.metric("LC50 (ppm)",f"{pred:.4f}")
    c2.metric("LC90 (ppm)",f"{lc90:.4f}")

    # =============================
    # DLS
    # =============================
    st.subheader("📊 DLS Distribution")

    data = np.random.normal(dls,dls*0.08,800)

    fig2, ax2 = plt.subplots()
    count,bins,_ = ax2.hist(data,bins=20,edgecolor='black')

    mu,sigma = norm.fit(data)
    x = np.linspace(min(bins),max(bins),200)
    y_curve = norm.pdf(x,mu,sigma)
    y_scaled = y_curve * max(count)/max(y_curve)

    ax2.plot(x,y_scaled,'r')
    ax2.set_xlabel("Diameter (nm)")
    ax2.set_ylabel("Number")

    st.pyplot(fig2)

    # =============================
    # ZETA
    # =============================
    st.subheader("⚡ Zeta Potential")

    x = np.linspace(-150,150,2000)
    power = np.exp(-(x - zeta)**2/(2*5**2))

    fig3, ax3 = plt.subplots()
    ax3.plot(x,power,'r')
    ax3.set_xlabel("Zeta (mV)")
    ax3.set_ylabel("Power")

    st.pyplot(fig3)

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown(
    "<center><b>Ahmed Abdulhakim</b><br>"
    "ahmed.abdulhakim_a015@agr.kfs.edu.eg</center>",
    unsafe_allow_html=True
)
