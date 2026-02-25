import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="AROGYAM", layout="wide", page_icon="🛡️")

# --- TACTICAL STYLING (Professional Animations & Dark Theme) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    h1, h2, h3 { color: #00E5FF; text-transform: uppercase; letter-spacing: 1px; }
    
    .stButton>button { 
        width: 100%; border-radius: 4px; height: 3em; font-weight: bold; 
        text-transform: uppercase; letter-spacing: 1px; transition: all 0.2s ease-in-out; 
    }
    .stButton>button:hover { 
        transform: translateY(-2px); box-shadow: 0px 4px 12px rgba(0, 229, 255, 0.2); 
    }
    button[kind="primary"] { background-color: #007BFF; color: white; border: none; }
    button[kind="primary"]:hover { background-color: #0056b3; }

    .req { color: #EF4444; font-weight: bold; }
    
    .spo2-container { 
        border: 4px solid #00E5FF; border-radius: 50%; width: 150px; height: 150px; 
        display: flex; flex-direction: column; align-items: center; justify-content: center; 
        margin: auto; background: rgba(0, 229, 255, 0.1); 
    }
    
    /* Custom Sidebar Branding Glow */
    .brand-glow {
        text-align: center; font-size: 2.2rem; font-weight: 900; color: #00E5FF; 
        letter-spacing: 2px; transition: all 0.3s ease; cursor: default;
    }
    .brand-glow:hover {
        text-shadow: 0px 0px 15px rgba(0, 229, 255, 0.8), 0px 0px 30px rgba(0, 229, 255, 0.5);
    }
    .brand-sub {
        text-align: center; color: #38bdf8; font-weight: bold; margin-top: -15px; font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD ASSETS ---
working_dir = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))
scaler = pickle.load(open(f'{working_dir}/scaler.pkl', 'rb'))

# --- SESSION STATE MANAGEMENT ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'page_step' not in st.session_state: st.session_state['page_step'] = 1

# Permanent State initialization
default_vals = {
    'age': 30, 'sex': None, 's_bp': 120, 'd_bp': 80, 'pulse': 72, 'resp': 16, 'spo2': 98,
    'cp_yn': None, 'cp_type': None, 'rad': None, 'sweat': None, 'nausea': None, 'doe': None, 'syncope': None,
    'comorb': None, 'fam_hx': None, 'per_hx': None, 'ecg_opt': False, 'ecg_val': None, 'hb_opt': False, 'hb_val': 14.0, 'trop_val': None
}
for key, val in default_vals.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Helper for Radio/Selectbox indexing
def get_idx(options, val):
    return options.index(val) if val in options else None

# --- MEDICAL TOOLTIP DICTIONARY (Expanded) ---
tt_lib = {
    'age': "Age is a primary risk factor. Men >45 and Women >55 have a naturally higher risk of arterial plaque buildup.",
    'sex': "Men generally have a higher risk of heart attacks earlier in life. Women's risk catches up post-menopause.",
    's_bp': "Systolic BP measures pressure in arteries when the heart beats. Normal < 120. >180 is a Hypertensive Crisis requiring immediate meds.",
    'd_bp': "Diastolic BP measures pressure between beats. Normal < 80. >120 is a Hypertensive Crisis. High BP damages vessel walls over time.",
    'pulse': "Resting pulse should be 60-100 BPM. >120 at rest means the heart is struggling. <50 can cause fainting.",
    'resp': "Normal is 12-20 breaths/min. >25 indicates the lungs are failing to get enough oxygen to the blood (respiratory distress/fluid in lungs).",
    'spo2': "Measures oxygen saturation in the blood. Normal > 95%. <92% is severe Hypoxia, meaning organs are starving for oxygen. Needs immediate supplemental O2.",
    'cp_yn': "Any discomfort, squeezing, heaviness, or burning in the chest. Not all heart attacks have pain, but it is the #1 warning sign.",
    'cp_type': "Tight/Heavy usually means Angina (partial blockage). Crushing/Tearing pain is a severe red flag for a full Heart Attack (Myocardial Infarction).",
    'rad': "Pain starting in the chest but traveling to the left arm, jaw, neck, or back. Caused by nerve pathways sharing signals. Classic sign of heart muscle dying.",
    'sweat': "Sudden, heavy cold sweats when resting. The body is going into shock due to heart strain, dumping adrenaline into the blood.",
    'nausea': "Feeling sick to the stomach or vomiting without a GI bug. Often accompanies heart attacks in the lower (inferior) wall of the heart.",
    'doe': "Severe shortness of breath with minimal physical activity (like walking a few steps). The heart is too weak to pump oxygenated blood.",
    'syncope': "Sudden loss of consciousness or fainting spells. Caused by a sudden drop in blood pressure, starving the brain of oxygen.",
    'comorb': "Pre-existing conditions. Hypertension (High BP) damages arteries. Diabetes ruins blood vessels. Dyslipidemia (High Cholesterol) blocks arteries.",
    'fam_hx': "Did parents/siblings have heart attacks before age 50? Genetic predisposition significantly lowers the threshold for an event.",
    'per_hx': "Has this person had heart surgery, stents, or a heart attack before? They are at an extremely high risk of a recurrent event.",
    'ecg_val': "ECG reads the heart's electrical signals. ST Elevation means an artery is completely blocked right now. ST Depression means severe lack of oxygen.",
    'hb_val': "Measures protein in red blood cells that carries O2. Normal: 13-18. <10 = Severe anemia. >18 = Blood is sludging/thick, massive risk of clotting.",
    'trop_val': "A rapid blood test. If POSITIVE, it means heart muscle cells have literally burst and are leaking proteins into the blood. Confirmed Heart Attack."
}

# --- 1. LOGIN PAGE ---
def login_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='brand-glow'> <h1 style='text-align: center;'>🛡️AROGYAM</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #38bdf8;'>SELF ASSESSMENT HEALTH GUIDE</h4>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #38bdf8;'>BY BARHE CHALO</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94A3B8;'>High-Altitude Cardiovascular Decision Support Tool</p>", unsafe_allow_html=True)
        with st.form("login_form"):
            user = st.text_input("USER ID")
            passwd = st.text_input("PASSWORD KEY", type="password")
            if st.form_submit_button("ACCESS SYSTEM"):
                if user == "admin" and passwd == "admin":
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("INVALID CREDENTIALS")

# --- MAIN APPLICATION ---
def main_app():
    with st.sidebar:
        st.markdown("<div class='brand-glow'>🛡️AROGYAM</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-sub'>BY BARHE CHALO 🩺</div>", unsafe_allow_html=True)
        st.markdown("---")
        
        selected = option_menu("Diagnosis", ["Heart Disease", "Brain Stroke"], icons=["heart-pulse", "lightning-charge"], menu_icon="cast", default_index=0)
        
        st.write("<br><br><br><br>", unsafe_allow_html=True)
        _, c_btn, _ = st.columns([1, 4, 1])
        with c_btn:
            if st.button("SAFE LOGOUT", type="secondary"):
                st.session_state.clear()
                st.rerun()

    if selected == "Brain Stroke (WIP)":
        st.title("🧠 Brain Stroke Prediction")
        st.info("Module under development. Awaiting clinical parameters.")
        return

    st.markdown(f"### PAGE {st.session_state['page_step']} OF 4")
    cols = st.columns(4)
    for i in range(4): cols[i].progress(100 if st.session_state['page_step'] > i else (0 if st.session_state['page_step'] <= i else 50))
    st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)

    opts_yn = ["No", "Yes"]
    opts_cp = ["Tight", "Heavy", "Crushing"]
    opts_comorb = ["None", "Hypertension", "Diabetes", "Dyslipidemia"]
    opts_trop = ["Negative", "Positive"]
    opts_ecg = ["Normal", "ST Elevation", "ST Depression", "T Wave Inversion", "LBBB", "Pathological Q Waves"]

    # --- PAGE 1: CORE VITALS ---
    if st.session_state['page_step'] == 1:
        st.header("Core Vitals")
        st.caption("Fields marked with * are mandatory. Hover over the ? for clinical information.")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.session_state['age'] = st.number_input("Age *", 18, 90, st.session_state['age'], help=tt_lib['age'])
            st.session_state['sex'] = st.radio("Sex *", ["Male", "Female"], index=get_idx(["Male", "Female"], st.session_state['sex']), horizontal=True, help=tt_lib['sex'])
            
            bp1, bp2 = st.columns(2)
            with bp1:
                st.session_state['s_bp'] = st.number_input("Systolic BP *", 60, 240, st.session_state['s_bp'], help=tt_lib['s_bp'])
            with bp2:
                st.session_state['d_bp'] = st.number_input("Diastolic BP *", 40, 150, st.session_state['d_bp'], help=tt_lib['d_bp'])
            
            st.session_state['pulse'] = st.number_input("Pulse Rate (BPM) *", 40, 220, st.session_state['pulse'], help=tt_lib['pulse'])
            st.session_state['resp'] = st.number_input("Resp Rate (/min) *", 8, 50, st.session_state['resp'], help=tt_lib['resp'])

        with col_b:
            st.markdown(f"<div style='text-align: center; font-weight: bold;' title='{tt_lib['spo2']}'>SpO2 Levels (%) <span class='req'>*</span> ❓</div>", unsafe_allow_html=True)
            st.markdown("<div class='spo2-container'><h3>🫧 O₂</h3></div>", unsafe_allow_html=True)
            st.session_state['spo2'] = st.select_slider("spo2_hidden", options=list(range(60, 101)), value=st.session_state['spo2'], label_visibility="collapsed")
            st.write(f"<h1 style='text-align:center;'>{st.session_state['spo2']}%</h1>", unsafe_allow_html=True)

    # --- PAGE 2: CLINICAL FACTORS ---
    elif st.session_state['page_step'] == 2:
        st.header("Clinical Factors")
        st.caption("Hover over the ? for clinical information.")
        
        st.session_state['cp_yn'] = st.radio("Chest Pain Present? *", opts_yn, index=get_idx(opts_yn, st.session_state['cp_yn']), horizontal=True, help=tt_lib['cp_yn'])
        st.session_state['cp_type'] = st.selectbox("Type of Pain *", opts_cp, index=get_idx(opts_cp, st.session_state['cp_type']) if st.session_state['cp_yn'] == "Yes" else None, disabled=(st.session_state['cp_yn'] != "Yes"), help=tt_lib['cp_type'])
        
        c1, c2 = st.columns(2)
        with c1:
            st.session_state['rad'] = st.radio("Radiation of Pain *", opts_yn, index=get_idx(opts_yn, st.session_state['rad']), horizontal=True, help=tt_lib['rad'])
            st.session_state['sweat'] = st.radio("Sweating (Diaphoresis) *", opts_yn, index=get_idx(opts_yn, st.session_state['sweat']), horizontal=True, help=tt_lib['sweat'])
            st.session_state['nausea'] = st.radio("Nausea / Vomiting *", opts_yn, index=get_idx(opts_yn, st.session_state['nausea']), horizontal=True, help=tt_lib['nausea'])
        with c2:
            st.session_state['doe'] = st.radio("Dyspnoea on Exertion (DOE) *", opts_yn, index=get_idx(opts_yn, st.session_state['doe']), horizontal=True, help=tt_lib['doe'])
            st.session_state['syncope'] = st.radio("Syncope (Fainting) *", opts_yn, index=get_idx(opts_yn, st.session_state['syncope']), horizontal=True, help=tt_lib['syncope'])

    # --- PAGE 3: HISTORY & DIAGNOSTICS ---
    elif st.session_state['page_step'] == 3:
        st.header("History & Diagnostics")
        c1, c2 = st.columns(2)
        
        with c1:
            with st.container(border=True):
                st.subheader("History")
                st.session_state['comorb'] = st.selectbox("Primary Comorbidity *", opts_comorb, index=get_idx(opts_comorb, st.session_state['comorb']), help=tt_lib['comorb'])
                st.session_state['fam_hx'] = st.radio("Family History of CVD *", opts_yn, index=get_idx(opts_yn, st.session_state['fam_hx']), horizontal=True, help=tt_lib['fam_hx'])
                st.session_state['per_hx'] = st.radio("Personal History of CVD *", opts_yn, index=get_idx(opts_yn, st.session_state['per_hx']), horizontal=True, help=tt_lib['per_hx'])
        
        with c2:
            with st.container(border=True):
                st.subheader("Field Diagnostics")
                st.session_state['ecg_opt'] = st.toggle("Is ECG Available?", value=st.session_state['ecg_opt'])
                if st.session_state['ecg_opt']:
                    st.session_state['ecg_val'] = st.selectbox("ECG Finding *", opts_ecg, index=get_idx(opts_ecg, st.session_state.get('ecg_val')), help=tt_lib['ecg_val'])
                else:
                    st.session_state['ecg_val'] = "Normal"
                
                st.session_state['hb_opt'] = st.toggle("Is Hb Test Available?", value=st.session_state['hb_opt'])
                if st.session_state['hb_opt']:
                    st.session_state['hb_val'] = st.slider("Hemoglobin (g/dL) *", 5.0, 25.0, st.session_state['hb_val'], help=tt_lib['hb_val'])
                else:
                    st.session_state['hb_val'] = 14.0

                st.session_state['trop_val'] = st.radio("Troponin T (Rapid Kit) *", opts_trop, index=get_idx(opts_trop, st.session_state['trop_val']), horizontal=True, help=tt_lib['trop_val'])

    # --- PAGE 4: DETAILED HYBRID DIAGNOSIS ---
    elif st.session_state['page_step'] == 4:
        st.header("Diagnostic Triage Results")
        
        sex_val = 1 if st.session_state['sex'] == "Male" else 0
        cp_yn_val = 1 if st.session_state['cp_yn'] == "Yes" else 0
        cp_map = {"Tight": 0, "Heavy": 1, "Crushing": 2}
        cp_type_val = cp_map.get(st.session_state['cp_type'], 0) if cp_yn_val == 1 else 0
        
        rad_val = 1 if st.session_state['rad'] == "Yes" else 0
        sweat_val = 1 if st.session_state['sweat'] == "Yes" else 0
        nau_val = 1 if st.session_state['nausea'] == "Yes" else 0
        doe_val = 1 if st.session_state['doe'] == "Yes" else 0
        sync_val = 1 if st.session_state['syncope'] == "Yes" else 0
        
        com_map = {"None": 0, "Hypertension": 1, "Diabetes": 2, "Dyslipidemia": 3}
        fam_val = 1 if st.session_state['fam_hx'] == "Yes" else 0
        per_val = 1 if st.session_state['per_hx'] == "Yes" else 0
        
        ecg_map = {"Normal": 0, "ST Elevation": 1, "ST Depression": 2, "T Wave Inversion": 3, "LBBB": 4, "Pathological Q Waves": 5}
        ecg_mapped = ecg_map.get(st.session_state['ecg_val'], 0)
        trop_mapped = 1 if st.session_state['trop_val'] == "Positive" else 0

        features = [
            st.session_state['age'], sex_val, st.session_state['s_bp'], st.session_state['d_bp'], 
            st.session_state['pulse'], st.session_state['resp'], st.session_state['spo2'],
            cp_yn_val, cp_type_val, rad_val, sweat_val, nau_val, doe_val, sync_val,
            com_map.get(st.session_state['comorb'], 0), fam_val, per_val, ecg_mapped, 
            st.session_state['hb_val'], trop_mapped
        ]
        
        input_scaled = scaler.transform([features])
        ml_prediction = model.predict(input_scaled)[0]
        
        abnormal_flags = []
        critical_flags = []
        
        if st.session_state['s_bp'] > 160 or st.session_state['d_bp'] > 100: abnormal_flags.append({"name": f"High BP ({st.session_state['s_bp']}/{st.session_state['d_bp']})", "act": "Monitor closely. Keep subject seated. Do not give stimulants (tea/coffee)."})
        if st.session_state['pulse'] > 120 or st.session_state['pulse'] < 50: abnormal_flags.append({"name": f"Abnormal Pulse ({st.session_state['pulse']} BPM)", "act": "Assess for shock or severe dehydration. Hydrate slowly if conscious."})
        if st.session_state['spo2'] < 92: abnormal_flags.append({"name": f"Hypoxia (SpO2: {st.session_state['spo2']}%)", "act": "Administer Supplemental Oxygen via mask. Prepare for descent if at high altitude."})
        if st.session_state['resp'] > 25: abnormal_flags.append({"name": f"Tachypnea (Resp: {st.session_state['resp']}/min)", "act": "Patient is struggling to breathe. Sit them upright. Administer O2."})
        
        if st.session_state['fam_hx'] == "Yes": abnormal_flags.append({"name": "Family History of Heart Disease", "act": "Lowers threshold for evacuation. Treat minor symptoms more seriously."})
        if st.session_state['per_hx'] == "Yes": abnormal_flags.append({"name": "Previous Heart Issues", "act": "Extremely high risk of recurrence. Subject should not do heavy lifting/patrols."})
        
        if 18.0 < st.session_state['hb_val'] <= 19.5: abnormal_flags.append({"name": f"Elevated Hemoglobin ({st.session_state['hb_val']} g/dL)", "act": "Blood is thickening. Hydrate heavily. Restrict physical exertion to prevent clotting."})
        elif st.session_state['hb_val'] > 19.5: critical_flags.append({"name": f"CRITICAL Hemoglobin ({st.session_state['hb_val']} g/dL)", "act": "Severe risk of stroke/thrombosis due to blood sludging. Immediate hydration and CAS EVAC."})
        elif st.session_state['hb_val'] < 10.0: abnormal_flags.append({"name": f"Low Hemoglobin ({st.session_state['hb_val']} g/dL)", "act": "Anemia. Blood cannot carry enough oxygen. Do not deploy to high altitude."})
        
        if st.session_state['nausea'] == "Yes": abnormal_flags.append({"name": "Nausea / Vomiting", "act": "Ensure airway is clear. Do not force feed."})
        if st.session_state['doe'] == "Yes": abnormal_flags.append({"name": "Dyspnoea on Exertion", "act": "Strict bed rest. Administer O2. Check for HAPE."})
        
        if st.session_state['cp_yn'] == "Yes": critical_flags.append({"name": f"Active Chest Pain ({st.session_state.get('cp_type', '')})", "act": "Assume Heart Attack. Administer 300mg chewable Aspirin immediately (if not allergic). Give O2."})
        if st.session_state['rad'] == "Yes": critical_flags.append({"name": "Radiating Pain", "act": "Classic Ischemia. Administer Sorbitrate/Nitroglycerin under tongue if BP > 100."})
        if st.session_state['sweat'] == "Yes": critical_flags.append({"name": "Diaphoresis (Cold Sweats)", "act": "Subject is in clinical shock. Elevate legs slightly, keep warm."})
        if st.session_state['syncope'] == "Yes": critical_flags.append({"name": "Syncope (Fainting)", "act": "Check pulse and breathing. Be prepared to start CPR."})
        
        if st.session_state['ecg_val'] in ["ST Elevation", "ST Depression", "Pathological Q Waves"]: critical_flags.append({"name": f"Severe ECG Finding ({st.session_state['ecg_val']})", "act": "Confirmed cardiac event. CAS EVAC is mandatory."})
        if st.session_state['trop_val'] == "Positive": critical_flags.append({"name": "Troponin T POSITIVE", "act": "Confirmed death of heart muscle cells. Time is tissue. Immediate CAS EVAC."})

        total_abnormal = len(abnormal_flags) + len(critical_flags)
        
        if len(critical_flags) > 0 or total_abnormal >= 3 or ml_prediction == 0:
            st.markdown("<h2 style='color: #EF4444; border: 2px solid #EF4444; padding: 15px; text-align: center; border-radius: 5px; background: #450a0a;'>🔴 ZONE RED: IMMEDIATE CAS EVACUATION</h2>", unsafe_allow_html=True)
            if ml_prediction == 0: st.error("🚨 **ML Model Alert:** The analytical algorithm has detected a high probability of a severe cardiovascular event.")
            
            st.write("### 🚨 CRITICAL PARAMETERS DETECTED")
            for flag in critical_flags:
                st.markdown(f"**<span style='color:#EF4444'>{flag['name']}</span>**", unsafe_allow_html=True)
                st.caption(f"👉 **Action:** {flag['act']}")
            
            if abnormal_flags:
                st.write("### ⚠️ CONTRIBUTING FACTORS")
                for flag in abnormal_flags:
                    st.markdown(f"**<span style='color:#F59E0B'>{flag['name']}</span>**", unsafe_allow_html=True)
                    st.caption(f"👉 **Action:** {flag['act']}")
            
            st.error("**FINAL ORDER:** Initiate emergency MEDEVAC protocol. Keep patient calm, seated, and warm. Continuous monitoring required.")
            
        elif total_abnormal >= 1:
            st.markdown("<h2 style='color: #F59E0B; border: 2px solid #F59E0B; padding: 15px; text-align: center; border-radius: 5px; background: #451a03;'>🟡 ZONE AMBER: CAUTION & MONITORING</h2>", unsafe_allow_html=True)
            st.info("The ML model shows low baseline risk, but specific abnormal parameters require attention.")
            
            st.write("### ⚠️ ABNORMAL PARAMETERS DETECTED")
            for flag in abnormal_flags:
                st.markdown(f"**<span style='color:#F59E0B'>{flag['name']}</span>**", unsafe_allow_html=True)
                st.caption(f"👉 **Action:** {flag['act']}")
                
            st.warning("**FINAL ORDER:** Subject is stable but requires close monitoring. Withhold from heavy physical exertion. Reassess vitals every 4 hours. Consult MO.")
            
        else:
            st.markdown("<h2 style='color: #10B981; border: 2px solid #10B981; padding: 15px; text-align: center; border-radius: 5px; background: #064e3b;'>🟢 ZONE GREEN: STABLE / FIT FOR DUTY</h2>", unsafe_allow_html=True)
            st.write("Machine Learning analysis and clinical rule-checks show all inputted vital signs and markers are perfectly normal.")
            st.success("**FINAL ORDER:** Continue standard acclimatization and monitoring protocols. No immediate medical intervention required.")

    # --- NAVIGATION CHECKERS ---
    def validate(step):
        if step == 1: return st.session_state.get('sex') is not None
        if step == 2:
            r = [st.session_state.get(k) for k in ['cp_yn', 'rad', 'sweat', 'nausea', 'doe', 'syncope']]
            if any(x is None for x in r): return False
            if st.session_state['cp_yn'] == "Yes" and st.session_state.get('cp_type') is None: return False
            return True
        if step == 3:
            r = [st.session_state.get(k) for k in ['comorb', 'fam_hx', 'per_hx', 'trop_val']]
            if any(x is None for x in r): return False
            if st.session_state['ecg_opt'] and st.session_state.get('ecg_val') is None: return False
            return True
        return True

    # --- BOTTOM NAVIGATION BUTTONS ---
    st.write("<br><br>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns([1, 2, 1])
    
    with b1:
        if 1 < st.session_state['page_step'] <= 4:
            if st.button("PREVIOUS PAGE"):
                st.session_state['page_step'] -= 1
                st.rerun()
                
    with b3:
        if st.session_state['page_step'] < 4:
            action_text = "NEXT PAGE" if st.session_state['page_step'] < 3 else "RUN DIAGNOSIS"
            if st.button(action_text, type="primary"):
                if validate(st.session_state['page_step']):
                    st.session_state['page_step'] += 1
                    st.rerun()
                else:
                    st.error("⚠️ Please complete all mandatory (*) fields.")
        elif st.session_state['page_step'] == 4:
            if st.button("NEW ASSESSMENT"):
                st.session_state.clear()
                st.session_state['logged_in'] = True
                st.session_state['page_step'] = 1
                st.rerun()

if not st.session_state['logged_in']:
    login_page()
else:
    main_app()