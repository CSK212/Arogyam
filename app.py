import os
import io
import tempfile  # <--- Add this missing line right here!
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import urllib.parse
from datetime import datetime
from PIL import Image
import streamlit as st
from cryptography.fernet import Fernet
import time

# --- SUPABASE IMPORT ---
from supabase import create_client, Client

# ==========================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ==========================================
st.set_page_config(page_title="AROGYAM", layout="wide", page_icon="🛡️")

working_dir = os.path.dirname(os.path.abspath(__file__))

# --- SECURE LOCAL FOLDERS FOR TEMP FILES ---
TEMP_IMG_DIR = os.path.join(working_dir, 'temp_images')
os.makedirs(TEMP_IMG_DIR, exist_ok=True)

# ==========================================
# SUPABASE INITIALIZATION
# ==========================================
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase: Client = init_supabase()

# ==========================================
# GLOBAL POST VARIABLES (15 POSTS)
# ==========================================
GLOBAL_POSTS = [
    "Alpha Post", "Bravo Post", "Charlie Post", "Delta Post", "Echo Post", 
    "Foxtrot Post", "Golf Post", "Hotel Post", "India Post", "Juliet Post", 
    "Kilo Post", "Lima Post", "Mike Post", "November Post", "Oscar Post"
]

# ==========================================
# ENCRYPTION (AES-256 via Fernet)
# ==========================================
CIPHER_KEY = b'uP5mY2n4D8J7k9L1H6s3V0x2B5N8M7Q4W1Z9R5T6y2I=' 
cipher = Fernet(CIPHER_KEY)

def encrypt_data(data):
    if not data: return data
    try: return cipher.encrypt(data.encode()).decode()
    except Exception: return data

def decrypt_data(data):
    if not data or data == "N/A": return data
    try: return cipher.decrypt(data.encode()).decode()
    except Exception: return data

# ==========================================
# CLOUD DATA FETCHING & HELPERS
# ==========================================
def get_patient_record(army_no_query):
    if not army_no_query: return None
    res = supabase.table("patient_registry").select("*").execute()
    df = pd.DataFrame(res.data)
    
    if df.empty: return None
    
    df['army_no'] = df['army_no'].apply(decrypt_data)
    match = df[df['army_no'].str.upper() == army_no_query.strip().upper()]
    
    if not match.empty:
        row = match.iloc[0].to_dict()
        row['name'] = decrypt_data(row['name'])
        row['nok_name'] = decrypt_data(row['nok_name'])
        row['nok_phone'] = decrypt_data(row['nok_phone'])
        return row
    return None

def clean_all_duplicates():
    """Silently scans and removes encrypted duplicate records across all tables on startup."""
    # 1. Clean Patient Registry (Only 1 profile allowed per Army No)
    try:
        res = supabase.table("patient_registry").select("army_no").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            df['dec_army'] = df['army_no'].apply(decrypt_data).str.strip().str.upper()
            dups = df[df.duplicated(subset=['dec_army'], keep='first')]
            for raw_army in dups['army_no']: 
                supabase.table("patient_registry").delete().eq("army_no", raw_army).execute()
    except Exception: pass

    # 2. Clean Patient History (Prevents double-clicking save on the same triage form)
    try:
        res = supabase.table("patient_history").select("id, army_no, timestamp, module").execute()
        if res.data:
            df = pd.DataFrame(res.data)
            df['dec_army'] = df['army_no'].apply(decrypt_data).str.strip().str.upper()
            dups = df[df.duplicated(subset=['dec_army', 'timestamp', 'module'], keep='first')]
            for rid in dups['id']: 
                supabase.table("patient_history").delete().eq("id", rid).execute()
    except Exception: pass

    # 3. Clean Weekly Vitals & Acclimatization
    for table in ["weekly_vitals", "acclimatization_details"]:
        try:
            res = supabase.table(table).select("id, army_no, timestamp").execute()
            if res.data:
                df = pd.DataFrame(res.data)
                df['dec_army'] = df['army_no'].apply(decrypt_data).str.strip().str.upper()
                dups = df[df.duplicated(subset=['dec_army', 'timestamp'], keep='first')]
                for rid in dups['id']: 
                    supabase.table(table).delete().eq("id", rid).execute()
        except Exception: pass

# Run cleanup silently once when the app boots up
if 'dedup_done' not in st.session_state:
    clean_all_duplicates()
    st.session_state['dedup_done'] = True

def parse_date_safe(date_str, default_year=2000):
    if not date_str or date_str == "N/A":
        return datetime.now().date()
    try:
        return datetime.strptime(str(date_str), '%Y-%m-%d').date()
    except:
        return datetime(default_year, 1, 1).date()

def render_whatsapp_alert(module_name, rank, name, army_no):
    res = supabase.table("med_contacts").select("*").execute()
    contacts = res.data if res.data else []
    
    if not contacts:
        st.warning("⚠️ No Medical Chain of Command contacts configured in Admin Settings.")
        return
        
    st.markdown("### 📲 INITIATE MEDEVAC / SPECIALIST ALERTS")
    cols = st.columns(len(contacts))
    
    for idx, contact in enumerate(contacts):
        role = contact.get("role", "")
        c_rank = contact.get("rank", "")
        c_name = contact.get("name", "")
        phone = contact.get("phone", "")
        
        if phone and len(phone) > 5:
            wa_number = ''.join(filter(lambda x: x.isdigit() or x == '+', phone))
            wa_text = f"🚨 *CRITICAL CASUALTY ALERT* 🚨\n\n*To:* {role} ({c_rank} {c_name})\n*Post:* {st.session_state['post_name']}\n*Patient:* {rank} {name} ({army_no})\n*Diagnosis:* {module_name} - ZONE RED\n*Action:* IMMEDIATE EVACUATION REQUIRED\n\n_Please check Arogyam MDSS Dashboard for full PDF report._"
            wa_link = f"https://wa.me/{wa_number}?text={urllib.parse.quote(wa_text)}"
            with cols[idx]:
                st.markdown(f'<a href="{wa_link}" target="_blank" style="display: block; width: 100%; text-align: center; padding: 0.8em; color: white; background-color: #25D366; text-decoration: none; border-radius: 4px; font-weight: bold; margin-top: 10px;">Alert {role}</a>', unsafe_allow_html=True)

try:
    from fpdf import FPDF
    fpdf_available = True
except ImportError:
    fpdf_available = False

from streamlit_option_menu import option_menu

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    h1, h2, h3 { color: #00E5FF; text-transform: uppercase; letter-spacing: 1px; }
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; transition: all 0.2s ease-in-out; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0px 4px 12px rgba(0, 229, 255, 0.2); }
    button[kind="primary"] { background-color: #007BFF; color: white; border: none; }
    button[kind="primary"]:hover { background-color: #0056b3; }
    .spo2-wrapper { background: linear-gradient(145deg, rgba(0, 229, 255, 0.05), rgba(0, 123, 255, 0.1)); border: 2px solid rgba(0, 229, 255, 0.3); border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2); margin-bottom: 10px; transition: transform 0.2s; }
    .spo2-wrapper:hover { transform: scale(1.02); border-color: #00E5FF; }
    .spo2-title { color: #38bdf8; font-size: 1.1rem; font-weight: bold; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;}
    .spo2-val { font-size: 3.5rem; font-weight: 900; color: #00E5FF; line-height: 1; margin: 10px 0; text-shadow: 0 0 10px rgba(0,229,255,0.4); }
    .brand-glow { text-align: center; font-size: 2.2rem; font-weight: 900; color: #00E5FF; letter-spacing: 2px; transition: all 0.3s ease; cursor: default; }
    .brand-glow:hover { text-shadow: 0px 0px 15px rgba(0, 229, 255, 0.8), 0px 0px 30px rgba(0, 229, 255, 0.5); }
    .brand-sub { text-align: center; color: #38bdf8; font-weight: bold; margin-top: -15px; font-size: 1.1rem; }
    .demo-badge { background: linear-gradient(90deg, #1e3a8a, #3b82f6); color: white; padding: 10px 20px; border-radius: 8px; text-align: center; font-weight: bold; letter-spacing: 1px; border: 1px solid #60a5fa; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .pain-map-container { background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; text-align: center; margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        rf_model = pickle.load(open(f'{working_dir}/heart_disease_model.sav', 'rb'))
        rf_scaler = pickle.load(open(f'{working_dir}/scaler.pkl', 'rb'))
        return rf_model, rf_scaler
    except Exception: return None, None 

@st.cache_resource
def load_cnn():
    import tensorflow as tf
    try: return tf.keras.models.load_model(f'{working_dir}/demo_ecg_model.h5')
    except Exception: return None

model, scaler = load_models()

# ==========================================
# GLOBAL OPTIONS & VARIABLES
# ==========================================
alt_opts = ["< 9000", "9000-12000", "12000-15000", "15000-18000", "> 18000"]
opts_yn = ["No", "Yes"]
time_opts = ["Select Time", "Less than 3 hours ago", "3 to 4.5 hours ago", "More than 4.5 hours ago", "Unknown / Woke up with symptoms"]
opts_cp = ["Tight", "Heavy", "Crushing"]
opts_comorb = ["None", "Hypertension/hypotension", "Hypothyroid/hyperthyroid", "Diabetes", "Dyslipidemia", "Cardiovascular diseases"]
opts_trop = ["Negative", "Positive"]
opts_ecg = ["Normal", "ST Elevation", "ST Depression", "T Wave Inversion", "LBBB", "Pathological Q Waves"]

# ==========================================
# SESSION STATE MANAGEMENT & DEFAULTS
# ==========================================
if 'logged_in' not in st.session_state: 
    st.session_state['logged_in'] = False
    st.session_state['bfna_id'] = None
    st.session_state['post_name'] = None

nav_states = ['page_step', 'bshc_page_step', 'ams_page_step', 'hape_page_step', 'ci_page_step']
for state in nav_states:
    if state not in st.session_state: st.session_state[state] = 1

default_vals = {
    'p_rank': '', 'p_name': '', 'p_num': '', 'p_loc': '',
    
    # HD Core & SOCRATES
    'age': 30, 'sex': "Male", 's_bp': 120, 'd_bp': 80, 'pulse': 72, 'resp': 16, 'spo2': 98, 'temp': 98.6, 'alt': "< 9000",
    'cp_yn': "No", 'cp_site': [], 'cp_onset': "Unknown", 'cp_char': "Unknown", 'cp_rad': "No", 'cp_rad_text': "",
    'cp_assoc_sweat': "No", 'cp_assoc_nau': "No", 'cp_assoc_cough': "No", 'cp_assoc_doe': "No", 'cp_assoc_sync': "No", 'cp_assoc_bowel': "No", 'cp_assoc_slur': "No", 'cp_assoc_focal': "No", 'cp_assoc_other': "",
    'cp_timing': "Constant pain", 'cp_exac': "No", 'cp_exac_text': "", 'cp_relieve': "No", 'cp_relieve_text': "", 'cp_severity': 0,
    'comorb': "None", 'fam_hx_cond': "None", 'hx_alcohol': "No", 'hx_smoking': "No",
    'ecg_opt': False, 'ecg_val': "Normal", 'hb_opt': False, 'hb_val': 14.0, 'trop_val': "Negative",
    'ecg_img_path': None, 'ecg_ai_result': "N/A", 'trop_img_path': None,
    
    # HD Physical Exam
    'hd_pe_tenderness': "No", 'hd_pe_bowel': "No", 'hd_pe_pulsations': "Present (+)", 'hd_pe_discolor': "No", 'hd_pe_distension': "No",
    
    # Brain Stroke & HACE Unified
    'bshc_sex': "Male", 'bshc_s_bp': 120, 'bshc_d_bp': 80, 'bshc_pulse': 72, 'bshc_resp': 16, 'bshc_spo2': 90, 'bshc_temp': 98.6, 'bshc_alt': "< 9000", 'bshc_age': 30,
    'bshc_balance': "No", 'bshc_eyes': "No", 'bshc_face': "No", 'bshc_arms': "No", 'bshc_speech': "No", 'bshc_time': "Select Time",
    'bshc_vertigo': "No", 'bshc_nystagmus': "No", 'bshc_tremor': "No", 'bshc_slur': "No", 'bshc_hypotonia': "No", 'bshc_gait': "No", 'bshc_dysdia': "Yes", 'bshc_ftn': "Yes", 'bshc_hts': "Yes", 'bshc_rebound': "Yes", 'bshc_romberg': "No",
    'bshc_headache': 0, 'bshc_mental': "No", 'bshc_vomit': "No", 'bshc_nausea': "No", 'bshc_dizzy': "No", 'bshc_sensation': "No", 'bshc_pupils': "Yes", 'bshc_dtr': "Normal",
    
    # AMS
    'ams_s_bp': 120, 'ams_d_bp': 80, 'ams_pulse': 72, 'ams_resp': 16, 'ams_spo2': 90, 'ams_temp': 98.6, 'ams_alt': "< 9000",
    'll_headache': 0, 'll_gi': 0, 'll_fatigue': 0, 'll_dizzy': 0, 'll_sleep': 0, 'ams_urine': "Clear / Pale Yellow",
    
    # HAPE
    'hape_s_bp': 120, 'hape_d_bp': 80, 'hape_pulse': 72, 'hape_resp': 16, 'hape_spo2': 90, 'hape_temp': 98.6, 'hape_alt': "< 9000",
    'hape_dyspnea': "Normal", 'hape_resp_qual': "Normal", 'hape_activity': "Normal", 'hape_mobility': "Normal", 
    'hape_mental': "Normal", 'hape_cough': "None", 'hape_cyanosis': "None", 
    'hape_nausea': "No", 'hape_rales': "No", 'hape_headache': "No",
    
    # Cold Injuries
    'ci_s_bp': 120, 'ci_d_bp': 80, 'ci_pulse': 72, 'ci_resp': 16, 'ci_spo2': 90, 'ci_temp': 98.6, 'ci_alt': "< 9000",
    'ci_mental_alt': "No", 'ci_breathing': "No", 'ci_shiver': "Yes",
    'ci_assoc_sweat': "No", 'ci_assoc_nau': "No", 'ci_assoc_cough': "No", 'ci_assoc_doe': "No", 'ci_assoc_sync': "No", 'ci_assoc_bowel': "No", 'ci_assoc_slur': "No", 'ci_assoc_focal': "No", 'ci_assoc_other': "",
    'ci_site': "", 'ci_img_path': None, 'ci_skin_color': "Normal", 'ci_tenderness': "No", 'ci_sensation': "No", 'ci_cap_refill': "No",
    'ci_frostbite_stage': "None", 'ci_blister_type': "None"
}

for key, val in default_vals.items():
    if key not in st.session_state: st.session_state[key] = val

def get_idx(options, val):
    if val is None: return 0
    try: return options.index(val)
    except ValueError: return 0

tt_lib = {
    'age': "HOW TO CHECK: Ask the patient their age.\nWHAT IT MEANS: Older patients have stiffer arteries, increasing heart attack and stroke risks.",
    'sex': "HOW TO CHECK: Select Male or Female.\nWHAT IT MEANS: Men and women experience heart attacks differently. Men usually have crushing chest pain, women may just feel nausea or back pain.",
    'bp': "HOW TO CHECK: Have patient sit still for 5 mins. Wrap cuff on bare upper arm at heart level. Top number is Systolic, bottom is Diastolic.\nWHAT IT MEANS: Pressure of blood. Over 140/90 is a warning. Over 180/110 is a severe emergency.",
    'pulse': "HOW TO CHECK: Place two fingers on the inside of the wrist. Count beats for 60 seconds.\nWHAT IT MEANS: Normal is 60-100. Over 120 means the body is under severe stress or shock.",
    'resp': "HOW TO CHECK: Pretend to check their pulse, but watch their chest rise and fall. Count breaths for 60 seconds.\nWHAT IT MEANS: Normal is 12-20. Over 25 means they are struggling for air.",
    'spo2': "HOW TO CHECK: Ensure finger is warm and clean. Clip oximeter and wait 30 seconds.\nWHAT IT MEANS: Oxygen in blood. Below 90% at altitude is bad. Below 85% needs immediate oxygen.",
    'temp': "HOW TO CHECK: Use a digital thermometer under the tongue or armpit.\nWHAT IT MEANS: Normal is 98.6F. Below 95F is hypothermia. Over 100.4F is a fever.",
    'alt': "HOW TO CHECK: Check your GPS or post data for current elevation.\nWHAT IT MEANS: Higher altitude equals less oxygen, increasing risk for AMS, HAPE, and HACE.",
    
    'cp_yn': "HOW TO CHECK: Ask 'Are you feeling any pain, tightness, heaviness, or burning in your chest?'",
    'cp_site': "HOW TO CHECK: Click all the areas on the diagram where the patient feels pain.",
    'cp_onset': "HOW TO CHECK: Ask 'Exactly how long ago did this pain start?'",
    'cp_char': "HOW TO CHECK: Ask 'What does it feel like?' Heavy/Crushing is a classic heart attack sign.",
    'cp_rad': "HOW TO CHECK: Ask 'Does the pain travel from the chest into the neck, jaw, or left arm?'",
    'cp_timing': "HOW TO CHECK: Ask 'Is the pain there all the time, or does it come and go?'",
    'cp_exac': "HOW TO CHECK: Ask 'Does anything make the pain worse, like taking a deep breath or walking?'",
    'cp_relieve': "HOW TO CHECK: Ask 'Does anything make the pain better, like sitting down or resting?'",
    'cp_severity': "HOW TO CHECK: Ask 'On a scale from 0 to 10, where 10 is the worst pain of your life, what number is it?'",
    'cp_assoc': "HOW TO CHECK: Look at the patient and ask if they feel these specific things along with the chest pain.",
    
    'comorb': "HOW TO CHECK: Ask 'Do you have high blood pressure, diabetes, or cholesterol problems?'",
    'fam_hx': "HOW TO CHECK: Ask 'Did your parents or siblings ever have a heart attack or stroke before age 60?'",
    'hx_alcohol': "HOW TO CHECK: Ask 'Do you drink alcohol regularly?'",
    'hx_smoking': "HOW TO CHECK: Ask 'Do you smoke cigarettes or bidis?'",
    'ecg_opt': "HOW TO CHECK: Do you have a portable ECG machine and a printed strip available?",
    'ecg_val': "WHAT IT MEANS: ST Elevation means an active, massive heart attack. ST Depression means the heart is starving for oxygen.",
    'hb_opt': "HOW TO CHECK: Do you have a rapid Hemoglobin (Hb) blood testing kit available?",
    'hb_val': "WHAT IT MEANS: Normal is 13-18. Over 19.5 means the blood is thick like sludge and can cause a stroke.",
    'trop_val': "HOW TO CHECK: Prick the finger and drop blood onto the Troponin kit.\nWHAT IT MEANS: A POSITIVE result means heart muscle is actively dying. Immediate Evac.",

    'hd_pe_tenderness': "HOW TO CHECK: Gently press on their chest and stomach. Ask 'Does it hurt when I press here?'",
    'hd_pe_bowel': "HOW TO CHECK: Put your stethoscope on their stomach. Listen for gurgling. If you hear absolutely nothing for 2 mins, or very loud rushing sounds, select Yes.",
    'hd_pe_pulsations': "HOW TO CHECK: Feel for the pulse on the top of their foot or their wrist. Is it completely missing or very weak? Select Absent (-).",
    'hd_pe_discolor': "HOW TO CHECK: Look at their bare chest. Do you see any large bruises, unusual redness, or dark purple patches?",
    'hd_pe_distension': "HOW TO CHECK: Look at their bare stomach. Is it unusually swollen, tight, or blown up like a balloon?",

    'bs_balance': "HOW TO CHECK: Ask 'Did you suddenly feel dizzy or lose your balance?'",
    'bs_eyes': "HOW TO CHECK: Hold up 2 fingers on their left, then right. Ask 'Can you see my fingers clearly?'",
    'bs_face': "HOW TO CHECK: Tell them 'Smile big and show me your teeth.' Look to see if one side of the mouth is drooping.",
    'bs_arms': "HOW TO CHECK: Ask them to close their eyes and hold both arms straight out for 10 seconds. Does one arm drop?",
    'bs_speech': "HOW TO CHECK: Ask them to repeat 'The sky is blue in Jammu today.' Do they sound drunk or slurred?",
    'bs_time': "HOW TO CHECK: Ask everyone nearby: 'What was the exact time you last saw this person acting 100% normal?'",
    
    'bshc_vertigo': "HOW TO CHECK: Ask 'Does it feel like the room is spinning around you?'",
    'bshc_nystagmus': "HOW TO CHECK: Hold your finger 1 foot from their face. Move it left and right. Do their eyes violently jerk or bounce?",
    'bshc_tremor': "HOW TO CHECK: Ask them to reach out and touch your finger. Does their hand shake wildly as it gets close?",
    'bshc_slur': "HOW TO CHECK: Listen to them talk. Are they struggling to form words clearly?",
    'bshc_hypotonia': "HOW TO CHECK: Lift their arm and let it go. Does it drop completely dead and floppy, like a ragdoll?",
    'bshc_gait': "HOW TO CHECK: Ask them to walk 10 steps normally. Are they stumbling, dragging a foot, or unable to walk straight?",
    'bshc_dysdia': "HOW TO CHECK: Ask them to put their hand palm-up on their thigh, then quickly flip it palm-down. Can they do it fast?",
    'bshc_ftn': "HOW TO CHECK: Ask them to touch their nose, then touch your finger. Are they missing your finger?",
    'bshc_hts': "HOW TO CHECK: Ask them to slide the heel of one foot down the shin bone of their other leg. Select 'No' if their heel keeps falling off the shin.",
    'bshc_rebound': "HOW TO CHECK: Have them pull their fist towards their face while you hold their arm back. Suddenly let go. Select 'No' if they cannot stop their arm and end up hitting themselves.",
    'bshc_romberg': "HOW TO CHECK: Have them stand feet together, arms crossed, eyes closed for 20s. Do they instantly lose balance?",
    'bshc_headache': "HOW TO CHECK: Ask 'Rate your head pain from 0 to 10.' 10 is unbearable pressure.",
    'bshc_mental': "HOW TO CHECK: Are they confused about where they are, acting aggressive, or talking nonsense?",
    'bshc_vomit': "HOW TO CHECK: Have they thrown up violently?",
    'bshc_nausea': "HOW TO CHECK: Do they feel sick to their stomach?",
    'bshc_dizzy': "HOW TO CHECK: Do they feel lightheaded when sitting up?",
    'bshc_sensation': "HOW TO CHECK: Touch their arms and legs. Ask if any area feels totally numb or dead.",
    'bshc_pupils': "HOW TO CHECK: Shine a flashlight into their eyes. Do the black circles shrink quickly?",
    'bshc_dtr': "HOW TO CHECK: Tap the tendon just below the kneecap with a tool. Does the leg kick normally?",

    'll_headache': "HOW TO CHECK: Do they have a headache? Rate it: None, Mild, Moderate, Severe.",
    'll_gi': "HOW TO CHECK: Ask 'Are you hungry? Do you feel sick to your stomach?'",
    'll_fatigue': "HOW TO CHECK: Are they completely exhausted from doing simple things like walking 10 steps?",
    'll_dizzy': "HOW TO CHECK: Have them lay down, then sit up. Do they feel lightheaded?",
    'll_sleep': "HOW TO CHECK: Ask 'Did you wake up a lot last night feeling like you couldn't breathe?'",
    'ams_urine': "HOW TO CHECK: Ask the color of their urine. Dark yellow means dehydration, which perfectly mimics AMS symptoms.",
    
    'hape_dyspnea': "HOW TO CHECK: Are they panting heavily? Being out of breath while sitting totally still is a massive red flag.",
    'hape_resp_qual': "HOW TO CHECK: Put your ear near their mouth. Do you hear high-pitched whistling (wheezing)?",
    'hape_activity': "HOW TO CHECK: Ask them to put on a jacket. Are they too exhausted to do it?",
    'hape_mobility': "HOW TO CHECK: Ask them to stand and walk to you. Do their legs buckle from weakness?",
    'hape_mental': "HOW TO CHECK: Are they confused? Stupor means they only wake up when you pinch them hard.",
    'hape_cough': "HOW TO CHECK: Listen to them cough. Are they spitting up pink, frothy, or bloody spit? (Critical if yes).",
    'hape_cyanosis': "HOW TO CHECK: Look at their lips and fingernails. Are they turning blue, purple, or dark gray?",
    'hape_rales': "HOW TO CHECK: Put your ear to their bare back. If you hear a crackling sound like bubbling water, their lungs are filled with fluid.",
    
    'ci_mental_alt': "HOW TO CHECK: Look for the Umbles: Mumbles, Grumbles, Stumbles, Fumbles. Shows brain is freezing.",
    'ci_breathing': "HOW TO CHECK: Are they gasping or taking very shallow, slow breaths?",
    'ci_shiver': "HOW TO CHECK: If a freezing patient SUDDENLY STOPS shivering, their body has exhausted its energy. Critical emergency.",
    'ci_site': "HOW TO CHECK: Look at the patient and type the exact body part (e.g., Left Index Finger, Right Ear).",
    'ci_skin_color': "HOW TO CHECK: Look at the frozen skin. Red = Frostnip. White = Frostbite. Black = Dead tissue.",
    'ci_tenderness': "HOW TO CHECK: Does it hurt intensely when you press gently on the frozen part?",
    'ci_sensation': "HOW TO CHECK: Can they feel a light touch on the frozen part, or is it completely numb and dead?",
    'ci_cap_refill': "HOW TO CHECK: Press their fingernail until it turns white. Does it take longer than 2 seconds to turn pink again?",
    'ci_frostbite_stage': "HOW TO CHECK: Look and feel the skin gently. 1st deg is red. 2nd deg is stiff. 3rd deg is hard like wood. 4th is black/rubbery.",
    'ci_blister': "HOW TO CHECK: Clear blisters = surface damage. Blood-filled blisters = deep tissue death (high amputation risk)."
}

def show_doctrine_table(module):
    with st.expander("📊 Clinical Parameters & Doctrine Limits (Click to Expand)", expanded=False):
        if module == "Heart Disease":
            st.markdown("""
            | Parameter | Normal Range | High Altitude / Warning Limits | Clinical Signs if Abnormal | Verified Reference |
            | :--- | :--- | :--- | :--- | :--- |
            | **Systolic BP** | 90 - 120 mmHg | > 140 mmHg (Warning) / > 180 mmHg (Critical) | Hypertension, throbbing headache, risk of stroke | Whelton PK et al. Hypertension 2018 |
            | **Diastolic BP** | 60 - 80 mmHg | > 90 mmHg (Warning) / > 110 mmHg (Critical) | Hypertension, organ strain | Whelton PK et al. Hypertension 2018 |
            | **Pulse Rate** | 60 - 100 BPM | < 50 or > 120 BPM at rest | Tachycardia/Bradycardia, palpitations, shock | Page RL et al. SVT Guidelines 2015 |
            | **SpO2 Level** | 95 - 100% | < 90% (Warning) / < 85% (Critical) | Cyanosis, confusion, severe hypoxia | High Alt Med Biol 2011 |
            | **Troponin T/I** | Negative | POSITIVE (Any trace is Critical) | Active death of heart muscle (Myocardial Infarction) | Amsterdam EA et al. JACC 2014 |
            | **Hemoglobin** | 13 - 18 g/dL | > 18.0 (Warning) / > 19.5 (Critical) | Blood sludging, extremely high thrombosis/stroke risk | AHA/ASA Guidelines |
            """)
        elif module == "Brain Stroke / HACE":
            st.markdown("""
            | Parameter | Normal Range | High Altitude / Warning Limits | Clinical Signs if Abnormal | Verified Reference |
            | :--- | :--- | :--- | :--- | :--- |
            | **BEFAST Test** | Negative | Any 1 Positive Sign = Critical | Facial droop, arm weakness, slurred speech | Stroke 2019 Guidelines AHA/ASA |
            | **Romberg / Ataxia**| Steady, Negative | Positive (Falling) / Unsteady Gait | Loss of cerebellar control, severe HACE | WMS Guidelines 2019 (PubMed: 31248818) |
            | **Mental Status** | Alert, Oriented | Altered / Stupor = Critical | Confusion, hallucinations, increasing intracranial pressure | Bärtsch P et al. NEJM 2013 |
            | **Pupil Reflex** | Reactive | Sluggish / Unreactive = Critical | Brain stem compression, severe hemorrhage | Stroke 2019 Guidelines AHA/ASA |
            """)
        elif module == "AMS":
            st.markdown("""
            | Parameter | Normal Range | High Altitude / Warning Limits | Clinical Signs if Abnormal | Verified Reference |
            | :--- | :--- | :--- | :--- | :--- |
            | **Lake Louise Score**| 0 | 1-4 (Mild), 5-10 (Moderate), >10 (Severe) | Headache, GI distress, severe fatigue, dizziness | Roach RC et al. High Alt Med Biol 2018 |
            | **Urine Color** | Clear / Pale | Dark Yellow / Brown | Severe dehydration mimicking or exacerbating AMS | WMS Guidelines 2019 |
            """)
        elif module == "HAPE":
            st.markdown("""
            | Parameter | Normal Range | High Altitude / Warning Limits | Clinical Signs if Abnormal | Verified Reference |
            | :--- | :--- | :--- | :--- | :--- |
            | **Resting Dyspnea** | None | Present at Rest = Critical | Lung filling with fluid, failure to oxygenate | Bärtsch P et al. NEJM 2013 |
            | **Resp Rate** | 12 - 20 /min | > 30 /min (Resting) = Serious/Critical | Gasping, heavy accessory muscle use | WMS Guidelines 2019 (PubMed: 31248818) |
            | **Lung Sounds** | Clear | Bubbling Rales / Wheezing = Critical | Fluid accumulating in alveoli | WMS Guidelines 2019 |
            | **Sputum / Cough** | Dry / None | Pink, frothy, or bloody sputum = Critical| Capillary leakage into lungs, severe HAPE | High Alt Med Biol 2011 |
            """)
        elif module == "Cold Injuries":
            st.markdown("""
            | Parameter | Normal Range | High Altitude / Warning Limits | Clinical Signs if Abnormal | Verified Reference |
            | :--- | :--- | :--- | :--- | :--- |
            | **Core Temp** | ~98.6°F | < 95.0°F (Mild), < 82.4°F (Severe) | Shivering stops (critical), Umbles, unconsciousness | WMS Hypothermia Guidelines 2019 |
            | **Frostbite Grade** | Normal | 1st/2nd (Superficial), 3rd/4th (Deep) | Clear vs bloody blisters, waxy vs woody/hard skin | WMS Frostbite Guidelines 2019 |
            | **Capillary Refill**| < 2 seconds | > 2 seconds (Warning) | Poor localized vascular perfusion | WMS Frostbite Guidelines 2019 |
            """)

def check_temp_rule(temp_val, abnormal_list):
    if temp_val < 95.0: abnormal_list.append({"name": f"Hypothermia ({temp_val}°F)", "act": "Patient temp is low. Warm the patient, prevent heat loss."})
    elif temp_val > 100.4: abnormal_list.append({"name": f"Hyperthermia/Fever ({temp_val}°F)", "act": "Patient temp is high. Monitor for infection or heat injury."})

def render_triage_results(module_name, critical_flags, abnormal_flags, mild_flags=None, final_order_override=None):
    if mild_flags is None: mild_flags = []
    
    if len(critical_flags) > 0:
        status_tier = f"ZONE RED: CRITICAL ({module_name.upper()})"
        final_order = final_order_override if final_order_override else "INITIATE IMMEDIATE EMERGENCY EVAC. Keep patient stable and monitor continuously."
        st.markdown(f"<h2 style='color: #EF4444; border: 2px solid #EF4444; padding: 15px; text-align: center; border-radius: 5px; background: #450a0a;'>🔴 {status_tier}</h2>", unsafe_allow_html=True)
    elif len(abnormal_flags) > 0:
        status_tier = f"ZONE AMBER: MODERATE/ABNORMAL ({module_name.upper()})"
        final_order = final_order_override if final_order_override else "Subject is stable but requires close monitoring. Withhold from heavy physical exertion."
        st.markdown(f"<h2 style='color: #F59E0B; border: 2px solid #F59E0B; padding: 15px; text-align: center; border-radius: 5px; background: #451a03;'>🟠 {status_tier}</h2>", unsafe_allow_html=True)
    elif len(mild_flags) > 0:
        status_tier = f"ZONE YELLOW: MILD ({module_name.upper()})"
        final_order = final_order_override if final_order_override else "Halt ascent. Observe carefully and treat symptoms."
        st.markdown(f"<h2 style='color: #FBBF24; border: 2px solid #FBBF24; padding: 15px; text-align: center; border-radius: 5px; background: #422006;'>🟡 {status_tier}</h2>", unsafe_allow_html=True)
    else:
        status_tier = f"ZONE GREEN: NORMAL ({module_name.upper()})"
        final_order = final_order_override if final_order_override else "Continue standard acclimatization and monitoring protocols. No immediate medical intervention required."
        st.markdown(f"<h2 style='color: #10B981; border: 2px solid #10B981; padding: 15px; text-align: center; border-radius: 5px; background: #064e3b;'>🟢 {status_tier}</h2>", unsafe_allow_html=True)

    if critical_flags:
        st.write("### 🚨 CRITICAL PARAMETERS DETECTED")
        for flag in critical_flags:
            st.markdown(f"**<span style='color:#EF4444'>{flag['name']}</span>**", unsafe_allow_html=True)
            st.caption(f"👉 **Action:** {flag['act']}")
            
    if abnormal_flags:
        st.write("### ⚠️ MODERATE / ABNORMAL FLAGS")
        for flag in abnormal_flags:
            st.markdown(f"**<span style='color:#F59E0B'>{flag['name']}</span>**", unsafe_allow_html=True)
            st.caption(f"👉 **Action:** {flag['act']}")
            
    if mild_flags:
        st.write("### ⚠️ MILD / CAUTION FLAGS")
        for flag in mild_flags:
            st.markdown(f"**<span style='color:#FBBF24'>{flag['name']}</span>**", unsafe_allow_html=True)
            st.caption(f"👉 **Action:** {flag['act']}")

    if len(critical_flags) > 0: st.error(f"**FINAL ORDER:** {final_order}")
    elif len(abnormal_flags) > 0: st.warning(f"**FINAL ORDER:** {final_order}")
    elif len(mild_flags) > 0: st.warning(f"**FINAL ORDER:** {final_order}")
    else: st.success(f"**FINAL ORDER:** {final_order}")
        
    pdf_flags = [{"name": f["name"], "act": f["act"], "level": "CRITICAL"} for f in critical_flags] + \
                [{"name": f["name"], "act": f["act"], "level": "ABNORMAL"} for f in abnormal_flags] + \
                [{"name": f["name"], "act": f["act"], "level": "MILD"} for f in mild_flags]
                
    return status_tier, pdf_flags, final_order

def generate_ams_graph(army_no, current_score):
    res = supabase.table("patient_history").select("timestamp, flags, army_no").eq("module", "AMS").execute()
    df = pd.DataFrame(res.data)
    
    if df.empty: return None

    df['army_no'] = df['army_no'].apply(decrypt_data)
    df = df[df['army_no'] == army_no].sort_values(by='timestamp')

    dates = []
    scores = []
    for _, row in df.iterrows():
        flags = row['flags']
        score = 0
        if "Lake Louise Score:" in flags:
            try: score = int([x for x in flags.split(',') if "Lake Louise Score:" in x][0].split(":")[1].strip())
            except Exception: score = 0
        dates.append(row['timestamp'])
        scores.append(score)

    current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    dates.append(current_timestamp)
    scores.append(current_score)

    if len(dates) <= 1: return None 

    plt.figure(figsize=(8, 4))
    plt.plot(dates, scores, marker='o', color='#EF4444', linestyle='-', linewidth=2)
    plt.title(f"Lake Louise Score (AMS) History for {army_no}")
    plt.xlabel("Date/Time")
    plt.ylabel("Score (0-15)")
    plt.ylim(0, 15)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    tmpfile = tempfile.NamedTemporaryFile(dir=TEMP_IMG_DIR, delete=False, suffix='.png')
    plt.savefig(tmpfile.name)
    plt.close()
    return tmpfile.name

def create_pdf_report(module_name, status_tier, flags_list, final_order, temp_val=None, alt_val=None, army_no=None, current_score=None, has_audio=False, patient_info=None):
    if not fpdf_available: return None
        
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 16)
            self.cell(0, 10, 'AROGYAM - MEDICAL TRIAGE REPORT', 0, 1, 'C')
            self.set_font('Arial', 'I', 10)
            self.cell(0, 5, 'High-Altitude Medical Decision Support System', 0, 1, 'C')
            self.ln(5)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
            
    pdf = PDF()
    pdf.add_page()
    
    def add_param_row(p1, p2=""):
        pdf.cell(95, 6, str(p1), border=0)
        pdf.cell(95, 6, str(p2), border=0, ln=1)

    if patient_info:
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(220, 230, 245)
        pdf.cell(0, 8, ' PATIENT REGISTRATION DOSSIER', 0, 1, 'L', fill=True)
        pdf.set_font('Arial', '', 10)
        
        try:
            dob = datetime.strptime(patient_info['dob'], '%Y-%m-%d').date()
            age = (datetime.now().date() - dob).days // 365
        except: age = "Unknown"
        
        try:
            ind_date = datetime.strptime(patient_info['induction_date'], '%Y-%m-%d').date()
            haa_days = max(0, (datetime.now().date() - ind_date).days)
        except: haa_days = "Unknown"
        
        add_param_row(f"Rank & Name: {patient_info['rank']} {patient_info['name']}", f"Army No: {patient_info['army_no']}")
        add_param_row(f"Unit/Coy: {patient_info['company']}", f"Age: {age} yrs | Blood Group: {patient_info['blood_group']}")
        add_param_row(f"Days in HAA: {haa_days} days", f"SHAPE Category: {patient_info['shape_category']}")
        add_param_row(f"Stage 1 Acclim: {patient_info['acclimatization_1']}", f"Stage 2 Acclim: {patient_info['acclimatization_2']}")
        add_param_row(f"Height/Weight: {patient_info['height']}cm / {patient_info['weight']}kg", f"Leaves This Year: {patient_info['leaves_this_year']}")
        add_param_row(f"Past Surgery/Admissions: {patient_info['surgery_history']}", f"AME/PME Done: {patient_info['ame_pme_done']} ({patient_info['ame_pme_date']})")
        add_param_row(f"NOK Name: {patient_info['nok_name']}", f"NOK Phone: {patient_info['nok_phone']}")
        pdf.ln(5)
    else:
        pdf.set_font('Arial', 'B', 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, ' PATIENT DETAILS', 0, 1, 'L', fill=True)
        pdf.set_font('Arial', '', 11)
        rank_name = f"{st.session_state.get('p_rank', '')} {st.session_state.get('p_name', '')}".strip()
        if not rank_name: rank_name = "Not Provided"
        pdf.cell(0, 6, f"Rank & Name: {rank_name}", 0, 1)
        pdf.cell(0, 6, f"Service / Army No: {st.session_state.get('p_num', 'Not Provided')}", 0, 1)
        pdf.cell(0, 6, f"Location / Post: {st.session_state.get('p_loc', 'Not Provided')}", 0, 1)
        pdf.ln(5)
        
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 6, f"Date & Time of Exam: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
    pdf.ln(2)

    pdf.set_font('Arial', 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, f' ASSESSMENT MODULE: {module_name.upper()}', 0, 1, 'L', fill=True)
    
    # NEW PDF FORMAT: Highlight critical status in Red/Bold/Big
    if "RED" in status_tier.upper() or "CRITICAL" in status_tier.upper():
        pdf.set_text_color(220, 0, 0)
        pdf.set_font('Arial', 'B', 14)
    elif "AMBER" in status_tier.upper() or "MODERATE" in status_tier.upper():
        pdf.set_text_color(220, 100, 0)
        pdf.set_font('Arial', 'B', 12)
    else:
        pdf.set_text_color(0, 150, 0)
        pdf.set_font('Arial', 'B', 12)
        
    pdf.cell(0, 8, f"Triage Status: {status_tier}", 0, 1)
    pdf.set_text_color(0, 0, 0) # Reset color
    
    if has_audio:
        pdf.set_text_color(0, 102, 204)
        pdf.cell(0, 8, "[AUDIO] VOICE HANDOVER ATTACHED IN DIGITAL LEDGER", 0, 1)
        pdf.set_text_color(0, 0, 0)
        
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 6, f"Final Order: {final_order}")
    pdf.ln(5)
    
    if flags_list:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 6, "Detected Abnormalities / Clinical Flags:", 0, 1)
        pdf.set_font('Arial', '', 10)
        for flag in flags_list:
            if flag['level'] == 'CRITICAL': pdf.set_text_color(220, 0, 0)
            elif flag['level'] == 'ABNORMAL': pdf.set_text_color(220, 100, 0)
            elif flag['level'] == 'MILD': pdf.set_text_color(200, 150, 0)
            pdf.multi_cell(0, 6, f"- {flag['name']}: {flag['act']}")
        pdf.set_text_color(0, 0, 0) 
            
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 8, ' CLINICAL PARAMETERS & INPUTS', 0, 1, 'L', fill=True)
    pdf.set_font('Arial', '', 10)
    
    def add_param_row(p1, p2=""):
        pdf.cell(95, 6, str(p1), border=0)
        pdf.cell(95, 6, str(p2), border=0, ln=1)

    add_param_row(f"Temperature: {temp_val} F", f"Altitude: {alt_val} ft")

    if module_name == "Heart Disease":
        add_param_row(f"Age: {st.session_state.get('age')}", f"Sex: {st.session_state.get('sex')}")
        add_param_row(f"Blood Pressure: {st.session_state.get('s_bp')}/{st.session_state.get('d_bp')} mmHg", f"Pulse Rate: {st.session_state.get('pulse')} BPM")
        add_param_row(f"Resp Rate: {st.session_state.get('resp')} /min", f"SpO2 Level: {st.session_state.get('spo2')}%")
        
        if st.session_state.get('cp_yn') == "Yes":
            add_param_row("--- SOCRATES CHEST PAIN ASSESSMENT ---", "")
            add_param_row(f"Site: {', '.join(st.session_state.get('cp_site', []))}", f"Onset: {st.session_state.get('cp_onset')}")
            add_param_row(f"Character: {st.session_state.get('cp_char')}", f"Radiation: {st.session_state.get('cp_rad')} ({st.session_state.get('cp_rad_text')})")
            add_param_row(f"Timing: {st.session_state.get('cp_timing')}", f"Severity (1-10): {st.session_state.get('cp_severity')}")
            add_param_row(f"Exac: {st.session_state.get('cp_exac')} ({st.session_state.get('cp_exac_text')})", f"Relieve: {st.session_state.get('cp_relieve')} ({st.session_state.get('cp_relieve_text')})")
            add_param_row(f"Assoc Sweat: {st.session_state.get('cp_assoc_sweat')}", f"Assoc Nausea: {st.session_state.get('cp_assoc_nau')}")
            add_param_row(f"Assoc Cough: {st.session_state.get('cp_assoc_cough')}", f"Assoc DOE: {st.session_state.get('cp_assoc_doe')}")
            add_param_row(f"Assoc Syncope: {st.session_state.get('cp_assoc_sync')}", f"Assoc Bowel/Blad: {st.session_state.get('cp_assoc_bowel')}")
            add_param_row(f"Assoc Slurring: {st.session_state.get('cp_assoc_slur')}", f"Assoc Focal: {st.session_state.get('cp_assoc_focal')}")
            if st.session_state.get('cp_assoc_other'): add_param_row(f"Other: {st.session_state.get('cp_assoc_other')}", "")
        else:
            add_param_row("Chest Pain: No", "")

        add_param_row("--- HISTORY & DIAGNOSTICS ---", "")
        add_param_row(f"Comorbidity: {st.session_state.get('comorb')}", f"Family Hx: {st.session_state.get('fam_hx_cond')}")
        add_param_row(f"Alcohol Hx: {st.session_state.get('hx_alcohol')}", f"Smoking Hx: {st.session_state.get('hx_smoking')}")
        add_param_row(f"Hemoglobin: {st.session_state.get('hb_val')} g/dL", f"Troponin T/I: {st.session_state.get('trop_val')}")

        add_param_row("--- PHYSICAL EXAM ---", "")
        add_param_row(f"Chest/Abd Tenderness: {st.session_state.get('hd_pe_tenderness')}", f"Abnormal Bowel: {st.session_state.get('hd_pe_bowel')}")
        add_param_row(f"Peripheral Pulsations: {st.session_state.get('hd_pe_pulsations')}", f"Chest Discoloration: {st.session_state.get('hd_pe_discolor')}")
        add_param_row(f"Abdomen Distension: {st.session_state.get('hd_pe_distension')}", "")
        
        if st.session_state.get('trop_val') == "Positive" and st.session_state.get('trop_img_path'):
            if os.path.exists(st.session_state.get('trop_img_path')):
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 6, "Positive Troponin Kit Image:", 0, 1)
                try: pdf.image(st.session_state.get('trop_img_path'), x=10, w=100)
                except: pdf.cell(0, 6, "(Image render failed)", 0, 1)

        if st.session_state.get('ecg_opt'):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, ' ECG STRIP & INTERPRETATION', 0, 1, 'L', fill=True)
            pdf.ln(5)
            img_path = st.session_state.get('ecg_img_path')
            if img_path and os.path.exists(img_path):
                try: 
                    pdf.image(img_path, x=10, w=190)
                    pdf.ln(130) 
                except Exception as e:
                    pdf.cell(0, 6, "(ECG Image could not be rendered in PDF)", 0, 1)
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 6, f"AI Interpretation: {st.session_state.get('ecg_ai_result', 'N/A')}", 0, 1)
            pdf.cell(0, 6, f"User Interpretation: {st.session_state.get('ecg_val', 'N/A')}", 0, 1)

    elif module_name == "BRAIN STROKE / HACE":
        add_param_row(f"Blood Pressure: {st.session_state.get('bshc_s_bp')}/{st.session_state.get('bshc_d_bp')} mmHg", f"Pulse Rate: {st.session_state.get('bshc_pulse')} BPM")
        add_param_row(f"Resp Rate: {st.session_state.get('bshc_resp')} /min", f"SpO2 Level: {st.session_state.get('bshc_spo2')}%")
        add_param_row("--- BEFAST EXAM ---", "")
        add_param_row(f"Balance Loss: {st.session_state.get('bshc_balance')}", f"Vision Loss: {st.session_state.get('bshc_eyes')}")
        add_param_row(f"Facial Droop: {st.session_state.get('bshc_face')}", f"Arm Weakness: {st.session_state.get('bshc_arms')}")
        add_param_row(f"Speech Difficulty: {st.session_state.get('bshc_speech')}", f"Time Since Normal: {st.session_state.get('bshc_time')}")
        add_param_row("--- PHYSICAL EXAM ---", "")
        add_param_row(f"Vertigo: {st.session_state.get('bshc_vertigo')}", f"Nystagmus: {st.session_state.get('bshc_nystagmus')}")
        add_param_row(f"Tremor: {st.session_state.get('bshc_tremor')}", f"Slurring: {st.session_state.get('bshc_slur')}")
        add_param_row(f"Hypotonia: {st.session_state.get('bshc_hypotonia')}", f"Gait Abnormality: {st.session_state.get('bshc_gait')}")
        add_param_row(f"Dysdiadochokinesia: {st.session_state.get('bshc_dysdia')}", f"Finger to Nose: {st.session_state.get('bshc_ftn')}")
        add_param_row(f"Heel to Shin: {st.session_state.get('bshc_hts')}", f"Rebound Phenomenon: {st.session_state.get('bshc_rebound')}")
        add_param_row(f"Romberg Sign: {st.session_state.get('bshc_romberg')}", "")
        add_param_row("--- PATIENT COMPLAINTS ---", "")
        add_param_row(f"Headache Severity: {st.session_state.get('bshc_headache')}/10", f"Altered Mental: {st.session_state.get('bshc_mental')}")
        add_param_row(f"Vomiting: {st.session_state.get('bshc_vomit')}", f"Nausea: {st.session_state.get('bshc_nausea')}")
        add_param_row(f"Dizziness: {st.session_state.get('bshc_dizzy')}", f"Loss of Sensation: {st.session_state.get('bshc_sensation')}")
        add_param_row(f"Pupils Reactive: {st.session_state.get('bshc_pupils')}", f"Deep Tendon Reflex: {st.session_state.get('bshc_dtr')}")

    elif module_name == "AMS":
        add_param_row(f"Blood Pressure: {st.session_state.get('ams_s_bp')}/{st.session_state.get('ams_d_bp')} mmHg", f"Pulse Rate: {st.session_state.get('ams_pulse')} BPM")
        add_param_row(f"Resp Rate: {st.session_state.get('ams_resp')} /min", f"SpO2 Level: {st.session_state.get('ams_spo2')}%")
        add_param_row(f"Urine Color: {st.session_state.get('ams_urine')}", "")
        add_param_row(f"Headache Score: {st.session_state.get('ll_headache')} / 3", f"GI Symptoms Score: {st.session_state.get('ll_gi')} / 3")
        add_param_row(f"Fatigue Score: {st.session_state.get('ll_fatigue')} / 3", f"Dizziness Score: {st.session_state.get('ll_dizzy')} / 3")
        add_param_row(f"Sleep Difficulty: {st.session_state.get('ll_sleep')} / 3", f"TOTAL LL SCORE: {current_score}")

    elif module_name == "HAPE":
        add_param_row(f"Blood Pressure: {st.session_state.get('hape_s_bp')}/{st.session_state.get('hape_d_bp')} mmHg", f"Pulse Rate: {st.session_state.get('hape_pulse')} BPM")
        add_param_row(f"Resp Rate: {st.session_state.get('hape_resp')} /min", f"SpO2 Level: {st.session_state.get('hape_spo2')}%")
        add_param_row(f"Dyspnoea: {st.session_state.get('hape_dyspnea')}", "")
        add_param_row(f"Respiration Quality: {st.session_state.get('hape_resp_qual')}", "")
        add_param_row(f"Activity Level: {st.session_state.get('hape_activity')}", "")
        add_param_row(f"Mobility: {st.session_state.get('hape_mobility')}", "")
        add_param_row(f"Mental Status: {st.session_state.get('hape_mental')}", f"Cyanosis: {st.session_state.get('hape_cyanosis')}")
        add_param_row(f"Cough Status: {st.session_state.get('hape_cough')}", f"Rales: {st.session_state.get('hape_rales')}")
        add_param_row(f"Nausea at Rest: {st.session_state.get('hape_nausea')}", f"Headache: {st.session_state.get('hape_headache')}")

    elif module_name == "COLD INJURY":
        add_param_row(f"Blood Pressure: {st.session_state.get('ci_s_bp')}/{st.session_state.get('ci_d_bp')} mmHg", f"Pulse Rate: {st.session_state.get('ci_pulse')} BPM")
        add_param_row(f"Resp Rate: {st.session_state.get('ci_resp')} /min", f"SpO2 Level: {st.session_state.get('ci_spo2')}%")
        add_param_row("--- HYPOTHERMIA EVAL ---", "")
        add_param_row(f"Shivering: {st.session_state.get('ci_shiver')}", f"Altered Mental: {st.session_state.get('ci_mental_alt')}")
        add_param_row(f"Diff Breathing: {st.session_state.get('ci_breathing')}", "")
        add_param_row("--- FROSTBITE / CHILBLAINS EVAL ---", "")
        add_param_row(f"Site of Injury: {st.session_state.get('ci_site')}", f"Skin Color: {st.session_state.get('ci_skin_color')}")
        add_param_row(f"Severe Tenderness: {st.session_state.get('ci_tenderness')}", f"Loss of Sensation: {st.session_state.get('ci_sensation')}")
        add_param_row(f"Delayed Cap Refill: {st.session_state.get('ci_cap_refill')}", f"Frostbite Stage: {st.session_state.get('ci_frostbite_stage')}")
        add_param_row(f"Blister Type: {st.session_state.get('ci_blister_type')}", "")
        
        if st.session_state.get('ci_img_path') and os.path.exists(st.session_state.get('ci_img_path')):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, ' FROSTBITE INJURY IMAGE', 0, 1, 'L', fill=True)
            pdf.ln(5)
            try: 
                pdf.image(st.session_state.get('ci_img_path'), x=10, w=150)
            except Exception:
                pdf.cell(0, 6, "(Image could not be rendered in PDF)", 0, 1)

    if module_name == "AMS" and army_no:
        graph_path = generate_ams_graph(army_no, current_score)
        if graph_path:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 8, ' AMS PROGRESSION HISTORY', 0, 1, 'L', fill=True)
            pdf.image(graph_path, x=10, y=pdf.get_y()+5, w=190)
            try: os.remove(graph_path) 
            except Exception: pass
    
    try: return pdf.output(dest='S').encode('latin-1')
    except Exception: return bytes(pdf.output())
    

def save_to_ledger(rank, name, army_no, location, module, status_tier, flags_list, final_order, audio_bytes=None):
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # --- DOUBLE-CLICK PREVENTION ---
    res_dup = supabase.table("patient_history").select("army_no, timestamp, module").eq("module", module).eq("timestamp", timestamp).execute()
    existing_logs = [decrypt_data(r['army_no']).strip().upper() for r in (res_dup.data if res_dup.data else [])]
    if army_no.strip().upper() in existing_logs:
        return  # Silently ignore the duplicate click because the record is already saved
    # -------------------------------
    
    flags_str = ", ".join(flags_list) if flags_list else "None"
    enc_name = encrypt_data(name)
    enc_army_no = encrypt_data(army_no)
    enc_location = encrypt_data(location)
    
    audio_path = "None"
    if audio_bytes is not None:
        audio_filename = f"audio_{int(datetime.now().timestamp())}.wav"
        try:
            audio_data = audio_bytes.read()
            supabase.storage.from_("arogyam-audio").upload(
                path=audio_filename, 
                file=audio_data, 
                file_options={"content-type": "audio/wav"}
            )
            audio_path = supabase.storage.from_("arogyam-audio").get_public_url(audio_filename)
        except Exception as e:
            st.error(f"Audio upload failed: {e}")
            audio_path = "None"
            
    insert_data = {
        "timestamp": timestamp,
        "bfna_id": st.session_state['bfna_id'],
        "post_name": st.session_state['post_name'],
        "rank": rank,
        "name": enc_name,
        "army_no": enc_army_no,
        "location": enc_location,
        "module": module,
        "status_tier": status_tier,
        "flags": flags_str,
        "final_order": final_order,
        "audio_path": audio_path
    }
    
    try:
        supabase.table('patient_history').insert(insert_data).execute()
    except Exception as e:
        st.error(f"Error saving to ledger: {e}")

# ==========================================
# PAGE COMPONENTS
# ==========================================
def login_page():
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.write("<br><br><br>", unsafe_allow_html=True)
        st.markdown("<div class='brand-glow'> <h1 style='text-align: center;'>🛡️AROGYAM</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #38bdf8;'>MEDICAL DECISION SUPPORT SYSTEM</h4>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            user = st.text_input("USER ID (e.g., admin, bfna_alpha)")
            passwd = st.text_input("PASSWORD KEY", type="password")
            if st.form_submit_button("SECURE LOGIN"):
                try:
                    res = supabase.table('users').select("*").eq('user_id', user.strip()).execute()
                    row = res.data[0] if res.data else None
                    
                    if row and row['password'] == passwd:
                        st.session_state['logged_in'] = True
                        st.session_state['bfna_id'] = row['bfna_id']
                        st.session_state['post_name'] = row['post_name']
                        st.rerun() 
                    else:
                        st.error("INVALID CREDENTIALS")
                except Exception as e:
                    st.error(f"Login failed. Check internet connection or Supabase settings. Error: {e}")

def main_app():
    
    menu_items = ["Heart Disease", "Brain Stroke / HACE", "AMS", "HAPE", "Cold Injuries", "Weekly Vitals", "Acclimatization", "Patient History", "Patient Registration"]
    if st.session_state['bfna_id'] in ['RMO', 'MASTER_ADMIN']:
        menu_items.append("RMO Dashboard")
        menu_items.append("Admin Settings")
        
    with st.sidebar:
        st.markdown("<div class='brand-glow'>🛡️AROGYAM</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-sub'>BY BARHE CHALO 🩺</div>", unsafe_allow_html=True)
        st.markdown("---")
        st.success(f"👤 **BFNA ID:** {st.session_state['bfna_id']}\n\n📍 **POST:** {st.session_state['post_name']}")
        
        icons_list = ["heart-pulse", "lightning-charge", "triangle-half", "lungs", "thermometer-snow", "clipboard2-pulse", "activity", "journal-medical", "person-badge"]
        if st.session_state['bfna_id'] in ['RMO', 'MASTER_ADMIN']:
            icons_list.extend(["bar-chart-fill", "gear"])
            
        selected = option_menu("Diagnosis Modules", menu_items, icons=icons_list, menu_icon="cast", default_index=0)
        
        st.write("<br><br>", unsafe_allow_html=True)
        if st.button("SAFE LOGOUT", type="secondary"):
            st.session_state['logged_in'] = False
            st.session_state['bfna_id'] = None
            st.session_state['post_name'] = None
            st.rerun()

    st.markdown(f"<div class='demo-badge'>🌐 ACTIVE SENSOR NET: DATA ISOLATED TO {st.session_state['post_name']}</div>", unsafe_allow_html=True)

    # ------------------------------------------
    # 1. HEART DISEASE MODULE
    # ------------------------------------------
    if selected == "Heart Disease":
        st.markdown(f"### PAGE {st.session_state['page_step']} OF 5")
        cols = st.columns(5)
        for i in range(5): cols[i].progress(100 if st.session_state['page_step'] > i else (0 if st.session_state['page_step'] <= i else 50))
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        show_doctrine_table("Heart Disease")

        if st.session_state['page_step'] == 1:
            st.header("Core Vitals & Environment")
            col_a, col_b = st.columns(2)
            with col_a:
                st.session_state['age'] = st.number_input("Age *", 18, 90, st.session_state['age'], help=tt_lib.get('age', ''))
                st.session_state['sex'] = st.radio("Sex *", ["Male", "Female"], index=get_idx(["Male", "Female"], st.session_state['sex']), horizontal=True, help=tt_lib.get('sex', ''))
                bp1, bp2 = st.columns(2)
                with bp1: st.session_state['s_bp'] = st.number_input("Systolic BP *", 60, 240, st.session_state['s_bp'], help=tt_lib.get('bp', ''))
                with bp2: st.session_state['d_bp'] = st.number_input("Diastolic BP *", 40, 150, st.session_state['d_bp'], help=tt_lib.get('bp', ''))
                pr1, rr1 = st.columns(2)
                with pr1: st.session_state['pulse'] = st.number_input("Pulse Rate (BPM) *", 40, 220, st.session_state['pulse'], help=tt_lib.get('pulse', ''))
                with rr1: st.session_state['resp'] = st.number_input("Resp Rate (/min) *", 8, 50, st.session_state['resp'], help=tt_lib.get('resp', ''))
                t1, a1 = st.columns(2)
                with t1: st.session_state['temp'] = st.number_input("Temperature (°F) *", 70.0, 110.0, st.session_state['temp'], help=tt_lib.get('temp', ''))
                with a1: st.session_state['alt'] = st.selectbox("Altitude (ft) *", alt_opts, index=get_idx(alt_opts, st.session_state['alt']), help=tt_lib.get('alt', ''))
            with col_b:
                st.markdown(f"<div class='spo2-wrapper'><div class='spo2-title'>🫧 Blood Oxygen</div><div class='spo2-val'>{st.session_state['spo2']}%</div></div>", unsafe_allow_html=True)
                st.session_state['spo2'] = st.slider("SpO2 Levels (%) *", 40, 100, st.session_state['spo2'], help=tt_lib.get('spo2', ''))

        elif st.session_state['page_step'] == 2:
            st.header("Clinical Factors (SOCRATES)")
            st.session_state['cp_yn'] = st.radio("Chest Pain Present? *", opts_yn, index=get_idx(opts_yn, st.session_state['cp_yn']), horizontal=True, help=tt_lib.get('cp_yn', ''))
            
            if st.session_state['cp_yn'] == "Yes":
                with st.container(border=True):
                    
                    st.markdown("**Visual Chest Pain Map (Select all affected areas):**")
                    st.markdown("<div class='pain-map-container'>", unsafe_allow_html=True)
                    pm1, pm2, pm3 = st.columns(3)
                    with pm1:
                        r_chest = st.checkbox("🫁 Right Chest", value="Right" in st.session_state.get('cp_site', []))
                    with pm2:
                        c_chest = st.checkbox("❤️ Centre Sternum", value="Centre" in st.session_state.get('cp_site', []))
                    with pm3:
                        l_chest = st.checkbox("🫁 Left Chest", value="Left" in st.session_state.get('cp_site', []))
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    cp_site_list = []
                    if r_chest: cp_site_list.append("Right")
                    if c_chest: cp_site_list.append("Centre")
                    if l_chest: cp_site_list.append("Left")
                    if not cp_site_list: cp_site_list.append("None")
                    st.session_state['cp_site'] = cp_site_list
                    
                    st.session_state['cp_onset'] = st.selectbox("Onset *", time_opts, index=get_idx(time_opts, st.session_state.get('cp_onset')), help=tt_lib.get('cp_onset', ''))
                    st.session_state['cp_char'] = st.selectbox("Character of pain *", opts_cp, index=get_idx(opts_cp, st.session_state.get('cp_char')), help=tt_lib.get('cp_char', ''))
                    r1, r2 = st.columns(2)
                    with r1: st.session_state['cp_rad'] = st.radio("Radiation *", opts_yn, horizontal=True, index=get_idx(opts_yn, st.session_state.get('cp_rad')), help=tt_lib.get('cp_rad', ''))
                    with r2: 
                        if st.session_state['cp_rad'] == "Yes": st.session_state['cp_rad_text'] = st.text_input("Radiation Area", st.session_state.get('cp_rad_text', ''))
                    
                    st.markdown("**Associated Symptoms ***")
                    a1, a2, a3, a4 = st.columns(4)
                    with a1:
                        st.session_state['cp_assoc_sweat'] = st.radio("Sweating", opts_yn, index=get_idx(opts_yn, st.session_state.get('cp_assoc_sweat', 'No')), horizontal=True, help=tt_lib.get('cp_assoc', ''))
                        st.session_state['cp_assoc_sync'] = st.radio("Syncope", opts_yn, index=get_idx(opts_yn, st.session_state.get('cp_assoc_sync', 'No')), horizontal=True, help=tt_lib.get('cp_assoc', ''))
                    with a2:
                        st.session_state['cp_assoc_nau'] = st.radio("Nausea", opts_yn, index=get_idx(opts_yn, st.session_state.get('cp_assoc_nau', 'No')), horizontal=True, help=tt_lib.get('cp_assoc', ''))
                        st.session_state['cp_assoc_bowel'] = st.radio("Abnormal Bowel/Bladder", opts_yn, index=get_idx(opts_yn, st.session_state.get('cp_assoc_bowel', 'No')), horizontal=True, help=tt_lib.get('cp_assoc', ''))
                    with a3:
                        st.session_state['cp_assoc_cough'] = st.radio("Coughing", opts_yn, index=get_idx(opts_yn, st.session_state.get('cp_assoc_cough', 'No')), horizontal=True, help=tt_lib.get('cp_assoc', ''))
                        st.session_state['cp_assoc_slur'] = st.radio("Slurring of Speech", opts_yn, index=get_idx(opts_yn, st.session_state.get('cp_assoc_slur', 'No')), horizontal=True, help=tt_lib.get('cp_assoc', ''))
                    with a4:
                        st.session_state['cp_assoc_doe'] = st.radio("Dyspnoea on Exertion", opts_yn, index=get_idx(opts_yn, st.session_state.get('cp_assoc_doe', 'No')), horizontal=True, help=tt_lib.get('cp_assoc', ''))
                        st.session_state['cp_assoc_focal'] = st.radio("Focal Deficits", opts_yn, index=get_idx(opts_yn, st.session_state.get('cp_assoc_focal', 'No')), horizontal=True, help=tt_lib.get('cp_assoc', ''))
                    st.session_state['cp_assoc_other'] = st.text_input("Other Symptoms (Specify)", st.session_state.get('cp_assoc_other', ''))
                    
                    st.session_state['cp_timing'] = st.selectbox("Timing of pain *", ["Constant pain", "Intermittent pain"], index=get_idx(["Constant pain", "Intermittent pain"], st.session_state.get('cp_timing')), help=tt_lib.get('cp_timing', ''))
                    e1, e2 = st.columns(2)
                    with e1: st.session_state['cp_exac'] = st.radio("Exacerbating factors? *", opts_yn, horizontal=True, index=get_idx(opts_yn, st.session_state.get('cp_exac')), help=tt_lib.get('cp_exac', ''))
                    with e2: 
                        if st.session_state['cp_exac'] == "Yes": st.session_state['cp_exac_text'] = st.text_input("Exacerbating Description", st.session_state.get('cp_exac_text', ''))
                    
                    re1, re2 = st.columns(2)
                    with re1: st.session_state['cp_relieve'] = st.radio("Relieving factors? *", opts_yn, horizontal=True, index=get_idx(opts_yn, st.session_state.get('cp_relieve')), help=tt_lib.get('cp_relieve', ''))
                    with re2:
                        if st.session_state['cp_relieve'] == "Yes": st.session_state['cp_relieve_text'] = st.text_input("Relieving Description", st.session_state.get('cp_relieve_text', ''))
                    
                    st.session_state['cp_severity'] = st.slider("Severity of pain (1-10) *", 0, 10, st.session_state.get('cp_severity', 0), help=tt_lib.get('cp_severity', ''))

        elif st.session_state['page_step'] == 3:
            st.header("History & Diagnostics")
            c1, c2 = st.columns(2)
            with c1:
                with st.container(border=True):
                    st.subheader("History")
                    st.session_state['comorb'] = st.selectbox("Primary Comorbidity *", opts_comorb, index=get_idx(opts_comorb, st.session_state['comorb']), help=tt_lib.get('comorb', ''))
                    st.session_state['fam_hx_cond'] = st.selectbox("Family History of Comorbidity *", opts_comorb, index=get_idx(opts_comorb, st.session_state['fam_hx_cond']), help=tt_lib.get('fam_hx', ''))
                    hx1, hx2 = st.columns(2)
                    with hx1: st.session_state['hx_alcohol'] = st.radio("History of Alcohol *", opts_yn, index=get_idx(opts_yn, st.session_state['hx_alcohol']), horizontal=True, help=tt_lib.get('hx_alcohol', ''))
                    with hx2: st.session_state['hx_smoking'] = st.radio("History of Smoking *", opts_yn, index=get_idx(opts_yn, st.session_state['hx_smoking']), horizontal=True, help=tt_lib.get('hx_smoking', ''))
            
            with c2:
                with st.container(border=True):
                    st.subheader("Field Diagnostics")
                    ecg_checked = st.checkbox("Is ECG Available?", value=st.session_state['ecg_opt'], help=tt_lib.get('ecg_opt', ''))
                    st.session_state['ecg_opt'] = ecg_checked
                    if ecg_checked:
                        input_method = st.radio("ECG Input Method", ["Upload File", "Use Camera"], horizontal=True)
                        ecg_file = st.file_uploader("📸 Upload Raw ECG Image", type=['png', 'jpg', 'jpeg']) if input_method == "Upload File" else st.camera_input("📸 Take picture of ECG Strip")
                        if ecg_file is not None:
                            temp_ecg_path = os.path.join(TEMP_IMG_DIR, f"temp_ecg_{int(datetime.now().timestamp())}.jpg")
                            try:
                                with open(temp_ecg_path, "wb") as f: f.write(ecg_file.getvalue())
                                st.session_state['ecg_img_path'] = temp_ecg_path
                            except Exception: pass

                            if st.button("🔍 ANALYZE ECG WITH AI", type="secondary"):
                                with st.spinner("Processing image..."):
                                    try:
                                        from tensorflow.keras.preprocessing.image import img_to_array
                                        img = Image.open(io.BytesIO(ecg_file.getvalue())).convert('RGB').resize((224, 224))
                                        img_array = img_to_array(img)
                                        input_data = np.expand_dims(img_array, axis=0)
                                        cnn_model = load_cnn()
                                        if cnn_model:
                                            raw_preds = cnn_model.predict(input_data)[0]
                                            best_idx = np.argmax(raw_preds)
                                            best_diagnosis = ["Normal", "Abnormal", "Class 3", "Class 4"][best_idx]
                                            st.session_state['ecg_ai_result'] = best_diagnosis
                                            if "normal" in best_diagnosis.lower(): st.success(f"**AI Diagnosis:** {best_diagnosis}")
                                            else: st.error(f"**AI Diagnosis:** {best_diagnosis} (Abnormal)")
                                        else: st.error("Model failed to load.")
                                    except Exception as e: st.error(f"Analysis Failed: {e}")
                        
                        st.info("Select final ECG interpretation:")
                        st.session_state['ecg_val'] = st.selectbox("ECG Interpretation *", opts_ecg, index=get_idx(opts_ecg, st.session_state['ecg_val']), help=tt_lib.get('ecg_val', ''))
                    
                    st.markdown("---") 
                    hb_checked = st.checkbox("Is Hb Test Available?", value=st.session_state['hb_opt'], help=tt_lib.get('hb_opt', ''))
                    st.session_state['hb_opt'] = hb_checked
                    if hb_checked: st.session_state['hb_val'] = st.slider("Hemoglobin (g/dL) *", 5.0, 25.0, float(st.session_state['hb_val']), help=tt_lib.get('hb_val', ''))
                    
                    st.markdown("---")
                    st.session_state['trop_val'] = st.radio("Troponin T/I (Rapid Kit) *", opts_trop, index=get_idx(opts_trop, st.session_state['trop_val']), horizontal=True, help=tt_lib.get('trop_val', ''))                
                    if st.session_state['trop_val'] == "Positive":
                        trop_file = st.file_uploader("Upload Troponin Test Image for PDF", type=['png', 'jpg', 'jpeg'])
                        if trop_file is not None:
                            t_path = os.path.join(TEMP_IMG_DIR, f"temp_trop_{int(datetime.now().timestamp())}.jpg")
                            try:
                                with open(t_path, "wb") as f: f.write(trop_file.getvalue())
                                st.session_state['trop_img_path'] = t_path
                                st.success("Trop T image saved for report.")
                            except Exception: pass

        elif st.session_state['page_step'] == 4:
            st.header("Physical Exam")
            with st.container(border=True):
                st.session_state['hd_pe_tenderness'] = st.radio("Tenderness over chest/abdomen? *", opts_yn, index=get_idx(opts_yn, st.session_state['hd_pe_tenderness']), horizontal=True, help=tt_lib.get('hd_pe_tenderness', ''))
                st.session_state['hd_pe_bowel'] = st.radio("Abnormal Bowel sounds? *", opts_yn, index=get_idx(opts_yn, st.session_state['hd_pe_bowel']), horizontal=True, help=tt_lib.get('hd_pe_bowel', ''))
                st.session_state['hd_pe_pulsations'] = st.radio("Peripheral pulsations? *", ["Present (+)", "Absent (-)"], index=get_idx(["Present (+)", "Absent (-)"], st.session_state['hd_pe_pulsations']), horizontal=True, help=tt_lib.get('hd_pe_pulsations', ''))
                st.session_state['hd_pe_discolor'] = st.radio("Any discolouration over chest area? *", opts_yn, index=get_idx(opts_yn, st.session_state['hd_pe_discolor']), horizontal=True, help=tt_lib.get('hd_pe_discolor', ''))
                st.session_state['hd_pe_distension'] = st.radio("Abdomen distension? *", opts_yn, index=get_idx(opts_yn, st.session_state['hd_pe_distension']), horizontal=True, help=tt_lib.get('hd_pe_distension', ''))
            
        elif st.session_state['page_step'] == 5:
            st.header("Diagnostic Triage Results")
            
            abnormal_flags = []
            critical_flags = []
            
            # Core Vitals Flags
            if st.session_state['s_bp'] > 160 or st.session_state['d_bp'] > 100: abnormal_flags.append({"name": f"High BP ({st.session_state['s_bp']}/{st.session_state['d_bp']})", "act": "Monitor closely. Keep seated."})
            if st.session_state['pulse'] > 120 or st.session_state['pulse'] < 50: abnormal_flags.append({"name": f"Abnormal Pulse ({st.session_state['pulse']} BPM)", "act": "Assess for shock/dehydration."})
            if st.session_state['spo2'] < 85: abnormal_flags.append({"name": f"Hypoxia (SpO2: {st.session_state['spo2']}%)", "act": "Administer O2 via mask. Prepare for descent."})
            if st.session_state['resp'] > 25: abnormal_flags.append({"name": f"Tachypnea (Resp: {st.session_state['resp']}/min)", "act": "Patient struggling to breathe. Sit upright."})
            check_temp_rule(st.session_state['temp'], abnormal_flags)

            # SOCRATES Flags
            if st.session_state['cp_yn'] == "Yes":
                if "Left" in st.session_state.get('cp_site', []) or "Centre" in st.session_state.get('cp_site', []): critical_flags.append({"name": "Central/Left Chest Pain", "act": "High suspicion of cardiac event."})
                elif "Right" in st.session_state.get('cp_site', []): abnormal_flags.append({"name": "Right Chest Pain", "act": "Monitor closely. Rule out pleuritic pain."})
                
                if st.session_state.get('cp_onset') != "Unknown": critical_flags.append({"name": f"Pain Onset: {st.session_state['cp_onset']}", "act": "Evaluate timeline for intervention."})
                
                if st.session_state.get('cp_char') == "Crushing/heavy": critical_flags.append({"name": "Crushing Chest Pain", "act": "Classic Ischemia sign."})
                elif st.session_state.get('cp_char') in ["Stabbing/sharp", "Burning"]: abnormal_flags.append({"name": f"Pain Character: {st.session_state['cp_char']}", "act": "Evaluate for other causes."})
                
                if st.session_state.get('cp_rad') == "Yes": critical_flags.append({"name": "Radiating Pain", "act": "Classic Ischemia. Check jaw/arm."})
                
                assoc_list = [st.session_state.get(k) for k in ['cp_assoc_sweat', 'cp_assoc_nau', 'cp_assoc_cough', 'cp_assoc_doe', 'cp_assoc_sync', 'cp_assoc_bowel', 'cp_assoc_slur', 'cp_assoc_focal']]
                if "Yes" in assoc_list: critical_flags.append({"name": "Associated High-Risk Symptoms Present", "act": "Complex presentation, possible shock or stroke concurrent."})
                
                if st.session_state.get('cp_timing') in ["Constant pain", "Intermittent pain"]: critical_flags.append({"name": f"Timing: {st.session_state['cp_timing']}", "act": "Ongoing distress."})
                if st.session_state.get('cp_exac') == "Yes": critical_flags.append({"name": "Exacerbating Factors Present", "act": "Limit physical movement entirely."})
                if st.session_state.get('cp_relieve') == "No": critical_flags.append({"name": "No Relieving Factors", "act": "Pain unmanageable. Evacuate."})
                elif st.session_state.get('cp_relieve') == "Yes": abnormal_flags.append({"name": "Has Relieving Factors", "act": "Apply relieving factors if safe."})
                
                if st.session_state.get('cp_severity', 1) >= 6: critical_flags.append({"name": f"High Pain Severity ({st.session_state['cp_severity']}/10)", "act": "Administer analgesics per protocol."})
                else: abnormal_flags.append({"name": f"Mild Pain Severity ({st.session_state['cp_severity']}/10)", "act": "Monitor for escalation."})

            # History Flags
            if st.session_state.get('comorb') != "None": critical_flags.append({"name": f"Comorbidity: {st.session_state['comorb']}", "act": "High baseline risk."})
            if st.session_state.get('fam_hx_cond') != "None": abnormal_flags.append({"name": f"Fam Hx: {st.session_state['fam_hx_cond']}", "act": "Elevated genetic risk."})
            if st.session_state.get('hx_alcohol') == "Yes": abnormal_flags.append({"name": "Alcohol History", "act": "Note for medication contraindications."})
            if st.session_state.get('hx_smoking') == "Yes": abnormal_flags.append({"name": "Smoking History", "act": "Vascular risk factor."})

            # Physical Exam Flags
            if st.session_state.get('hd_pe_tenderness') == "Yes": abnormal_flags.append({"name": "Chest/Abdomen Tenderness", "act": "Possible musculoskeletal pain or internal injury. Monitor."})
            if st.session_state.get('hd_pe_bowel') == "Yes": abnormal_flags.append({"name": "Abnormal Bowel Sounds", "act": "Possible GI involvement. Withhold oral feeding."})
            if st.session_state.get('hd_pe_pulsations') == "Absent (-)": abnormal_flags.append({"name": "Absent/Weak Peripheral Pulsations", "act": "Poor blood flow to extremities. Check for shock."})
            if st.session_state.get('hd_pe_discolor') == "Yes": abnormal_flags.append({"name": "Chest Discoloration", "act": "Possible trauma or internal bleeding. Inspect closely."})
            if st.session_state.get('hd_pe_distension') == "Yes": abnormal_flags.append({"name": "Abdomen Distension", "act": "Possible internal bleeding or GI block. Evacuate if severe."})

            # Diagnostics Flags
            if st.session_state.get('hb_opt'):
                hb = st.session_state['hb_val']
                if hb < 11.0: critical_flags.append({"name": f"Severe Anemia (Hb: {hb})", "act": "Critical oxygen transport failure."})
                elif 11.0 <= hb < 13.0: abnormal_flags.append({"name": f"Mild Anemia (Hb: {hb})", "act": "Reduced stamina."})
                elif 18.0 <= hb <= 19.5: abnormal_flags.append({"name": f"Elevated Hb ({hb})", "act": "Blood thickening. Hydrate."})
                elif hb > 19.5: critical_flags.append({"name": f"CRITICAL Hb ({hb})", "act": "Severe risk of stroke/thrombosis."})

            if st.session_state.get('ecg_val') in ["ST Elevation", "ST Depression", "Pathological Q Waves"]: critical_flags.append({"name": f"Severe ECG Finding ({st.session_state['ecg_val']})", "act": "Confirmed cardiac event. CAS EVAC."})
            if st.session_state.get('trop_val') == "Positive": critical_flags.append({"name": "Troponin POSITIVE", "act": "Confirmed tissue death. Immediate CAS EVAC."})

            features = [30, 0, 120, 80, 72, 16, 98, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14.0, 0] 
            ml_prediction = 1
            if scaler and model:
                try: ml_prediction = model.predict(scaler.transform([features]))[0]
                except: pass
            if ml_prediction == 0: critical_flags.append({"name": "AI Model Alert", "act": "Baseline high risk detected."})

            status_tier, pdf_flags, final_order = render_triage_results("HEART DISEASE", critical_flags, abnormal_flags)

            st.markdown("---")
            st.subheader("🎤 Voice Handover (Optional)")
            audio_note = st.audio_input("Record descriptive symptoms or ground situation for the MO", help="Press the microphone to start recording.")
            
            st.markdown("---")
            st.subheader("💾 Finalize Assessment")
            
            with st.container(border=True):
                p_num_input = st.text_input("Army / Service No. *", key="num_hd")
                p_rec = get_patient_record(p_num_input)
                
                if p_rec: st.success(f"✅ Patient Record Linked: {p_rec['rank']} {p_rec['name']}")
                elif p_num_input: st.warning("⚠️ Patient not found in Registry. PDF will not contain full medical history.")

                c_p1, c_p2 = st.columns(2)
                p_rank_val = c_p1.text_input("Rank", value=p_rec['rank'] if p_rec else "", key="rank_hd")
                p_name_val = c_p2.text_input("Name", value=p_rec['name'] if p_rec else "", key="name_hd")
                
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    p_loc_val = c_p1.text_input("Location / Post", key="loc_hd")
                else:
                    p_loc_val = st.session_state['post_name']
                    c_p1.text_input("Location / Post", value=p_loc_val, disabled=True, key="loc_hd_dis")
                
                pdf_data = create_pdf_report("Heart Disease", status_tier, pdf_flags, final_order, temp_val=st.session_state['temp'], alt_val=st.session_state['alt'], has_audio=(audio_note is not None), patient_info=p_rec)
                
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("💾 SAVE RECORD TO LEDGER", type="primary", key="save_hd"):
                        if p_num_input.strip() == '': st.error("⚠️ Army / Service No. is required.")
                        else:
                            save_to_ledger(p_rank_val, p_name_val, p_num_input, p_loc_val, "Heart Disease", status_tier, [f['name'] for f in pdf_flags], final_order, audio_bytes=audio_note)
                            st.session_state['saved_success_hd'] = True
                            st.success("✅ Medical record securely saved.")
                
                with action_col2:
                    if st.session_state.get('saved_success_hd', False) and pdf_data:
                        st.download_button("📄 DOWNLOAD PDF REPORT", pdf_data, f"Arogyam_Heart_{p_num_input}.pdf", "application/pdf", type="secondary", key="dl_hd")
                
                if st.session_state.get('saved_success_hd', False) and len(critical_flags) > 0:
                    st.markdown("---")
                    render_whatsapp_alert("Heart Disease", p_rank_val, p_name_val, p_num_input)

        def validate_hd(step):
            if step == 1:
                r = [st.session_state.get(k) for k in ['sex', 'alt']]
                return not any(x is None for x in r)
            if step == 2:
                if st.session_state.get('cp_yn') is None: return False
                if st.session_state.get('cp_yn') == "Yes":
                    r = [st.session_state.get(k) for k in ['cp_site', 'cp_onset', 'cp_char', 'cp_rad', 'cp_assoc_sweat', 'cp_assoc_nau', 'cp_assoc_cough', 'cp_assoc_doe', 'cp_assoc_sync', 'cp_assoc_bowel', 'cp_assoc_slur', 'cp_assoc_focal', 'cp_timing', 'cp_exac', 'cp_relieve']]
                    if any(x is None or x == [] for x in r): return False
                return True
            if step == 3:
                r = [st.session_state.get(k) for k in ['comorb', 'fam_hx_cond', 'hx_alcohol', 'hx_smoking', 'trop_val']]
                if any(x is None for x in r): return False
                if st.session_state.get('ecg_opt') and st.session_state.get('ecg_val') is None: return False
                return True
            if step == 4:
                r = [st.session_state.get(k) for k in ['hd_pe_tenderness', 'hd_pe_bowel', 'hd_pe_pulsations', 'hd_pe_discolor', 'hd_pe_distension']]
                if any(x is None for x in r): return False
                return True
            return True

        b1, b2, b3 = st.columns([1, 2, 1])
        with b1:
            if 1 < st.session_state['page_step'] <= 5 and st.button("PREVIOUS PAGE"): st.session_state['page_step'] -= 1; st.rerun()
        with b3:
            if st.session_state['page_step'] < 5:
                action_text = "RUN DIAGNOSIS" if st.session_state['page_step'] == 4 else "NEXT PAGE"
                if st.button(action_text, type="primary"):
                    if validate_hd(st.session_state['page_step']):
                        st.session_state['page_step'] += 1; st.rerun()
                    else: st.error("⚠️ Please complete all mandatory (*) fields.")
            elif st.session_state['page_step'] == 5:
                st.markdown("<br>", unsafe_allow_html=True)
                if not st.session_state.get('confirm_new_hd', False):
                    if st.button("🔄 START NEW ASSESSMENT", key="new_hd"): st.session_state['confirm_new_hd'] = True; st.rerun()
                else:
                    st.warning("⚠️ Are you sure? Unsaved data will be lost.")
                    c_yes, c_no = st.columns(2)
                    if c_yes.button("✅ Yes, Proceed", key="yes_hd"):
                        st.session_state['confirm_new_hd'] = False
                        st.session_state['saved_success_hd'] = False
                        for key in default_vals.keys():
                            if key in st.session_state and not key.startswith(('bshc_', 'ams_', 'hape_', 'ci_', 'p_')): del st.session_state[key]
                        st.session_state['page_step'] = 1; st.rerun()
                    if c_no.button("❌ Cancel", key="no_hd"):
                        st.session_state['confirm_new_hd'] = False; st.rerun()

    # ------------------------------------------
    # 2. BRAIN STROKE / HACE MODULE (UNIFIED)
    # ------------------------------------------
    elif selected == "Brain Stroke / HACE":
        st.markdown(f"### NEUROLOGICAL TRIAGE (STROKE/HACE) - PAGE {st.session_state['bshc_page_step']} OF 5")
        cols = st.columns(5)
        for i in range(5): cols[i].progress(100 if st.session_state['bshc_page_step'] > i else (0 if st.session_state['bshc_page_step'] <= i else 50))
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        show_doctrine_table("Brain Stroke / HACE")

        if st.session_state['bshc_page_step'] == 1:
            st.header("Core Vitals & Environment")
            col_a, col_b = st.columns(2)
            with col_a:
                st.session_state['bshc_age'] = st.number_input("Age *", 18, 90, st.session_state['bshc_age'], key='in_bshc_age', help=tt_lib.get('age', ''))
                st.session_state['bshc_sex'] = st.radio("Sex *", ["Male", "Female"], index=get_idx(["Male", "Female"], st.session_state['bshc_sex']), horizontal=True, key='in_bshc_sex', help=tt_lib.get('sex', ''))
                bp1, bp2 = st.columns(2)
                with bp1: st.session_state['bshc_s_bp'] = st.number_input("Systolic BP *", 60, 260, st.session_state['bshc_s_bp'], key='in_bshc_sbp', help=tt_lib.get('bp', ''))
                with bp2: st.session_state['bshc_d_bp'] = st.number_input("Diastolic BP *", 40, 160, st.session_state['bshc_d_bp'], key='in_bshc_dbp', help=tt_lib.get('bp', ''))
                pr1, rr1 = st.columns(2)
                with pr1: st.session_state['bshc_pulse'] = st.number_input("Pulse Rate (BPM) *", 40, 220, st.session_state['bshc_pulse'], key='in_bshc_pulse', help=tt_lib.get('pulse', ''))
                with rr1: st.session_state['bshc_resp'] = st.number_input("Resp Rate (/min) *", 8, 50, st.session_state['bshc_resp'], key='in_bshc_resp', help=tt_lib.get('resp', ''))
                t1, a1 = st.columns(2)
                with t1: st.session_state['bshc_temp'] = st.number_input("Temperature (°F) *", 70.0, 110.0, st.session_state['bshc_temp'], key='in_bshc_temp', help=tt_lib.get('temp', ''))
                with a1: st.session_state['bshc_alt'] = st.selectbox("Altitude (ft) *", alt_opts, index=get_idx(alt_opts, st.session_state['bshc_alt']), key='in_bshc_alt', help=tt_lib.get('alt', ''))
            with col_b:
                st.markdown(f"<div class='spo2-wrapper'><div class='spo2-title'>🫧 Blood Oxygen</div><div class='spo2-val'>{st.session_state['bshc_spo2']}%</div></div>", unsafe_allow_html=True)
                st.session_state['bshc_spo2'] = st.slider("SpO2 Levels (%) *", 40, 100, st.session_state['bshc_spo2'], key='in_bshc_spo2', help=tt_lib.get('spo2', ''))

        elif st.session_state['bshc_page_step'] == 2:
            st.header("BEFAST Neurological Assessment")
            c1, c2 = st.columns(2)
            with c1:
                st.session_state['bshc_balance'] = st.radio("⚖️ **B**alance: Sudden loss of balance/dizzy? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_balance']), horizontal=True, key='in_bshc_bal', help=tt_lib.get('bs_balance', ''))
                st.session_state['bshc_eyes'] = st.radio("👁️ **E**yes: Sudden blurred or lost vision? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_eyes']), horizontal=True, key='in_bshc_eyes', help=tt_lib.get('bs_eyes', ''))
                st.session_state['bshc_face'] = st.radio("😐 **F**ace: Is one side drooping? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_face']), horizontal=True, key='in_bshc_face', help=tt_lib.get('bs_face', ''))
            with c2:
                st.session_state['bshc_arms'] = st.radio("💪 **A**rms: Arm or leg weakness/numbness? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_arms']), horizontal=True, key='in_bshc_arms', help=tt_lib.get('bs_arms', ''))
                st.session_state['bshc_speech'] = st.radio("💬 **S**peech: Slurred speech? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_speech']), horizontal=True, key='in_bshc_speech', help=tt_lib.get('bs_speech', ''))
            st.markdown("---")
            st.session_state['bshc_time'] = st.selectbox("⏱️ **T**ime: When was the subject last seen acting normally? *", time_opts, index=get_idx(time_opts, st.session_state['bshc_time']), key='in_bshc_time', help=tt_lib.get('bs_time', ''))

        elif st.session_state['bshc_page_step'] == 3:
            st.header("Physical Exam (Cerebellar/Neuro)")
            with st.container(border=True):
                p1, p2 = st.columns(2)
                with p1:
                    st.session_state['bshc_vertigo'] = st.radio("Vertigo (Room spinning)? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_vertigo']), horizontal=True, help=tt_lib.get('bshc_vertigo', ''))
                    st.session_state['bshc_nystagmus'] = st.radio("Nystagmus (Eyes jerking)? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_nystagmus']), horizontal=True, help=tt_lib.get('bshc_nystagmus', ''))
                    st.session_state['bshc_tremor'] = st.radio("Intentional Tremor? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_tremor']), horizontal=True, help=tt_lib.get('bshc_tremor', ''))
                    st.session_state['bshc_slur'] = st.radio("Slurring of Speech? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_slur']), horizontal=True, help=tt_lib.get('bshc_slur', ''))
                    st.session_state['bshc_hypotonia'] = st.radio("Hypotonia (Floppy muscles)? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_hypotonia']), horizontal=True, help=tt_lib.get('bshc_hypotonia', ''))
                    st.session_state['bshc_gait'] = st.radio("Gait Abnormality? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_gait']), horizontal=True, help=tt_lib.get('bshc_gait', ''))
                with p2:
                    st.session_state['bshc_dysdia'] = st.radio("Dysdiadochokinesia (Can perform fast alternating hand movements)? *", ["Yes", "No"], index=get_idx(["Yes", "No"], st.session_state['bshc_dysdia']), horizontal=True, help=tt_lib.get('bshc_dysdia', ''))
                    st.session_state['bshc_ftn'] = st.radio("Finger to Nose Test (Smooth/Accurate)? *", ["Yes", "No"], index=get_idx(["Yes", "No"], st.session_state['bshc_ftn']), horizontal=True, help=tt_lib.get('bshc_ftn', ''))
                    st.session_state['bshc_hts'] = st.radio("Heel to Shin Test (Smooth)? *", ["Yes", "No"], index=get_idx(["Yes", "No"], st.session_state['bshc_hts']), horizontal=True, help=tt_lib.get('bshc_hts', ''))
                    st.session_state['bshc_rebound'] = st.radio("Normal Rebound Phenomenon (Arm stops when resistance removed)? *", ["Yes", "No"], index=get_idx(["Yes", "No"], st.session_state['bshc_rebound']), horizontal=True, help=tt_lib.get('bshc_rebound', ''))
                    st.session_state['bshc_romberg'] = st.radio("Positive Romberg Sign (Loses balance with eyes closed)? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_romberg']), horizontal=True, help=tt_lib.get('bshc_romberg', ''))

        elif st.session_state['bshc_page_step'] == 4:
            st.header("Patient Complaints")
            with st.container(border=True):
                st.session_state['bshc_headache'] = st.slider("Headache Severity (0-10) *", 0, 10, st.session_state.get('bshc_headache', 0), help=tt_lib.get('bshc_headache', ''))
                c1, c2 = st.columns(2)
                with c1:
                    st.session_state['bshc_mental'] = st.radio("Altered Mental Status? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_mental']), horizontal=True, help=tt_lib.get('bshc_mental', ''))
                    st.session_state['bshc_vomit'] = st.radio("Vomiting? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_vomit']), horizontal=True, help=tt_lib.get('bshc_vomit', ''))
                    st.session_state['bshc_nausea'] = st.radio("Nausea? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_nausea']), horizontal=True, help=tt_lib.get('bshc_nausea', ''))
                with c2:
                    st.session_state['bshc_dizzy'] = st.radio("Dizziness? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_dizzy']), horizontal=True, help=tt_lib.get('bshc_dizzy', ''))
                    st.session_state['bshc_sensation'] = st.radio("Loss of Sensation? *", opts_yn, index=get_idx(opts_yn, st.session_state['bshc_sensation']), horizontal=True, help=tt_lib.get('bshc_sensation', ''))
                    st.session_state['bshc_pupils'] = st.radio("Pupils Reactive? *", ["Yes", "No"], index=get_idx(["Yes", "No"], st.session_state['bshc_pupils']), horizontal=True, help=tt_lib.get('bshc_pupils', ''))
                
                st.markdown("---")
                dtr_opts = ["Normal", "Decreased", "Increased", "Absent"]
                st.session_state['bshc_dtr'] = st.selectbox("Deep Tendon Reflexes *", dtr_opts, index=get_idx(dtr_opts, st.session_state['bshc_dtr']), help=tt_lib.get('bshc_dtr', ''))

        elif st.session_state['bshc_page_step'] == 5:
            st.header("Diagnostic Triage Results")
            abnormal_flags = []
            critical_flags = []
            
            # Vitals
            if st.session_state['bshc_s_bp'] > 160 or st.session_state['bshc_d_bp'] > 100: abnormal_flags.append({"name": f"Hypertension ({st.session_state['bshc_s_bp']}/{st.session_state['bshc_d_bp']})", "act": "Do NOT lower BP drastically if stroke suspected."})
            if st.session_state['bshc_spo2'] < 85: abnormal_flags.append({"name": f"Hypoxia (SpO2: {st.session_state['bshc_spo2']}%)", "act": "Administer O2 to maintain SpO2 > 90%."})
            check_temp_rule(st.session_state['bshc_temp'], abnormal_flags)

            # BEFAST
            if st.session_state['bshc_balance'] == "Yes": critical_flags.append({"name": "Loss of Balance", "act": "Stroke indicator."})
            if st.session_state['bshc_eyes'] == "Yes": critical_flags.append({"name": "Vision Loss", "act": "Stroke indicator."})
            if st.session_state['bshc_face'] == "Yes": critical_flags.append({"name": "Facial Droop", "act": "Stroke indicator."})
            if st.session_state['bshc_arms'] == "Yes": critical_flags.append({"name": "Arm Weakness", "act": "Stroke indicator."})
            if st.session_state['bshc_speech'] == "Yes": critical_flags.append({"name": "Speech Difficulty", "act": "Stroke indicator."})

            # Physical Exam
            if st.session_state['bshc_vertigo'] == "Yes": critical_flags.append({"name": "Vertigo", "act": "Neurological sign."})
            if st.session_state['bshc_nystagmus'] == "Yes": critical_flags.append({"name": "Nystagmus", "act": "Neurological sign."})
            if st.session_state['bshc_tremor'] == "Yes": critical_flags.append({"name": "Intentional Tremor", "act": "Neurological sign."})
            if st.session_state['bshc_slur'] == "Yes": critical_flags.append({"name": "Slurring of Speech", "act": "Neurological sign."})
            if st.session_state['bshc_hypotonia'] == "Yes": critical_flags.append({"name": "Hypotonia", "act": "Neurological sign."})
            if st.session_state['bshc_gait'] == "Yes": critical_flags.append({"name": "Gait Abnormality", "act": "Neurological sign. Prevent movement."})
            if st.session_state['bshc_dysdia'] == "No": critical_flags.append({"name": "Failed Dysdiadochokinesia", "act": "Cerebellar sign."})
            if st.session_state['bshc_ftn'] == "No": critical_flags.append({"name": "Failed Finger-Nose Test", "act": "Cerebellar sign."})
            if st.session_state['bshc_hts'] == "No": critical_flags.append({"name": "Failed Heel-Shin Test", "act": "Cerebellar sign."})
            if st.session_state['bshc_rebound'] == "No": critical_flags.append({"name": "Absent Rebound Phenomenon", "act": "Cerebellar sign."})
            if st.session_state['bshc_romberg'] == "Yes": critical_flags.append({"name": "Positive Romberg", "act": "Balance failure."})

            # Complaints
            if st.session_state['bshc_headache'] >= 6: critical_flags.append({"name": f"Severe Headache ({st.session_state['bshc_headache']}/10)", "act": "Suspect HACE or Hemorrhage."})
            elif st.session_state['bshc_headache'] >= 1: abnormal_flags.append({"name": f"Mild Headache ({st.session_state['bshc_headache']}/10)", "act": "Monitor closely."})
            if st.session_state['bshc_mental'] == "Yes": critical_flags.append({"name": "Altered Mental Status", "act": "Immediate Evac. High ICP risk."})
            if st.session_state['bshc_vomit'] == "Yes": critical_flags.append({"name": "Vomiting", "act": "High ICP risk."})
            if st.session_state['bshc_nausea'] == "Yes": critical_flags.append({"name": "Nausea", "act": "Neurological distress."})
            if st.session_state['bshc_dizzy'] == "Yes": critical_flags.append({"name": "Dizziness", "act": "Neurological distress."})
            if st.session_state['bshc_sensation'] == "Yes": critical_flags.append({"name": "Loss of Sensation", "act": "Focal deficit."})
            if st.session_state['bshc_pupils'] == "No": critical_flags.append({"name": "Unreactive Pupils", "act": "Critical Brain Stem sign."})
            
            if st.session_state['bshc_dtr'] == "Absent": critical_flags.append({"name": "Absent Deep Tendon Reflexes", "act": "Severe neurological suppression."})
            elif st.session_state['bshc_dtr'] in ["Increased", "Decreased"]: abnormal_flags.append({"name": f"DTR: {st.session_state['bshc_dtr']}", "act": "Neurological alteration."})

            status_tier, pdf_flags, final_order = render_triage_results("BRAIN STROKE / HACE", critical_flags, abnormal_flags)

            st.markdown("---")
            st.subheader("🎤 Voice Handover (Optional)")
            audio_note = st.audio_input("Record descriptive symptoms or ground situation for the MO", help="Press the microphone to start recording.")

            st.markdown("---")
            st.subheader("💾 Finalize Assessment")
            
            with st.container(border=True):
                p_num_input = st.text_input("Army / Service No. *", key="num_bshc")
                p_rec = get_patient_record(p_num_input)
                
                if p_rec: st.success(f"✅ Patient Record Linked: {p_rec['rank']} {p_rec['name']}")
                elif p_num_input: st.warning("⚠️ Patient not found in Registry. PDF will not contain full medical history.")

                c_p1, c_p2 = st.columns(2)
                p_rank_val = c_p1.text_input("Rank", value=p_rec['rank'] if p_rec else "", key="rank_bshc")
                p_name_val = c_p2.text_input("Name", value=p_rec['name'] if p_rec else "", key="name_bshc")
                
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    p_loc_val = c_p1.text_input("Location / Post", key="loc_bshc")
                else:
                    p_loc_val = st.session_state['post_name']
                    c_p1.text_input("Location / Post", value=p_loc_val, disabled=True, key="loc_bshc_dis")
                
                pdf_data = create_pdf_report("BRAIN STROKE / HACE", status_tier, pdf_flags, final_order, temp_val=st.session_state['bshc_temp'], alt_val=st.session_state['bshc_alt'], has_audio=(audio_note is not None), patient_info=p_rec)
                
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("💾 SAVE RECORD TO LEDGER", type="primary", key="save_bshc"):
                        if p_num_input.strip() == '': st.error("⚠️ Army / Service No. is required.")
                        else:
                            save_to_ledger(p_rank_val, p_name_val, p_num_input, p_loc_val, "Brain Stroke / HACE", status_tier, [f['name'] for f in pdf_flags], final_order, audio_bytes=audio_note)
                            st.session_state['saved_success_bshc'] = True
                            st.success("✅ Medical record securely saved.")
                
                with action_col2:
                    if st.session_state.get('saved_success_bshc', False) and pdf_data:
                        st.download_button("📄 DOWNLOAD PDF REPORT", pdf_data, f"Arogyam_Neuro_{p_num_input}.pdf", "application/pdf", type="secondary", key="dl_bshc")
                
                if st.session_state.get('saved_success_bshc', False) and len(critical_flags) > 0:
                    st.markdown("---")
                    render_whatsapp_alert("Brain Stroke / HACE", p_rank_val, p_name_val, p_num_input)

        def validate_bshc(step):
            if step == 1:
                r = [st.session_state.get(k) for k in ['bshc_sex', 'bshc_alt']]
                return not any(x is None for x in r)
            if step == 2:
                r = [st.session_state.get(k) for k in ['bshc_balance', 'bshc_eyes', 'bshc_face', 'bshc_arms', 'bshc_speech', 'bshc_time']]
                if any(x is None for x in r) or st.session_state['bshc_time'] == "Select Time": return False
                return True
            if step == 3:
                r = [st.session_state.get(k) for k in ['bshc_vertigo', 'bshc_nystagmus', 'bshc_tremor', 'bshc_slur', 'bshc_hypotonia', 'bshc_gait', 'bshc_dysdia', 'bshc_ftn', 'bshc_hts', 'bshc_rebound', 'bshc_romberg']]
                return not any(x is None for x in r)
            if step == 4:
                r = [st.session_state.get(k) for k in ['bshc_mental', 'bshc_vomit', 'bshc_nausea', 'bshc_dizzy', 'bshc_sensation', 'bshc_pupils', 'bshc_dtr']]
                return not any(x is None for x in r)
            return True

        b1, b2, b3 = st.columns([1, 2, 1])
        with b1:
            if 1 < st.session_state['bshc_page_step'] <= 5 and st.button("PREVIOUS PAGE", key="bshc_prev"): st.session_state['bshc_page_step'] -= 1; st.rerun()
        with b3:
            if st.session_state['bshc_page_step'] < 5:
                action_text = "RUN DIAGNOSIS" if st.session_state['bshc_page_step'] == 4 else "NEXT PAGE"
                if st.button(action_text, type="primary", key="bshc_next"):
                    if validate_bshc(st.session_state['bshc_page_step']): st.session_state['bshc_page_step'] += 1; st.rerun()
                    else: st.error("⚠️ Please complete all mandatory (*) fields.")
            elif st.session_state['bshc_page_step'] == 5:
                st.markdown("<br>", unsafe_allow_html=True)
                if not st.session_state.get('confirm_new_bshc', False):
                    if st.button("🔄 START NEW ASSESSMENT", key="new_bshc"): st.session_state['confirm_new_bshc'] = True; st.rerun()
                else:
                    st.warning("⚠️ Are you sure? Unsaved data will be lost.")
                    c_yes, c_no = st.columns(2)
                    if c_yes.button("✅ Yes, Proceed", key="yes_bshc"):
                        st.session_state['confirm_new_bshc'] = False
                        st.session_state['saved_success_bshc'] = False
                        for key in list(st.session_state.keys()):
                            if key.startswith('bshc_') and key != 'bshc_page_step': del st.session_state[key]
                        st.session_state['bshc_page_step'] = 1; st.rerun()
                    if c_no.button("❌ Cancel", key="no_bshc"):
                        st.session_state['confirm_new_bshc'] = False; st.rerun()
    
    # ------------------------------------------
    # 3. AMS (Acute Mountain Sickness) Triage
    # ------------------------------------------
    elif selected == "AMS":
        st.markdown(f"### ACUTE MOUNTAIN SICKNESS (AMS) PROTOCOL - PAGE {st.session_state['ams_page_step']} OF 3")
        cols = st.columns(3)
        for i in range(3): cols[i].progress(100 if st.session_state['ams_page_step'] > i else (0 if st.session_state['ams_page_step'] <= i else 50))
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        show_doctrine_table("AMS")

        if st.session_state['ams_page_step'] == 1:
            st.header("Core Vitals & Environment")
            col_a, col_b = st.columns(2)
            with col_a:
                bp1, bp2 = st.columns(2)
                with bp1: st.session_state['ams_s_bp'] = st.number_input("Systolic BP *", 60, 260, st.session_state['ams_s_bp'], key='in_ams_sbp', help=tt_lib.get('bp', ''))
                with bp2: st.session_state['ams_d_bp'] = st.number_input("Diastolic BP *", 40, 160, st.session_state['ams_d_bp'], key='in_ams_dbp', help=tt_lib.get('bp', ''))
                pr1, rr1 = st.columns(2)
                with pr1: st.session_state['ams_pulse'] = st.number_input("Pulse Rate (BPM) *", 40, 220, st.session_state['ams_pulse'], key='in_ams_pulse', help=tt_lib.get('pulse', ''))
                with rr1: st.session_state['ams_resp'] = st.number_input("Resp Rate (/min) *", 8, 50, st.session_state['ams_resp'], key='in_ams_resp', help=tt_lib.get('resp', ''))
                t1, a1 = st.columns(2)
                with t1: st.session_state['ams_temp'] = st.number_input("Temperature (°F) *", 70.0, 110.0, st.session_state['ams_temp'], key='in_ams_temp', help=tt_lib.get('temp', ''))
                with a1: st.session_state['ams_alt'] = st.selectbox("Altitude (ft) *", alt_opts, index=get_idx(alt_opts, st.session_state['ams_alt']), key='in_ams_alt', help=tt_lib.get('alt', ''))
            with col_b:
                st.markdown(f"<div class='spo2-wrapper'><div class='spo2-title'>🫧 Blood Oxygen</div><div class='spo2-val'>{st.session_state['ams_spo2']}%</div></div>", unsafe_allow_html=True)
                st.session_state['ams_spo2'] = st.slider("SpO2 Levels (%) *", 40, 100, st.session_state['ams_spo2'], key='in_ams_spo2', help=tt_lib.get('spo2', ''))

        elif st.session_state['ams_page_step'] == 2:
            st.header("Lake Louise Scoring System & Hydration")
            urine_opts = ["Clear / Pale Yellow", "Dark Yellow", "Very Dark / Brown"]
            st.session_state['ams_urine'] = st.selectbox("💧 Urine Color *", urine_opts, index=get_idx(urine_opts, st.session_state['ams_urine']), key='in_ams_urine', help=tt_lib.get('ams_urine', ''))
            st.markdown("---")
            scores = [0, 1, 2, 3]
            st.session_state['ll_headache'] = st.selectbox("Headache *", scores, format_func=lambda x: f"{x} - Severity", index=get_idx(scores, st.session_state['ll_headache']), key='in_ll_head', help=tt_lib.get('ll_headache', ''))
            st.session_state['ll_gi'] = st.selectbox("Gastrointestinal Symptoms *", scores, format_func=lambda x: f"{x} - Severity", index=get_idx(scores, st.session_state['ll_gi']), key='in_ll_gi', help=tt_lib.get('ll_gi', ''))
            st.session_state['ll_fatigue'] = st.selectbox("Fatigue / Weakness *", scores, format_func=lambda x: f"{x} - Severity", index=get_idx(scores, st.session_state['ll_fatigue']), key='in_ll_fat', help=tt_lib.get('ll_fatigue', ''))
            st.session_state['ll_dizzy'] = st.selectbox("Dizziness / Lightheadedness *", scores, format_func=lambda x: f"{x} - Severity", index=get_idx(scores, st.session_state['ll_dizzy']), key='in_ll_diz', help=tt_lib.get('ll_dizzy', ''))
            st.session_state['ll_sleep'] = st.selectbox("Difficulty Sleeping *", scores, format_func=lambda x: f"{x} - Severity", index=get_idx(scores, st.session_state['ll_sleep']), key='in_ll_sleep', help=tt_lib.get('ll_sleep', ''))
        
        elif st.session_state['ams_page_step'] == 3:
            st.header("Acute Mountain Sickness (AMS) Triage Results")
            lls_total = st.session_state['ll_headache'] + st.session_state['ll_gi'] + st.session_state['ll_fatigue'] + st.session_state['ll_dizzy'] + st.session_state['ll_sleep']
            
            abnormal_flags = []
            critical_flags = []
            mild_flags = []

            if st.session_state['ams_urine'] != "Clear / Pale Yellow":
                abnormal_flags.append({"name": f"Dehydration Suspected ({st.session_state['ams_urine']})", "act": "Hydrate via ORS before escalating medications if score is mild."})

            if lls_total > 10:
                critical_flags.append({"name": f"Severe AMS (Score: {lls_total})", "act": "Halt ascent. Evaluate for HACE/HAPE. Descend safely."})
            elif 5 <= lls_total <= 10:
                abnormal_flags.append({"name": f"Moderate to Severe AMS (Score: {lls_total})", "act": "Halt ascent immediately. Rest. Administer protocol meds."})
            elif 1 <= lls_total <= 4:
                mild_flags.append({"name": f"Mild AMS (Score: {lls_total})", "act": "Halt ascent until symptoms resolve. Treat symptoms."})

            check_temp_rule(st.session_state['ams_temp'], abnormal_flags)
            status_tier, pdf_flags, final_order = render_triage_results("AMS", critical_flags, abnormal_flags, mild_flags)

            st.markdown("---")
            st.subheader("🎤 Voice Handover (Optional)")
            audio_note = st.audio_input("Record descriptive symptoms or ground situation for the MO", help="Press the microphone to start recording.")

            st.markdown("---")
            st.subheader("💾 Finalize Assessment")
            
            with st.container(border=True):
                p_num_input = st.text_input("Army / Service No. *", key="num_ams")
                p_rec = get_patient_record(p_num_input)
                
                if p_rec: st.success(f"✅ Patient Record Linked: {p_rec['rank']} {p_rec['name']}")
                elif p_num_input: st.warning("⚠️ Patient not found in Registry. PDF will not contain full medical history.")

                c_p1, c_p2 = st.columns(2)
                p_rank_val = c_p1.text_input("Rank", value=p_rec['rank'] if p_rec else "", key="rank_ams")
                p_name_val = c_p2.text_input("Name", value=p_rec['name'] if p_rec else "", key="name_ams")
                
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    p_loc_val = c_p1.text_input("Location / Post", key="loc_ams")
                else:
                    p_loc_val = st.session_state['post_name']
                    c_p1.text_input("Location / Post", value=p_loc_val, disabled=True, key="loc_ams_dis")
                
                pdf_data = create_pdf_report("AMS", status_tier, pdf_flags, final_order, temp_val=st.session_state['ams_temp'], alt_val=st.session_state['ams_alt'], army_no=p_num_input, current_score=lls_total, has_audio=(audio_note is not None), patient_info=p_rec)
                
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("💾 SAVE RECORD TO LEDGER", type="primary", key="save_ams"):
                        if p_num_input.strip() == '': st.error("⚠️ Army / Service No. is required.")
                        else:
                            save_to_ledger(p_rank_val, p_name_val, p_num_input, p_loc_val, "AMS", status_tier, [f['name'] for f in pdf_flags], final_order, audio_bytes=audio_note)
                            st.session_state['saved_success_ams'] = True
                            st.success("✅ Medical record securely saved.")
                
                with action_col2:
                    if st.session_state.get('saved_success_ams', False) and pdf_data:
                        st.download_button("📄 DOWNLOAD PDF REPORT", pdf_data, f"Arogyam_AMS_{p_num_input}.pdf", "application/pdf", type="secondary", key="dl_ams")
                
                if st.session_state.get('saved_success_ams', False) and len(critical_flags) > 0:
                    st.markdown("---")
                    render_whatsapp_alert("AMS", p_rank_val, p_name_val, p_num_input)

        def validate_ams(step):
            if step == 1:
                r = [st.session_state.get(k) for k in ['ams_alt']]
                return not any(x is None for x in r)
            if step == 2:
                r = [st.session_state.get(k) for k in ['ams_urine', 'll_headache', 'll_gi', 'll_fatigue', 'll_dizzy', 'll_sleep']]
                return not any(x is None for x in r)
            return True

        b1, b2, b3 = st.columns([1, 2, 1])
        with b1:
            if 1 < st.session_state['ams_page_step'] <= 3 and st.button("PREVIOUS PAGE", key="ams_prev"): st.session_state['ams_page_step'] -= 1; st.rerun()
        with b3:
            if st.session_state['ams_page_step'] < 3:
                action_text = "RUN DIAGNOSIS" if st.session_state['ams_page_step'] == 2 else "NEXT PAGE"
                if st.button(action_text, type="primary", key="ams_next"): 
                    if validate_ams(st.session_state['ams_page_step']): st.session_state['ams_page_step'] += 1; st.rerun()
                    else: st.error("⚠️ Please complete all mandatory (*) fields.")
            elif st.session_state['ams_page_step'] == 3:
                st.markdown("<br>", unsafe_allow_html=True)
                if not st.session_state.get('confirm_new_ams', False):
                    if st.button("🔄 START NEW ASSESSMENT", key="new_ams"): st.session_state['confirm_new_ams'] = True; st.rerun()
                else:
                    st.warning("⚠️ Are you sure? Unsaved data will be lost.")
                    c_yes, c_no = st.columns(2)
                    if c_yes.button("✅ Yes, Proceed", key="yes_ams"):
                        st.session_state['confirm_new_ams'] = False
                        st.session_state['saved_success_ams'] = False
                        for key in list(st.session_state.keys()):
                            if (key.startswith('ams_') or key.startswith('ll_')) and key != 'ams_page_step': del st.session_state[key]
                        st.session_state['ams_page_step'] = 1; st.rerun()
                    if c_no.button("❌ Cancel", key="no_ams"):
                        st.session_state['confirm_new_ams'] = False; st.rerun()

    # ------------------------------------------
    # 4. HAPE (High Altitude Pulmonary Edema) Triage
    # ------------------------------------------
    elif selected == "HAPE":
        dyspnea_opts = ["Normal", "Mild Exertion", "At Rest", "Severe"]
        resp_opts = ["Normal", "Wheezy", "Severe Distress"]
        activity_opts = ["Normal", "Able to perform light activity", "Cannot perform light activity"]
        mobility_opts = ["Normal", "Weakness, fatigue on slight effort", "Weakness", "Unable to stand/ walk"]
        mental_opts = ["Normal", "Clouded consciousness", "Stupor", "Coma"]
        cough_opts = ["None", "Headache with cough", "Loose recurrent productive cough", "Copious bloody sputum"]
        cyanosis_opts = ["None", "Obvious cyanosis", "Severe cyanosis"]

        st.markdown(f"### HAPE (PULMONARY EDEMA) PROTOCOL - PAGE {st.session_state['hape_page_step']} OF 3")
        cols = st.columns(3)
        for i in range(3): cols[i].progress(100 if st.session_state['hape_page_step'] > i else (0 if st.session_state['hape_page_step'] <= i else 50))
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        show_doctrine_table("HAPE")

        if st.session_state['hape_page_step'] == 1:
            st.header("Core Vitals & Environment")
            col_a, col_b = st.columns(2)
            with col_a:
                bp1, bp2 = st.columns(2)
                with bp1: st.session_state['hape_s_bp'] = st.number_input("Systolic BP *", 60, 260, st.session_state['hape_s_bp'], key='in_hape_sbp', help=tt_lib.get('bp', ''))
                with bp2: st.session_state['hape_d_bp'] = st.number_input("Diastolic BP *", 40, 160, st.session_state['hape_d_bp'], key='in_hape_dbp', help=tt_lib.get('bp', ''))
                pr1, rr1 = st.columns(2)
                with pr1: st.session_state['hape_pulse'] = st.number_input("Pulse Rate (BPM) *", 40, 220, st.session_state['hape_pulse'], key='in_hape_pulse', help=tt_lib.get('pulse', ''))
                with rr1: st.session_state['hape_resp'] = st.number_input("Resp Rate (/min) *", 8, 50, st.session_state['hape_resp'], key='in_hape_resp', help=tt_lib.get('resp', ''))
                t1, a1 = st.columns(2)
                with t1: st.session_state['hape_temp'] = st.number_input("Temperature (°F) *", 70.0, 110.0, st.session_state['hape_temp'], key='in_hape_temp', help=tt_lib.get('temp', ''))
                with a1: st.session_state['hape_alt'] = st.selectbox("Altitude (ft) *", alt_opts, index=get_idx(alt_opts, st.session_state['hape_alt']), key='in_hape_alt', help=tt_lib.get('alt', ''))
            with col_b:
                st.markdown(f"<div class='spo2-wrapper'><div class='spo2-title'>🫧 Blood Oxygen</div><div class='spo2-val'>{st.session_state['hape_spo2']}%</div></div>", unsafe_allow_html=True)
                st.session_state['hape_spo2'] = st.slider("SpO2 Levels (%) *", 40, 100, st.session_state['hape_spo2'], key='in_hape_spo2', help=tt_lib.get('spo2', ''))

        elif st.session_state['hape_page_step'] == 2:
            st.header("Pulmonary Assessment")
            c_d, c_r = st.columns(2)
            with c_d: st.session_state['hape_dyspnea'] = st.selectbox("Dyspnoea *", dyspnea_opts, index=get_idx(dyspnea_opts, st.session_state['hape_dyspnea']), key='in_h_dys', help=tt_lib.get('hape_dyspnea', ''))
            with c_r: st.session_state['hape_resp_qual'] = st.selectbox("Respiration Quality *", resp_opts, index=get_idx(resp_opts, st.session_state['hape_resp_qual']), key='in_h_rq', help=tt_lib.get('hape_resp_qual', ''))
            c_a, c_m = st.columns(2)
            with c_a: st.session_state['hape_activity'] = st.selectbox("Activity Level *", activity_opts, index=get_idx(activity_opts, st.session_state['hape_activity']), key='in_h_act', help=tt_lib.get('hape_activity', ''))
            with c_m: st.session_state['hape_mobility'] = st.selectbox("Mobility / Strength *", mobility_opts, index=get_idx(mobility_opts, st.session_state['hape_mobility']), key='in_h_mob', help=tt_lib.get('hape_mobility', ''))
            st.session_state['hape_mental'] = st.selectbox("Mental Status *", mental_opts, index=get_idx(mental_opts, st.session_state['hape_mental']), key='in_h_men', help=tt_lib.get('hape_mental', ''))
            st.session_state['hape_cough'] = st.selectbox("Cough Status *", cough_opts, index=get_idx(cough_opts, st.session_state['hape_cough']), key='in_h_cou', help=tt_lib.get('hape_cough', ''))
            st.session_state['hape_cyanosis'] = st.selectbox("Cyanosis *", cyanosis_opts, index=get_idx(cyanosis_opts, st.session_state['hape_cyanosis']), key='in_h_cya', help=tt_lib.get('hape_cyanosis', ''))
            st.markdown("---")
            c1, c2, c3 = st.columns(3)
            with c1: st.session_state['hape_nausea'] = st.radio("Nausea at rest? *", opts_yn, index=get_idx(opts_yn, st.session_state['hape_nausea']), horizontal=True, key='in_h_nau', help=tt_lib.get('hape_nausea', ''))
            with c2: st.session_state['hape_rales'] = st.radio("Bubbling rales? *", opts_yn, index=get_idx(opts_yn, st.session_state['hape_rales']), horizontal=True, key='in_h_ral', help=tt_lib.get('hape_rales', ''))
            with c3: st.session_state['hape_headache'] = st.radio("Standalone Headache? *", opts_yn, index=get_idx(opts_yn, st.session_state['hape_headache']), horizontal=True, key='in_h_head', help=tt_lib.get('hape_headache', ''))

        elif st.session_state['hape_page_step'] == 3:
            st.header("HAPE Triage Results")
            abnormal_flags = []
            critical_flags = []
            mild_flags = []

            hr = st.session_state['hape_pulse']
            rr = st.session_state['hape_resp']
            spo2 = st.session_state['hape_spo2']

            if hr > 140 or rr > 40 or st.session_state['hape_dyspnea'] == "Severe" or st.session_state['hape_resp_qual'] == "Severe Distress" or st.session_state['hape_cyanosis'] == "Severe cyanosis" or st.session_state['hape_mental'] in ["Stupor", "Coma"] or st.session_state['hape_mobility'] == "Unable to stand/ walk" or st.session_state['hape_cough'] == "Copious bloody sputum":
                critical_flags.append({"name": "Critical HAPE Symptoms/Vitals", "act": "Immediate, rapid descent. Oxygen and Nifedipine immediately."})
            elif (121 <= hr <= 140) or (31 <= rr <= 40) or spo2 < 80 or st.session_state['hape_rales'] == "Yes" or st.session_state['hape_dyspnea'] == "At Rest" or st.session_state['hape_cyanosis'] == "Obvious cyanosis" or st.session_state['hape_cough'] == "Loose recurrent productive cough":
                critical_flags.append({"name": "Serious HAPE Symptoms/Vitals", "act": "Immediate descent is mandatory. Administer Nifedipine. Administer O2."})
            elif (110 <= hr <= 120) or (20 <= rr <= 30) or spo2 < 85 or st.session_state['hape_dyspnea'] == "Mild Exertion" or st.session_state['hape_resp_qual'] == "Wheezy" or st.session_state['hape_mobility'] != "Normal" or st.session_state['hape_activity'] != "Normal" or st.session_state['hape_nausea'] == "Yes" or st.session_state['hape_headache'] == "Yes":
                abnormal_flags.append({"name": "Moderate HAPE Symptoms/Vitals", "act": "Halt ascent. Begin descent. Rest strictly. Administer O2."})
            elif st.session_state['hape_dyspnea'] == "Mild Exertion" or st.session_state['hape_activity'] == "Able to perform light activity":
                mild_flags.append({"name": "Mild HAPE Symptoms", "act": "Observe closely. Do not ascend."})

            check_temp_rule(st.session_state['hape_temp'], abnormal_flags)
            status_tier, pdf_flags, final_order = render_triage_results("HAPE", critical_flags, abnormal_flags, mild_flags)

            st.markdown("---")
            st.subheader("🎤 Voice Handover (Optional)")
            audio_note = st.audio_input("Record descriptive symptoms or ground situation for the MO", help="Press the microphone to start recording.")

            st.markdown("---")
            st.subheader("💾 Finalize Assessment")
            
            with st.container(border=True):
                p_num_input = st.text_input("Army / Service No. *", key="num_hape")
                p_rec = get_patient_record(p_num_input)
                
                if p_rec: st.success(f"✅ Patient Record Linked: {p_rec['rank']} {p_rec['name']}")
                elif p_num_input: st.warning("⚠️ Patient not found in Registry. PDF will not contain full medical history.")

                c_p1, c_p2 = st.columns(2)
                p_rank_val = c_p1.text_input("Rank", value=p_rec['rank'] if p_rec else "", key="rank_hape")
                p_name_val = c_p2.text_input("Name", value=p_rec['name'] if p_rec else "", key="name_hape")
                
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    p_loc_val = c_p1.text_input("Location / Post", key="loc_hape")
                else:
                    p_loc_val = st.session_state['post_name']
                    c_p1.text_input("Location / Post", value=p_loc_val, disabled=True, key="loc_hape_dis")
                
                pdf_data = create_pdf_report("HAPE", status_tier, pdf_flags, final_order, temp_val=st.session_state['hape_temp'], alt_val=st.session_state['hape_alt'], has_audio=(audio_note is not None), patient_info=p_rec)
                
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("💾 SAVE RECORD TO LEDGER", type="primary", key="save_hape"):
                        if p_num_input.strip() == '': st.error("⚠️ Army / Service No. is required.")
                        else:
                            save_to_ledger(p_rank_val, p_name_val, p_num_input, p_loc_val, "HAPE", status_tier, [f['name'] for f in pdf_flags], final_order, audio_bytes=audio_note)
                            st.session_state['saved_success_hape'] = True
                            st.success("✅ Medical record securely saved.")
                
                with action_col2:
                    if st.session_state.get('saved_success_hape', False) and pdf_data:
                        st.download_button("📄 DOWNLOAD PDF REPORT", pdf_data, f"Arogyam_HAPE_{p_num_input}.pdf", "application/pdf", type="secondary", key="dl_hape")
                
                if st.session_state.get('saved_success_hape', False) and len(critical_flags) > 0:
                    st.markdown("---")
                    render_whatsapp_alert("HAPE", p_rank_val, p_name_val, p_num_input)

        def validate_hape(step):
            if step == 1:
                r = [st.session_state.get(k) for k in ['hape_alt']]
                return not any(x is None for x in r)
            if step == 2:
                r = [st.session_state.get(k) for k in ['hape_dyspnea', 'hape_resp_qual', 'hape_activity', 'hape_mobility', 'hape_mental', 'hape_cough', 'hape_cyanosis', 'hape_nausea', 'hape_rales', 'hape_headache']]
                return not any(x is None for x in r)
            return True

        b1, b2, b3 = st.columns([1, 2, 1])
        with b1:
            if 1 < st.session_state['hape_page_step'] <= 3 and st.button("PREVIOUS PAGE", key="hape_prev"): st.session_state['hape_page_step'] -= 1; st.rerun()
        with b3:
            if st.session_state['hape_page_step'] < 3:
                action_text = "RUN DIAGNOSIS" if st.session_state['hape_page_step'] == 2 else "NEXT PAGE"
                if st.button(action_text, type="primary", key="hape_next"):
                    if validate_hape(st.session_state['hape_page_step']): st.session_state['hape_page_step'] += 1; st.rerun()
                    else: st.error("⚠️ Please complete all mandatory (*) fields.")
            elif st.session_state['hape_page_step'] == 3:
                st.markdown("<br>", unsafe_allow_html=True)
                if not st.session_state.get('confirm_new_hape', False):
                    if st.button("🔄 START NEW ASSESSMENT", key="new_hape"): st.session_state['confirm_new_hape'] = True; st.rerun()
                else:
                    st.warning("⚠️ Are you sure? Unsaved data will be lost.")
                    c_yes, c_no = st.columns(2)
                    if c_yes.button("✅ Yes, Proceed", key="yes_hape"):
                        st.session_state['confirm_new_hape'] = False
                        st.session_state['saved_success_hape'] = False
                        for key in list(st.session_state.keys()):
                            if key.startswith('hape_') and key != 'hape_page_step': del st.session_state[key]
                        st.session_state['hape_page_step'] = 1; st.rerun()
                    if c_no.button("❌ Cancel", key="no_hape"):
                        st.session_state['confirm_new_hape'] = False; st.rerun()

    # ------------------------------------------
    # 6. COLD INJURIES MODULE
    # ------------------------------------------
    elif selected == "Cold Injuries":
        st.markdown(f"### COLD INJURIES (HYPOTHERMIA & FROSTBITE) - PAGE {st.session_state['ci_page_step']} OF 3")
        cols = st.columns(3)
        for i in range(3): cols[i].progress(100 if st.session_state['ci_page_step'] > i else (0 if st.session_state['ci_page_step'] <= i else 50))
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        show_doctrine_table("Cold Injuries")

        if st.session_state['ci_page_step'] == 1:
            st.header("Core Vitals & Environment")
            col_a, col_b = st.columns(2)
            with col_a:
                bp1, bp2 = st.columns(2)
                with bp1: st.session_state['ci_s_bp'] = st.number_input("Systolic BP *", 60, 260, st.session_state['ci_s_bp'], key='in_ci_sbp', help=tt_lib.get('bp', ''))
                with bp2: st.session_state['ci_d_bp'] = st.number_input("Diastolic BP *", 40, 160, st.session_state['ci_d_bp'], key='in_ci_dbp', help=tt_lib.get('bp', ''))
                pr1, rr1 = st.columns(2)
                with pr1: st.session_state['ci_pulse'] = st.number_input("Pulse Rate (BPM) *", 40, 220, st.session_state['ci_pulse'], key='in_ci_pulse', help=tt_lib.get('pulse', ''))
                with rr1: st.session_state['ci_resp'] = st.number_input("Resp Rate (/min) *", 8, 50, st.session_state['ci_resp'], key='in_ci_resp', help=tt_lib.get('resp', ''))
                t1, a1 = st.columns(2)
                with t1: st.session_state['ci_temp'] = st.number_input("Temperature (°F) *", 70.0, 110.0, st.session_state['ci_temp'], step=0.1, key='in_ci_temp', help=tt_lib.get('temp', ''))
                with a1: st.session_state['ci_alt'] = st.selectbox("Altitude (ft) *", alt_opts, index=get_idx(alt_opts, st.session_state['ci_alt']), key='in_ci_alt', help=tt_lib.get('alt', ''))
            with col_b:
                st.markdown(f"<div class='spo2-wrapper'><div class='spo2-title'>🫧 Blood Oxygen</div><div class='spo2-val'>{st.session_state['ci_spo2']}%</div></div>", unsafe_allow_html=True)
                st.session_state['ci_spo2'] = st.slider("SpO2 Levels (%) *", 40, 100, st.session_state['ci_spo2'], key='in_ci_spo2', help=tt_lib.get('spo2', ''))

        elif st.session_state['ci_page_step'] == 2:
            st.header("Field Clinical Assessment")
            
            with st.container(border=True):
                st.subheader("🥶 Systemic (Hypothermia)")
                c1, c2, c3 = st.columns(3)
                with c1: st.session_state['ci_mental_alt'] = st.radio("Mental Status Altered? *", opts_yn, index=get_idx(opts_yn, st.session_state['ci_mental_alt']), horizontal=True, key='in_ci_mental_alt', help=tt_lib.get('ci_mental_alt', ''))
                with c2: st.session_state['ci_breathing'] = st.radio("Difficulty Breathing? *", opts_yn, index=get_idx(opts_yn, st.session_state['ci_breathing']), horizontal=True, key='in_ci_breathing', help=tt_lib.get('ci_breathing', ''))
                with c3: st.session_state['ci_shiver'] = st.radio("Is patient shivering? *", opts_yn, index=get_idx(opts_yn, st.session_state['ci_shiver']), horizontal=True, key='in_ci_shiver', help=tt_lib.get('ci_shiver', ''))
                
                st.markdown("**Associated Symptoms (Hypothermia)**")
                a1, a2, a3, a4 = st.columns(4)
                with a1:
                    st.session_state['ci_assoc_sweat'] = st.radio("Sweating", opts_yn, index=get_idx(opts_yn, st.session_state.get('ci_assoc_sweat', 'No')), horizontal=True)
                    st.session_state['ci_assoc_sync'] = st.radio("Syncope", opts_yn, index=get_idx(opts_yn, st.session_state.get('ci_assoc_sync', 'No')), horizontal=True)
                with a2:
                    st.session_state['ci_assoc_nau'] = st.radio("Nausea", opts_yn, index=get_idx(opts_yn, st.session_state.get('ci_assoc_nau', 'No')), horizontal=True)
                    st.session_state['ci_assoc_bowel'] = st.radio("Abnormal Bowel/Bladder", opts_yn, index=get_idx(opts_yn, st.session_state.get('ci_assoc_bowel', 'No')), horizontal=True)
                with a3:
                    st.session_state['ci_assoc_cough'] = st.radio("Coughing", opts_yn, index=get_idx(opts_yn, st.session_state.get('ci_assoc_cough', 'No')), horizontal=True)
                    st.session_state['ci_assoc_slur'] = st.radio("Slurring of Speech", opts_yn, index=get_idx(opts_yn, st.session_state.get('ci_assoc_slur', 'No')), horizontal=True)
                with a4:
                    st.session_state['ci_assoc_doe'] = st.radio("Dyspnoea on Exertion", opts_yn, index=get_idx(opts_yn, st.session_state.get('ci_assoc_doe', 'No')), horizontal=True)
                    st.session_state['ci_assoc_focal'] = st.radio("Focal Deficits", opts_yn, index=get_idx(opts_yn, st.session_state.get('ci_assoc_focal', 'No')), horizontal=True)
                st.session_state['ci_assoc_other'] = st.text_input("Other Symptoms (Specify)", st.session_state.get('ci_assoc_other', ''))

            with st.container(border=True):
                st.subheader("🧊 Localized (Frostbite / Chilblains)")
                st.session_state['ci_site'] = st.text_input("Site of Injury (Body Part) *", st.session_state.get('ci_site', ''), key='in_ci_site', help=tt_lib.get('ci_site', ''))
                
                if st.session_state['ci_site'].strip() != "":
                    ci_file = st.file_uploader("📸 Upload Image of Injury for MO Review", type=['png', 'jpg', 'jpeg'])
                    if ci_file is not None:
                        temp_ci_path = os.path.join(TEMP_IMG_DIR, f"temp_ci_{int(datetime.now().timestamp())}.jpg")
                        try:
                            with open(temp_ci_path, "wb") as f: f.write(ci_file.getvalue())
                            st.session_state['ci_img_path'] = temp_ci_path
                            st.success("Injury image saved for report.")
                        except Exception: pass
                
                cl_opts = ["Normal", "Red/Flushed", "White/Waxy", "Mottled Blue/Black"]
                st.session_state['ci_skin_color'] = st.selectbox("Skin Color of Extremity *", cl_opts, index=get_idx(cl_opts, st.session_state['ci_skin_color']), key='in_ci_color', help=tt_lib.get('ci_skin_color', ''))
                
                f1, f2, f3 = st.columns(3)
                with f1: st.session_state['ci_tenderness'] = st.radio("Severe Tenderness? *", opts_yn, index=get_idx(opts_yn, st.session_state['ci_tenderness']), horizontal=True, key='in_ci_tenderness', help=tt_lib.get('ci_tenderness', ''))
                with f2: st.session_state['ci_sensation'] = st.radio("Loss of Sensation? *", opts_yn, index=get_idx(opts_yn, st.session_state['ci_sensation']), horizontal=True, key='in_ci_sensation', help=tt_lib.get('ci_sensation', ''))
                with f3: st.session_state['ci_cap_refill'] = st.radio("Delayed Capillary Refill? *", opts_yn, index=get_idx(opts_yn, st.session_state['ci_cap_refill']), horizontal=True, key='in_ci_cap', help=tt_lib.get('ci_cap_refill', ''))

                st.markdown("**Physical Examination Findings**")
                fb_stages = [
                    "None",
                    "First-degree (Red, numb, and tingly skin)",
                    "Second-degree (Skin feels stiff/frozen, aching/throbbing pain)",
                    "Third-degree (Skin feels hard, shooting pain)",
                    "Fourth-degree (Skin is dark/rubbery, deep tissue involvement)"
                ]
                st.session_state['ci_frostbite_stage'] = st.selectbox("Stage of Frostbite *", fb_stages, index=get_idx(fb_stages, st.session_state.get('ci_frostbite_stage', 'None')), key='in_ci_stage', help=tt_lib.get('ci_frostbite_stage', ''))
                
                b_opts = ['None', 'Clear Fluid', 'Blood-Filled']
                st.session_state['ci_blister_type'] = st.selectbox("Blister Type *", b_opts, index=get_idx(b_opts, st.session_state['ci_blister_type']), key='in_ci_bt', help=tt_lib.get('ci_blister', ''))

        elif st.session_state['ci_page_step'] == 3:
            st.header("Cold Injury Triage Results")
            abnormal_flags = []
            critical_flags = []

            # 1. Hypothermia Logic
            if st.session_state['ci_temp'] < 82.4 or st.session_state['ci_mental_alt'] == "Yes" or st.session_state['ci_breathing'] == "Yes":
                critical_flags.append({"name": "Severe Systemic Hypothermia", "act": "Patient is critical. Active core rewarming. Handle gently (V-Fib risk)."})
            elif (st.session_state['ci_shiver'] == "No" and st.session_state['ci_temp'] < 95.0):
                abnormal_flags.append({"name": "Moderate Hypothermia (Paradoxical Shivering Loss)", "act": "Active external rewarming needed."})
            
            check_temp_rule(st.session_state['ci_temp'], abnormal_flags)

            assoc_ci_list = [st.session_state.get(k) for k in ['ci_assoc_sweat', 'ci_assoc_nau', 'ci_assoc_cough', 'ci_assoc_doe', 'ci_assoc_sync', 'ci_assoc_bowel', 'ci_assoc_slur', 'ci_assoc_focal']]
            if "Yes" in assoc_ci_list: critical_flags.append({"name": "Associated High-Risk Systemic Symptoms", "act": "Complex presentation alongside cold exposure."})

            # 2. Frostbite / Chilblains Logic
            if st.session_state['ci_skin_color'] in ["Red/Flushed", "White/Waxy"]: abnormal_flags.append({"name": f"Skin Color: {st.session_state['ci_skin_color']}", "act": "Signs of localized freezing."})
            elif st.session_state['ci_skin_color'] == "Mottled Blue/Black": critical_flags.append({"name": "Skin Color: Mottled/Black", "act": "Necrotic tissue. Protect limb."})
            
            if st.session_state['ci_tenderness'] == "Yes": critical_flags.append({"name": "Severe Tenderness", "act": "Vascular/nerve damage."})
            if st.session_state['ci_sensation'] == "Yes": critical_flags.append({"name": "Loss of Sensation", "act": "Nerve freezing/death."})
            if st.session_state['ci_cap_refill'] == "Yes": abnormal_flags.append({"name": "Delayed Capillary Refill", "act": "Poor vascular circulation in extremity."})

            # Clinical Grading Algorithm
            stage = st.session_state['ci_frostbite_stage']
            bt = st.session_state['ci_blister_type']

            if "Fourth" in stage or "Third" in stage or bt == "Blood-Filled":
                critical_flags.append({"name": "Deep Frostbite (3rd/4th Degree)", "act": "High risk of tissue loss. Do NOT thaw if refreezing is possible. Immediate EVAC."})
            elif "Second" in stage or bt == "Clear Fluid":
                abnormal_flags.append({"name": "Superficial Frostbite (2nd Degree)", "act": "Apply active external rewarming. Protect blisters. Avoid massaging."})
            elif "First" in stage:
                abnormal_flags.append({"name": "Frostnip (1st Degree)", "act": "Passive rewarming. Move to warm environment."})

            status_tier, pdf_flags, final_order = render_triage_results("COLD INJURY", critical_flags, abnormal_flags)

            st.markdown("---")
            st.subheader("🎤 Voice Handover (Optional)")
            audio_note = st.audio_input("Record descriptive symptoms or ground situation for the MO", help="Press the microphone to start recording.")

            st.markdown("---")
            st.subheader("💾 Finalize Assessment")
            
            with st.container(border=True):
                p_num_input = st.text_input("Army / Service No. *", key="num_ci")
                p_rec = get_patient_record(p_num_input)
                
                if p_rec: st.success(f"✅ Patient Record Linked: {p_rec['rank']} {p_rec['name']}")
                elif p_num_input: st.warning("⚠️ Patient not found in Registry. PDF will not contain full medical history.")

                c_p1, c_p2 = st.columns(2)
                p_rank_val = c_p1.text_input("Rank", value=p_rec['rank'] if p_rec else "", key="rank_ci")
                p_name_val = c_p2.text_input("Name", value=p_rec['name'] if p_rec else "", key="name_ci")
                
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    p_loc_val = c_p1.text_input("Location / Post", key="loc_ci")
                else:
                    p_loc_val = st.session_state['post_name']
                    c_p1.text_input("Location / Post", value=p_loc_val, disabled=True, key="loc_ci_dis")
                
                pdf_data = create_pdf_report("COLD INJURY", status_tier, pdf_flags, final_order, temp_val=st.session_state['ci_temp'], alt_val=st.session_state['ci_alt'], has_audio=(audio_note is not None), patient_info=p_rec)
                
                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    if st.button("💾 SAVE RECORD TO LEDGER", type="primary", key="save_ci"):
                        if p_num_input.strip() == '': st.error("⚠️ Army / Service No. is required.")
                        else:
                            save_to_ledger(p_rank_val, p_name_val, p_num_input, p_loc_val, "Cold Injuries", status_tier, [f['name'] for f in pdf_flags], final_order, audio_bytes=audio_note)
                            st.session_state['saved_success_ci'] = True
                            st.success("✅ Medical record securely saved.")
                
                with action_col2:
                    if st.session_state.get('saved_success_ci', False) and pdf_data:
                        st.download_button("📄 DOWNLOAD PDF REPORT", pdf_data, f"Arogyam_coldinjuries_{p_num_input}.pdf", "application/pdf", type="secondary", key="dl_ci")
                
                if st.session_state.get('saved_success_ci', False) and len(critical_flags) > 0:
                    st.markdown("---")
                    render_whatsapp_alert("Cold Injuries", p_rank_val, p_name_val, p_num_input)

        def validate_ci(step):
            if step == 1:
                r = [st.session_state.get(k) for k in ['ci_alt']]
                return not any(x is None for x in r)
            if step == 2:
                r = [st.session_state.get(k) for k in ['ci_site']]
                if any(x is None or str(x).strip() == "" for x in r): return False
            return True

        b1, b2, b3 = st.columns([1, 2, 1])
        with b1:
            if 1 < st.session_state['ci_page_step'] <= 3 and st.button("PREVIOUS PAGE", key="ci_prev"): st.session_state['ci_page_step'] -= 1; st.rerun()
        with b3:
            if st.session_state['ci_page_step'] < 3:
                action_text = "RUN DIAGNOSIS" if st.session_state['ci_page_step'] == 2 else "NEXT PAGE"
                if st.button(action_text, type="primary", key="ci_next"): 
                    if validate_ci(st.session_state['ci_page_step']): st.session_state['ci_page_step'] += 1; st.rerun()
                    else: st.error("⚠️ Please complete all mandatory (*) fields.")
            elif st.session_state['ci_page_step'] == 3:
                st.markdown("<br>", unsafe_allow_html=True)
                if not st.session_state.get('confirm_new_ci', False):
                    if st.button("🔄 START NEW ASSESSMENT", key="new_ci"): st.session_state['confirm_new_ci'] = True; st.rerun()
                else:
                    st.warning("⚠️ Are you sure? Unsaved data will be lost.")
                    c_yes, c_no = st.columns(2)
                    if c_yes.button("✅ Yes, Proceed", key="yes_ci"):
                        st.session_state['confirm_new_ci'] = False
                        st.session_state['saved_success_ci'] = False
                        for key in list(st.session_state.keys()):
                            if key.startswith('ci_') and key != 'ci_page_step': del st.session_state[key]
                        st.session_state['ci_page_step'] = 1; st.rerun()
                    if c_no.button("❌ Cancel", key="no_ci"):
                        st.session_state['confirm_new_ci'] = False; st.rerun()

    # ------------------------------------------
    # 7. WEEKLY VITALS MODULE 
    # ------------------------------------------
    elif selected == "Weekly Vitals":
        st.markdown("### 📈 WEEKLY VITALS MODULE")
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["📝 Log Vitals", "🗄️ Ledger & Export", "📉 Patient Analytics"])
        
        with tab1:
            st.subheader("Quick-Entry Routine Log")
            with st.form("vitals_form"):
                v_army_no = st.text_input("Army / Service No. *")
                c1, c2, c3, c4, c5 = st.columns(5)
                v_sys = c1.number_input("Systolic BP", 60, 200, 120)
                v_dia = c2.number_input("Diastolic BP", 40, 150, 80)
                v_pul = c3.number_input("Pulse (BPM)", 40, 200, 72)
                v_spo2 = c4.number_input("SpO2 (%)", 50, 100, 95)
                v_rr = c5.number_input("Resp Rate (/min)", 8, 50, 16) 
                
                if st.form_submit_button("SAVE TO WEEKLY LEDGER", type="primary"):
                    if v_army_no.strip():
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
                        
                        # Prevent duplicate saving if user double-clicks fast
                        res_dup = supabase.table("weekly_vitals").select("army_no, timestamp").eq("timestamp", timestamp).execute()
                        existing_logs = [decrypt_data(r['army_no']).strip().upper() for r in (res_dup.data if res_dup.data else [])]
                        
                        if v_army_no.strip().upper() in existing_logs:
                            st.warning("⚠️ This vital record was just saved. Please wait a minute before logging again.")
                        else:
                            enc_v_army = encrypt_data(v_army_no)
                            insert_data = {
                                "timestamp": timestamp,
                                "bfna_id": st.session_state['bfna_id'],
                                "post_name": st.session_state['post_name'],
                                "army_no": enc_v_army,
                                "sys_bp": v_sys,
                                "dia_bp": v_dia,
                                "pulse": v_pul,
                                "spo2": v_spo2,
                                "resp_rate": v_rr
                            }
                            try:
                                supabase.table("weekly_vitals").insert(insert_data).execute()
                                st.success(f"✅ Vitals logged for {v_army_no.upper()} at {st.session_state['post_name']}.")
                            except Exception as e:
                                st.error(f"Error saving to Supabase: {e}")
                    else:
                        st.error("Army No is required.")
                        
        with tab2:
            try:
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    st.subheader("Battalion Overview: Weekly Vitals")
                    res = supabase.table("weekly_vitals").select("*").order("timestamp", desc=True).execute()
                    v_df = pd.DataFrame(res.data)
                    
                    if not v_df.empty:
                        v_df['army_no'] = v_df['army_no'].apply(decrypt_data)
                        posts = v_df['post_name'].dropna().unique()
                        if len(posts) > 0:
                            tabs = st.tabs([str(p).replace("17_RAJPUT_", "") for p in posts])
                            for i, p in enumerate(posts):
                                with tabs[i]:
                                    post_df = v_df[v_df['post_name'] == p]
                                    st.dataframe(post_df.drop(columns=['id']), use_container_width=True)
                                    csv = post_df.drop(columns=['id']).to_csv(index=False).encode('utf-8')
                                    st.download_button(f"📥 EXPORT {str(p).replace('17_RAJPUT_', '')} CSV", data=csv, file_name=f'WeeklyVitals_{p}.csv', mime='text/csv', key=f"dl_{p}")
                        else:
                            st.info("No valid post data found.")
                    else: st.info("No vitals recorded in the battalion yet.")
                else:
                    st.subheader(f"Weekly Logs for Post: {st.session_state['post_name']}")
                    res = supabase.table("weekly_vitals").select("*").eq("post_name", st.session_state['post_name']).order("timestamp", desc=True).execute()
                    v_df = pd.DataFrame(res.data)
                    
                    if not v_df.empty:
                        v_df['army_no'] = v_df['army_no'].apply(decrypt_data)
                        visual_df = v_df.drop(columns=['id'])
                        st.dataframe(visual_df, use_container_width=True)
                        
                        csv = visual_df.to_csv(index=False).encode('utf-8')
                        st.download_button("📥 EXPORT CSV FOR MO", data=csv, file_name=f'WeeklyVitals_{st.session_state["post_name"]}.csv', mime='text/csv', type="primary")

                # --- ENHANCED DROPDOWN DELETE LOGIC ---
                if not v_df.empty:
                    st.markdown("---")
                    st.write("### Manage Records")
                    
                    # RBAC: Only Admin/RMO can delete
                    if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                        
                        # 1. Format the records into a clean dropdown list
                        vitals_list = [f"ID: {row['id']} | Date: {row['timestamp']} | Army No: {row['army_no']} | BP: {row['sys_bp']}/{row['dia_bp']}" for _, row in v_df.iterrows()]
                        selected_vital = st.selectbox("Select Vitals Record to Delete", ["-- Select --"] + vitals_list, key="sel_del_vital")
                        
                        col_del1, col_del2 = st.columns(2)
                        with col_del1:
                            if st.button("🗑️ Delete Selected Record", type="secondary"):
                                if selected_vital != "-- Select --":
                                    try:
                                        # 2. Isolate the exact ID number from the dropdown string
                                        target_id = int(selected_vital.split("ID: ")[1].split(" |")[0])
                                        
                                        # 3. Fire the deletion command to Supabase
                                        supabase.table("weekly_vitals").delete().eq("id", target_id).execute()
                                        st.success("✅ Record successfully deleted.")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Deletion failed: {e}")
                                else:
                                    st.warning("⚠️ Please select a record from the dropdown first.")
                        
                        with col_del2:
                            st.warning("Clear entirely.")
                            if st.button("🚨 CLEAR ALL VITALS", type="primary"):
                                try:
                                    # Safely wipe the whole table
                                    supabase.table("weekly_vitals").delete().neq("id", 0).execute() 
                                    st.success("All records cleared.")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Clear failed: {e}")
                    else:
                        st.info("⚠️ BFNAs have View-Only access to the Weekly Vitals Ledger. Only the RMO can delete records.")

            except Exception as e:
                st.error(f"DB Error: {e}")
                
        with tab3:
            st.subheader("Health Trend Analytics")
            target_soldier = st.text_input("Enter Army No. to visualize trends:")
            if target_soldier:
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    res = supabase.table("weekly_vitals").select("timestamp, army_no, sys_bp, dia_bp, pulse").order("timestamp").execute()
                else:
                    res = supabase.table("weekly_vitals").select("timestamp, army_no, sys_bp, dia_bp, pulse").eq("post_name", st.session_state['post_name']).order("timestamp").execute()
                
                t_df = pd.DataFrame(res.data)
                
                if not t_df.empty:
                    t_df['army_no'] = t_df['army_no'].apply(decrypt_data)
                    t_df = t_df[t_df['army_no'] == target_soldier]
                    
                if not t_df.empty:
                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(t_df['timestamp'], t_df['sys_bp'], marker='o', label='Sys BP', color='#EF4444')
                    ax1.plot(t_df['timestamp'], t_df['dia_bp'], marker='o', label='Dia BP', color='#F59E0B')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Blood Pressure (mmHg)')
                    ax2 = ax1.twinx()
                    ax2.plot(t_df['timestamp'], t_df['pulse'], marker='s', linestyle='--', label='Pulse', color='#3B82F6')
                    ax2.set_ylabel('Pulse (BPM)')
                    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
                    plt.title(f"Vitals Trend for {target_soldier}")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                else:
                    st.warning("No data found for this soldier.")

    
    # ------------------------------------------
    # NEW MODULE: ACCLIMATIZATION TRACKER
    # ------------------------------------------
    elif selected == "Acclimatization":
        st.markdown("### 🏔️ ACCLIMATIZATION PROTOCOL (STAGE 1 & 2)")
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        tab_search, tab_entry, tab_records = st.tabs(["🔍 Patient Search", "📝 Acclimatization Entry", "📊 Post-Wise Records"])
        
        if 'acc_patient' not in st.session_state: st.session_state['acc_patient'] = None
        
        with tab_search:
            st.markdown("Enter Army No. to pull patient dossier for Acclimatization logging.")
            search_army = st.text_input("Army / Service No.")
            if st.button("Fetch Patient", type="primary"):
                pt_rec = get_patient_record(search_army)
                if pt_rec:
                    st.session_state['acc_patient'] = pt_rec
                    st.success(f"✅ Patient Found: {pt_rec['rank']} {pt_rec['name']}")
                else:
                    st.session_state['acc_patient'] = None
                    st.error("❌ Patient not found in Battalion Registry. Please register them first.")
                    
            if st.session_state['acc_patient']:
                p = st.session_state['acc_patient']
                st.info(f"**Selected Patient:** {p['rank']} {p['name']} | **Coy:** {p['company']} | **Inducted:** {p['induction_date']}")

        with tab_entry:
            if not st.session_state['acc_patient']:
                st.warning("Please search and select a patient in the first tab.")
            else:
                p = st.session_state['acc_patient']
                
                with st.form("acc_form"):
                    st.subheader("Stage 1 Acclimatization (Day 1 to 6)")
                    
                    st.markdown("**Daily Vitals**")
                    s1_vitals = {}
                    cols1 = st.columns(6)
                    for day in range(1, 7):
                        with cols1[day-1]:
                            st.markdown(f"**Day {day}**")
                            sys = st.number_input("Sys", 60, 200, 120, key=f"s1_s_{day}")
                            dia = st.number_input("Dia", 40, 120, 80, key=f"s1_d_{day}")
                            pul = st.number_input("Pul", 40, 150, 72, key=f"s1_p_{day}")
                            s1_vitals[f"Day_{day}"] = {"sys": sys, "dia": dia, "pulse": pul}
                            
                    st.markdown("---")
                    st.markdown("**Blood & Serum / Renal Profile**")
                    lc1, lc2, lc3, lc4 = st.columns(4)
                    hb = lc1.number_input("Hb (gm/dL)", 5.0, 25.0, 15.0, step=0.1)
                    tlc = lc2.number_input("TLC (/cumm)", 1000, 20000, 4800)
                    pltl = lc3.number_input("Platelets (x10³)", 50, 500, 180)
                    ldh = lc4.number_input("LDH (U/L)", 100, 1000, 410)
                    
                    bili = lc1.number_input("Total Bilirubin", 0.1, 10.0, 2.1, step=0.1)
                    sgot = lc2.number_input("SGOT (AST)", 0, 200, 25)
                    sgpt = lc3.number_input("SGPT (ALT)", 0, 200, 27)
                    
                    st.markdown("**Lipid Profile**")
                    l_c1, l_c2, l_c3 = st.columns(3)
                    chol = l_c1.number_input("Total Cholesterol", 100, 400, 190)
                    trig = l_c2.number_input("Triglyceride", 50, 500, 286)
                    ldl = l_c3.number_input("LDL Cholesterol", 50, 300, 130)
                    
                    s1_probs = st.text_area("Stage 1 Problems / Complications (Leave blank if normal)")
                    
                    st.markdown("---")
                    st.subheader("Stage 2 Acclimatization (Day 7 to 10)")
                    st.markdown("**Daily Vitals**")
                    s2_vitals = {}
                    cols2 = st.columns(4)
                    for day in range(7, 11):
                        with cols2[day-7]:
                            st.markdown(f"**Day {day}**")
                            sys2 = st.number_input("Sys", 60, 200, 120, key=f"s2_s_{day}")
                            dia2 = st.number_input("Dia", 40, 120, 80, key=f"s2_d_{day}")
                            pul2 = st.number_input("Pul", 40, 150, 72, key=f"s2_p_{day}")
                            s2_vitals[f"Day_{day}"] = {"sys": sys2, "dia": dia2, "pulse": pul2}
                            
                    s2_probs = st.text_area("Stage 2 Problems / Complications (Leave blank if normal)")
                    
                    status_opt = st.selectbox("Acclimatization Status", ["In Progress", "Completed Successfully", "Halted / Medical Issue"])
                    
                    if st.form_submit_button("💾 SAVE & GENERATE PDF", type="primary"):
                        enc_army = encrypt_data(p['army_no'])
                        enc_name = encrypt_data(p['name'])
                        
                        import json
                        insert_data = {
                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
                            "bfna_id": st.session_state['bfna_id'],
                            "post_name": st.session_state['post_name'],
                            "army_no": enc_army, "rank": p['rank'], "name": enc_name,
                            "s1_vitals": s1_vitals, "s2_vitals": s2_vitals,
                            "lab_hb": hb, "lab_tlc": tlc, "lab_platelets": pltl, "lab_ldh": ldh,
                            "lab_tot_bili": bili, "lab_sgot": sgot, "lab_sgpt": sgpt,
                            "lab_tot_chol": chol, "lab_triglycerides": trig, "lab_ldl": ldl,
                            "stage_1_problems": s1_probs if s1_probs else "None",
                            "stage_2_problems": s2_probs if s2_probs else "None",
                            "status": status_opt
                        }
                        
                        try:
                            supabase.table("acclimatization_details").insert(insert_data).execute()
                            st.success("✅ Record successfully saved to Supabase Cloud!")
                            
                            # Generate PDF
                            if fpdf_available:
                                class AccPDF(FPDF):
                                    def header(self):
                                        self.set_font('Arial', 'B', 15)
                                        self.cell(0, 10, 'ACCLIMATIZATION REPORT (STAGE 1 & 2)', 0, 1, 'C')
                                        self.ln(5)
                                pdf = AccPDF()
                                pdf.add_page()
                                pdf.set_font('Arial', 'B', 12)
                                pdf.set_fill_color(220, 230, 245)
                                pdf.cell(0, 8, ' PATIENT DETAILS', 0, 1, 'L', fill=True)
                                pdf.set_font('Arial', '', 10)
                                pdf.cell(95, 6, f"Rank & Name: {p['rank']} {p['name']}", border=0)
                                pdf.cell(95, 6, f"Army No: {p['army_no']}", border=0, ln=1)
                                pdf.cell(95, 6, f"Coy: {p['company']}", border=0)
                                pdf.cell(95, 6, f"Induction Date: {p['induction_date']}", border=0, ln=1)
                                pdf.ln(5)
                                
                                pdf.set_font('Arial', 'B', 12)
                                pdf.cell(0, 8, ' STAGE 1 ACCLIMATIZATION (DAY 1 - 6)', 0, 1, 'L', fill=True)
                                pdf.set_font('Arial', '', 10)
                                for d in range(1, 7):
                                    v = s1_vitals[f"Day_{d}"]
                                    pdf.cell(0, 6, f"Day {d}: BP {v['sys']}/{v['dia']} | Pulse {v['pulse']}", 0, 1)
                                pdf.ln(3)
                                pdf.set_font('Arial', 'B', 10)
                                pdf.cell(0, 6, 'Blood & Lipid Profile:', 0, 1)
                                pdf.set_font('Arial', '', 10)
                                pdf.cell(95, 6, f"Hb: {hb} | TLC: {tlc} | Platelets: {pltl}", 0)
                                pdf.cell(95, 6, f"LDH: {ldh} | Tot Bili: {bili}", 0, ln=1)
                                pdf.cell(0, 6, f"SGOT: {sgot} | SGPT: {sgpt} | Chol: {chol} | Trig: {trig} | LDL: {ldl}", 0, 1)
                                pdf.cell(0, 6, f"Stage 1 Problems: {s1_probs if s1_probs else 'None'}", 0, 1)
                                pdf.ln(5)
                                
                                pdf.set_font('Arial', 'B', 12)
                                pdf.cell(0, 8, ' STAGE 2 ACCLIMATIZATION (DAY 7 - 10)', 0, 1, 'L', fill=True)
                                pdf.set_font('Arial', '', 10)
                                for d in range(7, 11):
                                    v = s2_vitals[f"Day_{d}"]
                                    pdf.cell(0, 6, f"Day {d}: BP {v['sys']}/{v['dia']} | Pulse {v['pulse']}", 0, 1)
                                pdf.cell(0, 6, f"Stage 2 Problems: {s2_probs if s2_probs else 'None'}", 0, 1)
                                pdf.ln(5)
                                pdf.set_font('Arial', 'B', 11)
                                pdf.cell(0, 8, f"FINAL STATUS: {status_opt.upper()}", 0, 1)
                                
                                # Store PDF in memory safely
                                st.session_state['acc_pdf_data'] = pdf.output(dest='S').encode('latin-1')
                                st.session_state['acc_pdf_army'] = p['army_no']
                                st.session_state['acc_saved'] = True
                                
                        except Exception as e:
                            st.error(f"Save failed: {e}")

                # Safely generate the download button OUTSIDE the form
                if st.session_state.get('acc_saved', False):
                    st.download_button("📄 DOWNLOAD ACCLIMATIZATION PDF", st.session_state['acc_pdf_data'], f"Acclim_{st.session_state['acc_pdf_army']}.pdf", "application/pdf", type="secondary")
                            

        with tab_records:
            st.subheader(f"Acclimatization Records for {st.session_state['post_name']}")
            try:
                # We added 'id' to the select query so we can target it for deletion
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    res_acc = supabase.table("acclimatization_details").select("id, timestamp, army_no, rank, name, status, stage_1_problems, post_name").order("timestamp", desc=True).execute()
                else:
                    res_acc = supabase.table("acclimatization_details").select("id, timestamp, army_no, rank, name, status, stage_1_problems").eq("post_name", st.session_state['post_name']).order("timestamp", desc=True).execute()
                
                df_acc = pd.DataFrame(res_acc.data)
                if not df_acc.empty:
                    df_acc['army_no'] = df_acc['army_no'].apply(decrypt_data)
                    df_acc['name'] = df_acc['name'].apply(decrypt_data)
                    
                    # Show dataframe without the ID column for a cleaner look
                    st.dataframe(df_acc.drop(columns=['id']), use_container_width=True, hide_index=True)
                    
                    # Deletion Logic
                    if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                        st.markdown("---")
                        st.subheader("Delete Acclimatization Record")
                        
                        acc_list = [f"ID: {row['id']} | {row['timestamp']} | {row['army_no']} | {row['status']}" for _, row in df_acc.iterrows()]
                        selected_acc = st.selectbox("Select Record to Delete", ["-- Select --"] + acc_list, key="sel_del_acc")
                        
                        if st.button("🗑️ Delete Selected Record", type="secondary"):
                            if selected_acc != "-- Select --":
                                target_id = int(selected_acc.split("ID: ")[1].split(" |")[0])
                                supabase.table("acclimatization_details").delete().eq("id", target_id).execute()
                                st.success("Record successfully deleted.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("Please select a record first.")
                else:
                    st.info("No acclimatization records logged yet.")
            except Exception as e:
                st.error(f"Error loading records: {e}")

    # ------------------------------------------
    # 8. PATIENT HISTORY & LEDGER
    # ------------------------------------------
    elif selected == "Patient History":
        st.markdown(f"### 🗄️ BATTALION MEDICAL LEDGER: {st.session_state['post_name']}")
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        tab_view, tab_manage, tab_manual = st.tabs(["🗂️ View Ledger", "⚙️ Manage Records", "➕ Manual Entry"])
        
        try:
            if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                res = supabase.table("patient_history").select("*").order("timestamp", desc=True).execute()
            else:
                res = supabase.table("patient_history").select("*").eq("post_name", st.session_state['post_name']).order("timestamp", desc=True).execute()
            
            df = pd.DataFrame(res.data)
            if not df.empty:
                df['name'] = df['name'].apply(decrypt_data)
                df['army_no'] = df['army_no'].apply(decrypt_data)
                df['location'] = df['location'].apply(decrypt_data)
                
        except Exception as e:
            st.error(f"Database error: {e}")
            df = pd.DataFrame()

        with tab_view:
            if not df.empty:
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    st.subheader("Battalion Overview (Global Search)")
                    global_search = st.text_input("🔍 Search Army No. across ALL posts:")
                    
                    if global_search:
                        filtered_df = df[df['army_no'].str.contains(global_search, case=False, na=False)]
                        disp_df = filtered_df.drop(columns=['id', 'audio_path'], errors='ignore')
                        st.dataframe(disp_df, use_container_width=True, hide_index=True)
                        
                        audio_records = filtered_df[(filtered_df['audio_path'] != 'None') & (filtered_df['audio_path'].notnull())]
                        if not audio_records.empty:
                            st.markdown("### 🎧 Play Voice Notes")
                            selected_audio = st.selectbox("Select Patient Record:", audio_records['army_no'] + " - " + audio_records['timestamp'], key="audio_player_rmo")
                            if selected_audio:
                                path_to_play = audio_records[audio_records['army_no'] + " - " + audio_records['timestamp'] == selected_audio]['audio_path'].values[0]
                                if path_to_play and path_to_play != 'None':
                                    st.audio(path_to_play)
                                    st.markdown(f"[📥 Download Audio File]({path_to_play})")
                    else:
                        st.info("Enter an Army Number above to search across the Battalion, or export the full ledger below.")
                        disp_df = df.drop(columns=['id', 'audio_path'], errors='ignore')
                        st.dataframe(disp_df.head(50), use_container_width=True, hide_index=True)
                        st.caption("Showing top 50 recent records.")
                    
                    csv_ph = df.drop(columns=['id', 'audio_path'], errors='ignore').to_csv(index=False).encode('utf-8')
                    st.download_button("📥 EXPORT FULL BATTALION LEDGER (CSV)", data=csv_ph, file_name='Battalion_MedicalLedger.csv', mime='text/csv', type="primary")

                else:
                    search_no = st.text_input("🔍 Search by Army / Service No.")
                    filtered_df = df[df['army_no'].str.contains(search_no, case=False, na=False)] if search_no else df
                    
                    disp_df = filtered_df.drop(columns=['id', 'audio_path'], errors='ignore')
                    st.dataframe(disp_df, use_container_width=True, hide_index=True)
                    
                    csv_ph = disp_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 EXPORT LEDGER (CSV)", data=csv_ph, file_name=f'MedicalLedger_{st.session_state["post_name"]}.csv', mime='text/csv', type="primary")

                    audio_records = filtered_df[(filtered_df['audio_path'] != 'None') & (filtered_df['audio_path'].notnull())]
                    if not audio_records.empty:
                        st.markdown("### 🎧 Play Voice Notes")
                        selected_audio = st.selectbox("Select Patient Record:", audio_records['army_no'] + " - " + audio_records['timestamp'], key="audio_player_bfna")
                        if selected_audio:
                            path_to_play = audio_records[audio_records['army_no'] + " - " + audio_records['timestamp'] == selected_audio]['audio_path'].values[0]
                            if path_to_play and path_to_play != 'None':
                                st.audio(path_to_play)
                                st.markdown(f"[📥 Download Audio File]({path_to_play})")

            else:
                st.warning("No medical records found.")

        with tab_manage:
            if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                st.subheader("Manage Database Records")
                if not df.empty:
                    st.dataframe(df[['id', 'timestamp', 'name', 'army_no', 'module', 'post_name']])
                    
                    st.markdown("---")
                    st.subheader("Delete Specific Triage Record")
                    
                    hist_list = [f"ID: {row['id']} | {row['timestamp']} | {row['army_no']} | {row['module']}" for _, row in df.iterrows()]
                    selected_hist = st.selectbox("Select Record to Delete", ["-- Select --"] + hist_list, key="sel_del_hist")
                    
                    del_col1, del_col2 = st.columns(2)
                    with del_col1:
                        if st.button("🗑️ Delete Selected Record", type="secondary"):
                            if selected_hist != "-- Select --":
                                target_id = int(selected_hist.split("ID: ")[1].split(" |")[0])
                                supabase.table("patient_history").delete().eq("id", target_id).execute()
                                st.success("Record successfully deleted.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.warning("Please select a record from the dropdown first.")
                                
                    with del_col2:
                        st.warning("⚠️ This will permanently delete records.")
                        if st.button("🚨 CLEAR ALL RECORDS", type="primary"):
                            supabase.table("patient_history").delete().neq("id", 0).execute()
                            st.success("Records cleared.")
                            time.sleep(1)
                            st.rerun()
                else:
                    st.info("No records to manage.")
            else:
                st.info("⚠️ BFNAs have View-Only access to Manage Records. Please contact the RMO to delete historical entries.")

        with tab_manual:
            st.subheader("Add Record Manually")
            with st.form("manual_entry"):
                col_m1, col_m2 = st.columns(2)
                m_rank = col_m1.text_input("Rank")
                m_name = col_m2.text_input("Name")
                m_army_no = col_m1.text_input("Army / Service No. *")
                
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    m_loc = col_m2.text_input("Location")
                else:
                    m_loc = st.session_state['post_name']
                    col_m2.text_input("Location", m_loc, disabled=True)
                
                m_module = col_m1.selectbox("Module", ["Heart Disease", "Brain Stroke / HACE", "AMS", "HAPE", "Cold Injuries", "Manual/Other"])
                m_status = col_m2.text_input("Status Tier (e.g. ZONE GREEN: NORMAL)")
                m_flags = st.text_input("Clinical Flags (comma separated)")
                m_order = st.text_area("Final Order / Notes")
                
                if st.form_submit_button("Save Manual Entry", type="primary"):
                    if m_army_no.strip():
                        flags_list = [f.strip() for f in m_flags.split(',')] if m_flags else []
                        save_to_ledger(m_rank, m_name, m_army_no, m_loc, m_module, m_status, flags_list, m_order)
                        st.success("Manual entry safely added to Ledger.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Army No is required.")

    # ------------------------------------------
    # 9. PATIENT REGISTRATION
    # ------------------------------------------
    elif selected == "Patient Registration":
        st.markdown("### 📝 BATTALION PATIENT REGISTRY")
        st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
        
        tab_reg, tab_manage_reg = st.tabs(["📝 Register New Patient", "🗄️ Battalion Registry Base"])
        
        with tab_reg:
            st.markdown("Register a new patient into the Battalion database. Calculated fields (Age, HAA Days, BMI) update automatically.")
            col1, col2 = st.columns(2)
            reg_army_no = col1.text_input("Army / Service No. *")
            reg_rank = col2.text_input("Rank *")
            reg_name = col1.text_input("Name *")
            reg_coy = col2.text_input("Company / Unit *")
            
            st.markdown("---")
            reg_dob = col1.date_input("Date of Birth", min_value=datetime(1950, 1, 1).date(), max_value=datetime.now().date(), value=datetime(1995, 1, 1).date())
            age = (datetime.now().date() - reg_dob).days // 365 if reg_dob else 0
            col2.info(f"**Calculated Age:** {age} years")
            
            reg_bg = col1.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"])
            
            st.markdown("---")
            reg_ind_date = col1.date_input("Date of Induction to HAA", max_value=datetime.now().date())
            haa_days = (datetime.now().date() - reg_ind_date).days if reg_ind_date else 0
            if haa_days < 0: haa_days = 0
            col2.info(f"**Total Days in HAA:** {haa_days} days")
            
            reg_acc1 = col1.date_input("Stage 1 Acclimatization Date")
            reg_acc2 = col2.date_input("Stage 2 Acclimatization Date")
            
            post_acc2_days = (datetime.now().date() - reg_acc2).days if reg_acc2 else 0
            if post_acc2_days < 0: post_acc2_days = 0
            col2.success(f"**Days Post Stage-2 Acclimatization:** {post_acc2_days} days")
            
            st.markdown("---")
            reg_leaves = col1.number_input("Leaves Availed This Year (Days)", 0, 365, 0)
            reg_shape = col2.selectbox("SHAPE Category", ["SHAPE 1", "Low Medical Category (LMC)"])
            
            reg_weight = col1.number_input("Weight (kg)", 30.0, 150.0, 70.0)
            reg_height = col2.number_input("Height (cm)", 100.0, 250.0, 170.0)
            
            bmi = reg_weight / ((reg_height/100) ** 2) if reg_height > 0 else 0
            col2.info(f"**Calculated BMI:** {bmi:.1f}")
            
            st.markdown("---")
            reg_surg_yn = st.radio("Any past surgery or hospital admission?", ["No", "Yes"], horizontal=True)
            if reg_surg_yn == "Yes":
                reg_surg_desc = st.text_input("Provide details of surgery/admission:")
            else:
                reg_surg_desc = "None"
                
            col_pme1, col_pme2 = st.columns(2)
            reg_pme_yn = col_pme1.radio("AME/PME Done?", ["No", "Yes"], horizontal=True)
            if reg_pme_yn == "Yes":
                reg_pme_date = col_pme2.date_input("Date of AME/PME")
            else:
                reg_pme_date = "N/A"
                
            st.markdown("---")
            st.subheader("Next of Kin (NOK) Details")
            nok_name = st.text_input("NOK Name")
            nok_phone = st.text_input("NOK Phone Number")
            nok_dist = st.text_input("NOK District")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("💾 REGISTER PATIENT", type="primary"):
                if not reg_army_no.strip() or not reg_name.strip() or not reg_rank.strip() or not reg_coy.strip():
                    st.error("⚠️ Please fill all mandatory fields (Army No, Rank, Name, Company).")
                else:
                    # Bulletproof Cloud Duplicate Check (Decrypted & Case-Insensitive)
                    res_dup = supabase.table("patient_registry").select("army_no").execute()
                    existing_armies = [decrypt_data(row['army_no']).strip().upper() for row in (res_dup.data if res_dup.data else [])]
                    
                    if reg_army_no.strip().upper() in existing_armies:
                        st.error(f"⚠️ Registration Failed: Patient with Army No '{reg_army_no.upper()}' is already registered in the Battalion.")
                    else:
                        enc_army = encrypt_data(reg_army_no)
                        enc_name = encrypt_data(reg_name)
                        enc_nok_name = encrypt_data(nok_name)
                        enc_nok_phone = encrypt_data(nok_phone)
                        
                        reg_data = {
                            "army_no": enc_army, "rank": reg_rank, "name": enc_name, "company": reg_coy,
                            "dob": str(reg_dob), "blood_group": reg_bg, "induction_date": str(reg_ind_date),
                            "acclimatization_1": str(reg_acc1), "acclimatization_2": str(reg_acc2), "leaves_this_year": reg_leaves,
                            "shape_category": reg_shape, "weight": reg_weight, "height": reg_height, "surgery_history": reg_surg_desc,
                            "ame_pme_done": reg_pme_yn, "ame_pme_date": str(reg_pme_date), "nok_name": enc_nok_name, "nok_phone": enc_nok_phone,
                            "nok_district": nok_dist, "post_name": st.session_state['post_name']
                        }
                        
                        try:
                            supabase.table("patient_registry").upsert(reg_data).execute()
                            st.success(f"✅ Patient {reg_army_no.upper()} successfully registered in Battalion Database.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to register patient. Ensure internet connection. Error: {e}")

        with tab_manage_reg:
            st.subheader("Battalion Patient Database")
            
            res_manage = supabase.table("patient_registry").select("*").execute()
            df_manage = pd.DataFrame(res_manage.data)
            
            if not df_manage.empty:
                # 1. HOLD THE ORIGINAL ENCRYPTED ID FOR SUPABASE
                df_manage['raw_army_no'] = df_manage['army_no'] 
                
                df_manage['army_no'] = df_manage['army_no'].apply(decrypt_data)
                df_manage['name'] = df_manage['name'].apply(decrypt_data)
                df_manage['nok_name'] = df_manage['nok_name'].apply(decrypt_data)
                df_manage['nok_phone'] = df_manage['nok_phone'].apply(decrypt_data)
                
                search_reg = st.text_input("🔍 Search Registry by Army No or Name:")
                if search_reg:
                    df_manage = df_manage[df_manage['army_no'].str.contains(search_reg, case=False, na=False) | df_manage['name'].str.contains(search_reg, case=False, na=False)]
                
                st.dataframe(df_manage.drop(columns=['raw_army_no']), use_container_width=True, hide_index=True)
                
                # RBAC for Modifying or Deleting Patient Registry
                if st.session_state['bfna_id'] in ["MASTER_ADMIN", "RMO"]:
                    st.markdown("---")
                    st.subheader("Edit or Delete Patient Profile")
                    
                    patient_list = [f"{row['army_no']} - {row['rank']} {row['name']}" for _, row in df_manage.iterrows()]
                    selected_pt = st.selectbox("Select Patient to Modify", ["-- Select --"] + patient_list, key="rmo_pt_sel")
                    
                    if selected_pt != "-- Select --":
                        sel_army_no = selected_pt.split(" - ")[0]
                        pt_data = df_manage[df_manage['army_no'] == sel_army_no].iloc[0]
                        
                        # 2. ASSIGN THE TARGET HASH
                        target_db_army_no = pt_data['raw_army_no'] 
                        
                        with st.form("rmo_edit_pt_form"):
                            st.info("Modify any patient field below and click Update to save changes.")
                            
                            e_c1, e_c2, e_c3 = st.columns(3)
                            e_rank = e_c1.text_input("Rank", value=pt_data.get('rank', ''))
                            e_name = e_c2.text_input("Name", value=pt_data.get('name', ''))
                            e_coy = e_c3.text_input("Company/Unit", value=pt_data.get('company', ''))
                            
                            bg_opts = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"]
                            curr_bg = pt_data.get('blood_group', 'Unknown')
                            e_bg = e_c1.selectbox("Blood Group", bg_opts, index=bg_opts.index(curr_bg) if curr_bg in bg_opts else 8)
                            e_dob = e_c2.date_input("Date of Birth", value=parse_date_safe(pt_data.get('dob')))
                            
                            curr_post = pt_data.get('post_name', GLOBAL_POSTS[0])
                            e_post = e_c3.selectbox("Assigned Post", GLOBAL_POSTS, index=GLOBAL_POSTS.index(curr_post) if curr_post in GLOBAL_POSTS else 0)
                            
                            st.markdown("---")
                            i_c1, i_c2, i_c3 = st.columns(3)
                            e_ind_date = i_c1.date_input("Induction Date", value=parse_date_safe(pt_data.get('induction_date')))
                            e_acc1 = i_c2.date_input("Stage 1 Acclimatization", value=parse_date_safe(pt_data.get('acclimatization_1')))
                            e_acc2 = i_c3.date_input("Stage 2 Acclimatization", value=parse_date_safe(pt_data.get('acclimatization_2')))
                            
                            m_c1, m_c2, m_c3 = st.columns(3)
                            shape_opts = ["SHAPE 1", "Low Medical Category (LMC)"]
                            curr_shape = pt_data.get('shape_category', 'SHAPE 1')
                            e_shape = m_c1.selectbox("SHAPE Category", shape_opts, index=shape_opts.index(curr_shape) if curr_shape in shape_opts else 0)
                            e_leaves = m_c2.number_input("Leaves Availed", 0, 365, int(pt_data.get('leaves_this_year', 0)))
                            e_surg = m_c3.text_input("Surgery History", value=pt_data.get('surgery_history', 'None'))
                            
                            v_c1, v_c2, v_c3 = st.columns(3)
                            e_weight = v_c1.number_input("Weight (kg)", 30.0, 150.0, float(pt_data.get('weight', 70.0)))
                            e_height = v_c2.number_input("Height (cm)", 100.0, 250.0, float(pt_data.get('height', 170.0)))
                            
                            pme_opts = ["No", "Yes"]
                            curr_pme = pt_data.get('ame_pme_done', 'No')
                            e_pme = v_c3.radio("PME Done?", pme_opts, index=pme_opts.index(curr_pme) if curr_pme in pme_opts else 0, horizontal=True)
                            
                            e_pme_date = st.date_input("AME/PME Date", value=parse_date_safe(pt_data.get('ame_pme_date'))) if e_pme == "Yes" else "N/A"
                            
                            st.markdown("---")
                            st.markdown("**Next of Kin (NOK)**")
                            n_c1, n_c2, n_c3 = st.columns(3)
                            e_nok_name = n_c1.text_input("NOK Name", value=pt_data.get('nok_name', ''))
                            e_nok_phone = n_c2.text_input("NOK Phone", value=pt_data.get('nok_phone', ''))
                            e_nok_dist = n_c3.text_input("NOK District", value=pt_data.get('nok_district', ''))
                            
                            if st.form_submit_button("🔄 UPDATE ENTIRE PROFILE", type="primary"):
                                try:
                                    up_name = encrypt_data(e_name)
                                    up_nok_name = encrypt_data(e_nok_name)
                                    up_nok_phone = encrypt_data(e_nok_phone)
                                    
                                    update_payload = {
                                        "rank": e_rank, 
                                        "name": up_name, 
                                        "company": e_coy,
                                        "dob": str(e_dob),
                                        "blood_group": e_bg,
                                        "induction_date": str(e_ind_date),
                                        "acclimatization_1": str(e_acc1),
                                        "acclimatization_2": str(e_acc2),
                                        "leaves_this_year": e_leaves,
                                        "shape_category": e_shape,
                                        "weight": e_weight,
                                        "height": e_height,
                                        "surgery_history": e_surg,
                                        "ame_pme_done": e_pme,
                                        "ame_pme_date": str(e_pme_date) if e_pme == "Yes" else "N/A",
                                        "nok_name": up_nok_name,
                                        "nok_phone": up_nok_phone,
                                        "nok_district": e_nok_dist,
                                        "post_name": e_post
                                    }
                                    # 3. USE TARGET HASH TO UPDATE
                                    supabase.table("patient_registry").update(update_payload).eq("army_no", target_db_army_no).execute()
                                    
                                    st.success(f"Successfully updated complete record for {sel_army_no}.")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Update failed: {e}")
                                    
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🗑️ DELETE PATIENT COMPLETELY", type="secondary", key="rmo_del_pt"):
                            try:
                                # 4. USE TARGET HASH TO DELETE
                                supabase.table("patient_registry").delete().eq("army_no", target_db_army_no).execute()
                                st.success(f"Patient {sel_army_no} has been permanently deleted.")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Deletion failed: {e}")
                else:
                    st.info("⚠️ BFNAs have View-Only access to the Patient Registry. Please contact the RMO to modify or delete patient dossiers.")
            else:
                st.warning("Registry is currently empty.")

    # ------------------------------------------
    # 10. RMO DASHBOARD (ADMIN ONLY)
    # ------------------------------------------

    elif selected == "RMO Dashboard":
        if st.session_state['bfna_id'] not in ["MASTER_ADMIN", "RMO"]:
            st.error("⚠️ You do not have permission to access the RMO Dashboard.")
        else:
            tab_dash, tab_manage_pts = st.tabs(["📊 Readiness Dashboard", "🗄️ Modify/Delete Patients"])
            
            with tab_dash:
                st.markdown("### 📊 POST-WISE MEDICAL READINESS DASHBOARD")
                st.markdown("Real-time combat readiness and health surveillance overview.")
                
                # --- SAFE RMO-ONLY AUTO REFRESH ---
                col_ref, col_time = st.columns([1, 2])
                with col_ref:
                    live_sync = st.toggle("🔄 Live Auto-Refresh (30s)", value=False)
                with col_time:
                    st.caption(f"Last Synced: {datetime.now().strftime('%H:%M:%S')}")
                
                if live_sync:
                    try:
                        from streamlit_autorefresh import st_autorefresh
                        st_autorefresh(interval=30000, limit=None, key="rmo_dash_refresh")
                    except ImportError: pass
                st.markdown("---")
                # ----------------------------------

                # UNIQUE KEY ADDED HERE TO PREVENT CRASH
                view_post = st.selectbox("Select Post to View", ["All Posts"] + GLOBAL_POSTS, key="rmo_dash_post_sel_unique")
                
                res_reg = supabase.table("patient_registry").select("*").execute()
                res_hist = supabase.table("patient_history").select("*").execute()
                
                reg_df = pd.DataFrame(res_reg.data)
                hist_df = pd.DataFrame(res_hist.data)
                
                if not reg_df.empty:
                    reg_df['army_no'] = reg_df['army_no'].apply(decrypt_data)
                    reg_df['name'] = reg_df['name'].apply(decrypt_data)
                    
                    if view_post != "All Posts":
                        reg_df = reg_df[reg_df['post_name'] == view_post]
                    
                    total_troops = len(reg_df)
                    
                    # --- Rank Classification Logic ---
                    def categorize_rank(r):
                        r_str = str(r).lower()
                        if any(x in r_str for x in ['sub', 'nb', 'naib']): return 'JCOs'
                        elif any(x in r_str for x in ['lt', 'capt', 'maj', 'col', 'brig', 'gen']): return 'Officers'
                        else: return 'NCOs / ORs'
                    
                    reg_df['rank_category'] = reg_df['rank'].apply(categorize_rank)
                    off_df = reg_df[reg_df['rank_category'] == 'Officers']
                    jco_df = reg_df[reg_df['rank_category'] == 'JCOs']
                    or_df = reg_df[reg_df['rank_category'] == 'NCOs / ORs']
                    
                    # --- Medical Classification Logic ---
                    def calc_acclim(row):
                        try:
                            ind_date = datetime.strptime(row['induction_date'], '%Y-%m-%d').date()
                            days = (datetime.now().date() - ind_date).days
                            return days >= 14
                        except: return False
                    
                    acclim_mask = reg_df.apply(calc_acclim, axis=1)
                    fully_acclim_df = reg_df[acclim_mask]
                    pending_pme_df = reg_df[reg_df['ame_pme_done'] == 'No']
                    
                    under_obs = 0
                    obs_df = pd.DataFrame()
                    if not hist_df.empty:
                        hist_df['army_no'] = hist_df['army_no'].apply(decrypt_data)
                        hist_df['name'] = hist_df['name'].apply(decrypt_data) # <--- ADDED THIS LINE
                        latest_hist = hist_df.sort_values('timestamp').groupby('army_no').tail(1)
                        if view_post != "All Posts":
                            latest_hist = latest_hist[latest_hist['post_name'] == view_post]
                        obs_df = latest_hist[latest_hist['status_tier'].str.contains('AMBER|RED|YELLOW', case=False, na=False)]
                        under_obs = len(obs_df)
                    
                    # --- RENDER METRICS & LISTS ---
                    st.markdown("##### 👥 Troop Deployment & Rank Breakdown")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Registered Troops", total_troops)
                    c2.metric("Officers", len(off_df))
                    c3.metric("JCOs", len(jco_df))
                    c4.metric("NCOs / ORs", len(or_df))
                    
                    with st.expander("🔍 CLICK TO VIEW PERSONNEL LISTS (BY RANK)"):
                        t1, t2, t3, t4 = st.tabs(["All Troops", "Officers", "JCOs", "NCOs / ORs"])
                        with t1: st.dataframe(reg_df[['army_no', 'rank', 'name', 'company', 'post_name']], use_container_width=True, hide_index=True)
                        with t2: st.dataframe(off_df[['army_no', 'rank', 'name', 'company', 'post_name']], use_container_width=True, hide_index=True)
                        with t3: st.dataframe(jco_df[['army_no', 'rank', 'name', 'company', 'post_name']], use_container_width=True, hide_index=True)
                        with t4: st.dataframe(or_df[['army_no', 'rank', 'name', 'company', 'post_name']], use_container_width=True, hide_index=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##### ⚕️ Medical Readiness & Surveillance")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Fully Acclimatized (>14 Days)", len(fully_acclim_df))
                    m2.metric("Pending PME/AME", len(pending_pme_df))
                    m3.metric("Under Med Observation (Amber/Red)", under_obs)
                    
                    with st.expander("🔍 CLICK TO VIEW MEDICAL READINESS LISTS"):
                        mt1, mt2, mt3 = st.tabs(["Fully Acclimatized", "Pending PME/AME", "Under Med Observation"])
                        with mt1: st.dataframe(fully_acclim_df[['army_no', 'rank', 'name', 'induction_date', 'post_name']], use_container_width=True, hide_index=True)
                        with mt2: st.dataframe(pending_pme_df[['army_no', 'rank', 'name', 'company', 'post_name']], use_container_width=True, hide_index=True)
                        with mt3: 
                            if not obs_df.empty:
                                # Simply print the dataframe directly without merging
                                st.dataframe(obs_df[['army_no', 'rank', 'name', 'module', 'status_tier', 'timestamp']], use_container_width=True, hide_index=True)
                            else:
                                st.success("✅ No troops currently under medical observation.")
                    
                    st.markdown("---")
                    st.subheader("📋 Master Troop Health Roster")
                    
                    if not hist_df.empty:
                        hist_summary = hist_df.sort_values('timestamp').groupby('army_no').tail(1)[['army_no', 'status_tier', 'timestamp']]
                        hist_summary.rename(columns={'status_tier': 'Latest Triage', 'timestamp': 'Last Exam'}, inplace=True)
                        disp_df = pd.merge(reg_df, hist_summary, on='army_no', how='left')
                    else:
                        disp_df = reg_df.copy()
                        disp_df['Latest Triage'] = "No Data"
                        disp_df['Last Exam'] = "No Data"
                    
                    disp_df['Acclimatized'] = disp_df.apply(calc_acclim, axis=1).map({True: 'Yes', False: 'No'})
                    disp_df['Latest Triage'] = disp_df['Latest Triage'].fillna("Healthy / Unchecked")
                    
                    clean_df = disp_df[['army_no', 'rank', 'name', 'post_name', 'Acclimatized', 'ame_pme_done', 'Latest Triage']]
                    clean_df.columns = ['Army No', 'Rank', 'Name', 'Post', 'Acclimatized (>14d)', 'PME Done', 'Latest Med Status']
                    
                    def highlight_critical(row):
                        if 'RED' in str(row['Latest Med Status']):
                            return ['background-color: rgba(255, 0, 0, 0.2)'] * len(row)
                        elif 'AMBER' in str(row['Latest Med Status']):
                            return ['background-color: rgba(255, 165, 0, 0.2)'] * len(row)
                        return [''] * len(row)

                    st.dataframe(clean_df.style.apply(highlight_critical, axis=1), use_container_width=True, hide_index=True)
                else:
                    st.info("No troops registered in the Patient Registry yet.")
            
            with tab_manage_pts:
                st.subheader("Master Patient Registry Control")
                st.info("This mirrors the Battalion Registry Base. Changes here apply globally.")
                
                # Fetch fresh registry for editing
                res_rmo_reg = supabase.table("patient_registry").select("*").execute()
                df_rmo_reg = pd.DataFrame(res_rmo_reg.data)
                
                if not df_rmo_reg.empty:
                    df_rmo_reg['raw_army_no'] = df_rmo_reg['army_no'] # Hold original encrypted val
                    df_rmo_reg['army_no'] = df_rmo_reg['army_no'].apply(decrypt_data)
                    df_rmo_reg['name'] = df_rmo_reg['name'].apply(decrypt_data)
                    df_rmo_reg['nok_name'] = df_rmo_reg['nok_name'].apply(decrypt_data)
                    df_rmo_reg['nok_phone'] = df_rmo_reg['nok_phone'].apply(decrypt_data)
                    
                    patient_list = [f"{row['army_no']} - {row['rank']} {row['name']}" for _, row in df_rmo_reg.iterrows()]
                    selected_pt = st.selectbox("Select Patient to Modify", ["-- Select --"] + patient_list, key="rmo_pt_sel_manage")
                    
                    if selected_pt != "-- Select --":
                        sel_army_no = selected_pt.split(" - ")[0]
                        pt_data = df_rmo_reg[df_rmo_reg['army_no'] == sel_army_no].iloc[0]
                        target_db_army_no = pt_data['raw_army_no'] # The exact encrypted string in the cloud
                        
                        with st.form("rmo_edit_pt_form"):
                            st.info("Modify any patient field below and click Update to save changes.")
                            
                            e_c1, e_c2, e_c3 = st.columns(3)
                            e_rank = e_c1.text_input("Rank", value=pt_data.get('rank', ''))
                            e_name = e_c2.text_input("Name", value=pt_data.get('name', ''))
                            e_coy = e_c3.text_input("Company/Unit", value=pt_data.get('company', ''))
                            
                            bg_opts = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"]
                            curr_bg = pt_data.get('blood_group', 'Unknown')
                            e_bg = e_c1.selectbox("Blood Group", bg_opts, index=bg_opts.index(curr_bg) if curr_bg in bg_opts else 8)
                            e_dob = e_c2.date_input("Date of Birth", value=parse_date_safe(pt_data.get('dob')))
                            
                            curr_post = pt_data.get('post_name', GLOBAL_POSTS[0])
                            e_post = e_c3.selectbox("Assigned Post", GLOBAL_POSTS, index=GLOBAL_POSTS.index(curr_post) if curr_post in GLOBAL_POSTS else 0)
                            
                            st.markdown("---")
                            i_c1, i_c2, i_c3 = st.columns(3)
                            e_ind_date = i_c1.date_input("Induction Date", value=parse_date_safe(pt_data.get('induction_date')))
                            e_acc1 = i_c2.date_input("Stage 1 Acclimatization", value=parse_date_safe(pt_data.get('acclimatization_1')))
                            e_acc2 = i_c3.date_input("Stage 2 Acclimatization", value=parse_date_safe(pt_data.get('acclimatization_2')))
                            
                            m_c1, m_c2, m_c3 = st.columns(3)
                            shape_opts = ["SHAPE 1", "Low Medical Category (LMC)"]
                            curr_shape = pt_data.get('shape_category', 'SHAPE 1')
                            e_shape = m_c1.selectbox("SHAPE Category", shape_opts, index=shape_opts.index(curr_shape) if curr_shape in shape_opts else 0)
                            e_leaves = m_c2.number_input("Leaves Availed", 0, 365, int(pt_data.get('leaves_this_year', 0)))
                            e_surg = m_c3.text_input("Surgery History", value=pt_data.get('surgery_history', 'None'))
                            
                            v_c1, v_c2, v_c3 = st.columns(3)
                            e_weight = v_c1.number_input("Weight (kg)", 30.0, 150.0, float(pt_data.get('weight', 70.0)))
                            e_height = v_c2.number_input("Height (cm)", 100.0, 250.0, float(pt_data.get('height', 170.0)))
                            
                            pme_opts = ["No", "Yes"]
                            curr_pme = pt_data.get('ame_pme_done', 'No')
                            e_pme = v_c3.radio("PME Done?", pme_opts, index=pme_opts.index(curr_pme) if curr_pme in pme_opts else 0, horizontal=True)
                            
                            e_pme_date = st.date_input("AME/PME Date", value=parse_date_safe(pt_data.get('ame_pme_date'))) if e_pme == "Yes" else "N/A"
                            
                            st.markdown("---")
                            st.markdown("**Next of Kin (NOK)**")
                            n_c1, n_c2, n_c3 = st.columns(3)
                            e_nok_name = n_c1.text_input("NOK Name", value=pt_data.get('nok_name', ''))
                            e_nok_phone = n_c2.text_input("NOK Phone", value=pt_data.get('nok_phone', ''))
                            e_nok_dist = n_c3.text_input("NOK District", value=pt_data.get('nok_district', ''))
                            
                            if st.form_submit_button("🔄 UPDATE ENTIRE PROFILE", type="primary"):
                                try:
                                    up_name = encrypt_data(e_name)
                                    up_nok_name = encrypt_data(e_nok_name)
                                    up_nok_phone = encrypt_data(e_nok_phone)
                                    
                                    update_payload = {
                                        "rank": e_rank, 
                                        "name": up_name, 
                                        "company": e_coy,
                                        "dob": str(e_dob),
                                        "blood_group": e_bg,
                                        "induction_date": str(e_ind_date),
                                        "acclimatization_1": str(e_acc1),
                                        "acclimatization_2": str(e_acc2),
                                        "leaves_this_year": e_leaves,
                                        "shape_category": e_shape,
                                        "weight": e_weight,
                                        "height": e_height,
                                        "surgery_history": e_surg,
                                        "ame_pme_done": e_pme,
                                        "ame_pme_date": str(e_pme_date) if e_pme == "Yes" else "N/A",
                                        "nok_name": up_nok_name,
                                        "nok_phone": up_nok_phone,
                                        "nok_district": e_nok_dist,
                                        "post_name": e_post
                                    }
                                    
                                    supabase.table("patient_registry").update(update_payload).eq("army_no", target_db_army_no).execute()
                                    
                                    st.success(f"Successfully updated complete record for {sel_army_no}.")
                                    time.sleep(1)
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Update failed: {e}")
                                    
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("🗑️ DELETE PATIENT COMPLETELY", type="secondary", key="rmo_del_pt_dash"):
                            try:
                                supabase.table("patient_registry").delete().eq("army_no", target_db_army_no).execute()
                                st.success(f"Patient {sel_army_no} has been permanently deleted.")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Deletion failed: {e}")
                else:
                    st.warning("Registry is currently empty.")

    # ------------------------------------------
    # 11. ADMIN SETTINGS (RMO ONLY)
    # ------------------------------------------
    elif selected == "Admin Settings":
        if st.session_state['bfna_id'] not in ["MASTER_ADMIN", "RMO"]:
            st.error("⚠️ You do not have permission to access Admin Settings.")
        else:
            st.markdown("### ⚙️ BATTALION ADMIN SETTINGS")
            st.markdown("<hr style='margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            
            tab_rmo, tab_bfna = st.tabs(["📲 Medical Chain of Command Setup", "👥 Manage BFNA Accounts"])
            
            with tab_rmo:
                st.subheader("Emergency Contacts Configuration")
                st.markdown("Set the WhatsApp numbers that will receive casualty alerts from forward posts.")
                
                res = supabase.table("med_contacts").select("*").execute()
                existing_contacts = {row["role"]: {"rank": row.get("rank", ""), "name": row.get("name", ""), "phone": row.get("phone", "+91")} for row in (res.data if res.data else [])}

                def get_c_val(role, key, default=""): return existing_contacts.get(role, {}).get(key, default)

                with st.form("med_contacts_form"):
                    st.markdown("**1. RMO (Regimental Medical Officer)**")
                    r1, r2, r3 = st.columns(3)
                    rmo_r = r1.text_input("Rank", value=get_c_val("RMO", "rank"), key="r_rmo")
                    rmo_n = r2.text_input("Name", value=get_c_val("RMO", "name"), key="n_rmo")
                    rmo_p = r3.text_input("WhatsApp No (+91...)", value=get_c_val("RMO", "phone", "+91"), key="p_rmo")
                    
                    st.markdown("---")
                    st.markdown("**2. Medical Specialist**")
                    s1, s2, s3 = st.columns(3)
                    spec_r = s1.text_input("Rank", value=get_c_val("Medical Specialist", "rank"), key="r_spec")
                    spec_n = s2.text_input("Name", value=get_c_val("Medical Specialist", "name"), key="n_spec")
                    spec_p = s3.text_input("WhatsApp No (+91...)", value=get_c_val("Medical Specialist", "phone", "+91"), key="p_spec")

                    st.markdown("---")
                    st.markdown("**3. Col Med**")
                    c1, c2, c3 = st.columns(3)
                    col_r = c1.text_input("Rank", value=get_c_val("Col Med", "rank"), key="r_col")
                    col_n = c2.text_input("Name", value=get_c_val("Col Med", "name"), key="n_col")
                    col_p = c3.text_input("WhatsApp No (+91...)", value=get_c_val("Col Med", "phone", "+91"), key="p_col")
                    
                    if st.form_submit_button("💾 SAVE ALL CONTACTS", type="primary"):
                        try:
                            contacts_data = [
                                {"role": "RMO", "rank": rmo_r, "name": rmo_n, "phone": rmo_p},
                                {"role": "Medical Specialist", "rank": spec_r, "name": spec_n, "phone": spec_p},
                                {"role": "Col Med", "rank": col_r, "name": col_n, "phone": col_p}
                            ]
                            supabase.table("med_contacts").upsert(contacts_data).execute()
                            st.success("✅ Medical Chain of Command updated successfully.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error saving profile: {e}")

            with tab_bfna:
                st.subheader("Manage Active BFNA Credentials")
                
                res_users = supabase.table("users").select("user_id, bfna_id, post_name").neq("user_id", "admin").execute()
                df_users = pd.DataFrame(res_users.data)
                
                st.info(f"**Current Active BFNA Users:** {len(df_users)} / 15 Slots")
                
                if not df_users.empty:
                    df_users.rename(columns={'user_id': 'User ID', 'bfna_id': 'BFNA ID', 'post_name': 'Assigned Post'}, inplace=True)
                    st.dataframe(df_users, use_container_width=True, hide_index=True)
                    
                    st.markdown("**Delete BFNA User**")
                    del_u = st.text_input("Enter 'User ID' to remove access:")
                    if st.button("🗑️ Delete User"):
                        try:
                            supabase.table("users").delete().eq("user_id", del_u.strip()).execute()
                            st.success(f"User {del_u} deleted successfully.")
                            time.sleep(1)
                            st.rerun()
                        except Exception:
                            st.error("Error deleting user.")
                else:
                    st.warning("No forward BFNAs created yet.")
                
                st.markdown("---")
                st.subheader("Create New BFNA Account")
                with st.form("create_user_form"):
                    new_user = st.text_input("New User ID (e.g., bfna_charlie) *")
                    col_p1, col_p2 = st.columns(2)
                    new_pass = col_p1.text_input("Password *", type="password")
                    conf_pass = col_p2.text_input("Confirm Password *", type="password")
                    
                    new_loc = st.selectbox("Assign Post Location *", GLOBAL_POSTS)
                    
                    if st.form_submit_button("CREATE BFNA ACCOUNT", type="primary"):
                        if len(df_users) >= 15:
                            st.error("⚠️ User limit reached (15/15). Please delete old users above to free up slots.")
                        elif new_pass != conf_pass:
                            st.error("⚠️ Passwords do not match.")
                        elif not new_user.strip() or not new_pass.strip() or not new_loc.strip():
                            st.error("⚠️ All fields are required.")
                        else:
                            try:
                                bfna_id_gen = f"BFNA-{len(df_users)+1:02d}"
                                supabase.table("users").insert({
                                    "user_id": new_user.strip(),
                                    "password": new_pass,
                                    "bfna_id": bfna_id_gen,
                                    "post_name": new_loc
                                }).execute()
                                st.success(f"✅ Account `{new_user}` created successfully. Assigned to: **{new_loc}**.")
                                time.sleep(1)
                                st.rerun()
                            except Exception:
                                st.error(f"⚠️ User ID `{new_user}` already exists or there is a database error.")

if __name__ == "__main__":
    if not st.session_state.get('logged_in', False):
        login_page()
    else:
        main_app()
