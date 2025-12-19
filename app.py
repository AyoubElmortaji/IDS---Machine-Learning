import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ======================== CONFIG ========================
st.set_page_config(page_title="IDS-ENSAM", layout="wide", page_icon="shield")

# ======================== COULEURS ========================
BLUE = "#003087"
RED = "#D91E18"
BLACK = "#0F1621"
WHITE = "#FFFFFF"

# ======================== CSS PROPRE & PARFAIT ========================
st.markdown(f"""
<style>
    .main {{
        background-color: {BLACK};
        color: {WHITE};
        padding: 2rem;
    }}
    .main-header {{
        font-size: 3.8rem;
        font-weight: 800;
        text-align: center;
        color: {BLUE};
        margin: 20px 0 8px 0;
        letter-spacing: 1px;
    }}
    .subtitle {{
        text-align: center;
        color: #bbbbbb;
        font-size: 1.1rem;
        font-weight: 300;
        margin-bottom: 40px;
    }}
    .sidebar .sidebar-content {{
        background: #161b22;
        border-right: 4px solid {BLUE};
    }}

    .metric-card {{
        background: #1a1f2e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        border-left: 4px solid {BLUE};
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: visible;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        border-left-color: {RED};
        box-shadow: 0 10px 25px rgba(217,30,24,0.2);
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: 700;
        margin: 8px 0;
        color: white;
    }}
    .metric-label {{
        font-size: 0.95rem;
        color: #cccccc;
        font-weight: 500;
    }}

    .tooltip-box {{
        position: absolute;
        top: -110px;
        left: 50%;
        transform: translateX(-50%);
        background: #0f1621;
        color: white;
        padding: 12px 16px;
        border-radius: 8px;
        border: 2px solid {RED};
        font-size: 0.9rem;
        width: 280px;
        text-align: center;
        opacity: 0;
        visibility: hidden;
        transition: opacity 0.3s ease;
        z-index: 1000;
        box-shadow: 0 8px 20px rgba(0,0,0,0.6);
    }}
    .tooltip-box::after {{
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -8px;
        border: 8px solid transparent;
        border-top-color: {RED};
    }}
    .metric-card:hover .tooltip-box {{
        opacity: 1;
        visibility: visible;
    }}

    .stButton > button {{
        background: {BLUE};
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        height: 48px;
        font-size: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }}
    .stButton > button:hover {{
        background: {RED};
        transform: translateY(-2px);
    }}

    h2, h3, h4 {{
        color: {BLUE};
        font-weight: 600;
    }}
</style>
""", unsafe_allow_html=True)

# ======================== SIDEBAR ========================
with st.sidebar:
    st.image("logo ensam fr.png", width=240)
    st.markdown(f"<h2 style='color:{BLUE}; text-align:center; margin-top:-10px;'>IDS-ENSAM</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#ccc;'><strong>Détection d'Intrusion IA</strong></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Mode d'Analyse")
    analysis_mode = st.radio("Choisir le mode", ["Un seul modèle", "Comparer plusieurs modèles", "Tous les modèles"])

# ======================== HEADER ========================
st.markdown(f"<h1 class='main-header'>IDS-ENSAM</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Système Intelligent de Détection d'Intrusion – ENSAM Casablanca</p>", unsafe_allow_html=True)
st.markdown("---")

# ======================== UPLOAD ========================
uploaded_file = st.file_uploader("Importer Votre Dataset", type=["csv", "txt"])


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, header=None)
        st.success("Dataset chargé avec succès !")

        @st.cache_data
        def preprocess(df_raw):
            cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent",
                    "hot","num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root",
                    "num_file_creations","num_shells","num_access_files","num_outbound_cmds","is_host_login",
                    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
                    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

            if df_raw.shape[1] == len(cols):
                df_raw.columns = cols

            df_raw['label'] = df_raw['label'].apply(lambda x: 0 if str(x).strip() == 'normal' else 1)
            df_encoded = pd.get_dummies(df_raw, columns=['protocol_type', 'service', 'flag'], drop_first=True)
            X = df_encoded.drop(['label', 'difficulty'], axis=1, errors='ignore')
            y = df_encoded['label']
            scaler = RobustScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            return X_scaled, y

        X, y = preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # ======================== AFFICHAGE FINAL ========================
        def display_results(acc, prec, rec, f1, auc, safe, attack, y_pred, y_prob, model_name):
            st.markdown("### Performance du Modèle")

            # 5 métriques
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label' style='color:blue'>Accuracy</div>
                    <div class='metric-value'>{acc:.4f}</div>
                    <div class='tooltip-box'>Proportion totale de prédictions correctes</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label' style='color:white'>Precision</div>
                    <div class='metric-value' style='color:#00E0FF'>{prec:.4f}</div>
                    <div class='tooltip-box'>Parmi les alertes, combien étaient vraies ?</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label' style='color:{RED}'>Recall</div>
                    <div class='metric-value' style='color:{RED}'>{rec:.4f}</div>
                    <div class='tooltip-box'>Parmi les attaques, combien détectées ?</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label' style='color:#00E0FF'>F1-Score</div>
                    <div class='metric-value' style='color:#00E0FF'>{f1:.4f}</div>
                    <div class='tooltip-box'>Équilibre Precision / Recall</div>
                </div>
                """, unsafe_allow_html=True)
            with col5:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label' style='color:#FFD700'>AUC-ROC</div>
                    <div class='metric-value' style='color:#FFD700'>{auc:.4f}</div>
                    <div class='tooltip-box'>Qualité globale de séparation (1 = parfait)</div>
                </div>
                """, unsafe_allow_html=True)

            # Espacement propre
            st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

            # Paquets + Matrice
            col_left, col_right = st.columns([1, 1.3])
            with col_left:
                st.markdown(f"""
                <div style="
                    background: #1a1f2e;
                    border-radius: 12px;
                    padding: 30px 20px;
                    text-align: center;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.4);
                    border-left: 4px solid {BLUE};
                    height: 180px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                ">
                    <h3 style="color: white; margin: 0 0 20px 0; font-size: 1rem; font-weight: 600;">Paquets Analysés</h3>
                    <div style="font-size: 1rem; font-weight: 700; color: #4CAF50; margin: 8px 0;">Sûrs : {safe:,}</div>
                    <div style="font-size: 1rem; font-weight: 700; color: {RED}; margin: 8px 0;">Attaques : {attack:,}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_right:
                st.markdown("#### Matrice de Confusion")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(7, 5.5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                            xticklabels=['Normal', 'Attaque'], yticklabels=['Normal', 'Attaque'],
                            cbar=False, linewidths=2, linecolor='BLACK',
                            annot_kws={"size": 18, "color": "BLACK"})
                ax.set_xlabel("Prédit", color=BLACK)
                ax.set_ylabel("Réel", color=BLACK)
                ax.tick_params(colors=BLACK)
                ax.set_facecolor("#1a1f2e")
                for spine in ax.spines.values():
                    spine.set_color(BLACK)
                st.pyplot(fig)
                plt.close(fig)

            # Courbe ROC
            st.markdown("#### Courbe ROC")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(fpr, tpr, color=BLUE, linewidth=3, label=f'AUC = {auc:.4f}')
            ax.plot([0,1], [0,1], 'gray', linestyle='--', linewidth=2)
            ax.set_facecolor("#1a1f2e")
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Faux Positifs', color=WHITE)
            ax.set_ylabel('Vrais Positifs', color=WHITE)
            ax.tick_params(colors=WHITE)
            ax.legend(fontsize=14, loc='lower right', facecolor='#1a1f2e', edgecolor=RED)
            for spine in ax.spines.values():
                spine.set_color(WHITE)
            st.pyplot(fig)
            plt.close(fig)

        # ======================== LANCEMENT ========================
        if analysis_mode == "Un seul modèle":
            model_choice = st.sidebar.selectbox("Modèle", ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"])
            if st.sidebar.button("Lancer l'analyse", type="primary"):
                with st.spinner("Entraînement..."):
                    model_dict = {
                        "XGBoost": XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42, eval_metric='logloss'),
                        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1)
                    }
                    model = model_dict[model_choice]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    auc = roc_auc_score(y_test, y_prob)
                    safe = (y_pred == 0).sum()
                    attack = (y_pred == 1).sum()
                    display_results(acc, prec, rec, f1, auc, safe, attack, y_pred, y_prob, model_choice)

        else:
            if analysis_mode == "Comparer plusieurs modèles":
                selected = st.sidebar.multiselect("Modèles", ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"], default=["XGBoost", "Random Forest"])
            else:
                selected = ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression"]

            if st.sidebar.button("Lancer la Comparaison", type="primary"):
                with st.spinner("Comparaison..."):
                    models_dict = {
                        "XGBoost": XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42, eval_metric='logloss'),
                        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
                        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
                        "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1)
                    }
                    best_auc = 0
                    best_model_name = None
                    best_pred = None
                    best_prob = None

                    for name in selected:
                        model = models_dict[name]
                        model.fit(X_train, y_train)
                        y_prob = model.predict_proba(X_test)[:, 1]
                        auc = roc_auc_score(y_test, y_prob)
                        if auc > best_auc:
                            best_auc = auc
                            best_model_name = name
                            best_pred = model.predict(X_test)
                            best_prob = y_prob

                    model = models_dict[best_model_name]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    acc = accuracy_score(y_test, y_pred)
                    prec = precision_score(y_test, y_pred)
                    rec = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    safe = (y_pred == 0).sum()
                    attack = (y_pred == 1).sum()
                    st.success(f"Meilleur modèle : **{best_model_name}**")
                    display_results(acc, prec, rec, f1, best_auc, safe, attack, y_pred, y_prob, best_model_name)

    except Exception as e:
        st.error(f"Erreur : {e}")

else:
    st.info("Importez votre dataset pour commencer.")
    st.markdown(f"<div style='text-align:center; padding:120px; background:{BLACK}; border-radius:20px; border: 3px dashed {WHITE};'><h2 style='color:{WHITE};'>IDS-ENSAM</h2><h3 style='color:white'>Prêt à analyser les intrusions ?</h3></div>", unsafe_allow_html=True)

# ======================== FOOTER ========================
st.markdown("---")
st.markdown("<p style='text-align:center; color:#888; font-size:0.95rem;'>© 2025 ENSAM Casablanca – Réalisé par ELMORTAJI • BELBARAKA • KEBIYER</p>", unsafe_allow_html=True)