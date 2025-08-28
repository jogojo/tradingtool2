import streamlit as st
import pandas as pd
from pathlib import Path
from src.tools.audit_bronze_utc import audit_bronze


def render():
    """Affiche la page Audit Bronze UTC"""
    st.header("🔍 Audit Bronze: Timestamps en UTC")
    st.write("Vérifie que tous les fichiers Parquet de la Bronze ont une colonne 'timestamp' tz-aware en UTC.")

    base_dir = st.text_input("Répertoire des données", value="./data", key="audit_base")
    verbose = st.checkbox("Mode verbeux (afficher chaque fichier)", value=False, key="audit_verbose")

    if st.button("Lancer l'audit", type="primary"):
        try:
            with st.spinner("Audit en cours..."):
                results, issues_summary = audit_bronze(Path(base_dir))

            total = len(results)
            ok = sum(1 for r in results if r.get("arrow_ok") and not r.get("issues"))
            to_fix = total - ok

            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Fichiers scannés", total)
            with colB:
                st.metric("Fichiers OK", ok)
            with colC:
                st.metric("À corriger", to_fix)

            if issues_summary:
                st.subheader("Problèmes détectés")
                st.dataframe(
                    pd.DataFrame(
                        [{"issue": k, "count": v} for k, v in sorted(issues_summary.items(), key=lambda kv: (-kv[1], kv[0]))]
                    ),
                    use_container_width=True
                )

            if verbose:
                st.subheader("Détails par fichier")
                st.dataframe(pd.DataFrame(results), use_container_width=True)

            st.success("Audit terminé")
        except Exception as e:
            st.error(f"Erreur durant l'audit: {e}")
