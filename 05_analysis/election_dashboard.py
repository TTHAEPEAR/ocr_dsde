import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Setup paths
BASE_DIR = Path(__file__).parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

# Page config
st.set_page_config(page_title="Thailand Election 2026 Dashboard", page_icon="🇹🇭", layout="wide")

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #3b82f6;
        font-weight: 800;
        margin-bottom: -10px;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🇹🇭 Thailand Election 2026 Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated OCR Data Extraction & Analysis</p>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    csv_path = CLEANED_DIR / "election_results_cleaned.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)

df = load_data()

if df is None:
    st.error("❌ No cleaned data found. Please run the OCR pipeline first.")
    st.stop()

# Key Metrics
total_stations = df['station_id'].nunique() if 'station_id' in df.columns else len(df)
total_ballots = df['total_ballots'].sum() if 'total_ballots' in df.columns else 0
valid_ballots = df['good_ballots'].sum() if 'good_ballots' in df.columns else 0
invalid_ballots = df['bad_ballots'].sum() if 'bad_ballots' in df.columns else 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Documents Processed", f"{total_stations:,}")
with col2:
    st.metric("Total Ballots Counted", f"{total_ballots:,.0f}")
with col3:
    st.metric("Valid Ballots", f"{valid_ballots:,.0f}")
with col4:
    st.metric("Invalid Ballots", f"{invalid_ballots:,.0f}")

st.markdown("---")

# Party Performance
st.subheader("📊 Party Performance Overview")
vote_cols = [c for c in df.columns if c.endswith("_votes")]
if vote_cols:
    party_totals = df[vote_cols].sum().reset_index()
    party_totals.columns = ['Party/Candidate', 'Total Votes']
    # Clean column names for display
    party_totals['Party/Candidate'] = party_totals['Party/Candidate'].str.replace('_votes', '').str.replace('candidate_', 'พรรคเบอร์ ').str.replace('party_', 'พรรคเบอร์ ')
    
    party_totals = party_totals.sort_values('Total Votes', ascending=False)

    fig = px.bar(party_totals, x='Total Votes', y='Party/Candidate', orientation='h',
                 color='Total Votes', color_continuous_scale='Blues',
                 title="Top Parties by Total OCR Vote Count")
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=500)
    st.plotly_chart(fig, width='stretch')

# Raw Data Explorer
st.subheader("🗂️ Cleaned Dataset Explorer")
st.dataframe(df, width='stretch')

st.markdown("---")
st.caption("Powered by Antigravity AI OCR Pipeline & Streamlit")
