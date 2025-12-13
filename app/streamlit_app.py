# app.py
import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
#import sys
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🚦 NYC Collision Risk Predictor",
    page_icon="🚦",
    layout="wide"
)

# Description
st.title("🚦 NYC Traffic Collision Severity Predictor")
st.markdown("### Vision Zero Decision Support Tool")
st.caption("Predicts the likelihood of a collision resulting in **Killed or Seriously Injured (KSI)** outcomes")

# Load model and metadata
@st.cache_resource
def load_model():
    return joblib.load("/Users/Marcy_Student/Desktop/Marcy_Projects/NYC-TrafficSafety-Modeling/app/final_model.pkl")

@st.cache_data
def load_meta():
    return json.load(open("/Users/Marcy_Student/Desktop/Marcy_Projects/NYC-TrafficSafety-Modeling/app/ksi_model_meta.json"))

try:
    model = load_model()
    meta = load_meta()
except Exception as e:
    st.error(f"Error loading model or metadata: {e}")
    st.stop()

# Display model performance
st.sidebar.markdown("### 📊 Model Performance")
st.sidebar.metric("Recall (Catch KSI)", f"{meta['test_recall']*100:.1f}%")
#st.sidebar.metric("Precision", f"{meta['test_precision']*100:.1f}%")
#st.sidebar.metric("F1-Score", f"{meta['test_f1']:.3f}")
st.sidebar.caption(f"Trained on {meta['total_crashes']:,} NYC collisions") #pull from the json file

st.sidebar.markdown("---")

# Sidebar inputs
st.sidebar.header("🎛️ Input Collision Details")

# Time inputs
hour = st.sidebar.slider("Hour of Day", 0, 23, 12, help="0 = midnight, 12 = noon, 23 = 11pm")

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    options=[0, 1, 2, 3, 4, 5, 6],
    format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
    index=0
)

is_weekend = 1 if day_of_week >= 5 else 0

month = st.sidebar.slider("Month", 1, 12, 6, format="%d", help="1 = January, 12 = December")

hour_category = st.sidebar.selectbox(
    "Time Category",
    options=["Morning_Rush", "Midday", "Evening_Rush", "Night", "Late_Night"],
    index=2
)

season = st.sidebar.selectbox(
    "Season",
    options=["Winter", "Spring", "Summer", "Fall"],
    index=2
)

# Location
borough = st.sidebar.selectbox(
    "Borough",
    options=["Brooklyn", "Manhattan", "Queens", "Bronx", "Staten Island"],
    index=0
)

# Incident characteristics
num_vehicles = st.sidebar.number_input(
    "Number of Vehicles Involved",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

pedestrian_involved = st.sidebar.checkbox("Pedestrian Involved", value=False)
cyclist_involved = st.sidebar.checkbox("Cyclist Involved", value=False)
high_risk_factor = st.sidebar.checkbox(
    "High Risk Factor",
    value=False,
    help="Alcohol, drugs, speeding, or distraction involved"
)

# Predict button
predict_button = st.sidebar.button("🔮 Predict Risk", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Vision Zero Goal:** Eliminate traffic deaths and serious injuries in NYC")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Input Summary")
    
    input_data = pd.DataFrame([{
        "Borough": borough,
        #"Hour": hour,
        "Day": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week],
        "Time Period": hour_category,
        "Season": season,
        "Vehicles": num_vehicles,
        "Pedestrian": "Yes" if pedestrian_involved else "No",
        "Cyclist": "Yes" if cyclist_involved else "No",
        "High Risk": "Yes" if high_risk_factor else "No"
    }]).T
    
    input_data.columns = ["Value"]
    st.dataframe(input_data, use_container_width=True)

with col2:
    st.subheader("🎯 KSI Risk Prediction")
    
    if predict_button:
        try:
            input_dict = {
                'hour': hour,
                'day_of_week': day_of_week,
                'month': month,
                'num_vehicles': num_vehicles,
                'is_weekend': is_weekend,
                'pedestrian_involved': int(pedestrian_involved),
                'cyclist_involved': int(cyclist_involved),
                'high_risk': int(high_risk_factor),
                'borough': borough, 
                'hour_category': hour_category,
                'season': season
            }
            
            X_input = pd.DataFrame([input_dict])
            
            # Make prediction
            prediction = model.predict(X_input)[0]
            probability = model.predict_proba(X_input)[0]
            
            ksi_prob = probability[1] * 100
            no_ksi_prob = probability[0] * 100
            
            # Display result
            if prediction == 1:
                st.error("⚠️ **HIGH RISK: KSI Likely**")
                st.markdown(f"### {ksi_prob:.1f}% probability of severe outcome")
            else:
                st.success("✅ **LOW RISK: Minor collision likely**")
                st.markdown(f"### {no_ksi_prob:.1f}% probability of minor outcome")
            
            # Probability bar
            st.progress(ksi_prob / 100)
            
            st.caption("**Probabilities:**")
            col_a, col_b = st.columns(2)
            col_a.metric("No KSI", f"{no_ksi_prob:.1f}%")
            col_b.metric("KSI", f"{ksi_prob:.1f}%")
            
            st.markdown("---")
            
            # Interpretation for stakeholder
            st.markdown("### 📢 Interpretation for Vision Zero")
            
            if prediction == 1:
                st.markdown("""
                **This collision scenario has a HIGH risk of severe injury or death.**
                
                **Recommended Actions:**
                - 🚨 Prioritize rapid emergency response
                - 🏥 Dispatch advanced medical support
                - 🚔 Consider enhanced enforcement in this area/time
                - 📊 Log for hotspot analysis
                """)
                
                # Risk factors contributing
                risk_factors = []
                if pedestrian_involved:
                    risk_factors.append("Pedestrian involvement (major risk factor)")
                if cyclist_involved:
                    risk_factors.append("Cyclist involvement (vulnerable road user)")
                if high_risk_factor:
                    risk_factors.append("High-risk behavior detected (alcohol/speed/distraction)")
                if hour_category in ["Late_Night", "Night"]:
                    risk_factors.append(f"Nighttime collision ({hour_category})")
                if num_vehicles >= 3:
                    risk_factors.append(f"Multi-vehicle crash ({num_vehicles} vehicles)")
                
                if risk_factors:
                    st.markdown("**Key Risk Factors:**")
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
            else:
                st.markdown("""
                **This collision scenario has a LOW risk of severe outcomes.**
                
                **Typical characteristics of minor collisions:**
                - Standard emergency response appropriate
                - Monitor for any escalation
                - Continue routine safety protocols
                """)
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.error("Please check that all required features match the training data.")
    
    else:
        st.info("👈 Adjust collision details in the sidebar and click **Predict Risk**")

# Divider
st.markdown("---")

# Key statistics
st.subheader("📈 Model Performance Summary")

perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.metric(
        "Recall",
        f"{meta['test_recall']*100:.1f}%",
        help="% of actual KSI cases caught by model"
    )

with perf_col2:
    st.metric(
        "Precision",
        f"{meta['test_precision']*100:.1f}%",
        help="% of KSI predictions that are correct"
    )

st.markdown("---")

# Why these metrics matter
with st.expander("ℹ️ Understanding Model Metrics"):
    st.markdown("""
    ### Why Recall Matters Most for Vision Zero
    
    **Recall (57.5%)** tells us: "Of all severe crashes, we catch 57.5%"
    - This is our most important metric
    - For life-safety applications, catching severe cases is critical
    - Missing a KSI event has severe consequences
    
    **Precision (11.6%)** tells us: "When we predict high risk, we're right 11.6% of the time"
    - Lower than recall, but acceptable for this use case
    - False alarms (predicting high risk when it's actually low) waste resources
    - But they're far better than missing a severe crash because the Vision Zero's perspective is to save lives.
    """)

# Footer
st.markdown("---")
st.caption("🚦 NYC Vision Zero Traffic Safety Decision Support System | Model: Random Forest Classifier")
st.caption(f"📈 Model achieves {meta['test_recall']*100:.1f}% recall in identifying severe collisions")
