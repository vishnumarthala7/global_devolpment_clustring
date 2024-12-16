import streamlit as st
import joblib
import numpy as np

# Load your pre-trained DBSCAN model
model = joblib.load('dbscan_model.pkl')

# List of countries for the country selection dropdown
countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", 
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", 
    "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", 
    "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", 
    "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)", 
    "Congo (Democratic Republic)", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czech Republic", "Denmark", "Djibouti", 
    "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", 
    "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", 
    "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", 
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", 
    "Kiribati", "Korea, North", "Korea, South", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", 
    "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", 
    "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", 
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar (Burma)", "Namibia", "Nauru", "Nepal", "Netherlands", 
    "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Panama", 
    "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda", 
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", 
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", 
    "Somalia", "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", 
    "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", 
    "Turkey", "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", 
    "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
]

# Sidebar: Add country selection box
country_name = st.sidebar.selectbox("Select a Country", countries)

# Sidebar: Add feature inputs
st.sidebar.header("Enter World Development Factors")

# Model feature names based on the columns you're using
feature_names = [
    "Business_Tax_Rate", "CO2_Emissions", "Energy_Usage", "GDP", "Health_Exp_GDP", 
    "Hours_to_do_Tax", "Infant_Mortality_Rate", "Internet_Usage", "Lending_Interest", 
    "Life_Expectancy_Female", "Life_Expectancy_Male", "Mobile_Phone_Usage", "Population_15_to_64", 
    "Population_Total", "Population_Urban", "Tourism_Inbound", "Tourism_Outbound"
]

# Create a list to store feature inputs
features = []

# Collect input values for each feature
for feature in feature_names:
    min_value = 0.0
    max_value = 10000000000000.0 if "GDP" in feature or "Population" in feature or "Tourism" in feature or "Energy_Usage" in feature else 10000000000000.0
    value = st.sidebar.number_input(feature, min_value=min_value, max_value=max_value, value=0.0)
    features.append(value)

# Convert feature list to a numpy array
features = np.array(features).reshape(1, -1)

# Button to trigger prediction and insights
if st.button("Predict and Provide Insights"):
    # Fit the model and predict the cluster label
    labels = model.fit_predict(features)
    cluster = labels[0]  # Access the first predicted label

    # Define cluster insights based on the predicted label
    cluster_insights = {
        0: {"desc": "Emerging Economic Country",
            "suggestions": [
                "Maintain growth in innovation",
                "Continue investing in sustainability",
                "Invest in renewable energy sources",
                "Improve healthcare access"
            ]},
        1: {"desc": "Developing Economies",
            "suggestions": [
                "Improve healthcare access",
                "Increase foreign investment",
                "Expand social welfare programs",
                "Improve infrastructure"
            ]},
        2: {"desc": "Developed Country",
            "suggestions": [
                "Focus on basic infrastructure development",
                "Boost education and literacy",
                "Increase international aid and investment"
            ]},
        -1: {"desc": "Moderate Developed Country",
              "suggestions": [
                  "Focus on infrastructure and development",
                  "Boost education and literacy",
                  "Increase international aid and investment"
              ]}
    }

    # Display the prediction and insights
    st.info(f"Cluster Description: {country_name} {cluster_insights[cluster]['desc']}")
    
    st.subheader("Suggestions for Improvement:")
    for suggestion in cluster_insights[cluster]['suggestions']:
        st.write(f"- {suggestion}")
