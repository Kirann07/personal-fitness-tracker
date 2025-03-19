import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# 🖼️ Customizing Streamlit Page
st.set_page_config(
    page_title="Personal Fitness Tracker",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Apply a custom Streamlit style
st.markdown(
    """
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 10px; }
    .stProgress .st-bo { background-color: green; }
    </style>
    """,
    unsafe_allow_html=True
)

# 🏋️‍♂️ Add a logo and app title
st.image("https://i.pinimg.com/originals/96/16/16/961616be741eac249d85c28bbcfd9742.jpg", width=100)
st.title("🏋️Personal Fitness Tracker")
st.subheader("🔥Track your fitness progress and predict calories burned!")

# 🎚️ Sidebar for user input
st.sidebar.header("⚙️ Customize Your Input:")
def user_input_features():
    age = st.sidebar.slider("🎂Age", 10, 100, 30)
    bmi = st.sidebar.slider("📊BMI", 15, 40, 20)
    duration = st.sidebar.slider("🕛 Duration(min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("❤️Heart Rate", 60, 150, 80)
    body_temp = st.sidebar.slider("🌡️Body Temperature(°C)", 36, 42, 37)
    gender = st.sidebar.radio("🧍Gender", ("Male", "Female"))
    
    gender_encoded = 1 if gender == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender_encoded
    }
    
    return pd.DataFrame(data_model, index=[0])


df = user_input_features()

# 🎯 Goal Settings (Keep in Sidebar)
st.sidebar.header("🎯 Goal Settings")
goal_weight = st.sidebar.number_input("⚖️ Target Weight (kg)", min_value=30, max_value=150, value=70)
goal_workout = st.sidebar.slider("🔥 Weekly Workout Goal (minutes)", min_value=0, max_value=600, value=150)
goal_water = st.sidebar.slider("💧 Daily Water Intake Goal (liters)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)


st.write("---")
st.header("📊 Your Input Parameters")
st.dataframe(df.style.highlight_max(axis=0))

st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# 🏋️‍♂️ Load dataset (Assume files are in the same directory)
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# ✅ FIX: Convert 'Gender' column to numerical values (1 = Male, 0 = Female)
exercise_df["Gender"] = exercise_df["Gender"].map({"male": 1, "female": 0})

# Add BMI column to dataset
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# Prepare features and target variable
X = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]]
y = exercise_df["Calories"]

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔄 Data Preprocessing
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# 🔀 Splitting Data
X = exercise_df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]]
y = exercise_df["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🎯 Train the Model
model = RandomForestRegressor(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# 📈 Make Prediction
df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(df)

st.write("---")
st.header("🔮 Predicted Calories Burned:")
st.metric(label="🔥 Estimated Calories Burned", value=f"{round(prediction[0], 2)} kcal")

# ✅ Display Selected Goals in Sidebar
st.write("---")
st.header("🎯 Your Goal Settings")
st.write(f"📌 **Target Weight:** {goal_weight} kg")
st.write(f"📌 **Weekly Workout Goal:** {goal_workout} min")
st.write(f"📌 **Daily Water Intake Goal:** {goal_water} L")

# 📊 Similar Workouts (Show in Dataframe)
st.write("---")
st.header("📋 Workouts with Similar Calories Burned")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.dataframe(similar_data.sample(5))

# 📊 Visualization - BMI vs Calories
st.write("---")
st.header("📊 BMI vs. Calories Burned")

# 🔹 Allow user to select the number of data points
num_points = st.slider("🔢 Select Number of Data Points:", 
                       min_value=10, max_value=len(exercise_df), value=50, step=10)

# 🔹 Filter dataset based on selection
sample_data = exercise_df.sample(n=num_points, random_state=42)

# 🔹 Create the scatter plot
fig, ax = plt.subplots(figsize=(6, 4))
sns.scatterplot(data=sample_data, x="BMI", y="Calories", hue="Gender")

# 🔹 Add user’s BMI as a red vertical line
plt.axvline(df["BMI"][0], color="red", linestyle="--", label="Your BMI")
plt.legend()

# 🔹 Display the plot
st.pyplot(fig)

# 📊 Age comparison
st.write("---")
st.header("📈 Age vs. Workout Duration")
fig, ax = plt.subplots(figsize=(6,4))
sns.lineplot(data=exercise_df, x="Age", y="Duration", marker="o", color="blue")
plt.axvline(df["Age"][0], color="red", linestyle="--", label="Your Age")
plt.legend()
st.pyplot(fig)

# 🎖️ Workout Streak & Achievements
st.write("---")
st.header("🏆 Workout Streak & Achievements")

streak = st.number_input("🔥 Days You Worked Out This Week", 0, 7, 3)
if streak == 7:
    st.success("🎉 Perfect Streak! Keep Going!")
elif streak >= 5:
    st.info("💪 Almost There! Stay Consistent!")
else:
    st.warning("🏃‍♂️ Try More Workouts!")

# 🏆 Fitness Challenges
st.write("---")
st.header("🏋️ Fitness Challenges")
challenge = st.selectbox("🎯 Pick a Challenge:", ["7-Day Cardio", "10K Steps Daily", "30-Day Strength", "No Challenge"])
if challenge != "No Challenge":
    st.info(f"🔥 You selected **{challenge}**! Stay on track and complete it!")

# # 📩 Reminders for Goals
# st.write("---")
# st.header("📩 Goal Reminder System")

# if df["Duration"][0] < goal_workout / 7:
#     st.warning("⚠️ You haven't met your **daily workout goal**. Try a quick session!")

# if goal_water > 2 and goal_water < 2.5:
#     st.success("✅ Great! You're drinking enough water!")
# elif goal_water < 2:
#     st.warning("💧 Drink more water for better health!")


# 🎉 Motivational Message
st.write("---")
st.header("💡 Fitness Motivation")
if prediction > 500:
    st.success("🔥 Great job! You burned a lot of calories today!")
elif prediction > 300:
    st.info("💪 Keep pushing! You’re making progress.")
else:
    st.warning("🏃 Try to be more active today!")

st.write("---")
st.success("✅ Stay Fit & Keep Exercising!")
