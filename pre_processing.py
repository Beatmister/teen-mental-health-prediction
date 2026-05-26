import pandas as pd

def transform(raw_data):
    ### Delete Missing Data
    raw_data = raw_data.dropna()
    
    
    ### Feature Creation
    raw_data["gender"] = raw_data["gender"].replace({"male":0, "female":1})
    
    raw_data["social_interaction_level"] = raw_data["social_interaction_level"].replace({"low":0, "medium":1, "high":2})
    
    raw_data = pd.get_dummies(raw_data, columns=["platform_usage"], drop_first=True)
    
    ### Feature Selection based on Correlation Heapmap
    raw_data = raw_data.drop(columns='age')
    raw_data = raw_data.drop(columns='addiction_level')
    raw_data = raw_data.drop(columns='screen_time_before_sleep')
    raw_data = raw_data.drop(columns='physical_activity')
    raw_data = raw_data.drop(columns='academic_performance')
    
    return raw_data

if __name__ == "__main__":
    raw_data = pd.read_csv("Teen_Mental_Health_Dataset.csv") 
    processd_data = transform(raw_data)
    print(processd_data.head())