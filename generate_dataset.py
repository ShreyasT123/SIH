import numpy as np
import pandas as pd

# Parameters for human-like behavior
human_params = {
    'mouse_speed_mean': 2.0,
    'mouse_speed_stddev': 1.0,
    'click_frequency': 10,
    'click_interval_mean': 0.5,
    'click_interval_stddev': 0.2,
    'scroll_speed_mean': 1.5,
    'scroll_speed_stddev': 0.5,
    'typing_speed_mean': 0.5,
    'typing_speed_stddev': 0.1,
    'session_duration': 300,
    'activity_gap_mean': 0.8,
    'activity_gap_stddev': 0.3
}

# Parameters for bot-like behavior
bot_params = {
    'mouse_speed_mean': 1.5,
    'mouse_speed_stddev': 0.1,
    'click_frequency': 15,
    'click_interval_mean': 0.2,
    'click_interval_stddev': 0.05,
    'scroll_speed_mean': 1.0,
    'scroll_speed_stddev': 0.1,
    'typing_speed_mean': 1.0,
    'typing_speed_stddev': 0.01,
    'session_duration': 300,
    'activity_gap_mean': 0.1,
    'activity_gap_stddev': 0.01
}

# Function to generate synthetic data
def generate_synthetic_data(num_samples):
    data = []
    
    for i in range(num_samples):
        user_type = np.random.choice(['human', 'bot'])
        params = human_params if user_type == 'human' else bot_params
        
        entry = {
            'interaction_id': i,
            'user_type': 0 if user_type == 'human' else 1,
            'mouse_speed_mean': np.random.normal(params['mouse_speed_mean'], params['mouse_speed_stddev']),
            'mouse_speed_stddev': params['mouse_speed_stddev'],
            'click_frequency': params['click_frequency'],
            'click_interval_mean': np.random.normal(params['click_interval_mean'], params['click_interval_stddev']),
            'click_interval_stddev': params['click_interval_stddev'],
            'scroll_speed_mean': np.random.normal(params['scroll_speed_mean'], params['scroll_speed_stddev']),
            'scroll_speed_stddev': params['scroll_speed_stddev'],
            'typing_speed_mean': np.random.normal(params['typing_speed_mean'], params['typing_speed_stddev']),
            'typing_speed_stddev': params['typing_speed_stddev'],
            'session_duration': params['session_duration'],
            'activity_gap_mean': np.random.normal(params['activity_gap_mean'], params['activity_gap_stddev']),
            'activity_gap_stddev': params['activity_gap_stddev']
        }
        
        data.append(entry)
    
    return pd.DataFrame(data)

# Generate and save synthetic dataset
df_synthetic = generate_synthetic_data(num_samples=1000)
df_synthetic.to_csv('synthetic_ui_dynamics.csv', index=False)
