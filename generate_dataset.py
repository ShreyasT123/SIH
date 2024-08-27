import numpy as np
import pandas as pd

# Parameters for human-like behavior (increased randomness)
human_params = {
    'mouse_speed_mean': 2.5,
    'mouse_speed_stddev': 1.5,  # Increased variation in mouse speed
    'click_frequency': 8,  # Humans click less frequently, but with variation
    'click_interval_mean': 0.6,
    'click_interval_stddev': 0.4,  # Wider range for intervals between clicks
    'scroll_speed_mean': 1.8,  # Humans tend to scroll slower
    'scroll_speed_stddev': 0.7,
    'typing_speed_mean': 0.45,  # Slightly slower typing speed
    'typing_speed_stddev': 0.15,  # Increased variation in typing speed
    'session_duration': 300,  # Same session duration
    'activity_gap_mean': 1.0,  # Humans are more likely to have occasional gaps
    'activity_gap_stddev': 0.5  # Gaps have more variation
}

# Parameters for bot-like behavior (less predictable, more life-like)
bot_params = {
    'mouse_speed_mean': 2.2,  # Closer to human speed
    'mouse_speed_stddev': 0.4,  # Slightly more variation in bot mouse speed
    'click_frequency': 12,  # Bots still click more frequently but with variability
    'click_interval_mean': 0.3,  # Reduced precision in click intervals
    'click_interval_stddev': 0.2,
    'scroll_speed_mean': 1.6,  # Bots scroll more similarly to humans now
    'scroll_speed_stddev': 0.3,  # Less robotic scrolling
    'typing_speed_mean': 0.6,  # Slightly faster than humans, but not excessively
    'typing_speed_stddev': 0.2,  # More variation in typing
    'session_duration': 300,  # Same session duration
    'activity_gap_mean': 0.5,  # Bots have shorter gaps but now with some variability
    'activity_gap_stddev': 0.2
}

# Function to generate synthetic data
def generate_synthetic_data(num_samples):
    data = []
    
    for i in range(num_samples):
        user_type = np.random.choice(['human', 'bot'])
        params = human_params if user_type == 'human' else bot_params
        
        # Add variability: occasional outliers or unexpected behavior for humans
        if user_type == 'human' and np.random.rand() > 0.95:
            params['mouse_speed_mean'] *= 2  # Occasional faster mouse speeds
            params['click_frequency'] += 5  # Randomly increased click frequency

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
