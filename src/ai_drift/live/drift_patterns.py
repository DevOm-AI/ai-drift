import random
import numpy as np

# Human-readable registry of pattern functions. Each function accepts and returns a dict (row).
# Patterns are deterministic/stateless functions that transform a row dict.

def boost_income(row, intensity=1.6):
    row['income'] = float(row.get('income', 0)) * intensity
    return row

def income_spike(row, intensity=2.5):
    # large jump
    row['income'] = float(row.get('income', 0)) * intensity
    return row

def mobile_users_only(row):
    row['device_type'] = 'mobile'
    return row

def pune_only_traffic(row):
    row['geo_city'] = 'Pune'
    return row

def young_audience_surge(row):
    row['age'] = int(random.randint(18, 25))
    return row

def older_users_surge(row):
    row['age'] = int(random.randint(50, 70))
    return row

def cart_value_boom(row, add=500):
    row['cart_value'] = float(row.get('cart_value', 0)) + add
    return row

def spend_score_crash(row):
    row['spend_score'] = float(row.get('spend_score', 0)) * 0.2
    return row

def spend_score_spike(row):
    row['spend_score'] = float(row.get('spend_score', 0)) * 3.0
    return row

def purchase_spike(row):
    row['purchased'] = 1
    return row

def ub_payments_only(row):
    row['payment_method'] = 'UPI'
    return row

def session_drop(row):
    row['session_duration'] = max(1, float(row.get('session_duration', 0)) * 0.3)
    return row

def pageview_surge(row):
    row['page_views'] = int(max(1, int(row.get('page_views', 1)) * 4))
    return row

def device_tablet_only(row):
    row['device_type'] = 'tablet'
    return row

def random_noise(row, scale=0.02):
    # add small Gaussian noise to numeric fields
    for k in ['income','spend_score','session_duration','page_views','cart_value','membership_years']:
        if k in row:
            try:
                v = float(row[k])
                row[k] = v * (1 + np.random.normal(0, scale))
            except Exception:
                pass
    return row

# registry mapping string->function (and optionally default args)
PATTERNS = {
    'boost_income': boost_income,
    'income_spike': income_spike,
    'mobile_users_only': mobile_users_only,
    'pune_only_traffic': pune_only_traffic,
    'young_audience_surge': young_audience_surge,
    'older_users_surge': older_users_surge,
    'cart_value_boom': cart_value_boom,
    'spend_score_crash': spend_score_crash,
    'spend_score_spike': spend_score_spike,
    'purchase_spike': purchase_spike,
    'upi_only_payments': ub_payments_only,
    'session_drop': session_drop,
    'pageview_surge': pageview_surge,
    'device_tablet_only': device_tablet_only,
    'random_noise': random_noise,
}