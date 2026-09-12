def on_time_delivery_rate(delay_flag):
    return 100 * (1 - delay_flag.mean())

def average_lead_time(lead_time_days):
    return lead_time_days.mean()
