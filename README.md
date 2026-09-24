# SilentGuard AI —  Interactive Verification Update

## What changed

This update keeps the YAMNet sound-classification and risk-mapping code, but fixes the interaction flow.

### Interactive flow

1. Click **Analyze Sound**.
2. If the prototype risk is HIGH or CRITICAL, a **30-second countdown** starts automatically.
3. **I'm OK** → shows a reassurance message and cancels escalation.
4. **I Need Help** → shows simulated family-message, ambulance/emergency-call, and nearby-hospital notification statuses.
5. If no response is given for 30 seconds → the system automatically enters **NO RESPONSE** and shows the same simulated escalation statuses.

## Important

All family messages, ambulance calls, and hospital notifications are **UI simulations only**. No real SMS, phone call, hospital notification, ambulance dispatch, or external emergency API is contacted.

## Setup

Python 3.11 is recommended for the current TensorFlow/YAMNet environment.

```bash
python -m venv venv
venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

The first YAMNet run requires internet access to download/cache the pretrained model and class map.
