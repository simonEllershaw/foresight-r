# Note

Critical outcome is defined as a single endpoint
- ICU admission within 12 hours of ED discharge
- OR hospital mortality

This requires the use of 2 different windows to define the target window. Currently ACES cannot handle this (or at least I can't work it out!). Therefore, they are defined as 2 separate tasks and combined using the `combine-critical-outcome` script.
