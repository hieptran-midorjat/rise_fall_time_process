import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to identify pulse regions based on LED Voltage
def find_pulses(data, voltage_threshold=5.0, min_points=10):
    pulse_on = data["LED Voltage"] > voltage_threshold
    pulse_regions = []
    start_idx = None
    count = 0
    
    for i in range(len(data)):
        if pulse_on.iloc[i]:
            if start_idx is None:
                start_idx = i
            count += 1
        elif start_idx is not None and count >= min_points:
            end_idx = i
            # Extend slightly to capture falling edge
            while end_idx < len(data) and data["LED Voltage"].iloc[end_idx] > voltage_threshold * 0.01:
                print(f'{end_idx} - {data["LED Voltage"].iloc[end_idx]}')
                end_idx += 1
            pulse_regions.append((start_idx, end_idx))
            start_idx = None
            count = 0
        else:
            start_idx = None
            count = 0
    
    if start_idx is not None and count >= min_points:
        pulse_regions.append((start_idx, len(data)))
    
    return pulse_regions

def process_rise_time(t_rising, c_rising):
    if len(t_rising) >= 5:
        try:
            # Remove 0 and negative values
            t_rising = t_rising[c_rising > 0]
            c_rising = c_rising[c_rising > 0]
            # Find closest points to 0.1 and 0.9
            idx_10_rise = np.argmin(np.abs(c_rising - 0.1))
            idx_90_rise = np.argmin(np.abs(c_rising - 0.9))
            # Extrapolate the time value if can't find exact 0.1 or 0.9
            print(f"{idx_10_rise}-{c_rising[idx_10_rise]} ")
            print(f"{c_rising}")
            if 0.09 < c_rising[idx_10_rise] < 0.11:
                t_10_rise = t_rising[idx_10_rise]
            else:
                # IF the idx is 0, we don't have a previous point
                # So assume the first point is (first time - 0.1s, 0)
                if idx_10_rise == 0:
                    r_first_point= (t_rising[idx_10_rise]-0.1, 0)
                else:
                    r_first_point = (t_rising[idx_10_rise-1], c_rising[idx_10_rise-1])

                r_second_point = (t_rising[idx_10_rise], c_rising[idx_10_rise])
                change = (r_second_point[1] - r_first_point[1])/(r_second_point[0] - r_first_point[0])
                t_10_rise = r_first_point[0] + (0.1 - r_first_point[1])/change

            if not(0.89 < c_rising[idx_90_rise] < 0.99):
                t_90_rise = t_rising[idx_90_rise]
            else:
                first_point = c_rising[idx_90_rise-1]
                second_point = c_rising[idx_90_rise]
                delta = second_point - first_point
                change = delta/(t_rising[idx_90_rise]- t_rising[idx_90_rise-1])
                t_90_rise = t_rising[idx_90_rise-1] + (0.9 - first_point)/change

            rise_time = t_90_rise - t_10_rise
            return t_10_rise, t_90_rise, rise_time
        except Exception as e:
            print(f"Rise time calculation failed: {e}")
    return None

def process_fall_time(t_falling, c_falling):
    if len(t_falling) >= 5:
        try:
            idx_90_fall = np.argmin(np.abs(c_falling - 0.9))
            idx_10_fall = np.argmin(np.abs(c_falling - 0.1))
            if not (0.09 < c_falling[idx_10_fall] < 0.11):
                print(f"{c_falling[idx_10_fall]}")
                print(f"{c_falling}")
                first_point = c_falling[idx_10_fall-1]
                idx = idx_10_fall
                while idx < len(c_falling) and (c_falling[idx] > 0.1 or t_falling[idx_10_fall-1] == t_falling[idx]):
                    idx += 1
                second_point = c_falling[idx]
                print(f"f {t_falling[idx_10_fall-1]} {first_point} - s {t_falling[idx]} {second_point}")
                delta = second_point - first_point
                change = delta/(t_falling[idx_10_fall-1]- t_falling[idx])
                print(f"delta {change}")
                t_10_fall = t_falling[idx_10_fall] + (0.1 - first_point)/change
            else:
                t_10_fall = t_falling[idx_10_fall]

            if not(0.89 < c_falling[idx_90_fall] < 0.91):
                first_point = c_falling[idx_90_fall]
                idx = idx_90_fall+1
                while idx < len(c_falling) and (c_falling[idx] > 0.9 or t_falling[idx_90_fall] == t_falling[idx]):
                    idx += 1
                second_point = c_falling[idx]
                delta = first_point - second_point
                change = delta/(t_falling[idx]- t_falling[idx_90_fall])
                t_90_fall = t_falling[idx_90_fall]+(first_point-0.9)/change
            else:
                t_90_fall = t_falling[idx_90_fall]

            fall_time =  t_10_fall - t_90_fall
            return t_90_fall, t_10_fall, fall_time
        except Exception as e:
            print(f"Fall time calculation failed: {e}")
    return None

def get_current_at_time(target_time, time_series, current_series, default_value):
    """
    Returns the first normalized current value at the given time.
    If not found, returns the default_value.
    """
    if target_time in time_series.values:
        current = current_series[time_series == target_time]
        return current.iloc[0] if not current.empty else default_value
    else:
        return default_value

# Function to calculate rise and fall times using data points
def calculate_rise_fall_times(time, current, led_voltage, pulse_start, pulse_end):
    pulse_data = (time >= pulse_start) & (time <= pulse_end)
    t = time[pulse_data].to_numpy()
    c = current[pulse_data].to_numpy()
    v = led_voltage[pulse_data].to_numpy()

    if len(t) < 10:
        print(f"Warning: Only {len(t)} points in pulse window, may affect accuracy.")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Find peak at the maximum current, refined by LED Voltage drop
    peak_idx = np.argmax(np.abs(c))
    voltage_threshold = 5.0
    for i in range(peak_idx, len(v) - 1):
        if v[i] >= voltage_threshold > v[i + 1]:
            peak_idx = i
            break
    peak_time = t[peak_idx]
    print('Peak found at index:', peak_idx, 'Time:', peak_time)
    print(f"current data {c}")

    # Ensure enough points on both sides
    if peak_idx < 2:
        peak_idx = 2
    elif peak_idx > len(c) - 5:
        peak_idx = len(c) - 5

    # Rising edge (before peak)
    t_rising = np.round(t[:peak_idx+1], 3)
    c_rising = np.round(c[:peak_idx+1], 3)
    # Set negative values to 0
    t_rising = np.where(t_rising < 0, 0, t_rising)
    c_rising = np.where(c_rising < 0, 0, c_rising)

    # Falling edge (after peak)
    t_falling = np.round(t[peak_idx:], 3)
    c_falling = np.round(c[peak_idx:], 3)
    t_falling = np.where(t_falling < 0, 0, t_falling)
    c_falling = np.where(c_falling < 0, 0, c_falling)

    # Debug: Plot pulse window and peak
    plt.axvline(pulse_start, color='gray', linestyle='--', alpha=0.3, label='Pulse Window')
    plt.axvline(pulse_end, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(peak_time, color='black', linestyle=':', alpha=0.3, label='Peak')

    # Debug: Check data range
    print(f"Rising edge current range: {c_rising.min():.3f} to {c_rising.max():.3f}")
    print(f"Falling edge current range: {c_falling.min():.3f} to {c_falling.max():.3f}")

    # Process rise time
    rise_time = np.nan
    t_10_rise = np.nan
    t_90_rise = np.nan
    rise_time_data = process_rise_time(t_rising, c_rising)
    if rise_time_data is not None:
        t_10_rise, t_90_rise, rise_time = rise_time_data

    # Process fall time
    fall_time = np.nan
    t_90_fall = np.nan
    t_10_fall = np.nan
    fall_time_daa = process_fall_time(t_falling, c_falling)
    if fall_time_daa is not None:
        t_90_fall, t_10_fall, fall_time = fall_time_daa

    # Debug output
    print(f"Peak index: {peak_idx}, Time at peak: {peak_time:.3f}s")
    print(f"Rise edge points: {len(t_rising)}, Fall edge points: {len(t_falling)}")
    print(f"10% rise time: {t_10_rise:.3f}s, 90% rise time: {t_90_rise:.3f}s")
    print(f"90% fall time: {t_90_fall:.3f}s, 10% fall time: {t_10_fall:.3f}s")
    print(f"Rise time: {rise_time:.3f}s, Fall time: {fall_time:.3f}s")

    return rise_time, fall_time, t_10_rise, t_90_rise, t_90_fall, t_10_fall

SECOND_PULSE_DATA = (290, 410)

def main():
    # Load the dataset
    data = pd.read_csv("5um_gap_2_pulses_50ms.csv")

    # Select the second pulse
    second_pulse = SECOND_PULSE_DATA
    pulse_start_idx, pulse_end_idx = second_pulse
    pulse_start_time = data["Device Time"].iloc[pulse_start_idx]
    pulse_end_time = data["Device Time"].iloc[pulse_end_idx - 1]

    print(f"Start time {pulse_start_time}, End time {pulse_end_time}")

    # Extend the pulse end time slightly to capture more falling edge data
    pulse_end_time_extended = pulse_end_time + 0.2  # 200ms buffer

    # Define the plotting window: 2s before start to 2s after extended end
    plot_start_time = pulse_start_time - 4.0
    plot_end_time = pulse_end_time_extended + 4.0

    # Filter data for the plotting window
    filtered_data = data[(data["Device Time"] >= plot_start_time) & (data["Device Time"] <= plot_end_time)]

    # Check if filtered data is empty
    if filtered_data.empty:
        raise ValueError("No data found in the specified time window.")

    # Normalize the Device Current
    time = filtered_data["Device Time"]
    pulse_mask = (time >= pulse_start_time) & (time <= pulse_end_time)
    current = filtered_data["Device Current"]
    # Calculate baseline as median of non-pulse data
    baseline_mask = ~pulse_mask
    baseline = np.median(current[baseline_mask]) if baseline_mask.any() else 0
    # Subtract baseline
    current_corrected = current - baseline
    # Normalize based on peak within pulse
    pulse_current = current_corrected[pulse_mask]
    amplitude = np.max(np.abs(pulse_current))
    if amplitude == 0:
        raise ValueError("Amplitude is zero, cannot normalize.")
    normalized_current = current_corrected / amplitude

    # Export normalized current to CSV
    normalized_data = pd.DataFrame({
        "Time (s)": time,
        "Normalized Current": normalized_current
    })
    normalized_data.to_csv("normalized_second_pulse.csv", index=False)

    # Create a high-quality figure
    plt.figure(figsize=(8, 6), dpi=300)

    # Plot normalized Device Current
    plt.plot(time, normalized_current,
             label="Normalized Device Current (Second Pulse)", color='blue', linewidth=2, alpha=0.8)

    # Calculate rise/fall times for the second pulse
    rt, ft, t_10_r, t_90_r, t_90_f, t_10_f = calculate_rise_fall_times(
        time, normalized_current, filtered_data["LED Voltage"], pulse_start_time, pulse_end_time_extended)

    c_10_r = get_current_at_time(t_10_r, time, normalized_current, 0.1)
    c_90_r = get_current_at_time(t_90_r, time, normalized_current, 0.9)
    c_90_f = get_current_at_time(t_90_f, time, normalized_current, 0.9)
    c_10_f = get_current_at_time(t_10_f, time, normalized_current, 0.1)

    # Annotate rise time points
    if not np.isnan(t_10_r):
        plt.plot(t_10_r, c_10_r, 'go', label='10% Rise')
        plt.text(t_10_r, c_10_r + 0.05, '10% Rise', fontsize=8, color='green', ha='center', va='bottom')
    if not np.isnan(t_90_r):
        plt.plot(t_90_r, c_90_r, 'go', label='90% Rise')
        plt.text(t_90_r,c_90_r + 0.05, '90% Rise', fontsize=8, color='green', ha='center', va='bottom')
    if not np.isnan(rt):
        plt.text((t_10_r + t_90_r)/2, 0.5, f'Rise: {rt:.3f}s', fontsize=8, color='green', ha='center')

    # Annotate fall time points
    if not np.isnan(t_90_f):
        plt.plot(t_90_f, c_90_f, 'ro', label='90% Fall')
        plt.text(t_90_f, c_90_f + 0.05, '90% Fall', fontsize=8, color='red', ha='center', va='bottom')
    if not np.isnan(t_10_f):
        plt.plot(t_10_f, c_10_f, 'ro', label='10% Fall')
        plt.text(t_10_f, c_10_f + 0.05, '10% Fall', fontsize=8, color='red', ha='center', va='bottom')
    if not np.isnan(ft):
        plt.text((t_90_f + t_10_f)/2, 0.3, f'Fall: {ft:.3f}s', fontsize=8, color='red', ha='center')

    # Print rise and fall times
    print(f"Second Pulse Rise Time (10% to 90%): {rt:.3f}s" if not np.isnan(rt) else "Second Pulse Rise Time: N/A")
    print(f"Second Pulse Fall Time (90% to 10%): {ft:.3f}s" if not np.isnan(ft) else "Second Pulse Fall Time: N/A")

    # Set x-axis ticks at 1-second intervals
    plt.xticks(np.arange(round(plot_start_time), round(plot_end_time) + 1, 1))

    # Customize the plot
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Current", fontsize=12)
    plt.legend(fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10, direction='in', length=5)
    plt.tight_layout()

    # Save the plot
    plt.savefig("5um_pulse.tiff", format="tiff", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
