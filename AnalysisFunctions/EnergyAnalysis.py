import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

# ==================== DATA LOADING & PREPROCESSING ====================

def load_and_clean_weather_data(filepath, date_col='datetime', 
                               required_cols=None, dropna=True):
    """
    Load weather/energy CSV data with basic cleaning and validation.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    date_col : str
        Name of datetime column
    required_cols : list
        List of required column names
    dropna : bool
        Whether to drop rows with missing values
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with datetime index
    """
    try:
        df = pd.read_csv(filepath)
        
        # Convert datetime column
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        
        # Check for required columns
        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Basic cleaning
        if dropna:
            df.dropna(inplace=True)
        
        # Remove any obvious outliers (negative irradiance, impossible wind speeds)
        if 'solar_irradiance' in df.columns:
            df = df[df['solar_irradiance'] >= 0]
        if 'wind_speed' in df.columns:
            df = df[(df['wind_speed'] >= 0) & (df['wind_speed'] <= 100)]
        
        print(f"Data loaded: {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def add_time_features(df, datetime_col=None):
    """
    Add useful time-based features for energy analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    datetime_col : str
        Datetime column name (if not index)
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with added time features
    """
    df_copy = df.copy()
    
    # Use index if datetime_col not specified
    dt_series = df_copy[datetime_col] if datetime_col else df_copy.index
    
    df_copy['hour'] = dt_series.hour
    df_copy['day_of_year'] = dt_series.dayofyear
    df_copy['month'] = dt_series.month
    df_copy['season'] = df_copy['month'].map({12:4, 1:4, 2:4,  # Winter
                                             3:1, 4:1, 5:1,    # Spring
                                             6:2, 7:2, 8:2,    # Summer
                                             9:3, 10:3, 11:3}) # Fall
    df_copy['is_weekend'] = dt_series.weekday >= 5
    
    return df_copy

# ==================== SOLAR ANALYSIS FUNCTIONS ====================

def solar_position(day_of_year, hour, latitude):
    """
    Calculate basic solar position parameters.
    
    Parameters:
    -----------
    day_of_year : int or array
        Day of year (1-365)
    hour : float or array
        Hour of day (0-24)
    latitude : float
        Latitude in degrees
    
    Returns:
    --------
    dict
        Dictionary with solar_elevation and solar_azimuth
    """
    # Solar declination angle
    declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
    
    # Hour angle
    hour_angle = 15 * (hour - 12)
    
    # Solar elevation angle
    lat_rad = np.radians(latitude)
    dec_rad = np.radians(declination)
    hour_rad = np.radians(hour_angle)
    
    elevation = np.degrees(np.arcsin(
        np.sin(lat_rad) * np.sin(dec_rad) + 
        np.cos(lat_rad) * np.cos(dec_rad) * np.cos(hour_rad)
    ))
    
    return {'solar_elevation': elevation, 'declination': declination}

def calculate_solar_pv_output(irradiance, temperature, panel_area=1.0, 
                             efficiency=0.20, temp_coeff=-0.004):
    """
    Calculate photovoltaic power output using basic model.
    
    Parameters:
    -----------
    irradiance : array-like
        Solar irradiance in W/m²
    temperature : array-like
        Cell temperature in °C
    panel_area : float
        Panel area in m²
    efficiency : float
        Panel efficiency at STC (25°C)
    temp_coeff : float
        Temperature coefficient (%/°C)
    
    Returns:
    --------
    array
        Power output in watts
    """
    # Temperature correction
    temp_correction = 1 + temp_coeff * (temperature - 25)
    
    # Power calculation
    power = irradiance * panel_area * efficiency * temp_correction
    
    return np.maximum(power, 0)  # No negative power

def solar_thermal_efficiency(collector_temp, ambient_temp, irradiance,
                           a0=0.8, a1=3.0, a2=0.01):
    """
    Calculate solar thermal collector efficiency.
    
    Parameters:
    -----------
    collector_temp : array-like
        Collector temperature (°C)
    ambient_temp : array-like
        Ambient temperature (°C)
    irradiance : array-like
        Solar irradiance (W/m²)
    a0, a1, a2 : float
        Collector efficiency parameters
    
    Returns:
    --------
    array
        Collector efficiency (0-1)
    """
    # Avoid division by zero
    irradiance = np.maximum(irradiance, 1)
    
    # Reduced temperature
    T_star = (collector_temp - ambient_temp) / irradiance
    
    # Efficiency equation
    efficiency = a0 - a1 * T_star - a2 * irradiance * T_star**2
    
    return np.clip(efficiency, 0, 1)

# ==================== WIND ANALYSIS FUNCTIONS ====================

def wind_power_calculation(wind_speed, air_density=1.225, rotor_diameter=90,
                          cut_in=3, cut_out=25, rated_speed=12, rated_power=2000):
    """
    Calculate wind turbine power output using simplified power curve.
    
    Parameters:
    -----------
    wind_speed : array-like
        Wind speed in m/s
    air_density : float
        Air density in kg/m³
    rotor_diameter : float
        Rotor diameter in meters
    cut_in, cut_out : float
        Cut-in and cut-out wind speeds
    rated_speed : float
        Rated wind speed
    rated_power : float
        Rated power in kW
    
    Returns:
    --------
    array
        Power output in kW
    """
    rotor_area = np.pi * (rotor_diameter / 2)**2
    power = np.zeros_like(wind_speed)
    
    # Operating region
    operating = (wind_speed >= cut_in) & (wind_speed <= cut_out)
    
    # Below rated speed: cubic relationship
    below_rated = operating & (wind_speed <= rated_speed)
    if np.any(below_rated):
        # Simplified cubic power curve
        power[below_rated] = rated_power * (wind_speed[below_rated] / rated_speed)**3
    
    # Above rated speed: constant power
    above_rated = operating & (wind_speed > rated_speed)
    power[above_rated] = rated_power
    
    return power

def air_density_correction(temperature, pressure=101325, humidity=0):
    """
    Calculate air density based on temperature, pressure, and humidity.
    
    Parameters:
    -----------
    temperature : array-like
        Temperature in °C
    pressure : float or array-like
        Atmospheric pressure in Pa
    humidity : float or array-like
        Relative humidity (0-1)
    
    Returns:
    --------
    array
        Air density in kg/m³
    """
    # Convert temperature to Kelvin
    T_K = temperature + 273.15
    
    # Dry air density
    rho_dry = pressure / (287.05 * T_K)
    
    # Humidity correction (simplified)
    rho = rho_dry * (1 - 0.378 * humidity)
    
    return rho

# ==================== THERMODYNAMIC ANALYSIS ====================

def carnot_efficiency(T_hot, T_cold):
    """
    Calculate theoretical Carnot efficiency.
    
    Parameters:
    -----------
    T_hot : float or array
        Hot reservoir temperature (°C)
    T_cold : float or array
        Cold reservoir temperature (°C)
    
    Returns:
    --------
    float or array
        Carnot efficiency (0-1)
    """
    T_hot_K = T_hot + 273.15
    T_cold_K = T_cold + 273.15
    
    return 1 - (T_cold_K / T_hot_K)

def rankine_cycle_analysis(T_steam, T_condenser, steam_flow_rate, 
                          turbine_efficiency=0.85, pump_efficiency=0.8):
    """
    Basic Rankine cycle analysis for steam power plants.
    
    Parameters:
    -----------
    T_steam : float
        Steam temperature (°C)
    T_condenser : float
        Condenser temperature (°C)
    steam_flow_rate : float
        Steam mass flow rate (kg/s)
    turbine_efficiency : float
        Turbine isentropic efficiency
    pump_efficiency : float
        Pump efficiency
    
    Returns:
    --------
    dict
        Cycle performance parameters
    """
    # Simplified analysis - assumes saturated conditions
    # In practice, would use steam tables or CoolProp
    
    # Approximate specific enthalpies (kJ/kg)
    h1 = 2800  # Steam at turbine inlet (approximation)
    h2s = 2200  # Isentropic turbine exit
    h2 = h1 - turbine_efficiency * (h1 - h2s)  # Actual turbine exit
    h3 = 200   # Saturated liquid at condenser
    h4 = h3 + 10  # After pump (approximation)
    
    # Work and heat calculations
    w_turbine = h1 - h2  # kJ/kg
    w_pump = (h4 - h3) / pump_efficiency  # kJ/kg
    q_in = h1 - h4  # kJ/kg
    q_out = h2 - h3  # kJ/kg
    
    net_work = w_turbine - w_pump
    efficiency = net_work / q_in
    
    # Power calculations
    power_output = steam_flow_rate * net_work  # kW
    heat_input = steam_flow_rate * q_in  # kW
    
    return {
        'thermal_efficiency': efficiency,
        'power_output_kW': power_output,
        'heat_input_kW': heat_input,
        'carnot_efficiency': carnot_efficiency(T_steam, T_condenser)
    }

def heat_exchanger_effectiveness(mass_flow_hot, mass_flow_cold, 
                               cp_hot, cp_cold, UA):
    """
    Calculate heat exchanger effectiveness using NTU method.
    
    Parameters:
    -----------
    mass_flow_hot, mass_flow_cold : float
        Mass flow rates (kg/s)
    cp_hot, cp_cold : float
        Specific heat capacities (kJ/kg·K)
    UA : float
        Overall heat transfer coefficient × area (kW/K)
    
    Returns:
    --------
    float
        Heat exchanger effectiveness
    """
    C_hot = mass_flow_hot * cp_hot
    C_cold = mass_flow_cold * cp_cold
    C_min = min(C_hot, C_cold)
    C_max = max(C_hot, C_cold)
    C_ratio = C_min / C_max
    
    NTU = UA / C_min
    
    # Counter-flow heat exchanger effectiveness
    if C_ratio < 1:
        effectiveness = (1 - np.exp(-NTU * (1 - C_ratio))) / (1 - C_ratio * np.exp(-NTU * (1 - C_ratio)))
    else:
        effectiveness = NTU / (1 + NTU)
    
    return effectiveness

# ==================== DATA ANALYSIS & VISUALIZATION ====================

def monthly_energy_summary(df, energy_col, datetime_col=None):
    """
    Create monthly energy production summary.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    energy_col : str
        Column name for energy data
    datetime_col : str
        Datetime column (if not index)
    
    Returns:
    --------
    pd.DataFrame
        Monthly summary statistics
    """
    df_copy = df.copy()
    
    if datetime_col:
        df_copy.set_index(datetime_col, inplace=True)
    
    monthly_stats = df_copy.groupby(df_copy.index.month)[energy_col].agg([
        'mean', 'sum', 'std', 'min', 'max'
    ]).round(2)
    
    monthly_stats.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_stats)]
    
    return monthly_stats

def capacity_factor(actual_output, rated_capacity, time_period_hours):
    """
    Calculate capacity factor for renewable energy systems.
    
    Parameters:
    -----------
    actual_output : float or array
        Actual energy output
    rated_capacity : float
        Rated power capacity
    time_period_hours : float
        Time period in hours
    
    Returns:
    --------
    float
        Capacity factor (0-1)
    """
    if isinstance(actual_output, (list, np.ndarray)):
        actual_output = np.sum(actual_output)
    
    theoretical_max = rated_capacity * time_period_hours
    
    return actual_output / theoretical_max if theoretical_max > 0 else 0

def plot_energy_profile(df, energy_col, title="Energy Profile", 
                       resample_freq='D', figsize=(12, 6)):
    """
    Create energy production profile plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    energy_col : str
        Energy column name
    title : str
        Plot title
    resample_freq : str
        Resampling frequency ('H', 'D', 'M')
    figsize : tuple
        Figure size
    """
    plt.figure(figsize=figsize)
    
    # Resample data
    if resample_freq:
        plot_data = df[energy_col].resample(resample_freq).mean()
    else:
        plot_data = df[energy_col]
    
    plt.plot(plot_data.index, plot_data.values, linewidth=1.5)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(f'{energy_col} (Power/Energy)')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==================== EXAMPLE USAGE ====================

def example_analysis(csv_file):
    """
    Example analysis workflow for students.
    """
    print("=== Renewable Energy Analysis Example ===\n")
    
    # Load data
    df = load_and_clean_weather_data(csv_file, 
                                   required_cols=['solar_irradiance', 'wind_speed', 'temperature'])
    
    if df is None:
        return
    
    # Add time features
    df = add_time_features(df)
    
    # Solar analysis
    df['solar_power'] = calculate_solar_pv_output(
        df['solar_irradiance'], df['temperature'], 
        panel_area=10, efficiency=0.20
    )
    
    # Wind analysis
    df['wind_power'] = wind_power_calculation(
        df['wind_speed'], rated_power=2000
    )
    
    # Monthly summaries
    solar_monthly = monthly_energy_summary(df, 'solar_power')
    wind_monthly = monthly_energy_summary(df, 'wind_power')
    
    print("Monthly Solar Power Summary:")
    print(solar_monthly)
    print("\nMonthly Wind Power Summary:")
    print(wind_monthly)
    
    # Capacity factors
    hours_in_period = len(df)  # Assuming hourly data
    solar_cf = capacity_factor(df['solar_power'].sum(), 10*200, hours_in_period)  # 10m² × 200W/m²
    wind_cf = capacity_factor(df['wind_power'].sum(), 2000, hours_in_period)
    
    print(f"\nSolar Capacity Factor: {solar_cf:.1%}")
    print(f"Wind Capacity Factor: {wind_cf:.1%}")

# Example thermodynamic analysis
def example_thermal_analysis():
    """
    Example thermodynamic calculations for power cycles.
    """
    print("\n=== Thermodynamic Analysis Example ===")
    
    # Rankine cycle analysis
    cycle_results = rankine_cycle_analysis(
        T_steam=500,  # °C
        T_condenser=30,  # °C
        steam_flow_rate=100  # kg/s
    )
    
    print(f"Rankine Cycle Results:")
    print(f"Thermal Efficiency: {cycle_results['thermal_efficiency']:.1%}")
    print(f"Power Output: {cycle_results['power_output_kW']:.0f} kW")
    print(f"Carnot Efficiency: {cycle_results['carnot_efficiency']:.1%}")
