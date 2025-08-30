# Configuration System - Airport Queue Detection

The Airport Queue Detection System now includes a comprehensive configuration management system that saves all settings including passwords, making it extremely easy for new airport staff to get started.

## 🚀 Quick Start for New Users

1. **First Time Setup**: Configure all your settings once in the application
2. **Save Configuration**: Click the "💾 Save Config" button in the sidebar
3. **Future Sessions**: Just click "🚀 Quick Connect" to instantly connect with saved settings

## ✨ Features

### 🔐 Complete Settings Storage
- **Connection Details**: IP, port, username, password, camera brand
- **Detection Settings**: Model selection, confidence threshold, image quality
- **Display Preferences**: Gate name, colors, fonts, theme, brightness
- **Advanced Options**: Cropping settings, wait time calculation method

### 🛡️ Security Features
- **Password Encoding**: Passwords are base64 encoded (not plain text)
- **Atomic File Operations**: Configuration saves are atomic to prevent corruption
- **Error Handling**: Graceful fallback to defaults if config is corrupted

### 🎯 User Interface
- **Quick Connect**: One-click connection using saved settings
- **Save Config**: Save current settings with one click
- **Reload Config**: Restore saved settings
- **Reset Config**: Delete saved configuration
- **Visual Feedback**: Shows connection summary and save status

## 📁 Configuration File

Settings are saved in: `airport_queue_config.json`

### File Structure
```json
{
  "connection": {
    "connection_type": "NVR",
    "ip_address": "192.168.1.165",
    "port": 554,
    "username": "admin",
    "password": "base64_encoded_password",
    "channel": 1,
    "stream_quality": "sub",
    "camera_brand": "Generic",
    "custom_url": "rtsp://..."
  },
  "detection": {
    "selected_model": "YOLOv8m",
    "confidence_threshold": 0.5,
    "image_quality": "Medium (Balanced)",
    "detection_interval": 5,
    "enable_people_adjustment": false,
    "people_adjustment": 0
  },
  "display": {
    "gate_name": "GATE",
    "gate_number": "02",
    "theme_option": "Dark (Airport Standard)",
    "font_size_multiplier": 1.0,
    "font_weight": "Bold",
    "accent_color": "#1f77b4",
    "people_count_color": "#1f77b4",
    "brightness_level": 1.0,
    "contrast_level": 1.0
  },
  "fonts": {
    "people_label_font": "Arial",
    "people_label_color": "#ffffff",
    "people_count_font": "Arial",
    "wait_time_label_font": "Arial",
    "wait_time_label_color": "#ffffff",
    "wait_time_value_font": "Arial",
    "wait_time_value_color": "#dc3545"
  },
  "cropping": {
    "enable_cropping": false,
    "crop_left": 0,
    "crop_top": 0,
    "crop_right": 100,
    "crop_bottom": 100
  },
  "wait_time": {
    "wait_time_method": "Smart AI Prediction (Recommended)",
    "time_setting": "Auto-detect current time",
    "current_hour": 12
  },
  "metadata": {
    "created_at": "2024-01-01T12:00:00",
    "last_updated": "2024-01-01T12:30:00",
    "version": "1.0",
    "description": "Airport Queue Detection System Configuration"
  }
}
```

## 🔧 Usage Scenarios

### Scenario 1: Initial Setup (IT Admin)
1. Run the application: `streamlit run main.py`
2. Configure all connection and display settings
3. Click "💾 Save Config" to save everything
4. Test with "🚀 Quick Connect"

### Scenario 2: New Airport Staff
1. Run the application: `streamlit run main.py`
2. Click "🚀 Quick Connect" - that's it!
3. System automatically loads all saved settings and connects

### Scenario 3: Settings Update
1. Make changes to any settings in the UI
2. Click "💾 Save Config" to update saved configuration
3. Settings are immediately available for future sessions

### Scenario 4: Multiple Configurations
1. Export current config: Use the config manager to export settings
2. Import different config: Load settings from a different file
3. Useful for different shifts, gates, or camera setups

## 🛠️ Technical Details

### Configuration Manager Class
The `ConfigManager` class handles all configuration operations:
- `load_config()`: Load configuration from file
- `save_config()`: Save configuration to file
- `config_exists()`: Check if config file exists
- `get_connection_summary()`: Get readable summary of connection settings
- `update_session_state_from_config()`: Apply config to Streamlit session
- `collect_config_from_session_state()`: Collect current UI settings

### Session State Integration
All UI elements are now connected to Streamlit session state with keys matching the configuration structure. This ensures:
- Settings persist during the session
- Configuration loading updates the UI immediately
- Saving captures the current UI state

### Error Handling
- **File Not Found**: Uses default configuration
- **JSON Parsing Error**: Falls back to defaults, removes corrupted file
- **Missing Keys**: Merges with defaults to ensure all required settings exist
- **Invalid Values**: Validates and uses safe defaults

## 🚨 Important Notes

1. **Password Security**: Passwords are base64 encoded, not encrypted. For production use, consider implementing proper encryption.

2. **File Location**: Configuration file is created in the same directory as the application. Ensure write permissions.

3. **Backup**: Consider backing up the configuration file, especially after initial setup.

4. **Updates**: When updating the application, the configuration system will automatically merge new settings with existing ones.

## 🎯 Benefits for Airport Operations

- **Zero Training Time**: New staff can operate immediately
- **Consistent Settings**: Same configuration across all shifts
- **Quick Deployment**: Easy to replicate setup across multiple gates
- **Reduced Errors**: No manual entry of complex connection details
- **Professional Operation**: Seamless, one-click operation

The configuration system transforms the application from requiring technical setup each time to a professional, deployment-ready solution perfect for airport environments.
