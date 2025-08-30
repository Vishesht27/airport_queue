"""
Configuration Management System for Airport Queue Detection
Saves and loads all application settings including passwords
"""

import json
import os
import base64
from typing import Dict, Any, Optional
from datetime import datetime
import streamlit as st

class ConfigManager:
    def __init__(self, config_file: str = "airport_queue_config.json"):
        self.config_file = config_file
        self.default_config = self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            # Connection settings
            "connection": {
                "connection_type": "NVR",
                "ip_address": "192.168.1.165",
                "port": 554,
                "username": "admin",
                "password": "",
                "channel": 1,
                "stream_quality": "sub",
                "camera_brand": "Generic",
                "custom_url": "rtsp://admin:password@192.168.1.125:554/rtsp/defaultPrimary?streamType=u"
            },
            
            # Detection settings
            "detection": {
                "selected_model": "YOLOv8m",
                "confidence_threshold": 0.5,
                "image_quality": "Medium (Balanced)",
                "detection_interval": 5,
                "enable_people_adjustment": False,
                "people_adjustment": 0
            },
            
            # Wait time settings
            "wait_time": {
                "wait_time_method": "Smart AI Prediction (Recommended)",
                "time_setting": "Auto-detect current time",
                "current_hour": 12
            },
            
            # Display settings
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
            
            # Font and color settings
            "fonts": {
                "people_label_font": "Arial",
                "people_label_color": "#ffffff",
                "people_count_font": "Arial",
                "wait_time_label_font": "Arial",
                "wait_time_label_color": "#ffffff",
                "wait_time_value_font": "Arial",
                "wait_time_value_color": "#dc3545"
            },
            
            # Individual font sizes
            "font_sizes": {
                "gate_name_size": 1.0,
                "people_label_size": 1.0,
                "people_count_size": 1.0,
                "wait_time_label_size": 1.0,
                "wait_time_value_size": 1.0
            },
            
            # Cropping settings
            "cropping": {
                "enable_cropping": False,
                "crop_left": 0,
                "crop_top": 0,
                "crop_right": 100,
                "crop_bottom": 100
            },
            
            # Metadata
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Airport Queue Detection System Configuration"
            }
        }
    
    def _encode_password(self, password: str) -> str:
        """Simple base64 encoding for password (not secure encryption, but better than plain text)"""
        if not password:
            return ""
        return base64.b64encode(password.encode()).decode()
    
    def _decode_password(self, encoded_password: str) -> str:
        """Decode base64 encoded password"""
        if not encoded_password:
            return ""
        try:
            return base64.b64decode(encoded_password.encode()).decode()
        except:
            return ""
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file"""
        try:
            # Create a copy to avoid modifying original
            config_to_save = config.copy()
            
            # Encode password
            if "connection" in config_to_save and "password" in config_to_save["connection"]:
                config_to_save["connection"]["password"] = self._encode_password(
                    config_to_save["connection"]["password"]
                )
            
            # Update metadata
            config_to_save["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")
            return False
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Decode password
                if "connection" in config and "password" in config["connection"]:
                    config["connection"]["password"] = self._decode_password(
                        config["connection"]["password"]
                    )
                
                # Merge with defaults to ensure all keys exist
                merged_config = self._merge_configs(self.default_config, config)
                return merged_config
            else:
                # Return default config if file doesn't exist
                return self.default_config.copy()
                
        except Exception as e:
            st.warning(f"Failed to load configuration: {e}. Using defaults.")
            return self.default_config.copy()
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge loaded config with defaults"""
        merged = default.copy()
        
        for key, value in loaded.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def config_exists(self) -> bool:
        """Check if configuration file exists"""
        return os.path.exists(self.config_file)
    
    def delete_config(self) -> bool:
        """Delete configuration file"""
        try:
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            return True
        except Exception as e:
            st.error(f"Failed to delete configuration: {e}")
            return False
    
    def export_config(self, export_path: str) -> bool:
        """Export configuration to a different file"""
        try:
            config = self.load_config()
            with open(export_path, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """Import configuration from a file"""
        try:
            with open(import_path, 'r') as f:
                config = json.load(f)
            
            # Validate basic structure
            if "connection" in config and "detection" in config:
                return self.save_config(config)
            else:
                st.error("Invalid configuration file format")
                return False
                
        except Exception as e:
            st.error(f"Failed to import configuration: {e}")
            return False
    
    def get_connection_summary(self, config: Dict[str, Any]) -> str:
        """Get a summary of connection settings"""
        conn = config.get("connection", {})
        conn_type = conn.get("connection_type", "Unknown")
        ip = conn.get("ip_address", "Unknown")
        port = conn.get("port", "Unknown")
        username = conn.get("username", "Unknown")
        
        if conn_type == "Custom URL":
            url = conn.get("custom_url", "")
            return f"Custom URL: {url[:50]}{'...' if len(url) > 50 else ''}"
        else:
            return f"{conn_type}: {username}@{ip}:{port}"
    
    def update_session_state_from_config(self, config: Dict[str, Any]):
        """Update Streamlit session state with configuration values"""
        
        # Connection settings
        conn = config.get("connection", {})
        st.session_state.connection_type = conn.get("connection_type", "NVR")
        st.session_state.ip_address = conn.get("ip_address", "192.168.1.165")
        st.session_state.port = conn.get("port", 554)
        st.session_state.username = conn.get("username", "admin")
        st.session_state.password = conn.get("password", "")
        st.session_state.channel = conn.get("channel", 1)
        st.session_state.stream_quality = conn.get("stream_quality", "sub")
        st.session_state.camera_brand = conn.get("camera_brand", "Generic")
        st.session_state.custom_url = conn.get("custom_url", "")
        
        # Detection settings
        detect = config.get("detection", {})
        st.session_state.selected_model = detect.get("selected_model", "YOLOv8m")
        st.session_state.confidence_threshold = detect.get("confidence_threshold", 0.5)
        st.session_state.image_quality = detect.get("image_quality", "Medium (Balanced)")
        st.session_state.detection_interval = detect.get("detection_interval", 5)
        st.session_state.enable_people_adjustment = detect.get("enable_people_adjustment", False)
        st.session_state.people_adjustment = detect.get("people_adjustment", 0)
        
        # Wait time settings
        wait = config.get("wait_time", {})
        st.session_state.wait_time_method = wait.get("wait_time_method", "Smart AI Prediction (Recommended)")
        st.session_state.time_setting = wait.get("time_setting", "Auto-detect current time")
        st.session_state.current_hour = wait.get("current_hour", 12)
        
        # Display settings
        display = config.get("display", {})
        st.session_state.gate_name = display.get("gate_name", "GATE")
        st.session_state.gate_number = display.get("gate_number", "02")
        st.session_state.theme_option = display.get("theme_option", "Dark (Airport Standard)")
        st.session_state.font_size_multiplier = display.get("font_size_multiplier", 1.0)
        st.session_state.font_weight = display.get("font_weight", "Bold")
        st.session_state.accent_color = display.get("accent_color", "#1f77b4")
        st.session_state.people_count_color = display.get("people_count_color", "#1f77b4")
        st.session_state.brightness_level = display.get("brightness_level", 1.0)
        st.session_state.contrast_level = display.get("contrast_level", 1.0)
        
        # Font settings
        fonts = config.get("fonts", {})
        st.session_state.people_label_font = fonts.get("people_label_font", "Arial")
        st.session_state.people_label_color = fonts.get("people_label_color", "#ffffff")
        st.session_state.people_count_font = fonts.get("people_count_font", "Arial")
        st.session_state.wait_time_label_font = fonts.get("wait_time_label_font", "Arial")
        st.session_state.wait_time_label_color = fonts.get("wait_time_label_color", "#ffffff")
        st.session_state.wait_time_value_font = fonts.get("wait_time_value_font", "Arial")
        st.session_state.wait_time_value_color = fonts.get("wait_time_value_color", "#dc3545")
        
        # Font sizes
        font_sizes = config.get("font_sizes", {})
        st.session_state.gate_name_size = font_sizes.get("gate_name_size", 1.0)
        st.session_state.people_label_size = font_sizes.get("people_label_size", 1.0)
        st.session_state.people_count_size = font_sizes.get("people_count_size", 1.0)
        st.session_state.wait_time_label_size = font_sizes.get("wait_time_label_size", 1.0)
        st.session_state.wait_time_value_size = font_sizes.get("wait_time_value_size", 1.0)
        
        # Cropping settings
        crop = config.get("cropping", {})
        st.session_state.enable_cropping = crop.get("enable_cropping", False)
        st.session_state.crop_left = crop.get("crop_left", 0)
        st.session_state.crop_top = crop.get("crop_top", 0)
        st.session_state.crop_right = crop.get("crop_right", 100)
        st.session_state.crop_bottom = crop.get("crop_bottom", 100)
    
    def collect_config_from_session_state(self) -> Dict[str, Any]:
        """Collect current configuration from Streamlit session state"""
        
        config = self.default_config.copy()
        
        # Connection settings
        config["connection"].update({
            "connection_type": getattr(st.session_state, 'connection_type', "NVR"),
            "ip_address": getattr(st.session_state, 'ip_address', "192.168.1.165"),
            "port": getattr(st.session_state, 'port', 554),
            "username": getattr(st.session_state, 'username', "admin"),
            "password": getattr(st.session_state, 'password', ""),
            "channel": getattr(st.session_state, 'channel', 1),
            "stream_quality": getattr(st.session_state, 'stream_quality', "sub"),
            "camera_brand": getattr(st.session_state, 'camera_brand', "Generic"),
            "custom_url": getattr(st.session_state, 'custom_url', "")
        })
        
        # Detection settings
        config["detection"].update({
            "selected_model": getattr(st.session_state, 'selected_model', "YOLOv8m"),
            "confidence_threshold": getattr(st.session_state, 'confidence_threshold', 0.5),
            "image_quality": getattr(st.session_state, 'image_quality', "Medium (Balanced)"),
            "detection_interval": getattr(st.session_state, 'detection_interval', 5),
            "enable_people_adjustment": getattr(st.session_state, 'enable_people_adjustment', False),
            "people_adjustment": getattr(st.session_state, 'people_adjustment', 0)
        })
        
        # Wait time settings
        config["wait_time"].update({
            "wait_time_method": getattr(st.session_state, 'wait_time_method', "Smart AI Prediction (Recommended)"),
            "time_setting": getattr(st.session_state, 'time_setting', "Auto-detect current time"),
            "current_hour": getattr(st.session_state, 'current_hour', 12)
        })
        
        # Display settings
        config["display"].update({
            "gate_name": getattr(st.session_state, 'gate_name', "GATE"),
            "gate_number": getattr(st.session_state, 'gate_number', "02"),
            "theme_option": getattr(st.session_state, 'theme_option', "Dark (Airport Standard)"),
            "font_size_multiplier": getattr(st.session_state, 'font_size_multiplier', 1.0),
            "font_weight": getattr(st.session_state, 'font_weight', "Bold"),
            "accent_color": getattr(st.session_state, 'accent_color', "#1f77b4"),
            "people_count_color": getattr(st.session_state, 'people_count_color', "#1f77b4"),
            "brightness_level": getattr(st.session_state, 'brightness_level', 1.0),
            "contrast_level": getattr(st.session_state, 'contrast_level', 1.0)
        })
        
        # Font settings
        config["fonts"].update({
            "people_label_font": getattr(st.session_state, 'people_label_font', "Arial"),
            "people_label_color": getattr(st.session_state, 'people_label_color', "#ffffff"),
            "people_count_font": getattr(st.session_state, 'people_count_font', "Arial"),
            "wait_time_label_font": getattr(st.session_state, 'wait_time_label_font', "Arial"),
            "wait_time_label_color": getattr(st.session_state, 'wait_time_label_color', "#ffffff"),
            "wait_time_value_font": getattr(st.session_state, 'wait_time_value_font', "Arial"),
            "wait_time_value_color": getattr(st.session_state, 'wait_time_value_color', "#dc3545")
        })
        
        # Font sizes
        config["font_sizes"].update({
            "gate_name_size": getattr(st.session_state, 'gate_name_size', 1.0),
            "people_label_size": getattr(st.session_state, 'people_label_size', 1.0),
            "people_count_size": getattr(st.session_state, 'people_count_size', 1.0),
            "wait_time_label_size": getattr(st.session_state, 'wait_time_label_size', 1.0),
            "wait_time_value_size": getattr(st.session_state, 'wait_time_value_size', 1.0)
        })
        
        # Cropping settings
        config["cropping"].update({
            "enable_cropping": getattr(st.session_state, 'enable_cropping', False),
            "crop_left": getattr(st.session_state, 'crop_left', 0),
            "crop_top": getattr(st.session_state, 'crop_top', 0),
            "crop_right": getattr(st.session_state, 'crop_right', 100),
            "crop_bottom": getattr(st.session_state, 'crop_bottom', 100)
        })
        
        return config
