import streamlit as st
import json
import time
from datetime import datetime
import os

# Page configuration for TV display
st.set_page_config(
    page_title="Queue Display",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Force dark theme
st._config.set_option("theme.base", "dark")
st._config.set_option("theme.backgroundColor", "#0e1117")
st._config.set_option("theme.secondaryBackgroundColor", "#262730")
st._config.set_option("theme.textColor", "#fafafa")

# Custom CSS for huge text and TV-friendly display
st.markdown("""
<style>
    /* Force dark background */
    .stApp {
        background-color: #0e1117 !important;
    }
    
    .main {
        background-color: #0e1117 !important;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {display: none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    
    /* Full screen styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
        background-color: #0e1117 !important;
    }
    
    /* Huge text styles */
    .huge-number {
        font-size: 15rem !important;
        font-weight: bold !important;
        text-align: center !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    
    .huge-text {
        font-size: 8rem !important;
        font-weight: bold !important;
        text-align: center !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    
    .medium-text {
        font-size: 4rem !important;
        font-weight: bold !important;
        text-align: center !important;
        margin: 1rem 0 !important;
    }
    
    .status-text {
        font-size: 3rem !important;
        font-weight: bold !important;
        text-align: center !important;
        margin: 1rem 0 !important;
    }
    
    /* Color classes */
    .green-text { color: #28a745 !important; }
    .yellow-text { color: #ffc107 !important; }
    .red-text { color: #dc3545 !important; }
    .blue-text { color: #1f77b4 !important; }
    .white-text { color: #ffffff !important; }
    
    /* Background colors for better contrast */
    .green-bg { 
        background-color: #d4edda !important; 
        padding: 2rem !important;
        border-radius: 20px !important;
        margin: 1rem 0 !important;
    }
    .yellow-bg { 
        background-color: #fff3cd !important; 
        padding: 2rem !important;
        border-radius: 20px !important;
        margin: 1rem 0 !important;
    }
    .red-bg { 
        background-color: #f8d7da !important; 
        padding: 2rem !important;
        border-radius: 20px !important;
        margin: 1rem 0 !important;
    }
    .blue-bg { 
        background-color: #d1ecf1 !important; 
        padding: 2rem !important;
        border-radius: 20px !important;
        margin: 1rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

def load_detection_data():
    """Load the latest detection data from JSON file"""
    try:
        if os.path.exists('detection_data.json'):
            with open('detection_data.json', 'r') as f:
                content = f.read().strip()
                if not content:
                    # Empty file
                    return None
                data = json.loads(content)
                return data
        else:
            return None
    except json.JSONDecodeError as e:
        # JSON is corrupted, delete it and start fresh
        try:
            if os.path.exists('detection_data.json'):
                os.remove('detection_data.json')
        except:
            pass
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def format_time_ago(timestamp):
    """Format how long ago the detection was made"""
    if timestamp is None:
        return "No data"
    
    try:
        detection_time = datetime.fromtimestamp(timestamp)
        now = datetime.now()
        diff = (now - detection_time).total_seconds()
        
        if diff < 60:
            return f"{int(diff)}s ago"
        elif diff < 3600:
            return f"{int(diff/60)}m ago"
        else:
            return f"{int(diff/3600)}h ago"
    except:
        return "Unknown"

def main():
    """Main display page"""
    
    # Display info in sidebar
    with st.sidebar:
        st.header("📺 Display Info")
        st.info("💡 All display settings are configured in the main detection app")
        st.markdown("**Configured from main app:**")
        st.markdown("• Gate name & number")
        st.markdown("• Theme & colors") 
        st.markdown("• Font size & weight")
        st.markdown("• Brightness & contrast")
    
    # Default display settings (fallback)
    default_settings = {
        'theme': 'Dark (Airport Standard)',
        'font_size_multiplier': 1.0,
        'font_weight': 'Bold',
        'accent_color': '#1f77b4',
        'people_count_color': '#1f77b4',
        'brightness_level': 1.0,
        'contrast_level': 1.0
    }
    
    # Auto-refresh every 1 second for instant updates
    time.sleep(1)
    
    # Load detection data
    data = load_detection_data()
    
    if data is None:
        # No data available
        st.markdown('<div class="blue-bg">', unsafe_allow_html=True)
        st.markdown('<p class="huge-text blue-text">WAITING FOR DATA</p>', unsafe_allow_html=True)
        st.markdown('<p class="medium-text blue-text">Start the main detection app</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.rerun()
        return
    
    # Extract data
    people_count = data.get('people_count', 0)
    wait_time = data.get('wait_time', 0)
    last_update = data.get('timestamp', None)
    model_name = data.get('model', 'Unknown')
    gate_name = data.get('gate_name', 'GATE')
    gate_number = data.get('gate_number', '02')
    
    # Extract display settings
    display_settings = data.get('display_settings', default_settings)
    theme_option = display_settings.get('theme', 'Dark (Airport Standard)')
    font_size_multiplier = display_settings.get('font_size_multiplier', 1.0)
    font_weight = display_settings.get('font_weight', 'Bold')
    accent_color = display_settings.get('accent_color', '#1f77b4')
    people_count_color = display_settings.get('people_count_color', '#1f77b4')
    brightness_level = display_settings.get('brightness_level', 1.0)
    contrast_level = display_settings.get('contrast_level', 1.0)
    
    # Extract font and color settings for display parameters
    people_label_font = display_settings.get('people_label_font', 'Arial')
    people_label_color = display_settings.get('people_label_color', '#ffffff')
    people_count_font = display_settings.get('people_count_font', 'Arial')
    wait_time_label_font = display_settings.get('wait_time_label_font', 'Arial')
    wait_time_label_color = display_settings.get('wait_time_label_color', '#ffffff')
    wait_time_value_font = display_settings.get('wait_time_value_font', 'Arial')
    wait_time_value_color = display_settings.get('wait_time_value_color', '#dc3545')
    
    # Apply theme colors based on selection
    if theme_option == "Light (Bright Areas)":
        bg_color = "#ffffff"
        text_color = "#000000"
        card_bg = "#f8f9fa"
    elif theme_option == "High Contrast (Accessibility)":
        bg_color = "#000000"
        text_color = "#ffffff"
        card_bg = "#333333"
    else:  # Dark (Airport Standard)
        bg_color = "#0e1117"
        text_color = "#fafafa"
        card_bg = "#262730"
    
    # Convert font weight to CSS
    font_weight_css = {
        "Normal": "400",
        "Bold": "700",
        "Extra Bold": "900"
    }.get(font_weight, "700")
    
    # Calculate font sizes based on multiplier
    gate_font_size = 6 * font_size_multiplier
    huge_font_size = 15 * font_size_multiplier
    medium_font_size = 4 * font_size_multiplier
    
    # Apply brightness and contrast
    brightness_filter = f"brightness({brightness_level}) contrast({contrast_level})"
    
    # Update CSS with all settings
    st.markdown(f"""
    <style>
        .stApp, .main {{
            background-color: {bg_color} !important;
            filter: {brightness_filter};
        }}
        .main .block-container {{
            background-color: {bg_color} !important;
        }}
        .theme-text {{
            color: {text_color} !important;
        }}
        .theme-accent {{
            color: {accent_color} !important;
        }}
        .theme-card {{
            background-color: #d1ecf1 !important;
            padding: 2rem !important;
            border-radius: 20px !important;
            margin: 1rem 0 !important;
        }}
        .huge-number {{
            font-size: {huge_font_size}rem !important;
            font-weight: {font_weight_css} !important;
        }}
        .huge-text {{
            font-size: {huge_font_size * 0.5}rem !important;
            font-weight: {font_weight_css} !important;
        }}
        .medium-text {{
            font-size: {medium_font_size}rem !important;
            font-weight: {font_weight_css} !important;
        }}
        .gate-text {{
            font-size: {gate_font_size}rem !important;
            font-weight: {font_weight_css} !important;
            color: {accent_color} !important;
        }}
        .people-count-color {{
            color: {people_count_color} !important;
        }}
        .people-label-style {{
            font-family: {people_label_font}, sans-serif !important;
            color: {people_label_color} !important;
        }}
        .people-count-style {{
            font-family: {people_count_font}, sans-serif !important;
            color: {people_count_color} !important;
        }}
        .wait-time-label-style {{
            font-family: {wait_time_label_font}, sans-serif !important;
            color: {wait_time_label_color} !important;
        }}
        .wait-time-value-style {{
            font-family: {wait_time_value_font}, sans-serif !important;
            color: {wait_time_value_color} !important;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Determine color scheme based on wait time
    if wait_time is None or wait_time == 0:
        color_class = "blue-text"
        bg_class = "blue-bg"
        status = "NO QUEUE"
    elif wait_time < 15:
        color_class = "green-text"
        bg_class = "green-bg"
        status = "SHORT WAIT"
    elif wait_time < 45:
        color_class = "yellow-text"
        bg_class = "yellow-bg"
        status = "MODERATE WAIT"
    else:
        color_class = "red-text"
        bg_class = "red-bg"
        status = "LONG WAIT"
    
    # Gate header
    st.markdown(f'<p class="gate-text" style="text-align: center; margin: 0;">{gate_name} {gate_number}</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Main display layout
    col1, col2 = st.columns(2)
    
    # People count display
    with col1:
        st.markdown('<div class="theme-card">', unsafe_allow_html=True)
        st.markdown('<p class="medium-text people-label-style">PEOPLE IN QUEUE</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="huge-number people-count-style">{people_count}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Wait time display
    with col2:
        st.markdown(f'<div class="{bg_class}">', unsafe_allow_html=True)
        st.markdown(f'<p class="medium-text wait-time-label-style">WAIT TIME (in minutes)</p>', unsafe_allow_html=True)
        if wait_time is not None and wait_time > 0:
            st.markdown(f'<p class="huge-number wait-time-value-style">{int(wait_time)}</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="huge-text wait-time-value-style">NO WAIT</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    

    
    # JavaScript for automatic fullscreen
    st.markdown("""
    <script>
    // Auto fullscreen function
    function enterFullscreen() {
        if (document.documentElement.requestFullscreen) {
            document.documentElement.requestFullscreen();
        } else if (document.documentElement.webkitRequestFullscreen) {
            document.documentElement.webkitRequestFullscreen();
        } else if (document.documentElement.msRequestFullscreen) {
            document.documentElement.msRequestFullscreen();
        }
    }
    
    // Try to enter fullscreen immediately
    setTimeout(function() {
        if (!document.fullscreenElement) {
            enterFullscreen();
        }
    }, 1000);
    
    // Also try on any user interaction
    document.addEventListener('click', function() {
        if (!document.fullscreenElement) {
            enterFullscreen();
        }
    }, { once: true });
    
    // Hide cursor after 3 seconds of inactivity
    let cursorTimeout;
    document.addEventListener('mousemove', function() {
        document.body.style.cursor = 'default';
        clearTimeout(cursorTimeout);
        cursorTimeout = setTimeout(function() {
            document.body.style.cursor = 'none';
        }, 3000);
    });
    
    // Prevent right-click context menu
    document.addEventListener('contextmenu', function(e) {
        e.preventDefault();
    });
    
    // Prevent F5 refresh, Ctrl+R, etc.
    document.addEventListener('keydown', function(e) {
        if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
            e.preventDefault();
        }
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Auto refresh
    st.rerun()

if __name__ == "__main__":
    main()