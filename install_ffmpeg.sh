#!/bin/bash

# FFmpeg Installation Script for Meeting Transcription Suite
# Supports multiple installation methods for different environments

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check FFmpeg installation
check_ffmpeg() {
    if command_exists ffmpeg && command_exists ffprobe; then
        print_success "FFmpeg is already installed!"
        echo "FFmpeg version:"
        ffmpeg -version | head -n 1
        echo "FFprobe version:"
        ffprobe -version | head -n 1
        return 0
    else
        return 1
    fi
}

# Function to install via apt (Ubuntu/Debian)
install_apt() {
    print_status "Installing FFmpeg via apt (Ubuntu/Debian)..."
    
    if ! command_exists apt; then
        print_error "apt package manager not found. This method requires Ubuntu/Debian."
        return 1
    fi
    
    # Update package list
    print_status "Updating package list..."
    sudo apt update
    
    # Install FFmpeg
    print_status "Installing ffmpeg package..."
    sudo apt install -y ffmpeg
    
    return 0
}

# Function to install via Homebrew (Linux)
install_homebrew() {
    print_status "Installing FFmpeg via Homebrew..."
    
    if ! command_exists brew; then
        print_error "Homebrew not found. Please install Homebrew first or try another method."
        return 1
    fi
    
    # Install FFmpeg
    print_status "Installing ffmpeg via brew..."
    brew install ffmpeg
    
    return 0
}

# Function to install via snap
install_snap() {
    print_status "Installing FFmpeg via snap..."
    
    if ! command_exists snap; then
        print_error "Snap package manager not found."
        return 1
    fi
    
    # Install FFmpeg
    print_status "Installing ffmpeg via snap..."
    sudo snap install ffmpeg
    
    return 0
}

# Function to install from static build
install_static() {
    print_status "Installing FFmpeg from static build..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            FFMPEG_ARCH="amd64"
            ;;
        aarch64|arm64)
            FFMPEG_ARCH="arm64"
            ;;
        *)
            print_error "Unsupported architecture: $ARCH"
            return 1
            ;;
    esac
    
    # Download static build
    FFMPEG_URL="https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-${FFMPEG_ARCH}-static.tar.xz"
    print_status "Downloading FFmpeg static build from $FFMPEG_URL..."
    
    if command_exists wget; then
        wget -q "$FFMPEG_URL" -O ffmpeg-static.tar.xz
    elif command_exists curl; then
        curl -L -s "$FFMPEG_URL" -o ffmpeg-static.tar.xz
    else
        print_error "Neither wget nor curl found. Cannot download static build."
        return 1
    fi
    
    # Extract
    print_status "Extracting FFmpeg..."
    tar -xf ffmpeg-static.tar.xz
    
    # Find extracted directory
    FFMPEG_DIR=$(find . -name "ffmpeg-*-static" -type d | head -n 1)
    if [ -z "$FFMPEG_DIR" ]; then
        print_error "Failed to find extracted FFmpeg directory"
        return 1
    fi
    
    # Install to /usr/local/bin
    print_status "Installing FFmpeg binaries to /usr/local/bin..."
    sudo cp "$FFMPEG_DIR/ffmpeg" /usr/local/bin/
    sudo cp "$FFMPEG_DIR/ffprobe" /usr/local/bin/
    sudo chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe
    
    # Clean up
    cd - > /dev/null
    rm -rf "$TEMP_DIR"
    
    return 0
}

# Function to verify installation
verify_installation() {
    print_status "Verifying FFmpeg installation..."
    
    if check_ffmpeg; then
        print_success "FFmpeg installation verified successfully!"
        
        # Test basic functionality
        print_status "Testing FFmpeg functionality..."
        
        # Create a test audio file and check if FFmpeg can process it
        if ffmpeg -f lavfi -i "sine=frequency=440:duration=1" -f null - &>/dev/null; then
            print_success "FFmpeg basic functionality test passed!"
        else
            print_warning "FFmpeg installed but basic functionality test failed."
        fi
        
        return 0
    else
        print_error "FFmpeg installation verification failed!"
        return 1
    fi
}

# Function to show usage information
show_usage() {
    echo "FFmpeg Installation Script"
    echo ""
    echo "Usage: $0 [method]"
    echo ""
    echo "Methods:"
    echo "  apt       Install via apt package manager (Ubuntu/Debian)"
    echo "  brew      Install via Homebrew"
    echo "  snap      Install via snap package manager"
    echo "  static    Install from static build (universal)"
    echo "  auto      Automatically choose best method (default)"
    echo ""
    echo "Examples:"
    echo "  $0 apt      # Install via apt"
    echo "  $0 auto     # Auto-detect and install"
    echo "  $0          # Same as auto"
}

# Main installation function
main() {
    echo "FFmpeg Installation Script for Meeting Transcription Suite"
    echo "========================================================"
    
    # Check if already installed
    if check_ffmpeg; then
        echo ""
        echo "FFmpeg is already installed and working. No action needed."
        exit 0
    fi
    
    # Parse arguments
    METHOD=${1:-auto}
    
    case $METHOD in
        -h|--help|help)
            show_usage
            exit 0
            ;;
        apt)
            if install_apt && verify_installation; then
                print_success "FFmpeg installation completed via apt!"
                exit 0
            else
                print_error "Failed to install FFmpeg via apt."
                exit 1
            fi
            ;;
        brew|homebrew)
            if install_homebrew && verify_installation; then
                print_success "FFmpeg installation completed via Homebrew!"
                exit 0
            else
                print_error "Failed to install FFmpeg via Homebrew."
                exit 1
            fi
            ;;
        snap)
            if install_snap && verify_installation; then
                print_success "FFmpeg installation completed via snap!"
                exit 0
            else
                print_error "Failed to install FFmpeg via snap."
                exit 1
            fi
            ;;
        static)
            if install_static && verify_installation; then
                print_success "FFmpeg installation completed via static build!"
                exit 0
            else
                print_error "Failed to install FFmpeg via static build."
                exit 1
            fi
            ;;
        auto)
            print_status "Auto-detecting best installation method..."
            
            # Try methods in order of preference
            if command_exists apt; then
                print_status "Detected apt package manager, trying apt installation..."
                if install_apt && verify_installation; then
                    print_success "FFmpeg installation completed via apt!"
                    exit 0
                fi
                print_warning "apt installation failed, trying next method..."
            fi
            
            if command_exists brew; then
                print_status "Detected Homebrew, trying brew installation..."
                if install_homebrew && verify_installation; then
                    print_success "FFmpeg installation completed via Homebrew!"
                    exit 0
                fi
                print_warning "Homebrew installation failed, trying next method..."
            fi
            
            if command_exists snap; then
                print_status "Detected snap, trying snap installation..."
                if install_snap && verify_installation; then
                    print_success "FFmpeg installation completed via snap!"
                    exit 0
                fi
                print_warning "snap installation failed, trying next method..."
            fi
            
            print_status "Trying static build installation..."
            if install_static && verify_installation; then
                print_success "FFmpeg installation completed via static build!"
                exit 0
            fi
            
            print_error "All installation methods failed!"
            print_error "Please install FFmpeg manually or check your system configuration."
            exit 1
            ;;
        *)
            print_error "Unknown method: $METHOD"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"