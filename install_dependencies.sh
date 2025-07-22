#!/bin/bash

# Dependencies Installation Script for Meeting Transcription Suite
# Installs FFmpeg and other required system dependencies for audio transcription
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

# Function to check Python lzma module
check_lzma() {
    if python3 -c "import lzma" 2>/dev/null; then
        print_success "Python lzma module is available!"
        return 0
    else
        print_warning "Python lzma module not available (may cause _lzma import errors)"
        return 1
    fi
}

# Function to check libsndfile
check_libsndfile() {
    # Try to check if librosa can load audio files properly
    if python3 -c "import soundfile" 2>/dev/null; then
        print_success "Python soundfile module is available (libsndfile is working)!"
        return 0
    else
        print_warning "Python soundfile module not available (librosa may use audioread fallback)"
        return 1
    fi
}

# Function to install libsndfile
install_libsndfile() {
    print_status "Installing libsndfile for audio processing support..."
    
    if command_exists apt; then
        sudo apt install -y libsndfile1
        print_success "libsndfile1 installed via apt"
        return 0
    elif command_exists yum; then
        sudo yum install -y libsndfile
        print_success "libsndfile installed via yum"
        return 0
    elif command_exists dnf; then
        sudo dnf install -y libsndfile
        print_success "libsndfile installed via dnf"
        return 0
    elif command_exists brew; then
        brew install libsndfile
        print_success "libsndfile installed via brew"
        return 0
    else
        print_warning "Could not install libsndfile automatically. Please install manually:"
        print_warning "  Ubuntu/Debian: sudo apt install libsndfile1"
        print_warning "  CentOS/RHEL: sudo yum install libsndfile"
        print_warning "  macOS: brew install libsndfile"
        return 1
    fi
}

# Function to install liblzma-dev
install_lzma_dev() {
    print_status "Installing liblzma-dev for Python lzma module support..."
    
    if command_exists apt; then
        sudo apt install -y liblzma-dev
        print_success "liblzma-dev installed via apt"
        return 0
    elif command_exists yum; then
        sudo yum install -y xz-devel
        print_success "xz-devel installed via yum"
        return 0
    elif command_exists dnf; then
        sudo dnf install -y xz-devel
        print_success "xz-devel installed via dnf"
        return 0
    elif command_exists brew; then
        brew install xz
        print_success "xz installed via brew"
        return 0
    else
        print_warning "Could not install liblzma-dev automatically. Please install manually:"
        print_warning "  Ubuntu/Debian: sudo apt install liblzma-dev"
        print_warning "  CentOS/RHEL: sudo yum install xz-devel"
        print_warning "  macOS: brew install xz"
        return 1
    fi
}

# Function to install via apt (Ubuntu/Debian)
install_apt() {
    print_status "Installing dependencies via apt (Ubuntu/Debian)..."
    
    if ! command_exists apt; then
        print_error "apt package manager not found. This method requires Ubuntu/Debian."
        return 1
    fi
    
    # Update package list
    print_status "Updating package list..."
    sudo apt update
    
    # Install FFmpeg, audio libraries, and lzma development libraries
    print_status "Installing ffmpeg, libsndfile1, and liblzma-dev packages..."
    sudo apt install -y ffmpeg libsndfile1 liblzma-dev
    
    return 0
}

# Function to install via Homebrew (Linux/macOS)
install_homebrew() {
    print_status "Installing dependencies via Homebrew..."
    
    if ! command_exists brew; then
        print_error "Homebrew not found. Please install Homebrew first or try another method."
        return 1
    fi
    
    # Install FFmpeg, audio libraries, and xz (for lzma support)
    print_status "Installing ffmpeg, libsndfile, and xz via brew..."
    brew install ffmpeg libsndfile xz
    
    return 0
}

# Function to install via snap
install_snap() {
    print_status "Installing dependencies via snap..."
    
    if ! command_exists snap; then
        print_error "Snap package manager not found."
        return 1
    fi
    
    # Install FFmpeg
    print_status "Installing ffmpeg via snap..."
    sudo snap install ffmpeg
    
    # Note: snap doesn't provide development libraries, so install them separately if possible
    if ! install_lzma_dev; then
        print_warning "Could not install liblzma-dev via snap. You may need to install it manually."
    fi
    
    if ! install_libsndfile; then
        print_warning "Could not install libsndfile via snap. You may need to install it manually."
    fi
    
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
    
    # Try to install liblzma-dev separately
    if ! install_lzma_dev; then
        print_warning "Could not install liblzma-dev. You may need to install it manually."
    fi
    
    # Try to install libsndfile separately
    if ! install_libsndfile; then
        print_warning "Could not install libsndfile. You may need to install it manually."
    fi
    
    return 0
}

# Function to verify installation
verify_installation() {
    print_status "Verifying dependencies installation..."
    
    VERIFICATION_FAILED=0
    
    # Check FFmpeg
    if check_ffmpeg; then
        print_success "FFmpeg installation verified successfully!"
        
        # Test basic functionality
        print_status "Testing FFmpeg functionality..."
        
        # Create a test audio file and check if FFmpeg can process it
        if ffmpeg -f lavfi -i "sine=frequency=440:duration=1" -f null - &>/dev/null; then
            print_success "FFmpeg basic functionality test passed!"
        else
            print_warning "FFmpeg installed but basic functionality test failed."
            VERIFICATION_FAILED=1
        fi
    else
        print_error "FFmpeg installation verification failed!"
        VERIFICATION_FAILED=1
    fi
    
    # Check lzma module
    print_status "Checking Python lzma module..."
    if check_lzma; then
        print_success "Python lzma module verification passed!"
    else
        print_warning "Python lzma module not available. This may cause '_lzma' import errors."
        print_warning "You may need to rebuild Python after installing liblzma-dev."
        # Don't fail verification for lzma, just warn
    fi
    
    # Check libsndfile
    print_status "Checking libsndfile/soundfile module..."
    if check_libsndfile; then
        print_success "Python soundfile module verification passed!"
    else
        print_warning "Python soundfile module not available. Librosa will use audioread fallback."
        print_warning "You may need to: pip install soundfile"
        # Don't fail verification for soundfile, just warn
    fi
    
    if [ $VERIFICATION_FAILED -eq 0 ]; then
        print_success "Dependencies installation verification completed!"
        return 0
    else
        print_error "Some dependencies failed verification!"
        return 1
    fi
}

# Function to show usage information
show_usage() {
    echo "Dependencies Installation Script for Meeting Transcription Suite"
    echo ""
    echo "This script installs required system dependencies:"
    echo "  - FFmpeg (for audio/video processing)"
    echo "  - libsndfile (for audio file loading with librosa)"
    echo "  - liblzma-dev (for Python lzma module support)"
    echo ""
    echo "Usage: $0 [method]"
    echo ""
    echo "Methods:"
    echo "  apt       Install via apt package manager (Ubuntu/Debian)"
    echo "  brew      Install via Homebrew (Linux/macOS)"
    echo "  snap      Install via snap package manager"
    echo "  static    Install from static build (universal)"
    echo "  auto      Automatically choose best method (default)"
    echo ""
    echo "Examples:"
    echo "  $0 apt      # Install via apt"
    echo "  $0 auto     # Auto-detect and install"
    echo "  $0          # Same as auto"
    echo ""
    echo "Note: After installation, you may need to rebuild Python if lzma module"
    echo "      is not available (check with: python3 -c 'import lzma')"
}

# Main installation function
main() {
    echo "Dependencies Installation Script for Meeting Transcription Suite"
    echo "================================================================"
    
    # Check if already installed
    FFMPEG_OK=0
    LZMA_OK=0
    LIBSNDFILE_OK=0
    
    if check_ffmpeg; then
        FFMPEG_OK=1
    fi
    
    if check_lzma; then
        LZMA_OK=1
    fi
    
    if check_libsndfile; then
        LIBSNDFILE_OK=1
    fi
    
    if [ $FFMPEG_OK -eq 1 ] && [ $LZMA_OK -eq 1 ] && [ $LIBSNDFILE_OK -eq 1 ]; then
        echo ""
        echo "All dependencies are already installed and working. No action needed."
        exit 0
    elif [ $FFMPEG_OK -eq 1 ]; then
        echo ""
        print_warning "FFmpeg is installed but some Python modules may be missing."
        NEED_INSTALL=0
        
        if [ $LZMA_OK -eq 0 ]; then
            print_status "Attempting to install liblzma-dev..."
            if install_lzma_dev; then
                print_success "liblzma-dev installed. You may need to rebuild Python."
                print_warning "If you still get '_lzma' errors, try: pyenv install [version] (if using pyenv)"
            fi
            NEED_INSTALL=1
        fi
        
        if [ $LIBSNDFILE_OK -eq 0 ]; then
            print_status "Attempting to install libsndfile..."
            if install_libsndfile; then
                print_success "libsndfile installed. You may need to reinstall soundfile: pip install --force-reinstall soundfile"
            fi
            NEED_INSTALL=1
        fi
        
        if [ $NEED_INSTALL -eq 1 ]; then
            exit 0
        fi
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
                print_success "Dependencies installation completed via apt!"
                exit 0
            else
                print_error "Failed to install dependencies via apt."
                exit 1
            fi
            ;;
        brew|homebrew)
            if install_homebrew && verify_installation; then
                print_success "Dependencies installation completed via Homebrew!"
                exit 0
            else
                print_error "Failed to install dependencies via Homebrew."
                exit 1
            fi
            ;;
        snap)
            if install_snap && verify_installation; then
                print_success "Dependencies installation completed via snap!"
                exit 0
            else
                print_error "Failed to install dependencies via snap."
                exit 1
            fi
            ;;
        static)
            if install_static && verify_installation; then
                print_success "Dependencies installation completed via static build!"
                exit 0
            else
                print_error "Failed to install dependencies via static build."
                exit 1
            fi
            ;;
        auto)
            print_status "Auto-detecting best installation method..."
            
            # Try methods in order of preference
            if command_exists apt; then
                print_status "Detected apt package manager, trying apt installation..."
                if install_apt && verify_installation; then
                    print_success "Dependencies installation completed via apt!"
                    exit 0
                fi
                print_warning "apt installation failed, trying next method..."
            fi
            
            if command_exists brew; then
                print_status "Detected Homebrew, trying brew installation..."
                if install_homebrew && verify_installation; then
                    print_success "Dependencies installation completed via Homebrew!"
                    exit 0
                fi
                print_warning "Homebrew installation failed, trying next method..."
            fi
            
            if command_exists snap; then
                print_status "Detected snap, trying snap installation..."
                if install_snap && verify_installation; then
                    print_success "Dependencies installation completed via snap!"
                    exit 0
                fi
                print_warning "snap installation failed, trying next method..."
            fi
            
            print_status "Trying static build installation..."
            if install_static && verify_installation; then
                print_success "Dependencies installation completed via static build!"
                exit 0
            fi
            
            print_error "All installation methods failed!"
            print_error "Please install dependencies manually or check your system configuration."
            print_error "Required dependencies: ffmpeg, liblzma-dev (or xz-devel)"
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