# Face Distance Calculator - Replit Project

## Overview

This is a professional Streamlit web application designed for face distance calculation and identity verification. The application serves medical, construction, and corporate environments by providing a clean interface for comparing face encodings and determining similarity between reference and comparison images.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework
- **Interface**: Single-page application with dual image upload functionality
- **Responsive Design**: Mobile and desktop compatible
- **Theme**: Professional blue color scheme with clean white background

### Backend Architecture
- **Runtime**: Python 3.11
- **Core Libraries**: 
  - Streamlit for web interface
  - PIL (Pillow) for image processing
  - NumPy for numerical computations
- **Modular Design**: Separate utility module for face recognition functions

### Application Structure
- `app.py`: Main Streamlit application with UI components
- `face_recognition_utils.py`: Core face recognition and validation utilities
- `.streamlit/config.toml`: Streamlit server and theme configuration

## Key Components

### Image Processing Pipeline
1. **Image Upload**: Dual file uploaders for reference and comparison images
2. **Validation**: Comprehensive image validation including format, size, and content checks
3. **Preprocessing**: Image normalization and preparation for face recognition
4. **Face Recognition**: Face encoding extraction and distance calculation
5. **Results Display**: Professional presentation of comparison results

### Validation System
- Format validation (JPEG, PNG support)
- Size constraints (50x50 to 5000x5000 pixels)
- File size limits (10MB maximum)
- Image mode validation (RGB, RGBA, grayscale)

### User Interface Components
- Header with application branding
- Use case descriptions for context
- Two-column layout for image comparison
- Session state management for results persistence
- Progress indicators and error handling

## Data Flow

1. **Image Upload**: Users upload reference and comparison images
2. **Client-Side Validation**: Immediate validation feedback
3. **Image Processing**: Server-side preprocessing and face detection
4. **Distance Calculation**: Euclidean distance computation between face encodings
5. **Results Presentation**: Formatted output with confidence scores and recommendations

## External Dependencies

### Python Packages
- `streamlit>=1.46.0`: Web application framework
- `pillow>=11.2.1`: Image processing library
- `numpy>=2.3.1`: Numerical computations

### System Dependencies (Nix)
- `freetype`: Font rendering
- `lcms2`: Color management
- `libimagequant`: Image quantization
- `libjpeg`, `libtiff`, `libwebp`: Image format support
- `openjpeg`: JPEG 2000 support
- `tcl`, `tk`: GUI toolkit components
- `zlib`: Compression library

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Nix package manager
- **Port**: 5000 (configured for both development and production)
- **Auto-scaling**: Configured for autoscale deployment target

### Production Considerations
- Headless server configuration
- Network accessible on all interfaces (0.0.0.0)
- Optimized for container deployment
- Session state management for user experience

### Workflow Configuration
- Parallel execution mode
- Automated port detection and forwarding
- Shell-based execution for flexibility

## Changelog

```
Changelog:
- June 22, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```

### Notes for Code Agents

1. **Face Recognition Implementation**: The `face_recognition_utils.py` module contains placeholder implementations that need to be replaced with actual face recognition models
2. **Database Integration**: The application currently operates without persistent storage - consider adding database functionality for user management or audit trails
3. **Security Considerations**: Implement proper image sanitization and consider adding authentication for production use
4. **Performance Optimization**: Consider adding caching mechanisms for repeated face encodings
5. **Error Handling**: Robust validation is in place, but consider adding logging for production monitoring

The application follows a clean separation of concerns with UI logic in the main app file and business logic in utility modules. The architecture supports easy extension for additional features like batch processing, API endpoints, or database integration.